from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    from scipy.interpolate import InterpolatedUnivariateSpline
    HAS_SCIPY = True
except Exception:
    InterpolatedUnivariateSpline = None
    HAS_SCIPY = False


RAW_W = 1296
RAW_H = 972
CUT_HEIGHT = 445
IMG_W = 800
IMG_H = 320
NUM_POINTS = 72
N_STRIPS = NUM_POINTS - 1
N_OFFSETS = NUM_POINTS
NUM_PRIORS = 192
OUTPUT_DIM = 78
SAMPLE_Y = list(range(971, 444, -20))
IMAGE_CENTER_X = RAW_W / 2.0
DEFAULT_PAIR_BONUS_PX = 60.0
DEFAULT_PI_ORT_THREADS = 4


def fs_path(path):
    p = Path(path)
    s = str(p.resolve())
    if sys.platform.startswith("win") and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def read_json(path):
    with open(fs_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def imwrite_bgr(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buf.tofile(fs_path(path))
    return path


def is_pi():
    machine = platform.machine().lower()
    return machine in {"aarch64", "armv7l", "armv6l"} or "raspberry" in platform.platform().lower()


def load_package_contracts(package_root):
    package_root = Path(package_root)
    decode_contract = read_json(package_root / "c" / "decode_contract_v1.json")
    driving_contract = read_json(package_root / "c" / "driving_contract_v1.json")
    driving_contract["driving_contract"]["params"].setdefault("pair_bonus_px", DEFAULT_PAIR_BONUS_PX)
    model_info = read_json(package_root / "m" / "model_path.json")
    return decode_contract, driving_contract, model_info


def update_driving_params(driving_contract, overrides):
    if not overrides:
        return driving_contract
    pp = driving_contract["driving_contract"]["params"]
    unknown = sorted(set(overrides) - set(pp))
    if unknown:
        raise KeyError(f"Unknown drive_postprocess_overrides keys: {unknown}")
    for key, value in overrides.items():
        pp[key] = value
    return driving_contract


def make_ort_session(package_root, intra_op_num_threads=None):
    package_root = Path(package_root)
    _, _, model_info = load_package_contracts(package_root)
    model_path = package_root / model_info["onnx_rel"]
    data_path = package_root / model_info["external_data_rel"]
    if not os.path.exists(fs_path(model_path)):
        raise FileNotFoundError(model_path)
    if not os.path.exists(fs_path(data_path)):
        raise FileNotFoundError(data_path)

    if intra_op_num_threads is None and is_pi():
        intra_op_num_threads = DEFAULT_PI_ORT_THREADS
    so = ort.SessionOptions()
    if intra_op_num_threads is not None:
        so.intra_op_num_threads = int(intra_op_num_threads)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(fs_path(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name


def preprocess_bgr_for_model(bgr):
    if bgr.shape[0] != RAW_H or bgr.shape[1] != RAW_W:
        bgr = cv2.resize(bgr, (RAW_W, RAW_H), interpolation=cv2.INTER_LINEAR)
    crop = bgr[int(CUT_HEIGHT):, :, :]
    resized = cv2.resize(crop, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
    if tensor.shape != (1, 3, IMG_H, IMG_W):
        raise AssertionError(tensor.shape)
    return tensor


def run_onnx_raw(session, input_name, output_name, bgr):
    inp = preprocess_bgr_for_model(bgr)
    out = session.run([output_name], {input_name: inp})[0]
    if out.shape != (1, NUM_PRIORS, OUTPUT_DIM):
        raise AssertionError(out.shape)
    return out[0].astype(np.float32)


def warmup_ort_session(session, input_name, output_name, bgr, runs=3):
    inp = preprocess_bgr_for_model(bgr)
    for _ in range(int(runs)):
        _ = session.run([output_name], {input_name: inp})[0]


def softmax_positive_score(logits_2):
    logits = logits_2.astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=1)


def official_cuda_suppresses(a, b, threshold):
    start_a = int(float(a[2]) * N_STRIPS + 0.5)
    start_b = int(float(b[2]) * N_STRIPS + 0.5)
    start = max(start_a, start_b)
    len_a, len_b = float(a[4]), float(b[4])
    end_a = int(start_a + len_a - 1 + 0.5 - (1 if (len_a - 1) < 0 else 0))
    end_b = int(start_b + len_b - 1 + 0.5 - (1 if (len_b - 1) < 0 else 0))
    end = min(end_a, end_b, N_OFFSETS - 1)
    if end < start:
        return False
    dist = float(np.abs(a[5 + start:5 + end + 1] - b[5 + start:5 + end + 1]).sum())
    return dist < float(threshold) * float(end - start + 1)


def official_overlap_nms(nms_predictions, scores, nms_thres, top_k):
    order = np.argsort(-scores)
    kept = []
    for idx in order:
        duplicate = any(official_cuda_suppresses(nms_predictions[idx], nms_predictions[j], nms_thres) for j in kept)
        if not duplicate:
            kept.append(int(idx))
        if len(kept) >= int(top_k):
            break
    return kept


def resample_lane_xs(xs, ys, query_ys):
    if len(xs) < 2:
        return np.full_like(query_ys, -2.0, dtype=np.float32)
    order = np.argsort(ys)
    ys_sorted, xs_sorted = ys[order], xs[order]
    if HAS_SCIPY:
        spline = InterpolatedUnivariateSpline(ys_sorted, xs_sorted, k=min(3, len(xs_sorted) - 1))
        out = spline(query_ys).astype(np.float32)
    else:
        out = np.interp(query_ys, ys_sorted, xs_sorted).astype(np.float32)
    min_y, max_y = float(ys_sorted.min()) - 0.01, float(ys_sorted.max()) + 0.01
    out[(query_ys < min_y) | (query_ys > max_y)] = -2.0
    return out


def prediction_to_lane_array(prediction, confidence):
    pred = prediction.copy().astype(np.float32)
    lane_xs = pred[6:].copy()
    start = min(max(0, int(round(float(pred[2]) * N_STRIPS))), N_STRIPS)
    length = int(round(float(pred[5])))
    end = min(start + length - 1, N_OFFSETS - 1)
    lane_xs[end + 1:] = -2.0
    if start > 0:
        valid_prefix = ((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)).astype(np.int32)
        mask = ~(valid_prefix[::-1].cumprod()[::-1].astype(bool))
        prefix = lane_xs[:start].copy()
        prefix[mask] = -2.0
        lane_xs[:start] = prefix

    prior_ys = np.linspace(1.0, 0.0, N_OFFSETS, dtype=np.float32)
    valid = lane_xs >= 0.0
    if int(valid.sum()) <= 1:
        return None
    xs_norm = lane_xs[valid][::-1].astype(np.float32)
    ys_crop_norm = prior_ys[valid][::-1].astype(np.float32)
    ys_norm = (ys_crop_norm * float(RAW_H - CUT_HEIGHT) + float(CUT_HEIGHT)) / float(RAW_H)
    sample_ys_norm = np.array(SAMPLE_Y, dtype=np.float32) / float(RAW_H)
    sample_xs_norm = resample_lane_xs(xs_norm, ys_norm, sample_ys_norm)
    valid_sample = (sample_xs_norm >= 0.0) & (sample_xs_norm < 1.0)
    if int(valid_sample.sum()) <= 1:
        return None
    pts = np.stack([sample_xs_norm[valid_sample] * float(RAW_W), sample_ys_norm[valid_sample] * float(RAW_H)], axis=1)
    return {"points": pts.astype(np.float32), "conf": float(confidence)}


def decode_raw_to_lanes(raw_predictions, decode_contract):
    dc = decode_contract.get("decoder_contract", decode_contract)
    conf_threshold = float(dc["conf_threshold"])
    nms_thres = float(dc["nms_thres"])
    nms_topk = int(dc["nms_topk"])
    predictions = raw_predictions.astype(np.float32)
    scores = softmax_positive_score(predictions[:, :2])
    keep_mask = scores >= conf_threshold
    if int(keep_mask.sum()) == 0:
        return []
    pred_kept, score_kept = predictions[keep_mask].copy(), scores[keep_mask].copy()
    nms_predictions = np.concatenate([pred_kept[:, :4], pred_kept[:, 5:]], axis=1).astype(np.float32)
    nms_predictions[:, 4] *= float(N_STRIPS)
    nms_predictions[:, 5:] *= float(IMG_W - 1)
    keep = official_overlap_nms(nms_predictions, score_kept, nms_thres, nms_topk)
    selected, selected_scores = pred_kept[keep].copy(), score_kept[keep].copy()
    selected[:, 5] = np.round(selected[:, 5] * float(N_STRIPS))
    lanes = []
    for pred, score in zip(selected, selected_scores):
        lane = prediction_to_lane_array(pred, score)
        if lane is not None:
            lanes.append(lane)
    return lanes


def interp_or_nearest_x(points, y, max_y_distance):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    order = np.argsort(ys)
    ys, xs = ys[order], xs[order]
    if float(ys.min()) <= y <= float(ys.max()) and len(pts) >= 2:
        return float(np.interp(y, ys, xs))
    nearest_idx = int(np.argmin(np.abs(ys - y)))
    if abs(float(ys[nearest_idx]) - float(y)) <= float(max_y_distance):
        return float(xs[nearest_idx])
    return None


def feature_from_lane(lane, pp):
    pts = np.asarray(lane["points"], dtype=np.float32)
    if len(pts) < int(pp["min_points"]):
        return None
    if float(pts[:, 1].max() - pts[:, 1].min()) < float(pp["min_y_span"]):
        return None
    y_top = float((RAW_H - 1) * pp["top_ratio"])
    y_primary = float((RAW_H - 1) * pp["primary_ratio"])
    y_bottom = float((RAW_H - 1) * pp["bottom_ratio"])
    x_top = interp_or_nearest_x(pts, y_top, pp["max_y_distance"])
    x_primary = interp_or_nearest_x(pts, y_primary, pp["max_y_distance"])
    x_bottom = interp_or_nearest_x(pts, y_bottom, pp["max_y_distance"])
    if x_top is None or x_primary is None or x_bottom is None:
        return None
    return {
        "x_top": x_top,
        "x_primary": x_primary,
        "x_bottom": x_bottom,
        "heading": (x_top - x_bottom) / max(1.0, y_bottom - y_top),
        "y_top": y_top,
        "y_primary": y_primary,
        "y_bottom": y_bottom,
    }


def estimate_steer(center_x, heading, pp):
    pos_error = (float(center_x) - IMAGE_CENTER_X) / IMAGE_CENTER_X
    raw = float(pp["steer_gain"]) * (float(pp["k_pos"]) * pos_error + float(pp["k_heading"]) * float(heading))
    return float(np.clip(raw, -float(pp["max_steer_norm"]), float(pp["max_steer_norm"])))


def trajectory_from_pair(left, right):
    center_top = 0.5 * (left["x_top"] + right["x_top"])
    center_primary = 0.5 * (left["x_primary"] + right["x_primary"])
    center_bottom = 0.5 * (left["x_bottom"] + right["x_bottom"])
    return {
        "center_primary": center_primary,
        "heading": (center_top - center_bottom) / max(1.0, left["y_bottom"] - left["y_top"]),
        "mode": "both_stable",
    }


def trajectory_from_single(feat, side, pp):
    sign = 1.0 if side == "left" else -1.0
    offset = float(pp["safe_offset_px"])
    center_top = feat["x_top"] + sign * offset
    center_primary = feat["x_primary"] + sign * offset
    center_bottom = feat["x_bottom"] + sign * offset
    return {
        "center_primary": center_primary,
        "heading": (center_top - center_bottom) / max(1.0, feat["y_bottom"] - feat["y_top"]),
        "mode": f"single_{side}",
        "source_x": feat["x_primary"],
    }


def init_drive_memory():
    return {
        "smoothed_center_x": IMAGE_CENTER_X,
        "smoothed_heading": 0.0,
        "last_steer_norm": 0.0,
        "turn_bias": 0.0,
        "lost_frames": 0,
        "unstable_frames": 0,
    }


def choose_trajectory(features, memory, pp):
    if not features:
        return None
    lefts = [f for f in features if f["x_primary"] < IMAGE_CENTER_X]
    rights = [f for f in features if f["x_primary"] >= IMAGE_CENTER_X]
    left = max(lefts, key=lambda f: f["x_primary"]) if lefts else None
    right = min(rights, key=lambda f: f["x_primary"]) if rights else None
    candidates = []
    if left is not None and right is not None:
        gap = right["x_primary"] - left["x_primary"]
        if float(pp["gap_min_px"]) <= gap <= float(pp["gap_max_px"]):
            candidates.append(trajectory_from_pair(left, right))
    for feat in features:
        for side in ["left", "right"]:
            cand = trajectory_from_single(feat, side, pp)
            penalty = 0.0
            if feat["x_primary"] < IMAGE_CENTER_X - float(pp["single_side_deadband_px"]) and side == "right":
                penalty = float(pp["side_prior_penalty_px"])
            if feat["x_primary"] > IMAGE_CENTER_X + float(pp["single_side_deadband_px"]) and side == "left":
                penalty = float(pp["side_prior_penalty_px"])
            cand["side_penalty"] = penalty
            candidates.append(cand)
    prev_center = memory.get("smoothed_center_x", IMAGE_CENTER_X)
    prev_heading = memory.get("smoothed_heading", 0.0)
    prev_steer = memory.get("last_steer_norm", 0.0)

    def cost(c):
        center_cost = abs(c["center_primary"] - prev_center)
        heading_cost = abs(c["heading"] - prev_heading) * float(pp["heading_cost_px"])
        steer_cost = abs(estimate_steer(c["center_primary"], c["heading"], pp) - prev_steer) * float(pp["steer_cost_px"])
        pair_bonus = -float(pp.get("pair_bonus_px", DEFAULT_PAIR_BONUS_PX)) if c["mode"] == "both_stable" else 0.0
        return center_cost + heading_cost + steer_cost + c.get("side_penalty", 0.0) + pair_bonus

    best = min(candidates, key=cost)
    best["candidate_count"] = len(candidates)
    best["candidate_cost"] = float(cost(best))
    return best


def update_drive(lanes, memory, driving_contract):
    pp = driving_contract["driving_contract"]["params"] if "driving_contract" in driving_contract else driving_contract["params"]
    features = [f for f in (feature_from_lane(l, pp) for l in lanes) if f is not None]
    measured = choose_trajectory(features, memory, pp)
    raw_mode = "lost" if measured is None else measured["mode"]
    if measured is None:
        memory["lost_frames"] += 1
        memory["unstable_frames"] = 0
        if memory["lost_frames"] <= int(pp["lost_short_frames"]):
            mode = "lost_short_recovery"
            target = memory["last_steer_norm"] * float(pp["recovery_decay"])
        else:
            mode = "lost_active_recovery"
            sign = float(np.sign(memory["turn_bias"])) or float(np.sign(memory["last_steer_norm"]))
            ramp = float(pp["recovery_ramp"]) * max(0, memory["lost_frames"] - int(pp["lost_short_frames"]))
            target = memory["last_steer_norm"] * float(pp["recovery_decay"]) + float(pp["recovery_gain"]) * memory["turn_bias"] + ramp * sign
        target = float(np.clip(target, -float(pp["max_steer_norm"]), float(pp["max_steer_norm"])))
        memory["smoothed_heading"] *= float(pp["lost_heading_decay"])
        steer = float(pp["steer_alpha"]) * target + (1.0 - float(pp["steer_alpha"])) * memory["last_steer_norm"]
        steer = float(np.clip(steer, -float(pp["max_steer_norm"]), float(pp["max_steer_norm"])))
        memory["last_steer_norm"] = steer
        memory["turn_bias"] = float(pp["turn_bias_alpha"]) * steer + (1.0 - float(pp["turn_bias_alpha"])) * memory["turn_bias"]
        return {
            "raw_mode": raw_mode,
            "effective_mode": mode,
            "feature_count": len(features),
            "candidate_count": 0,
            "measured_center_x": np.nan,
            "measured_heading": np.nan,
            "smoothed_center_x": memory["smoothed_center_x"],
            "smoothed_heading": memory["smoothed_heading"],
            "steer_norm": steer,
            "turn_bias": memory["turn_bias"],
            "lost_frames": memory["lost_frames"],
        }

    memory["lost_frames"] = 0
    center, heading = float(measured["center_primary"]), float(measured["heading"])
    center_jump = abs(center - memory["smoothed_center_x"])
    heading_jump = abs(heading - memory["smoothed_heading"])
    unstable = center_jump > float(pp["jump_center_px"]) or heading_jump > float(pp["jump_heading"])
    if unstable:
        memory["unstable_frames"] += 1
        blend = min(float(pp["jump_blend_max"]), float(pp["jump_blend_start"]) + float(pp["jump_blend_step"]) * max(0, memory["unstable_frames"] - 1))
        effective_center = (1.0 - blend) * memory["smoothed_center_x"] + blend * center
        effective_heading = (1.0 - blend) * memory["smoothed_heading"] + blend * heading
        mode = "unstable_blend"
    else:
        memory["unstable_frames"] = 0
        effective_center, effective_heading, mode = center, heading, measured["mode"]
    memory["smoothed_center_x"] = float(pp["center_alpha"]) * effective_center + (1.0 - float(pp["center_alpha"])) * memory["smoothed_center_x"]
    memory["smoothed_heading"] = float(pp["heading_alpha"]) * effective_heading + (1.0 - float(pp["heading_alpha"])) * memory["smoothed_heading"]
    target = estimate_steer(memory["smoothed_center_x"], memory["smoothed_heading"], pp)
    steer = float(pp["steer_alpha"]) * target + (1.0 - float(pp["steer_alpha"])) * memory["last_steer_norm"]
    steer = float(np.clip(steer, -float(pp["max_steer_norm"]), float(pp["max_steer_norm"])))
    memory["last_steer_norm"] = steer
    memory["turn_bias"] = float(pp["turn_bias_alpha"]) * steer + (1.0 - float(pp["turn_bias_alpha"])) * memory["turn_bias"]
    return {
        "raw_mode": raw_mode,
        "effective_mode": mode,
        "feature_count": len(features),
        "candidate_count": int(measured.get("candidate_count", 0)),
        "measured_center_x": center,
        "measured_heading": heading,
        "smoothed_center_x": memory["smoothed_center_x"],
        "smoothed_heading": memory["smoothed_heading"],
        "steer_norm": steer,
        "turn_bias": memory["turn_bias"],
        "lost_frames": memory["lost_frames"],
    }


def draw_overlay_frame(bgr, lanes, row, command=None, latency_ms=None, fps=None):
    img = bgr.copy()
    for lane in lanes:
        pts = np.asarray(lane["points"], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 120), 3, cv2.LINE_AA)
    center_x = float(row.get("smoothed_center_x", IMAGE_CENTER_X))
    heading = float(row.get("smoothed_heading", 0.0))
    steer = float(row.get("steer_norm", 0.0))
    y0, y1 = RAW_H - 30, int((RAW_H - 1) * 0.78)
    x0 = int(IMAGE_CENTER_X)
    x1 = int(np.clip(center_x + heading * 220.0, 0, RAW_W - 1))
    cv2.line(img, (x0, y0), (x1, y1), (0, 165, 255), 4, cv2.LINE_AA)
    steer_x = int(np.clip(IMAGE_CENTER_X + steer * 420.0, 0, RAW_W - 1))
    cv2.arrowedLine(img, (x0, y0 - 35), (steer_x, y1 - 40), (255, 0, 255), 4, cv2.LINE_AA, tipLength=0.18)
    text = f"mode={row.get('effective_mode', '')} steer={steer:+.3f}"
    if command is not None:
        text += f" L={command.get('left', 0):+.2f} R={command.get('right', 0):+.2f}"
    if latency_ms is not None:
        text += f" pipe={latency_ms:.1f}ms"
    if fps is not None:
        text += f" fps={fps:.2f}"
    cv2.rectangle(img, (18, 18), (1180, 82), (0, 0, 0), -1)
    cv2.putText(img, text, (32, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return img
