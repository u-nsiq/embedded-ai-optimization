"""stable_center_tangent lane postprocess.

08g 후보.

핵심:
  - 두 lane이 안정적이면 local center line을 만들고 center error + center tangent로 주행한다.
  - 두 lane이 불안정하면 center를 버리고 평균 tangent와 memory만 섞는다.
  - 한 lane만 안정적으로 보이면 그 lane의 local tangent만 잠시 따른다.
  - lost는 마지막 steer를 hold/decay한다.

push 기반 safety band가 없어서 08f보다 단순하고, 상태머신과도 덜 싸운다.
"""

from __future__ import annotations

import math
import numpy as np

from lane.geometry import HALF_W, IMAGE_CENTER_X, RAW_H
from .common import clamp, normalize_lane_points, x_at_y


def _effective_center_x(cfg):
    # 카메라가 물리적으로 좌/우를 향하도록 장착된 경우,
    # 화면 중앙 대신 "차체가 보는 중앙"을 별도로 둔다.
    # 양수는 기준 중앙을 화면 오른쪽으로, 음수는 왼쪽으로 옮긴다.
    return IMAGE_CENTER_X + float(cfg.get("camera_center_offset_ratio", 0.0)) * HALF_W


def init_memory(cfg=None):
    center_x = _effective_center_x(cfg or {})
    return {
        "last_steer": 0.0,
        "last_heading": 0.0,
        "last_center_x": center_x,
        "lost_frames": 0,
        "seen_lane": False,
    }


def _sample_feature(lane, cfg):
    pts = normalize_lane_points(lane)
    if len(pts) < int(cfg["min_points"]):
        return None

    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    if y_span < float(cfg["min_y_span_ratio"]) * RAW_H:
        return None

    near_y = float(cfg["near_y_ratio"]) * RAW_H
    mid_y = float(cfg["mid_y_ratio"]) * RAW_H
    far_y = float(cfg["far_y_ratio"]) * RAW_H
    query_ys = [near_y, mid_y, far_y]

    min_y = float(pts[:, 1].min())
    max_y = float(pts[:, 1].max())
    extrapolate_dist = 0.0
    for yq in query_ys:
        if yq < min_y:
            extrapolate_dist = max(extrapolate_dist, min_y - yq)
        elif yq > max_y:
            extrapolate_dist = max(extrapolate_dist, yq - max_y)
    if extrapolate_dist > float(cfg["max_extrapolate_ratio"]) * RAW_H:
        return None

    xs = [x_at_y(pts, yq) for yq in query_ys]
    if any(x is None for x in xs):
        return None

    x_near, x_mid, x_far = [float(x) for x in xs]
    heading = (x_far - x_near) / max(1.0, near_y - far_y)
    effective_center_x = _effective_center_x(cfg)
    center_dist_ratio = (x_mid - effective_center_x) / HALF_W
    span_score = clamp(y_span / (RAW_H * 0.28), 0.0, 1.0)
    point_score = clamp(len(pts) / 22.0, 0.0, 1.0)
    conf_score = clamp(float(lane.get("conf", 0.0)) / 0.85, 0.0, 1.0)
    quality = 0.40 * conf_score + 0.35 * span_score + 0.25 * point_score

    return {
        "points": pts,
        "conf": float(lane.get("conf", 0.0)),
        "x_near": x_near,
        "x_mid": x_mid,
        "x_far": x_far,
        "heading": float(heading),
        "center_dist_ratio": float(center_dist_ratio),
        "quality": float(clamp(quality, 0.0, 1.0)),
    }


def _features(lanes, cfg):
    feats = []
    for lane in lanes:
        feat = _sample_feature(lane, cfg)
        if feat is not None:
            feats.append(feat)
    feats.sort(key=lambda f: f["x_mid"])
    return feats


def _choose_pair(features, cfg):
    if len(features) < 2:
        return None
    min_gap = float(cfg["min_pair_gap_ratio"]) * (2.0 * HALF_W)
    max_gap = float(cfg["max_pair_gap_ratio"]) * (2.0 * HALF_W)
    candidates = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = features[i], features[j]
            gap_mid = abs(b["x_mid"] - a["x_mid"])
            if min_gap <= gap_mid <= max_gap:
                center_mid = 0.5 * (a["x_mid"] + b["x_mid"])
                effective_center_x = _effective_center_x(cfg)
                score = abs(center_mid - effective_center_x) - 120.0 * (a["quality"] + b["quality"])
                candidates.append((score, a, b))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def _fit_center_line(left, right, memory, cfg):
    near_y = float(cfg["near_y_ratio"]) * RAW_H
    mid_y = float(cfg["mid_y_ratio"]) * RAW_H
    far_y = float(cfg["far_y_ratio"]) * RAW_H
    ys = np.array([near_y, mid_y, far_y], dtype=np.float32)
    centers = np.array([
        0.5 * (left["x_near"] + right["x_near"]),
        0.5 * (left["x_mid"] + right["x_mid"]),
        0.5 * (left["x_far"] + right["x_far"]),
    ], dtype=np.float32)

    coeff = np.polyfit(ys, centers, 1)
    # x = a*y + b. 화면 위쪽으로 갈 때 x 변화량을 heading 부호와 맞춘다.
    a, b = float(coeff[0]), float(coeff[1])
    fitted = a * ys + b
    residual_ratio = float(np.max(np.abs(fitted - centers)) / HALF_W)
    center_slope = (float(fitted[2]) - float(fitted[0])) / max(1.0, near_y - far_y)
    lookahead_y = float(cfg["lookahead_y_ratio"]) * RAW_H
    lookahead_x = float(a * lookahead_y + b)
    effective_center_x = _effective_center_x(cfg)
    center_error = (lookahead_x - effective_center_x) / HALF_W

    gaps = np.array([
        abs(right["x_near"] - left["x_near"]),
        abs(right["x_mid"] - left["x_mid"]),
        abs(right["x_far"] - left["x_far"]),
    ], dtype=np.float32)
    gap_change_ratio = float((np.max(gaps) - np.min(gaps)) / max(1.0, np.mean(gaps)))
    heading_diff = abs(float(left["heading"]) - float(right["heading"]))

    if memory.get("seen_lane", False):
        jump_ratio = abs(lookahead_x - float(memory.get("last_center_x", effective_center_x))) / HALF_W
    else:
        jump_ratio = 0.0

    unstable_reasons = []
    if residual_ratio > float(cfg["center_fit_residual_ratio"]):
        unstable_reasons.append("fit_residual")
    if gap_change_ratio > float(cfg["max_gap_change_ratio"]):
        unstable_reasons.append("gap_change")
    if jump_ratio > float(cfg["max_center_jump_ratio"]):
        unstable_reasons.append("center_jump")
    if heading_diff > float(cfg["max_heading_diff"]):
        unstable_reasons.append("heading_diff")

    return {
        "centers": [float(x) for x in centers],
        "center_fit": [float(x) for x in fitted],
        "center_slope": float(center_slope),
        "lookahead_x": float(lookahead_x),
        "lookahead_y": float(lookahead_y),
        "center_error": float(center_error),
        "effective_center_x": float(effective_center_x),
        "camera_center_offset_ratio": float(cfg.get("camera_center_offset_ratio", 0.0)),
        "residual_ratio": residual_ratio,
        "gap_change_ratio": gap_change_ratio,
        "heading_diff": float(heading_diff),
        "jump_ratio": float(jump_ratio),
        "unstable_reasons": unstable_reasons,
    }


def _ema(last, raw, alpha):
    return (1.0 - float(alpha)) * float(last) + float(alpha) * float(raw)


def update(lanes, memory, cfg):
    features = _features(lanes, cfg)
    pair = _choose_pair(features, cfg)
    debug = {
        "features": features,
        "weights": [],
        "reason": "",
    }

    if pair is not None:
        left, right = pair
        center = _fit_center_line(left, right, memory, cfg)
        avg_heading = 0.5 * (left["heading"] + right["heading"])
        debug.update(center)
        debug["avg_lane_heading"] = float(avg_heading)

        if not center["unstable_reasons"]:
            raw_steer = (
                float(cfg["center_gain"]) * center["center_error"]
                + float(cfg["slope_gain"]) * center["center_slope"]
            )
            alpha = float(cfg["steer_alpha"])
            lane_state = "both_stable"
            speed_scale = float(cfg["both_speed_scale"])
            quality = float(clamp(0.5 * (left["quality"] + right["quality"]) + 0.25, 0.0, 1.0))
            stable_forward = True
            debug["reason"] = "local_center_line_plus_tangent"
            memory["last_center_x"] = float(center["lookahead_x"])
            memory["last_heading"] = float(center["center_slope"])
        else:
            slope_steer = float(cfg["unstable_slope_gain"]) * avg_heading
            raw_steer = _ema(memory["last_steer"], slope_steer, float(cfg["unstable_blend"]))
            alpha = float(cfg["unstable_alpha"])
            lane_state = "both_unstable"
            speed_scale = float(cfg["unstable_speed_scale"])
            quality = float(clamp(0.5 * (left["quality"] + right["quality"]), 0.0, 0.65))
            stable_forward = False
            debug["reason"] = "unstable_pair_memory_plus_avg_tangent"
            memory["last_heading"] = _ema(memory.get("last_heading", 0.0), avg_heading, 0.35)

        memory["lost_frames"] = 0
        memory["seen_lane"] = True

    elif len(features) >= 2:
        # lane 후보가 2개 이상 있는데 정상 pair를 만들 수 없다는 뜻이다.
        # ㅓ/ㅏ 분기나 잡음 후보일 가능성이 높으므로, 아무 lane 하나를 single로 믿지 않는다.
        # 이전 조향을 유지하며 감속하고 다음 frame에서 안정 pair가 다시 잡히길 기다린다.
        raw_steer = float(memory["last_steer"])
        alpha = float(cfg["unstable_alpha"])
        lane_state = "both_unstable"
        speed_scale = float(cfg["unstable_speed_scale"])
        quality = float(clamp(max(f["quality"] for f in features), 0.0, 0.65))
        stable_forward = False
        debug["reason"] = "no_valid_pair_hold_memory"
        debug["no_pair_feature_count"] = int(len(features))
        memory["lost_frames"] = 0
        memory["seen_lane"] = True

    elif features:
        feat = features[0]
        slope_steer = float(cfg["single_slope_gain"]) * float(feat["heading"])
        raw_steer = _ema(memory["last_steer"], slope_steer, float(cfg["single_blend"]))
        alpha = float(cfg["single_alpha"])
        lane_state = "single_slope"
        speed_scale = float(cfg["single_speed_scale"])
        quality = float(clamp(feat["quality"], 0.0, 0.60))
        stable_forward = False
        debug["reason"] = "single_lane_local_tangent"
        debug["single_heading"] = float(feat["heading"])
        memory["lost_frames"] = 0
        memory["seen_lane"] = True
        memory["last_heading"] = _ema(memory.get("last_heading", 0.0), feat["heading"], 0.45)

    else:
        memory["lost_frames"] = int(memory.get("lost_frames", 0)) + 1
        if memory["lost_frames"] <= int(cfg["lost_hold_frames"]):
            raw_steer = float(memory["last_steer"])
            lane_state = "lost_hold"
            debug["reason"] = "lost_hold_last_steer"
        else:
            raw_steer = float(memory["last_steer"]) * float(cfg["lost_decay"])
            lane_state = "lost_decay"
            debug["reason"] = "lost_decay_last_steer"
        alpha = float(cfg["lost_alpha"])
        speed_scale = float(cfg["lost_speed_scale"])
        quality = 0.0
        stable_forward = False

    raw_steer = clamp(raw_steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))
    steer = _ema(memory["last_steer"], raw_steer, alpha)
    steer = clamp(steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))
    memory["last_steer"] = float(steer)

    return {
        "lane_state": lane_state,
        "steer_norm": float(steer),
        "raw_steer": float(raw_steer),
        "speed_scale": float(speed_scale),
        "stable_forward": bool(stable_forward),
        "quality": float(quality),
        "visible_lane_count": int(len(lanes)),
        "feature_count": int(len(features)),
        "debug": debug,
    }
