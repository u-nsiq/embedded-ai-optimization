"""Official-overlap lane decoder.

이 파일은 12/07 decode contract를 runtime용 numpy 코드로 옮긴 것이다.
주행 튜닝 대상은 아니다. 보통 config.py의 conf_threshold 정도만 만진다.
"""

from __future__ import annotations

import numpy as np

from .geometry import CUT_HEIGHT, IMG_W, N_OFFSETS, N_STRIPS, RAW_H, RAW_W, SAMPLE_Y

try:
    from scipy.interpolate import InterpolatedUnivariateSpline

    HAS_SCIPY = True
except Exception:
    InterpolatedUnivariateSpline = None
    HAS_SCIPY = False


def softmax_positive_score(logits_2):
    logits = logits_2.astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=1)


def official_cuda_suppresses(a, b, threshold):
    """공식 CUDA NMS kernel의 핵심 조건.

    두 lane이 공통 y 구간에서 평균 x 거리가 threshold보다 작으면 같은 lane으로 본다.
    """
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
        # scipy가 없으면 실행은 가능하지만 공식 Lane.to_array와 약간 달라질 수 있다.
        out = np.interp(query_ys, ys_sorted, xs_sorted).astype(np.float32)
    min_y, max_y = float(ys_sorted.min()) - 0.01, float(ys_sorted.max()) + 0.01
    out[(query_ys < min_y) | (query_ys > max_y)] = -2.0
    return out


def prediction_to_lane_array(prediction, confidence):
    """공식 predictions_to_pred + Lane.to_array를 Pi용 numpy로 합친 함수."""
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
    pts = np.stack(
        [sample_xs_norm[valid_sample] * float(RAW_W), sample_ys_norm[valid_sample] * float(RAW_H)],
        axis=1,
    )
    return {"points": pts.astype(np.float32), "conf": float(confidence)}


def decode_raw_to_lanes(raw_predictions, config):
    conf_threshold = float(config["conf_threshold"])
    nms_thres = float(config["nms_thres"])
    nms_topk = int(config["nms_topk"])

    predictions = raw_predictions.astype(np.float32)
    scores = softmax_positive_score(predictions[:, :2])
    keep_mask = scores >= conf_threshold
    if int(keep_mask.sum()) == 0:
        return []

    pred_kept = predictions[keep_mask].copy()
    score_kept = scores[keep_mask].copy()

    nms_predictions = np.concatenate([pred_kept[:, :4], pred_kept[:, 5:]], axis=1).astype(np.float32)
    nms_predictions[:, 4] *= float(N_STRIPS)
    nms_predictions[:, 5:] *= float(IMG_W - 1)
    keep = official_overlap_nms(nms_predictions, score_kept, nms_thres, nms_topk)

    selected = pred_kept[keep].copy()
    selected_scores = score_kept[keep].copy()
    selected[:, 5] = np.round(selected[:, 5] * float(N_STRIPS))

    lanes = []
    for pred, score in zip(selected, selected_scores):
        lane = prediction_to_lane_array(pred, score)
        if lane is not None:
            lanes.append(lane)
    return lanes
