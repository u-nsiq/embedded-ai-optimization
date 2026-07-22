from __future__ import annotations

import numpy as np

from lane.geometry import HALF_W, IMAGE_CENTER_X, RAW_H, RAW_W


REQUIRED_SIGNAL_KEYS = {
    "steer_norm",
    "raw_steer",
    "speed_scale",
    "lane_state",
    "stable_forward",
    "quality",
    "visible_lane_count",
    "feature_count",
    "debug",
}


def clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def normalize_lane_points(lane):
    pts = np.array(lane.get("points", []), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    valid = (
        np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
        & (pts[:, 0] >= -RAW_W * 0.25)
        & (pts[:, 0] <= RAW_W * 1.25)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] <= RAW_H)
    )
    return pts[valid]


def x_at_y(points, y_query):
    """lane point들을 x=f(y)로 보고 원하는 y의 x를 읽는다."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return None
    order = np.argsort(pts[:, 1])
    ys = pts[order, 1]
    xs = pts[order, 0]
    uniq_y, uniq_idx = np.unique(ys, return_index=True)
    ys = uniq_y
    xs = xs[uniq_idx]
    if len(ys) < 2:
        return None
    yq = float(y_query)
    if ys[0] <= yq <= ys[-1]:
        return float(np.interp(yq, ys, xs))
    if yq < ys[0]:
        y0, y1 = float(ys[0]), float(ys[1])
        x0, x1 = float(xs[0]), float(xs[1])
    else:
        y0, y1 = float(ys[-2]), float(ys[-1])
        x0, x1 = float(xs[-2]), float(xs[-1])
    if abs(y1 - y0) < 1e-6:
        return x0
    return float(x0 + (x1 - x0) * ((yq - y0) / (y1 - y0)))


def validate_signal(signal, name):
    missing = sorted(REQUIRED_SIGNAL_KEYS - set(signal.keys()))
    if missing:
        raise KeyError(f"lane postprocess '{name}' missed LaneSignal keys: {missing}")
    return signal
