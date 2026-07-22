"""fixed_base lane postprocess.

12/08b와 40_drive_runtime/team3_final_drive의 steering.py 기반 후보.

핵심:
  - both lane: 두 lane의 중앙 x 오차 + 평균 local slope를 사용한다.
  - single lane: left/right를 억지로 판단하지 않고, 중앙에 가까운 lane의 local slope만 쓴다.
  - lost: 마지막 steer를 hold/decay한다.

가장 단순한 후보라서 현장 비교용 fallback으로 좋다.
"""

from __future__ import annotations

import math
import numpy as np

from lane.geometry import IMAGE_CENTER_X, RAW_H, RAW_W
from .common import clamp, normalize_lane_points, x_at_y


def init_memory(cfg=None):
    return {
        "last_steer": 0.0,
        "lost_frames": 0,
        "last_state": "init",
    }


def _effective_center_x(cfg):
    # 카메라가 물리적으로 좌/우로 틀어진 경우, 조향 기준이 되는 화면 중앙만 보정한다.
    return IMAGE_CENTER_X + float(cfg.get("camera_center_offset_ratio", 0.0)) * IMAGE_CENTER_X


def _feature_from_lane(lane, cfg):
    pts = normalize_lane_points(lane)
    if len(pts) < int(cfg["min_points"]):
        return None

    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    if y_span < float(cfg["min_y_span_ratio"]) * RAW_H:
        return None

    y_top = float(cfg["top_y_ratio"]) * RAW_H
    y_mid = float(cfg["mid_y_ratio"]) * RAW_H
    y_bottom = float(cfg["bottom_y_ratio"]) * RAW_H

    min_y = float(pts[:, 1].min())
    max_y = float(pts[:, 1].max())
    max_ex = float(cfg["max_y_distance_ratio"]) * RAW_H
    for yq in (y_top, y_mid, y_bottom):
        if yq < min_y and min_y - yq > max_ex:
            return None
        if yq > max_y and yq - max_y > max_ex:
            return None

    x_top = x_at_y(pts, y_top)
    x_mid = x_at_y(pts, y_mid)
    x_bottom = x_at_y(pts, y_bottom)
    if x_top is None or x_mid is None or x_bottom is None:
        return None

    local_slope = (float(x_top) - float(x_bottom)) / max(1.0, y_bottom - y_top)
    span_score = clamp(y_span / (RAW_H * 0.28), 0.0, 1.0)
    point_score = clamp(len(pts) / 22.0, 0.0, 1.0)
    conf_score = clamp(float(lane.get("conf", 0.0)) / 0.85, 0.0, 1.0)
    quality = 0.40 * conf_score + 0.35 * span_score + 0.25 * point_score
    return {
        "x_top": float(x_top),
        "x_mid": float(x_mid),
        "x_bottom": float(x_bottom),
        "heading": float(local_slope),
        "quality": float(clamp(quality, 0.0, 1.0)),
    }


def _choose_geometry(features, cfg):
    effective_center_x = _effective_center_x(cfg)
    if not features:
        return {
            "lane_state": "lost",
            "raw_state": "lost",
            "center_error": 0.0,
            "local_slope": 0.0,
            "pair_gap_ratio": math.nan,
            "quality": 0.0,
        }

    best_pair = None
    best_cost = float("inf")
    min_gap = float(cfg["pair_gap_min_ratio"]) * RAW_W
    max_gap = float(cfg["pair_gap_max_ratio"]) * RAW_W
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = features[i], features[j]
            left, right = (a, b) if a["x_mid"] <= b["x_mid"] else (b, a)
            gap = right["x_mid"] - left["x_mid"]
            if not (min_gap <= gap <= max_gap):
                continue
            center_x = 0.5 * (left["x_mid"] + right["x_mid"])
            cost = abs(center_x - effective_center_x) - 80.0 * (left["quality"] + right["quality"])
            if cost < best_cost:
                best_cost = cost
                best_pair = (left, right, center_x, gap)

    if best_pair is not None:
        left, right, center_x, gap = best_pair
        return {
            "lane_state": "both",
            "raw_state": "both",
            "center_error": float((center_x - effective_center_x) / IMAGE_CENTER_X),
            "local_slope": float(0.5 * (left["heading"] + right["heading"])),
            "pair_gap_ratio": float(gap / RAW_W),
            "quality": float(clamp(0.5 * (left["quality"] + right["quality"]) + 0.20, 0.0, 1.0)),
        }

    # feature는 2개 이상인데 pair 조건을 통과하지 못하면 branch/noise일 가능성이 높다.
    # 여기서 single lane으로 바로 따라가면 ㅓ 구간에서 옆 가지를 따라갈 수 있으므로,
    # 일단 이전 steer를 줄이며 직진 성향으로 넘어가게 한다.
    if len(features) >= 2:
        return {
            "lane_state": "no_pair",
            "raw_state": "no_pair",
            "center_error": 0.0,
            "local_slope": 0.0,
            "pair_gap_ratio": math.nan,
            "quality": float(clamp(max(float(f["quality"]) for f in features), 0.0, 0.65)),
        }

    # 한쪽 lane에서는 side를 판단하지 않는다. 중앙에 가까운 lane의 tangent만 사용한다.
    feat = min(features, key=lambda f: abs(float(f["x_mid"]) - effective_center_x))
    return {
        "lane_state": "single",
        "raw_state": "single",
        "center_error": 0.0,
        "local_slope": float(feat["heading"]),
        "pair_gap_ratio": math.nan,
        "quality": float(clamp(feat["quality"], 0.0, 0.65)),
    }


def _raw_steer(geom, cfg):
    state = geom["lane_state"]
    if state == "both":
        steer = float(cfg["both_center_weight"]) * float(geom["center_error"])
        steer += float(cfg["both_slope_weight"]) * float(geom["local_slope"])
    elif state == "single":
        steer = float(cfg["single_slope_weight"]) * float(geom["local_slope"])
    else:
        steer = 0.0
    return clamp(steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))


def _speed_for_state(lane_state, cfg):
    if lane_state == "both":
        return float(cfg["both_speed_scale"])
    if lane_state == "single":
        return float(cfg["single_speed_scale"])
    if lane_state == "no_pair":
        return float(cfg.get("no_pair_speed_scale", cfg["single_speed_scale"]))
    return float(cfg["lost_speed_scale"])


def update(lanes, memory, cfg):
    features = [f for f in (_feature_from_lane(lane, cfg) for lane in lanes) if f is not None]
    geom = _choose_geometry(features, cfg)
    lane_state = geom["lane_state"]

    if lane_state == "lost":
        memory["lost_frames"] = int(memory.get("lost_frames", 0)) + 1
        if memory["lost_frames"] <= int(cfg["lost_hold_frames"]):
            raw_steer = float(memory["last_steer"])
            out_state = "lost_hold"
            reason = "lost_hold_last_steer"
        else:
            raw_steer = float(memory["last_steer"]) * float(cfg["lost_decay"])
            out_state = "lost_decay"
            reason = "lost_decay_last_steer"
        alpha = float(cfg["lost_alpha"])
        quality = 0.0
    elif lane_state == "no_pair":
        memory["lost_frames"] = 0
        raw_steer = float(memory["last_steer"]) * float(cfg.get("no_pair_keep_ratio", 0.35))
        alpha = float(cfg.get("no_pair_alpha", cfg["single_alpha"]))
        out_state = "no_pair"
        reason = "no_valid_pair_go_straight"
        quality = float(geom["quality"])
    else:
        memory["lost_frames"] = 0
        raw_steer = _raw_steer(geom, cfg)
        alpha = float(cfg["both_alpha"] if lane_state == "both" else cfg["single_alpha"])
        out_state = lane_state
        reason = "center_plus_slope" if lane_state == "both" else "single_slope_only"
        quality = float(geom["quality"])

    steer = alpha * raw_steer + (1.0 - alpha) * float(memory["last_steer"])
    steer = clamp(steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))
    memory["last_steer"] = float(steer)
    memory["last_state"] = out_state

    return {
        "lane_state": out_state,
        "steer_norm": float(steer),
        "raw_steer": float(raw_steer),
        "speed_scale": _speed_for_state(lane_state, cfg),
        "stable_forward": bool(out_state == "both" and quality >= float(cfg["stable_quality"])),
        "quality": float(quality),
        "visible_lane_count": int(len(lanes)),
        "feature_count": int(len(features)),
        "debug": {
            "features": features,
            "center_error": float(geom["center_error"]),
            "local_slope": float(geom["local_slope"]),
            "pair_gap_ratio": geom["pair_gap_ratio"],
            "lost_frames": int(memory["lost_frames"]),
            "reason": reason,
        },
    }
