"""slope_push lane postprocess.

기존 08f 계열 로직이다.

핵심:
  1. 보이는 lane들의 국소 기울기를 weighted average로 구한다.
  2. lane이 중앙 safety band 안으로 들어오면 반대 방향 push를 더한다.
  3. lost 때는 마지막 조향/기울기를 잠시 유지한다.

이 후보는 기존 패키지 동작을 유지하기 위한 기본값이다.
"""

from __future__ import annotations

import numpy as np

from lane.geometry import HALF_W, IMAGE_CENTER_X, RAW_H
from .common import clamp, normalize_lane_points, x_at_y


def init_memory(cfg=None):
    return {
        "last_slope": 0.0,
        "last_steer": 0.0,
        "lost_frames": 0,
        "seen_lane": False,
    }


def lane_feature(lane, cfg):
    pts = normalize_lane_points(lane)
    if len(pts) < int(cfg["min_points"]):
        return None

    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    if y_span < float(cfg["min_y_span_ratio"]) * RAW_H:
        return None

    near_y = float(cfg["near_y_ratio"]) * RAW_H
    mid_y = float(cfg["mid_y_ratio"]) * RAW_H
    far_y = float(cfg["far_y_ratio"]) * RAW_H
    query_ys = np.array([near_y, mid_y, far_y], dtype=np.float32)

    min_y = float(pts[:, 1].min())
    max_y = float(pts[:, 1].max())
    extrapolate_dist = 0.0
    for yq in query_ys:
        if yq < min_y:
            extrapolate_dist = max(extrapolate_dist, min_y - float(yq))
        elif yq > max_y:
            extrapolate_dist = max(extrapolate_dist, float(yq) - max_y)
    if extrapolate_dist > float(cfg["max_extrapolate_ratio"]) * RAW_H:
        return None

    x_near = x_at_y(pts, near_y)
    x_mid = x_at_y(pts, mid_y)
    x_far = x_at_y(pts, far_y)
    if x_near is None or x_mid is None or x_far is None:
        return None

    heading = (float(x_far) - float(x_near)) / max(1.0, near_y - far_y)
    center_dist_ratio = (float(x_mid) - IMAGE_CENTER_X) / HALF_W

    inside_count = int(((query_ys >= min_y) & (query_ys <= max_y)).sum())
    coverage_score = inside_count / 3.0
    span_score = clamp(y_span / (RAW_H * 0.28), 0.0, 1.0)
    point_score = clamp(len(pts) / 22.0, 0.0, 1.0)
    conf_score = clamp(float(lane.get("conf", 0.0)) / 0.85, 0.0, 1.0)

    return {
        "points": pts,
        "conf": float(lane.get("conf", 0.0)),
        "x_near": float(x_near),
        "x_mid": float(x_mid),
        "x_far": float(x_far),
        "heading": float(heading),
        "center_dist_ratio": float(center_dist_ratio),
        "coverage_score": float(coverage_score),
        "span_score": float(span_score),
        "point_score": float(point_score),
        "conf_score": float(conf_score),
    }


def extract_lane_features(lanes, cfg):
    features = []
    for lane in lanes:
        feat = lane_feature(lane, cfg)
        if feat is not None:
            features.append(feat)
    features.sort(key=lambda f: abs(f["center_dist_ratio"]))
    return features


def lane_trust(feat, memory, cfg):
    conf_part = feat["conf_score"]
    span_part = 0.55 * feat["span_score"] + 0.25 * feat["point_score"] + 0.20 * feat["coverage_score"]
    if memory.get("seen_lane", False):
        diff = abs(feat["heading"] - float(memory.get("last_slope", 0.0)))
        memory_part = clamp(1.0 - diff / float(cfg["memory_slope_tolerance"]), 0.0, 1.0)
    else:
        memory_part = 1.0
    center_part = clamp(1.0 - max(0.0, abs(feat["center_dist_ratio"]) - 0.55) / 0.45, 0.0, 1.0)
    weight = (
        float(cfg["trust_conf_weight"]) * conf_part
        + float(cfg["trust_span_weight"]) * span_part
        + float(cfg["trust_memory_weight"]) * memory_part
        + float(cfg["trust_center_weight"]) * center_part
    )
    return float(clamp(weight, 0.0, 1.0))


def weighted_slope(features, memory, cfg):
    if not features:
        return float(memory.get("last_slope", 0.0)), 0.0, []
    weights = [lane_trust(f, memory, cfg) for f in features]
    total = float(sum(weights))
    if total < 1e-6:
        return float(memory.get("last_slope", 0.0)), 0.0, weights
    slope = float(sum(f["heading"] * w for f, w in zip(features, weights)) / total)
    confidence = float(clamp(total / max(1, len(features)), 0.0, 1.0))
    return slope, confidence, weights


def safety_push(features, cfg):
    safe_dist = float(cfg["safe_distance_ratio"]) * HALF_W
    push = 0.0
    risk = 0.0
    closest_abs_dx = None

    for feat in features:
        dx = float(feat["x_near"] - IMAGE_CENTER_X)
        abs_dx = abs(dx)
        closest_abs_dx = abs_dx if closest_abs_dx is None else min(closest_abs_dx, abs_dx)
        if abs_dx < safe_dist:
            intensity = 1.0 - abs_dx / max(1.0, safe_dist)
            sign = 0.0 if abs(dx) < 1e-6 else -np.sign(dx)
            push += float(sign * intensity)
            risk = max(risk, float(intensity))

    push = clamp(push, -float(cfg["max_push"]), float(cfg["max_push"]))
    return float(push), float(risk), closest_abs_dx


def recommended_speed_scale(departure_risk, quality, cfg):
    speed = 1.0
    speed -= float(cfg["risk_slowdown"]) * float(departure_risk)
    speed -= float(cfg["low_quality_slowdown"]) * float(1.0 - quality)
    return float(clamp(speed, float(cfg["min_speed_scale"]), 1.0))


def update(lanes, memory, cfg):
    features = extract_lane_features(lanes, cfg)

    if features:
        memory["lost_frames"] = 0
        slope, quality, weights = weighted_slope(features, memory, cfg)
        push, risk, closest_abs_dx = safety_push(features, cfg)

        raw_steer = float(cfg["slope_gain"]) * slope + float(cfg["push_gain"]) * push
        raw_steer = clamp(raw_steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))

        steer = (1.0 - float(cfg["steer_alpha"])) * float(memory["last_steer"]) + float(cfg["steer_alpha"]) * raw_steer
        steer = clamp(steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))

        memory["last_slope"] = (
            (1.0 - float(cfg["slope_memory_alpha"])) * float(memory["last_slope"])
            + float(cfg["slope_memory_alpha"]) * slope
        )
        memory["last_steer"] = steer
        memory["seen_lane"] = True

        lane_state = "slope_push_risk" if risk >= float(cfg["risk_state_threshold"]) else "slope_push"
        speed_scale = recommended_speed_scale(risk, quality, cfg)
        return {
            "lane_state": lane_state,
            "steer_norm": float(steer),
            "raw_steer": float(raw_steer),
            "speed_scale": float(speed_scale),
            "stable_forward": bool(quality >= float(cfg["stable_quality"]) and risk <= float(cfg["stable_risk"])),
            "quality": float(quality),
            "visible_lane_count": int(len(lanes)),
            "feature_count": int(len(features)),
            "debug": {
                "weighted_slope": float(slope),
                "memory_slope": float(memory["last_slope"]),
                "push_term": float(push),
                "departure_risk": float(risk),
                "closest_abs_dx": None if closest_abs_dx is None else float(closest_abs_dx),
                "weights": weights,
                "features": features,
                "reason": "weighted_slope_plus_safety_push",
            },
        }

    memory["lost_frames"] = int(memory.get("lost_frames", 0)) + 1
    if memory["lost_frames"] <= int(cfg["lost_hold_frames"]):
        raw_steer = float(memory["last_steer"])
        lane_state = "lost_hold"
        reason = "no_lane_hold_last_steer"
    else:
        raw_steer = float(memory["last_steer"]) * float(cfg["lost_decay"]) + float(cfg["lost_slope_gain"]) * float(memory["last_slope"])
        lane_state = "lost_decay"
        reason = "no_lane_decay_with_memory_slope"

    raw_steer = clamp(raw_steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))
    steer = (1.0 - float(cfg["lost_steer_alpha"])) * float(memory["last_steer"]) + float(cfg["lost_steer_alpha"]) * raw_steer
    steer = clamp(steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))
    memory["last_steer"] = steer

    risk = 0.70
    quality = 0.0
    return {
        "lane_state": lane_state,
        "steer_norm": float(steer),
        "raw_steer": float(raw_steer),
        "speed_scale": recommended_speed_scale(risk, quality, cfg),
        "stable_forward": False,
        "quality": float(quality),
        "visible_lane_count": int(len(lanes)),
        "feature_count": 0,
        "debug": {
            "weighted_slope": float(memory.get("last_slope", 0.0)),
            "memory_slope": float(memory.get("last_slope", 0.0)),
            "push_term": 0.0,
            "departure_risk": float(risk),
            "closest_abs_dx": None,
            "weights": [],
            "features": [],
            "reason": reason,
        },
    }
