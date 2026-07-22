"""inside_soft lane postprocess.

12/08 원래 주행 후처리와 19번 prototype의 lane_steering.py를 런타임 v2
LaneSignal 인터페이스에 맞춰 다시 정리한 후보.

핵심 동작:
  - both lane이면 두 lane 중앙 trajectory + heading을 사용한다.
  - single lane이면 그 lane을 왼쪽/오른쪽 경계라고 각각 가정한 후보를 만들고,
    이전 center/heading/steer와 가장 연속적인 후보를 선택한다.
  - 갑자기 center/heading이 튀면 unstable_blend로 천천히 섞는다.
  - lost가 길어지면 최근 turn_bias 방향으로 회복 조향을 조금 더한다.

기존 08 로직을 유지하되 config 이름은 현장에서 읽기 쉬운 값 위주로 줄였다.
"""

from __future__ import annotations

import math
import numpy as np

from lane.geometry import IMAGE_CENTER_X, RAW_H, RAW_W
from .common import clamp, normalize_lane_points, x_at_y


# 현장에서 건드릴 가능성이 낮은 내부 기준값.
DEFAULT_INTERNAL = {
    "single_side_deadband_ratio": 0.10,
    "side_prior_penalty_ratio": 0.30,
    "heading_cost_ratio": 0.20,
    "steer_cost_ratio": 0.20,
    "pair_bonus_ratio": 0.06,
    "jump_blend_start": 0.30,
    "jump_blend_step": 0.20,
    "jump_blend_max": 0.70,
    "lost_heading_decay": 0.90,
    "recovery_decay": 0.90,
}


def _merged_cfg(cfg):
    out = dict(DEFAULT_INTERNAL)
    out.update(cfg)
    return out


def init_memory(cfg=None):
    return {
        "smoothed_center_x": IMAGE_CENTER_X,
        "smoothed_heading": 0.0,
        "last_steer": 0.0,
        "turn_bias": 0.0,
        "lost_frames": 0,
        "unstable_frames": 0,
        "seen_lane": False,
    }


def _feature_from_lane(lane, cfg):
    pts = normalize_lane_points(lane)
    if len(pts) < int(cfg["min_points"]):
        return None

    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    if y_span < float(cfg["min_y_span_ratio"]) * RAW_H:
        return None

    y_top = float(cfg["top_y_ratio"]) * RAW_H
    y_primary = float(cfg["primary_y_ratio"]) * RAW_H
    y_bottom = float(cfg["bottom_y_ratio"]) * RAW_H

    min_y = float(pts[:, 1].min())
    max_y = float(pts[:, 1].max())
    max_ex = float(cfg["max_extrapolate_ratio"]) * RAW_H
    for yq in (y_top, y_primary, y_bottom):
        if yq < min_y and min_y - yq > max_ex:
            return None
        if yq > max_y and yq - max_y > max_ex:
            return None

    x_top = x_at_y(pts, y_top)
    x_primary = x_at_y(pts, y_primary)
    x_bottom = x_at_y(pts, y_bottom)
    if x_top is None or x_primary is None or x_bottom is None:
        return None

    heading = (float(x_top) - float(x_bottom)) / max(1.0, y_bottom - y_top)
    span_score = clamp(y_span / (RAW_H * 0.28), 0.0, 1.0)
    point_score = clamp(len(pts) / 22.0, 0.0, 1.0)
    conf_score = clamp(float(lane.get("conf", 0.0)) / 0.85, 0.0, 1.0)
    quality = 0.40 * conf_score + 0.35 * span_score + 0.25 * point_score
    return {
        "points": pts,
        "conf": float(lane.get("conf", 0.0)),
        "x_top": float(x_top),
        "x_primary": float(x_primary),
        "x_bottom": float(x_bottom),
        "heading": float(heading),
        "quality": float(clamp(quality, 0.0, 1.0)),
    }


def _estimate_steer(center_x, heading, cfg):
    pos_error = (float(center_x) - IMAGE_CENTER_X) / IMAGE_CENTER_X
    raw = float(cfg["steer_gain"]) * (
        float(cfg["position_gain"]) * pos_error
        + float(cfg["heading_gain"]) * float(heading)
    )
    return float(np.clip(raw, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"])))


def _heading_denominator(cfg):
    return max(1.0, (float(cfg["bottom_y_ratio"]) - float(cfg["top_y_ratio"])) * RAW_H)


def _trajectory_from_pair(left, right, cfg):
    center_top = 0.5 * (left["x_top"] + right["x_top"])
    center_primary = 0.5 * (left["x_primary"] + right["x_primary"])
    center_bottom = 0.5 * (left["x_bottom"] + right["x_bottom"])
    return {
        "center_primary": float(center_primary),
        "heading": float((center_top - center_bottom) / _heading_denominator(cfg)),
        "lane_state": "both_stable",
        "source": "pair",
        "quality": float(clamp(0.5 * (left["quality"] + right["quality"]) + 0.20, 0.0, 1.0)),
    }


def _trajectory_from_single(feat, side, cfg):
    # side == "left": 이 lane이 왼쪽 경계라고 보고 오른쪽 안쪽으로 offset.
    # side == "right": 이 lane이 오른쪽 경계라고 보고 왼쪽 안쪽으로 offset.
    sign = 1.0 if side == "left" else -1.0
    offset = float(cfg["safe_offset_ratio"]) * RAW_W
    center_top = feat["x_top"] + sign * offset
    center_primary = feat["x_primary"] + sign * offset
    center_bottom = feat["x_bottom"] + sign * offset
    return {
        "center_primary": float(center_primary),
        "heading": float((center_top - center_bottom) / _heading_denominator(cfg)),
        "lane_state": f"single_{side}",
        "source": "single",
        "source_x": float(feat["x_primary"]),
        "quality": float(clamp(feat["quality"], 0.0, 0.65)),
    }


def _choose_trajectory(features, memory, cfg):
    if not features:
        return None, []

    lefts = [f for f in features if f["x_primary"] < IMAGE_CENTER_X]
    rights = [f for f in features if f["x_primary"] >= IMAGE_CENTER_X]
    left = max(lefts, key=lambda f: f["x_primary"]) if lefts else None
    right = min(rights, key=lambda f: f["x_primary"]) if rights else None

    candidates = []
    min_gap = float(cfg["min_pair_gap_ratio"]) * RAW_W
    max_gap = float(cfg["max_pair_gap_ratio"]) * RAW_W
    if left is not None and right is not None:
        gap = right["x_primary"] - left["x_primary"]
        if min_gap <= gap <= max_gap:
            cand = _trajectory_from_pair(left, right, cfg)
            cand["pair_gap"] = float(gap)
            candidates.append(cand)

    side_deadband = float(cfg["single_side_deadband_ratio"]) * RAW_W
    side_penalty = float(cfg["side_prior_penalty_ratio"]) * RAW_W
    for feat in features:
        for side in ("left", "right"):
            cand = _trajectory_from_single(feat, side, cfg)
            penalty = 0.0
            if feat["x_primary"] < IMAGE_CENTER_X - side_deadband and side == "right":
                penalty = side_penalty
            if feat["x_primary"] > IMAGE_CENTER_X + side_deadband and side == "left":
                penalty = side_penalty
            cand["side_penalty"] = float(penalty)
            candidates.append(cand)

    if not candidates:
        return None, []

    prev_center = float(memory.get("smoothed_center_x", IMAGE_CENTER_X))
    prev_heading = float(memory.get("smoothed_heading", 0.0))
    prev_steer = float(memory.get("last_steer", 0.0))
    heading_cost_px = float(cfg["heading_cost_ratio"]) * RAW_W
    steer_cost_px = float(cfg["steer_cost_ratio"]) * RAW_W
    pair_bonus = float(cfg["pair_bonus_ratio"]) * RAW_W

    def cost(c):
        center_cost = abs(float(c["center_primary"]) - prev_center)
        heading_cost = abs(float(c["heading"]) - prev_heading) * heading_cost_px
        steer_cost = abs(_estimate_steer(c["center_primary"], c["heading"], cfg) - prev_steer) * steer_cost_px
        bonus = -pair_bonus if c["lane_state"] == "both_stable" else 0.0
        return center_cost + heading_cost + steer_cost + float(c.get("side_penalty", 0.0)) + bonus

    for cand in candidates:
        cand["candidate_cost"] = float(cost(cand))
    best = min(candidates, key=lambda c: c["candidate_cost"])
    best["candidate_count"] = len(candidates)
    return best, candidates


def _speed_for_state(lane_state, cfg):
    if lane_state == "both_stable":
        return float(cfg["both_speed_scale"])
    if lane_state.startswith("single_"):
        return float(cfg["single_speed_scale"])
    if lane_state == "unstable_blend":
        return float(cfg["unstable_speed_scale"])
    return float(cfg["lost_speed_scale"])


def update(lanes, memory, cfg):
    cfg = _merged_cfg(cfg)
    features = [f for f in (_feature_from_lane(lane, cfg) for lane in lanes) if f is not None]
    measured, candidates = _choose_trajectory(features, memory, cfg)
    debug = {
        "features": [
            {
                "x_top": f["x_top"],
                "x_primary": f["x_primary"],
                "x_bottom": f["x_bottom"],
                "heading": f["heading"],
                "quality": f["quality"],
            }
            for f in features
        ],
        "candidates": candidates,
        "reason": "",
    }

    if measured is None:
        memory["lost_frames"] = int(memory.get("lost_frames", 0)) + 1
        memory["unstable_frames"] = 0
        if memory["lost_frames"] <= int(cfg["lost_hold_frames"]):
            lane_state = "lost_hold"
            raw_steer = float(memory["last_steer"]) * float(cfg["recovery_decay"])
            debug["reason"] = "lost_short_hold_decay"
        else:
            lane_state = "lost_active_recovery"
            sign = float(np.sign(memory["turn_bias"])) or float(np.sign(memory["last_steer"]))
            ramp = float(cfg["recovery_ramp"]) * max(0, memory["lost_frames"] - int(cfg["lost_hold_frames"]))
            raw_steer = (
                float(memory["last_steer"]) * float(cfg["recovery_decay"])
                + float(cfg["recovery_gain"]) * float(memory["turn_bias"])
                + ramp * sign
            )
            debug["reason"] = "lost_active_turn_bias_recovery"

        raw_steer = clamp(raw_steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))
        memory["smoothed_heading"] *= float(cfg["lost_heading_decay"])
        steer = float(cfg["steer_alpha"]) * raw_steer + (1.0 - float(cfg["steer_alpha"])) * float(memory["last_steer"])
        steer = clamp(steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))
        memory["last_steer"] = steer
        memory["turn_bias"] = float(cfg["turn_bias_alpha"]) * steer + (1.0 - float(cfg["turn_bias_alpha"])) * float(memory["turn_bias"])

        return {
            "lane_state": lane_state,
            "steer_norm": float(steer),
            "raw_steer": float(raw_steer),
            "speed_scale": _speed_for_state(lane_state, cfg),
            "stable_forward": False,
            "quality": 0.0,
            "visible_lane_count": int(len(lanes)),
            "feature_count": int(len(features)),
            "debug": {
                **debug,
                "smoothed_center_x": float(memory["smoothed_center_x"]),
                "smoothed_heading": float(memory["smoothed_heading"]),
                "turn_bias": float(memory["turn_bias"]),
                "lost_frames": int(memory["lost_frames"]),
            },
        }

    memory["lost_frames"] = 0
    center = float(measured["center_primary"])
    heading = float(measured["heading"])
    center_jump = abs(center - float(memory["smoothed_center_x"]))
    heading_jump = abs(heading - float(memory["smoothed_heading"]))
    unstable = (
        center_jump > float(cfg["jump_center_ratio"]) * RAW_W
        or heading_jump > float(cfg["jump_heading"])
    )

    if unstable:
        memory["unstable_frames"] = int(memory.get("unstable_frames", 0)) + 1
        blend = min(
            float(cfg["jump_blend_max"]),
            float(cfg["jump_blend_start"]) + float(cfg["jump_blend_step"]) * max(0, memory["unstable_frames"] - 1),
        )
        effective_center = (1.0 - blend) * float(memory["smoothed_center_x"]) + blend * center
        effective_heading = (1.0 - blend) * float(memory["smoothed_heading"]) + blend * heading
        lane_state = "unstable_blend"
        debug["reason"] = "center_or_heading_jump_blend"
    else:
        memory["unstable_frames"] = 0
        effective_center = center
        effective_heading = heading
        lane_state = str(measured["lane_state"])
        debug["reason"] = "selected_trajectory"

    memory["smoothed_center_x"] = (
        float(cfg["center_alpha"]) * effective_center
        + (1.0 - float(cfg["center_alpha"])) * float(memory["smoothed_center_x"])
    )
    memory["smoothed_heading"] = (
        float(cfg["heading_alpha"]) * effective_heading
        + (1.0 - float(cfg["heading_alpha"])) * float(memory["smoothed_heading"])
    )
    raw_steer = _estimate_steer(memory["smoothed_center_x"], memory["smoothed_heading"], cfg)
    steer = float(cfg["steer_alpha"]) * raw_steer + (1.0 - float(cfg["steer_alpha"])) * float(memory["last_steer"])
    steer = clamp(steer, -float(cfg["max_steer_norm"]), float(cfg["max_steer_norm"]))

    memory["last_steer"] = steer
    memory["turn_bias"] = float(cfg["turn_bias_alpha"]) * steer + (1.0 - float(cfg["turn_bias_alpha"])) * float(memory["turn_bias"])
    memory["seen_lane"] = True

    stable_forward = lane_state == "both_stable" and float(measured.get("quality", 0.0)) >= float(cfg["stable_quality"])
    quality = float(clamp(measured.get("quality", 0.0), 0.0, 1.0))
    debug.update({
        "selected": measured,
        "measured_center_x": center,
        "measured_heading": heading,
        "center_jump": float(center_jump),
        "heading_jump": float(heading_jump),
        "smoothed_center_x": float(memory["smoothed_center_x"]),
        "smoothed_heading": float(memory["smoothed_heading"]),
        "turn_bias": float(memory["turn_bias"]),
        "lost_frames": int(memory["lost_frames"]),
    })

    return {
        "lane_state": lane_state,
        "steer_norm": float(steer),
        "raw_steer": float(raw_steer),
        "speed_scale": _speed_for_state(lane_state, cfg),
        "stable_forward": bool(stable_forward),
        "quality": quality,
        "visible_lane_count": int(len(lanes)),
        "feature_count": int(len(features)),
        "debug": debug,
    }
