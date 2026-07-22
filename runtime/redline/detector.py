from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def init_redline_state(cfg):
    """redline voting/cooldown 상태를 만든다."""
    return {
        "arm_history": deque(maxlen=max(1, int(cfg.get("arm_vote_window", cfg.get("vote_window", 3))))),
        "stop_history": deque(maxlen=max(1, int(cfg.get("stop_vote_window", cfg.get("vote_window", 3))))),
        "armed": False,
        "armed_at": 0.0,
        "last_emit_time": -1e9,
        "last_observation": {"active": False, "contours": [], "best": None, "mask": None},
    }


def _kernel(size):
    w, h = int(size[0]), int(size[1])
    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w), max(1, h)))


def hsv_mask(bgr, cfg):
    """BGR frame에서 red HSV mask를 만든다."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for item in cfg.get("hsv_ranges", []):
        lower = np.array(item["lower"], dtype=np.uint8)
        upper = np.array(item["upper"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    if "open_kernel" in cfg:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(cfg["open_kernel"]))
    if "close_kernel" in cfg:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(cfg["close_kernel"]))
    if "horizontal_close_kernel" in cfg:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(cfg["horizontal_close_kernel"]))
    return mask


def _center_overlap_ratio(x, bw, image_w, band):
    band_x0 = float(band[0]) * float(image_w)
    band_x1 = float(band[1]) * float(image_w)
    overlap = max(0.0, min(float(x + bw), band_x1) - max(float(x), band_x0))
    return overlap / max(1.0, band_x1 - band_x0)


def _roi_bounds(roi, image_h):
    y0 = int(round(float(roi[0]) * float(image_h)))
    y1 = int(round(float(roi[1]) * float(image_h)))
    y0 = max(0, min(int(image_h), y0))
    y1 = max(0, min(int(image_h), y1))
    if y1 < y0:
        y0, y1 = y1, y0
    return y0, y1


def _detect_redline_in_roi(mask, cfg, image_w, image_h, roi):
    y0, y1 = _roi_bounds(roi, image_h)
    roi_mask = np.zeros_like(mask)
    roi_mask[y0:y1, :] = mask[y0:y1, :]
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    central_band = cfg.get("central_band", [0.35, 0.65])
    passed = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(cfg.get("min_area", 350)):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < int(cfg.get("min_height_px", 3)):
            continue
        if bw / max(1.0, float(image_w)) < float(cfg.get("min_width_ratio", 0.07)):
            continue
        if bw / max(1.0, float(bh)) < float(cfg.get("min_aspect_wh", 2.0)):
            continue

        center_overlap = _center_overlap_ratio(x, bw, image_w, central_band)
        if center_overlap < float(cfg.get("min_center_overlap", 0.10)):
            continue

        passed.append({
            "box_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
            "area": area,
            "width_ratio": float(bw / max(1.0, image_w)),
            "height_px": float(bh),
            "aspect_wh": float(bw / max(1.0, bh)),
            "center_overlap": float(center_overlap),
            "center_y_norm": float((y + bh * 0.5) / max(1.0, image_h)),
        })

    passed.sort(key=lambda item: (item["area"], item["width_ratio"]), reverse=True)
    return {
        "active": bool(passed),
        "contours": passed,
        "best": passed[0] if passed else None,
        "mask": roi_mask,
        "roi": [float(roi[0]), float(roi[1])],
        "roi_y0": y0,
        "roi_y1": y1,
    }


def detect_redline(bgr, cfg):
    """frame 하나에서 final redline 관측값을 만든다.

    반환값의 active=True는 아직 최종 정지 명령이 아니다.
    update_redline_event_state()가 vote/cooldown을 거쳐 final_redline 이벤트로 바꾼다.
    """
    if not bool(cfg.get("enabled", True)):
        return {"active": False, "contours": [], "best": None, "mask": None, "roi_y0": None}

    h, w = bgr.shape[:2]
    mask = hsv_mask(bgr, cfg)
    default_arm = [float(cfg.get("bottom_roi_y0", 0.70)), 1.0]
    arm = _detect_redline_in_roi(mask, cfg, w, h, cfg.get("arm_roi", default_arm))
    stop = _detect_redline_in_roi(mask, cfg, w, h, cfg.get("stop_roi", [0.55, default_arm[0]]))
    combined_mask = cv2.bitwise_or(arm["mask"], stop["mask"])

    return {
        # 기존 코드 호환용: active/best는 arm ROI 기준이다.
        "active": bool(arm["active"]),
        "contours": arm["contours"],
        "best": arm["best"],
        "mask": combined_mask,
        "roi_y0": arm["roi_y0"],
        "arm_active": bool(arm["active"]),
        "arm_contours": arm["contours"],
        "arm_best": arm["best"],
        "arm_roi": arm["roi"],
        "stop_active": bool(stop["active"]),
        "stop_contours": stop["contours"],
        "stop_best": stop["best"],
        "stop_roi": stop["roi"],
    }


def update_redline_event_state(state, observation, cfg, now_sec, is_new_observation=True):
    """redline 관측값을 state_machine용 one-shot final_redline 이벤트로 바꾼다.

    arm ROI에서 먼저 redline을 본 뒤, stop ROI에서 다시 확인되면 event를 낸다.
    stop ROI를 못 봐도 max_stop_sec이 지나면 기존 delay 방식처럼 fallback 정지한다.
    """
    if not bool(cfg.get("enabled", True)):
        return {"events": [], "observation": observation, "history_hits": 0, "cooldown_active": False}

    if is_new_observation:
        arm_active = bool((observation or {}).get("arm_active", (observation or {}).get("active", False)))
        stop_active = bool((observation or {}).get("stop_active", False))
        state["arm_history"].append(arm_active)
        state["stop_history"].append(stop_active)
        state["last_observation"] = observation
    else:
        observation = state.get("last_observation", observation)

    arm_hits = sum(bool(x) for x in state["arm_history"])
    stop_hits = sum(bool(x) for x in state["stop_history"])
    arm_required = int(cfg.get("arm_required_hits", cfg.get("required_hits", 1)))
    stop_required = int(cfg.get("stop_required_hits", cfg.get("required_hits", 1)))
    since_emit = float(now_sec) - float(state.get("last_emit_time", -1e9))
    cooldown = float(cfg.get("cooldown_sec", 4.0))
    cooldown_active = since_emit < cooldown

    events = []
    if (not bool(state.get("armed", False))) and arm_hits >= arm_required and not cooldown_active:
        state["armed"] = True
        state["armed_at"] = float(now_sec)
        state["stop_history"].clear()
        stop_hits = 0

    elapsed = float(now_sec) - float(state.get("armed_at", now_sec))
    can_stop = bool(state.get("armed", False)) and elapsed >= float(cfg.get("min_stop_sec", 0.0))
    stop_seen = stop_hits >= stop_required
    timed_out = bool(state.get("armed", False)) and elapsed >= float(cfg.get("max_stop_sec", 6.0))

    if can_stop and (stop_seen or timed_out) and not cooldown_active:
        state["last_emit_time"] = float(now_sec)
        state["arm_history"].clear()
        state["stop_history"].clear()
        state["armed"] = False
        events.append({
            "name": "final_redline",
            "source": "redline",
            "stop_delay_sec": float(cfg.get("stop_delay_sec", 0.8)),
            "best": (observation or {}).get("stop_best") or (observation or {}).get("arm_best") or (observation or {}).get("best"),
            "redline_reason": "stop_roi" if stop_seen else "max_stop_sec",
            "time_sec": float(now_sec),
        })

    return {
        "events": events,
        "observation": observation,
        "history_hits": int(arm_hits),
        "required_hits": int(arm_required),
        "arm_hits": int(arm_hits),
        "arm_required": int(arm_required),
        "stop_hits": int(stop_hits),
        "stop_required": int(stop_required),
        "armed": bool(state.get("armed", False)),
        "armed_elapsed": float(elapsed) if bool(state.get("armed", False)) else 0.0,
        "cooldown_active": bool(cooldown_active),
    }


def compact_redline_debug(observation, update=None):
    """main_log에서 한 줄로 보기 좋은 redline 상태 문자열."""
    obs = observation or {}
    update = update or {}
    best = obs.get("stop_best") or obs.get("arm_best") or obs.get("best") or {}
    arm_active = int(bool(obs.get("arm_active", obs.get("active", False))))
    stop_active = int(bool(obs.get("stop_active", False)))
    arm_hits = update.get("arm_hits", update.get("history_hits", 0))
    arm_required = update.get("arm_required", update.get("required_hits", "?"))
    stop_hits = update.get("stop_hits", 0)
    stop_required = update.get("stop_required", "?")
    armed = int(bool(update.get("armed", False)))
    elapsed = float(update.get("armed_elapsed", 0.0))
    if best:
        return (
            f"redline arm={arm_active} {arm_hits}/{arm_required} stop={stop_active} {stop_hits}/{stop_required} "
            f"armed={armed} {elapsed:.1f}s "
            f"w={float(best.get('width_ratio', 0.0)):.2f} "
            f"area={float(best.get('area', 0.0)):.0f} "
            f"delay={float(update.get('events', [{}])[0].get('stop_delay_sec', 0.0)):.1f}s"
            if update.get("events") else
            f"redline arm={arm_active} {arm_hits}/{arm_required} stop={stop_active} {stop_hits}/{stop_required} "
            f"armed={armed} {elapsed:.1f}s "
            f"w={float(best.get('width_ratio', 0.0)):.2f} area={float(best.get('area', 0.0)):.0f}"
        )
    return (
        f"redline arm={arm_active} {arm_hits}/{arm_required} "
        f"stop={stop_active} {stop_hits}/{stop_required} armed={armed} {elapsed:.1f}s"
    )

