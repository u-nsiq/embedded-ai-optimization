from __future__ import annotations


def init_sign_trigger_state(cfg):
    """class별 hit streak와 cooldown 시간을 저장한다."""
    classes = cfg.get("classes", {})
    return {
        "hit_streak": {name: 0 for name in classes},
        "last_fire_time": {name: -1e9 for name in classes},
        "exit_armed": {name: False for name in classes},
        "exit_missing": {name: 0 for name in classes},
        "exit_last_det": {name: None for name in classes},
        "last_debug": [],
        "last_events": [],
    }


def _merge_dict(base, override):
    out = dict(base or {})
    out.update(override or {})
    return out


def _merge_policy(base, override):
    """policy를 합치되, roi만 nested dict로 안전하게 합친다.

    현장 튜닝 중 class별로 roi={"x_min": 0.55}처럼 한 항목만 덮어써도
    x_max/y_min/y_max가 사라지지 않게 하기 위한 방어 코드다.
    """
    out = _merge_dict(base, override)
    if "roi" in (base or {}) or "roi" in (override or {}):
        out["roi"] = _merge_dict((base or {}).get("roi", {}), (override or {}).get("roi", {}))
    return out


def build_class_policy(class_name, cfg):
    """default -> group -> class 순서로 policy를 합친다."""
    default = dict(cfg.get("default", {}))
    class_cfg = dict(cfg.get("classes", {}).get(class_name, {}))
    group_name = class_cfg.get("group")
    group_cfg = dict(cfg.get("groups", {}).get(group_name, {})) if group_name else {}
    return _merge_policy(_merge_policy(default, group_cfg), class_cfg)


def _inside_roi(det, roi):
    cx = float(det.get("center_x_norm", 0.0))
    cy = float(det.get("center_y_norm", 0.0))
    return (
        float(roi.get("x_min", 0.0)) <= cx <= float(roi.get("x_max", 1.0))
        and float(roi.get("y_min", 0.0)) <= cy <= float(roi.get("y_max", 1.0))
    )


def evaluate_detection(det, policy):
    """detection 하나가 event trigger 조건을 만족하는지와 실패 이유를 반환한다."""
    conf = float(det.get("confidence", 0.0))
    box_size = float(det.get("box_size", 0.0))

    min_conf = float(policy.get("min_conf", 0.0))
    if conf < min_conf:
        return False, f"low_conf {conf:.2f} < {min_conf:.2f}"

    roi = policy.get("roi", {})
    if not _inside_roi(det, roi):
        cx = float(det.get("center_x_norm", 0.0))
        cy = float(det.get("center_y_norm", 0.0))
        return False, f"outside_roi cx={cx:.2f} cy={cy:.2f}"

    need_size = float(policy.get("fire_box_size", 0.0))
    if box_size < need_size:
        return False, f"too_far size={box_size:.2f} < {need_size:.2f}"

    # 아래 조건들은 기본 설계에는 거의 쓰지 않지만, 현장 emergency filter용으로 열어둔다.
    if "min_bottom_y_norm" in policy and float(det.get("bottom_y_norm", 0.0)) < float(policy["min_bottom_y_norm"]):
        return False, f"bottom_y {float(det.get('bottom_y_norm', 0.0)):.2f} < {float(policy['min_bottom_y_norm']):.2f}"
    if "max_top_y_norm" in policy and float(det.get("top_y_norm", 1.0)) > float(policy["max_top_y_norm"]):
        return False, f"top_y {float(det.get('top_y_norm', 1.0)):.2f} > {float(policy['max_top_y_norm']):.2f}"
    if "min_height_ratio" in policy and float(det.get("height_ratio", 0.0)) < float(policy["min_height_ratio"]):
        return False, f"height {float(det.get('height_ratio', 0.0)):.2f} < {float(policy['min_height_ratio']):.2f}"
    if "min_width_ratio" in policy and float(det.get("width_ratio", 0.0)) < float(policy["min_width_ratio"]):
        return False, f"width {float(det.get('width_ratio', 0.0)):.2f} < {float(policy['min_width_ratio']):.2f}"
    if "min_aspect_hw" in policy and float(det.get("aspect_hw", 0.0)) < float(policy["min_aspect_hw"]):
        return False, f"aspect_hw {float(det.get('aspect_hw', 0.0)):.2f} < {float(policy['min_aspect_hw']):.2f}"

    return True, "pass"


def _reject_should_reset_streak(reason):
    """표지판이 사라졌다고 볼 수 있는 reject만 streak를 끊는다.

    too_far/shape reject는 "계속 보이지만 아직 조건만 부족한 상태"라서
    threshold 근처 흔들림 때문에 required_hits를 영원히 못 채우는 일을 막기 위해 streak를 유지한다.
    """
    reason = str(reason or "")
    return (
        reason.startswith("not_seen")
        or reason.startswith("low_conf")
        or reason.startswith("outside_roi")
    )


def _reject_sort_key(item):
    """디버그용 rejected 후보 정렬.

    too_far는 confidence보다 box_size가 더 중요하다. 현장에서는
    "얼마나 가까워졌는데도 안 터졌는지"가 튜닝 핵심이기 때문이다.
    """
    det, reason = item
    conf = float(det.get("confidence", 0.0))
    box_size = float(det.get("box_size", 0.0))
    if str(reason).startswith("too_far"):
        return (1, box_size, conf)
    return (0, conf, box_size)


def _best_detection_for_class(detections, class_name, policy):
    """같은 class detection 중 event 조건을 가장 잘 만족하는 bbox를 고른다.

    반환값은 (passed_detection, reason, rejected_detection)이다.
    passed_detection이 없더라도 rejected_detection을 함께 넘겨서,
    현장 로그에 "보이긴 했는데 왜 안 터졌는지"를 찍을 수 있게 한다.
    """
    same_class = [d for d in detections if str(d.get("class_name")) == class_name]
    if not same_class:
        return None, f"not_seen {class_name}", None

    passed = []
    rejected = []
    for det in same_class:
        ok, reason = evaluate_detection(det, policy)
        if ok:
            passed.append(det)
        else:
            rejected.append((det, reason))

    if passed:
        # 가까운 표지판을 더 우선하고, 같은 거리면 confidence가 높은 것을 쓴다.
        passed.sort(key=lambda d: (float(d.get("box_size", 0.0)), float(d.get("confidence", 0.0))), reverse=True)
        return passed[0], "pass", None

    rejected.sort(key=_reject_sort_key, reverse=True)
    return None, rejected[0][1], rejected[0][0]


def _debug_row_from_reject(class_name, policy, reason, rejected_det, hit_streak):
    return {
        "class_name": class_name,
        "event_name": policy.get("event_name", class_name),
        "status": "reject",
        "reason": reason,
        "conf": None if rejected_det is None else float(rejected_det.get("confidence", 0.0)),
        "box_size": None if rejected_det is None else float(rejected_det.get("box_size", 0.0)),
        "cx": None if rejected_det is None else float(rejected_det.get("center_x_norm", 0.0)),
        "cy": None if rejected_det is None else float(rejected_det.get("center_y_norm", 0.0)),
        "required_hits": int(policy.get("required_hits", 1)),
        "hit_streak": int(hit_streak),
    }


def _event_from_detection(class_name, policy, det, now_sec):
    det = det or {}
    return {
        "name": str(policy.get("event_name", class_name)),
        "source": "sign",
        "class_name": class_name,
        "confidence": float(det.get("confidence", 0.0)),
        "box_size": float(det.get("box_size", 0.0)),
        "center_x_norm": float(det.get("center_x_norm", 0.0)),
        "center_y_norm": float(det.get("center_y_norm", 0.0)),
        "box_xyxy": det.get("box_xyxy"),
        "time_sec": float(now_sec),
    }


def _reset_exit_state(state, class_name):
    state["exit_armed"][class_name] = False
    state["exit_missing"][class_name] = 0
    state["exit_last_det"][class_name] = None


def _update_exit_after_arm(state, detections, class_name, policy, now_sec):
    """left/right용: 충분히 가까워지면 arm, 이후 FOV/ROI에서 사라질 때 fire.

    sign을 본 순간부터 일정 초를 기다리는 대신, 실제로 표지판을 지나쳐 화면에서
    사라지는 시점을 회전 기준으로 쓰기 위한 모드다.
    """
    visible_policy = dict(policy)
    visible_policy["fire_box_size"] = 0.0
    best, reason, rejected_det = _best_detection_for_class(detections or [], class_name, visible_policy)

    cooldown_sec = float(policy.get("cooldown_sec", 0.0))
    since_fire = float(now_sec) - float(state["last_fire_time"].get(class_name, -1e9))
    arm_box_size = float(policy.get("arm_box_size", policy.get("fire_box_size", 0.0)))
    exit_missing_frames = max(1, int(policy.get("exit_missing_frames", 1)))

    if since_fire < cooldown_sec:
        _reset_exit_state(state, class_name)
        state["hit_streak"][class_name] = 0
        return [], [{
            "class_name": class_name,
            "event_name": policy.get("event_name", class_name),
            "status": "cooldown",
            "reason": f"cooldown {since_fire:.1f}s < {cooldown_sec:.1f}s",
            "conf": None,
            "box_size": None,
            "cx": None,
            "cy": None,
            "required_hits": 1,
            "hit_streak": 0,
            "cooldown_sec": cooldown_sec,
            "since_fire": since_fire,
        }]

    if best is not None:
        state["exit_last_det"][class_name] = dict(best)
        state["exit_missing"][class_name] = 0
        box_size = float(best.get("box_size", 0.0))
        if box_size >= arm_box_size:
            state["exit_armed"][class_name] = True
            status = "armed"
            msg = f"armed size={box_size:.2f} >= {arm_box_size:.2f}"
        elif bool(state["exit_armed"].get(class_name, False)):
            status = "armed_visible"
            msg = f"still_visible size={box_size:.2f}"
        else:
            status = "wait_arm"
            msg = f"arm size={box_size:.2f} < {arm_box_size:.2f}"
        return [], [{
            "class_name": class_name,
            "event_name": policy.get("event_name", class_name),
            "status": status,
            "reason": msg,
            "conf": float(best.get("confidence", 0.0)),
            "box_size": box_size,
            "cx": float(best.get("center_x_norm", 0.0)),
            "cy": float(best.get("center_y_norm", 0.0)),
            "required_hits": exit_missing_frames,
            "hit_streak": int(state["exit_missing"].get(class_name, 0)),
            "cooldown_sec": cooldown_sec,
            "since_fire": since_fire,
        }]

    if bool(state["exit_armed"].get(class_name, False)):
        state["exit_missing"][class_name] = int(state["exit_missing"].get(class_name, 0)) + 1
        missing = int(state["exit_missing"][class_name])
        if missing >= exit_missing_frames:
            det = state["exit_last_det"].get(class_name) or rejected_det or {}
            state["last_fire_time"][class_name] = float(now_sec)
            _reset_exit_state(state, class_name)
            state["hit_streak"][class_name] = 0
            event = _event_from_detection(class_name, policy, det, now_sec)
            return [event], [{
                "class_name": class_name,
                "event_name": policy.get("event_name", class_name),
                "status": "fire",
                "reason": f"exit_after_arm missing={missing}/{exit_missing_frames}",
                "conf": event["confidence"],
                "box_size": event["box_size"],
                "cx": event["center_x_norm"],
                "cy": event["center_y_norm"],
                "required_hits": exit_missing_frames,
                "hit_streak": 0,
                "cooldown_sec": cooldown_sec,
                "since_fire": since_fire,
            }]
        return [], [{
            "class_name": class_name,
            "event_name": policy.get("event_name", class_name),
            "status": "armed_missing",
            "reason": f"missing {missing}/{exit_missing_frames}",
            "conf": None if rejected_det is None else float(rejected_det.get("confidence", 0.0)),
            "box_size": None if rejected_det is None else float(rejected_det.get("box_size", 0.0)),
            "cx": None if rejected_det is None else float(rejected_det.get("center_x_norm", 0.0)),
            "cy": None if rejected_det is None else float(rejected_det.get("center_y_norm", 0.0)),
            "required_hits": exit_missing_frames,
            "hit_streak": missing,
            "cooldown_sec": cooldown_sec,
            "since_fire": since_fire,
        }]

    state["hit_streak"][class_name] = 0
    return [], [_debug_row_from_reject(class_name, policy, reason, rejected_det, 0)]


def update_sign_trigger(state, detections, cfg, now_sec):
    """YOLO detections -> one-shot sign events.

    반환값:
      events:
        이번 frame에서 새로 발생한 이벤트 목록.

      debug:
        class별 판단 로그. 왜 안 터졌는지 보는 데 사용한다.
    """
    if not bool(cfg.get("enabled", True)):
        return {"events": [], "debug": []}

    events = []
    debug = []

    for class_name in cfg.get("classes", {}):
        policy = build_class_policy(class_name, cfg)
        if str(policy.get("trigger_mode", "immediate")) == "exit_after_arm":
            new_events, new_debug = _update_exit_after_arm(state, detections or [], class_name, policy, now_sec)
            events.extend(new_events)
            debug.extend(new_debug)
            continue

        best, reason, rejected_det = _best_detection_for_class(detections or [], class_name, policy)

        if best is None:
            if _reject_should_reset_streak(reason):
                state["hit_streak"][class_name] = 0
            hit_streak = int(state["hit_streak"].get(class_name, 0))
            debug.append(_debug_row_from_reject(class_name, policy, reason, rejected_det, hit_streak))
            continue

        required_hits = int(policy.get("required_hits", 1))
        cooldown_sec = float(policy.get("cooldown_sec", 0.0))
        since_fire = float(now_sec) - float(state["last_fire_time"].get(class_name, -1e9))

        debug_row = {
            "class_name": class_name,
            "event_name": policy.get("event_name", class_name),
            "status": "candidate",
            "reason": reason,
            "conf": float(best.get("confidence", 0.0)),
            "box_size": float(best.get("box_size", 0.0)),
            "cx": float(best.get("center_x_norm", 0.0)),
            "cy": float(best.get("center_y_norm", 0.0)),
            "required_hits": required_hits,
            "hit_streak": int(state["hit_streak"].get(class_name, 0)),
            "cooldown_sec": cooldown_sec,
            "since_fire": since_fire,
        }

        # cooldown 중에는 hit을 누적하지 않는다.
        # 같은 표지판을 지나치는 동안 streak가 쌓였다가 cooldown 종료 직후 재발화하는 일을 막는다.
        if since_fire < cooldown_sec:
            state["hit_streak"][class_name] = 0
            debug_row["status"] = "cooldown"
            debug_row["hit_streak"] = 0
            debug_row["reason"] = f"cooldown {since_fire:.1f}s < {cooldown_sec:.1f}s"
            debug.append(debug_row)
            continue

        state["hit_streak"][class_name] = int(state["hit_streak"].get(class_name, 0)) + 1
        hit_streak = int(state["hit_streak"][class_name])
        debug_row["hit_streak"] = hit_streak

        if hit_streak < required_hits:
            debug_row["status"] = "wait_hits"
            debug_row["reason"] = f"hits {hit_streak}/{required_hits}"
            debug.append(debug_row)
            continue

        state["last_fire_time"][class_name] = float(now_sec)
        state["hit_streak"][class_name] = 0
        debug_row["status"] = "fire"
        debug_row["hit_streak"] = 0
        debug_row["reason"] = "fire"
        debug.append(debug_row)

        events.append({
            "name": str(policy.get("event_name", class_name)),
            "source": "sign",
            "class_name": class_name,
            "confidence": float(best.get("confidence", 0.0)),
            "box_size": float(best.get("box_size", 0.0)),
            "center_x_norm": float(best.get("center_x_norm", 0.0)),
            "center_y_norm": float(best.get("center_y_norm", 0.0)),
            "box_xyxy": best.get("box_xyxy"),
            "time_sec": float(now_sec),
        })

    # Event priority는 state_machine이 configs/state.py 기준으로 정한다.
    # trigger 내부에서는 confidence/box_size 기준으로만 정렬한다.
    events.sort(key=lambda e: (float(e.get("confidence", 0.0)), float(e.get("box_size", 0.0))), reverse=True)
    state["last_debug"] = debug
    state["last_events"] = events
    return {"events": events, "debug": debug}


def _display_name(row):
    class_name = str(row.get("class_name"))
    event_name = str(row.get("event_name") or class_name)
    if event_name != class_name:
        return f"{event_name}({class_name})"
    return class_name


def compact_sign_debug(debug_rows, max_rows=4):
    """터미널에 한 줄로 보기 좋은 디버그 문자열을 만든다."""
    status_priority = {
        "fire": 0,
        "wait_hits": 1,
        "armed": 2,
        "armed_missing": 3,
        "cooldown": 4,
        "wait_arm": 5,
        "armed_visible": 6,
        "candidate": 7,
        "reject": 8,
    }
    interesting = [
        row
        for row in debug_rows
        if row.get("status") in {"fire", "wait_hits", "cooldown", "armed", "armed_missing", "wait_arm", "armed_visible"} or row.get("conf") is not None
    ]
    interesting = sorted(
        interesting,
        key=lambda row: (
            status_priority.get(row.get("status"), 9),
            -float(row.get("box_size") or 0.0),
            -float(row.get("conf") or 0.0),
        ),
    )[: int(max_rows)]

    parts = []
    for row in interesting:
        label = _display_name(row)
        status = row.get("status")
        if row.get("conf") is None:
            parts.append(f"{label}:{status}:{row.get('reason')}")
        else:
            parts.append(
                f"{label}:{status}:conf={row.get('conf'):.2f},size={row.get('box_size'):.2f},"
                f"hits={row.get('hit_streak')}/{row.get('required_hits')}:{row.get('reason')}"
            )
    return " | ".join(parts)


