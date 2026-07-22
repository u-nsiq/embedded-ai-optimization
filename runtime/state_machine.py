from __future__ import annotations

import math

PHASE_EVENTS = {"left", "right", "traffic_green", "traffic_red", "stop", "final_redline"}
EFFECT_EVENTS = {"speed_20", "straight", "horn"}


def clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _event_priority(event, cfg):
    """Event 우선순위는 configs/state.py의 STATE["event_priority"]에서만 관리한다."""
    name = str(event.get("name", ""))
    priorities = cfg.get("event_priority", {})
    return int(priorities.get(name, 0))

def init_state_machine(now_sec=0.0):
    now = float(now_sec)
    return {
        "race_phase": "START_IGNORE_REDLINE",
        "phase": "LANE",
        "started_at": now,
        "phase_started_at": now,
        "last_reason": "init",
        "last_events": [],
        "last_accepted_event": None,
        "stop_until": 0.0,
        "after_stop": "LANE",
        "turn_direction": None,
        "turn_source": "lane",
        "turn_mode": "curve",
        "turn_min_until": 0.0,
        "turn_max_until": 0.0,
        "turn_stable_frames": 0,
        "pending_turn_direction": None,
        "pending_turn_until": 0.0,
        "pending_turn_delay_sec": 0.0,
        "pending_traffic_action": None,
        "pending_traffic_until": 0.0,
        "post_turn_until": 0.0,
        "speed_limit_until": 0.0,
        "straight_until": 0.0,
        "horn_until": 0.0,
        "lost_since": None,
        "emergency_stop": False,
        "final_stop_at": math.inf,
        "final_stop_pending": False,
    }


def _set_phase(state, phase, now, reason):
    state["phase"] = str(phase)
    state["phase_started_at"] = float(now)
    state["last_reason"] = str(reason)


def _lane_is_lost(lane_signal):
    return str(lane_signal.get("lane_state", "")).startswith("lost")


def _start_pending_turn(state, direction, now, cfg, reason):
    direction = str(direction)
    _set_phase(state, "PENDING_LEFT" if direction == "left" else "PENDING_RIGHT", now, reason)
    state["pending_turn_direction"] = direction

    # left/right는 회전 시작 위치가 다를 수 있으므로 delay를 따로 둔다.
    # 없으면 기존 left_right_delay_sec를 fallback으로 사용한다.
    fallback_delay = float(cfg.get("left_right_delay_sec", 0.0))
    delay_key = "left_delay_sec" if direction == "left" else "right_delay_sec"
    delay_sec = float(cfg.get(delay_key, fallback_delay))
    state["pending_turn_delay_sec"] = delay_sec
    state["pending_turn_until"] = float(now) + delay_sec


def _start_pending_traffic(state, action, now, cfg, reason):
    """신호등을 본 뒤, 바로 행동하지 않고 lane으로 조금 더 전진한다.

    action:
      green_right : delay 후 즉시 우회전
      red_stop    : delay 후 3초 정지, 그 다음 우회전
    """
    action = str(action)
    phase = "PENDING_TRAFFIC_GREEN" if action == "green_right" else "PENDING_TRAFFIC_RED"
    _set_phase(state, phase, now, reason)
    state["pending_traffic_action"] = action
    state["pending_traffic_until"] = float(now) + float(cfg.get("traffic_light_delay_sec", 0.0))


def _start_turn(state, direction, now, cfg, reason, source="left_right"):
    direction = str(direction)
    source = str(source)
    _set_phase(state, "TURN_LEFT" if direction == "left" else "TURN_RIGHT", now, reason)
    state["turn_direction"] = direction
    state["turn_source"] = source
    if source == "traffic":
        state["turn_mode"] = str(cfg.get("traffic_turn_mode", "pivot"))
    else:
        state["turn_mode"] = str(cfg.get("left_right_turn_mode", "curve"))
    state["turn_min_until"] = float(now) + float(cfg["turn_min_sec"])
    state["turn_max_until"] = float(now) + float(cfg["turn_max_sec"])
    state["turn_stable_frames"] = 0


def _start_stop(state, now, cfg, stop_sec, after_stop, reason):
    _set_phase(state, "STOPPED", now, reason)
    state["stop_until"] = float(now) + float(stop_sec)
    state["after_stop"] = str(after_stop)


def _start_post_turn_straight(state, now, cfg, reason):
    # pivot으로 90도 회전한 뒤 바로 lane에 맡기면 흔들릴 수 있다.
    # 짧게 직진시켜 차체 방향을 새 도로축에 맞춘 뒤 LANE으로 복귀한다.
    _set_phase(state, "POST_TURN_STRAIGHT", now, reason)
    state["post_turn_until"] = float(now) + float(cfg.get("post_turn_straight_sec", 0.0))


def _phase_steer_value(phase, cfg):
    if phase == "TURN_LEFT":
        return -abs(float(cfg["turn_steer"]))
    if phase == "TURN_RIGHT":
        return abs(float(cfg["turn_steer"]))
    return 0.0


def _apply_race_timer(state, now, cfg):
    if state["race_phase"] == "START_IGNORE_REDLINE":
        if float(now) - float(state["started_at"]) >= float(cfg["redline_arm_sec"]):
            state["race_phase"] = "RUNNING"


def _apply_final_stop_timer(state, now):
    if state["race_phase"] == "RUNNING" and bool(state.get("final_stop_pending", False)):
        if float(now) >= float(state.get("final_stop_at", math.inf)):
            state["race_phase"] = "FINISHED"
            _set_phase(state, "FINISHED", now, "final_redline_stop")


def _split_events(events, cfg):
    phase_events = []
    effect_events = []
    for event in events or []:
        name = str(event.get("name", ""))
        if name in PHASE_EVENTS:
            phase_events.append(event)
        elif name in EFFECT_EVENTS:
            effect_events.append(event)
    phase_events.sort(key=lambda e: _event_priority(e, cfg), reverse=True)
    effect_events.sort(key=lambda e: _event_priority(e, cfg), reverse=True)
    return phase_events, effect_events


def _apply_effect_events(state, effect_events, now, cfg):
    # effect 지속 시간은 configs/state.py만 본다.
    # sign trigger는 "언제 event를 낼지"만 정하고, event 후 행동 시간은 여기서 통일한다.
    for event in effect_events:
        name = str(event.get("name", ""))
        if name == "speed_20":
            sec = float(cfg["speed_20_hold_sec"])
            state["speed_limit_until"] = max(float(state["speed_limit_until"]), float(now) + sec)
        elif name == "straight":
            sec = float(cfg["straight_hold_sec"])
            state["straight_until"] = max(float(state["straight_until"]), float(now) + sec)
        elif name == "horn":
            sec = float(cfg["horn_sec"])
            state["horn_until"] = max(float(state["horn_until"]), float(now) + sec)

def _apply_phase_event(state, event, now, cfg):
    name = str(event.get("name", ""))

    if name == "final_redline":
        if state["race_phase"] == "RUNNING":
            delay = max(0.0, float(event.get("stop_delay_sec", 0.0)))
            target = float(now) + delay
            state["final_stop_at"] = min(float(state.get("final_stop_at", math.inf)), target)
            state["final_stop_pending"] = True
            state["last_accepted_event"] = name
            state["last_reason"] = f"event:final_redline_pending:{delay:.1f}s"
        else:
            state["last_reason"] = "ignore_start_redline"
        return

    if state["phase"] not in {"LANE", "POST_TURN_STRAIGHT"}:
        state["last_reason"] = f"ignore_event_during_{state['phase'].lower()}:{name}"
        return

    if name == "left":
        _start_pending_turn(state, "left", now, cfg, "event:left_pending")
        state["last_accepted_event"] = name
    elif name == "right":
        _start_pending_turn(state, "right", now, cfg, "event:right_pending")
        state["last_accepted_event"] = name
    elif name == "traffic_green":
        _start_pending_traffic(state, "green_right", now, cfg, "event:traffic_green_pending")
        state["last_accepted_event"] = name
    elif name == "traffic_red":
        _start_pending_traffic(state, "red_stop", now, cfg, "event:traffic_red_pending")
        state["last_accepted_event"] = name
    elif name == "stop":
        _start_stop(state, now, cfg, float(cfg["stop_sec"]), "LANE", "event:stop")
        state["last_accepted_event"] = name


def _apply_events(state, events, now, cfg):
    phase_events, effect_events = _split_events(events, cfg)
    if state["phase"] in {"FINISHED", "EMERGENCY_STOP"}:
        return

    if state["phase"] == "STOPPED":
        for event in phase_events:
            if str(event.get("name", "")) == "final_redline":
                _apply_phase_event(state, event, now, cfg)
                break
        return

    _apply_effect_events(state, effect_events, now, cfg)
    if phase_events:
        _apply_phase_event(state, phase_events[0], now, cfg)


def _auto_transition(state, lane_signal, now, cfg):
    phase = state["phase"]

    if phase in {"PENDING_LEFT", "PENDING_RIGHT"}:
        if float(now) >= float(state.get("pending_turn_until", 0.0)):
            direction = str(state.get("pending_turn_direction") or ("left" if phase == "PENDING_LEFT" else "right"))
            _start_turn(state, direction, now, cfg, f"pending_done:{direction}", source="left_right")
        return

    if phase in {"PENDING_TRAFFIC_GREEN", "PENDING_TRAFFIC_RED"}:
        if float(now) >= float(state.get("pending_traffic_until", 0.0)):
            action = str(state.get("pending_traffic_action") or "")
            if action == "red_stop":
                _start_stop(state, now, cfg, float(cfg["traffic_red_stop_sec"]), "TURN_RIGHT", "traffic_pending_done:red_stop")
            else:
                _start_turn(state, "right", now, cfg, "traffic_pending_done:green_right", source="traffic")
        return

    if phase == "STOPPED":
        if float(now) >= float(state["stop_until"]):
            if state.get("after_stop") == "TURN_RIGHT":
                _start_turn(state, "right", now, cfg, "after_stop:turn_right", source="traffic")
            else:
                _set_phase(state, "LANE", now, "stop_done")
        return

    if phase in {"TURN_LEFT", "TURN_RIGHT"}:
        if float(now) < float(state["turn_min_until"]):
            state["turn_stable_frames"] = 0
            return

        lane_stable = (
            bool(lane_signal.get("stable_forward", False))
            and float(lane_signal.get("quality", 0.0)) >= float(cfg.get("turn_stable_quality", 0.35))
            and not _lane_is_lost(lane_signal)
        )
        if lane_stable:
            state["turn_stable_frames"] = int(state.get("turn_stable_frames", 0)) + 1
        else:
            state["turn_stable_frames"] = 0

        stable_done = (
            str(state.get("turn_mode", "curve")) == "curve"
            and int(state.get("turn_stable_frames", 0)) >= int(cfg.get("turn_stable_frames", 2))
        )
        timed_out = float(now) >= float(state["turn_max_until"])
        if stable_done or timed_out:
            reason = "turn_lane_stable:post_turn_straight" if stable_done else "turn_timeout:post_turn_straight"
            _start_post_turn_straight(state, now, cfg, reason)
        return

    if phase == "POST_TURN_STRAIGHT":
        if float(now) >= float(state.get("post_turn_until", 0.0)):
            _set_phase(state, "LANE", now, "post_turn_straight_done")
        return

def _update_lost_monitor(state, lane_signal, now, cfg):
    if state["phase"] != "LANE":
        state["lost_since"] = None
        return
    if _lane_is_lost(lane_signal):
        if state["lost_since"] is None:
            state["lost_since"] = float(now)
        elif float(now) - float(state["lost_since"]) >= float(cfg["emergency_stop_lost_sec"]):
            state["emergency_stop"] = True
            _set_phase(state, "EMERGENCY_STOP", now, "emergency_lost")
    else:
        state["lost_since"] = None


def _base_command_from_phase(state, lane_signal, now, cfg, use_lane_speed_scale):
    lane_steer = float(lane_signal.get("steer_norm", 0.0))
    lane_speed = float(lane_signal.get("speed_scale", 1.0)) if bool(use_lane_speed_scale) else 1.0
    phase = state["phase"]

    if phase in {"FINISHED", "EMERGENCY_STOP"}:
        return 0.0, 0.0, True, phase.lower()
    if phase == "STOPPED":
        return 0.0, 0.0, True, "stopped"
    if phase in {"PENDING_LEFT", "PENDING_RIGHT", "PENDING_TRAFFIC_GREEN", "PENDING_TRAFFIC_RED"}:
        # event 발생 후 pivot 지점까지 이동하는 구간이다.
        # 이때 lane steer가 branch/교차로 때문에 흔들리면 회전 시작 위치가 틀어지므로 조향만 작게 제한한다.
        pending_limit = abs(float(cfg.get("pending_max_abs_steer", 0.12)))
        pending_speed = float(cfg.get("pending_speed_scale", lane_speed))
        return clamp(lane_steer, -pending_limit, pending_limit), pending_speed, False, phase.lower()
    if phase in {"TURN_LEFT", "TURN_RIGHT"}:
        return _phase_steer_value(phase, cfg), float(cfg["turn_speed_scale"]), False, phase.lower()
    if phase == "POST_TURN_STRAIGHT":
        return 0.0, float(cfg.get("post_turn_straight_speed_scale", 0.85)), False, "post_turn_straight"
    return lane_steer, lane_speed, False, "lane"


def _apply_effects_to_command(state, steer, speed, phase, now, cfg):
    reason_suffix = []
    if phase == "LANE" and float(now) < float(state.get("straight_until", 0.0)):
        steer = clamp(steer, -float(cfg["straight_max_abs_steer"]), float(cfg["straight_max_abs_steer"]))
        speed = min(float(speed), float(cfg["straight_speed_scale"]))
        reason_suffix.append("straight")
    if float(now) < float(state.get("speed_limit_until", 0.0)):
        speed = min(float(speed), float(cfg["speed_20_scale"]))
        reason_suffix.append("speed20")
    horn_on = float(now) < float(state.get("horn_until", 0.0))
    return float(steer), float(speed), bool(horn_on), "+".join(reason_suffix)


def update_state_machine(state, lane_signal, events, now_sec, cfg, use_lane_speed_scale=True):
    now = float(now_sec)
    events = list(events or [])
    events.sort(key=lambda e: _event_priority(e, cfg), reverse=True)
    state["last_events"] = events
    state["last_accepted_event"] = None

    _apply_race_timer(state, now, cfg)
    _apply_events(state, events, now, cfg)
    _apply_final_stop_timer(state, now)
    _auto_transition(state, lane_signal, now, cfg)
    _update_lost_monitor(state, lane_signal, now, cfg)

    steer, speed, force_stop, phase_reason = _base_command_from_phase(
        state, lane_signal, now, cfg, use_lane_speed_scale=use_lane_speed_scale
    )
    steer, speed, horn_on, effect_reason = _apply_effects_to_command(state, steer, speed, state["phase"], now, cfg)

    # left/right는 curve mode면 normal motor + curve_boost로 크게 돈다.
    # traffic은 config에 따라 pivot으로 제자리 회전을 유지할 수 있다.
    if state["phase"] in {"TURN_LEFT", "TURN_RIGHT"}:
        turn_mode = str(state.get("turn_mode", cfg.get("left_right_turn_mode", "curve")))
        if turn_mode == "pivot":
            motor_mode = "pivot"
            pivot_pwm = float(cfg.get("turn_pivot_pwm", 0.0))
        else:
            motor_mode = "normal"
            pivot_pwm = 0.0
    else:
        motor_mode = "normal"
        pivot_pwm = 0.0

    if state["race_phase"] == "FINISHED":
        state["phase"] = "FINISHED"
        steer, speed, force_stop = 0.0, 0.0, True
        motor_mode = "normal"
        pivot_pwm = 0.0

    last_reason = str(state.get("last_reason", ""))
    if state["phase"] == "LANE" and last_reason in {"", "init", "stop_done", "post_turn_straight_done"}:
        reason = phase_reason
    else:
        reason = last_reason or phase_reason
    if effect_reason:
        reason = f"{reason}+{effect_reason}"

    lane_state = str(lane_signal.get("lane_state", ""))
    lane_quality = float(lane_signal.get("quality", 0.0))
    return {
        "steer_norm": clamp(steer, -1.0, 1.0),
        "speed_scale": clamp(speed, 0.0, 1.0),
        "force_stop": bool(force_stop),
        "horn_on": bool(horn_on),
        "motor_mode": str(motor_mode),
        "pivot_pwm": float(pivot_pwm),
        "turn_mode": str(state.get("turn_mode", "")),
        "turn_source": str(state.get("turn_source", "")),
        "turn_stable_frames": int(state.get("turn_stable_frames", 0)),
        "phase": str(state["phase"]),
        "race_phase": str(state["race_phase"]),
        "reason": str(reason),
        "accepted_event": state.get("last_accepted_event"),
        "events": events,
        "lane_steer": float(lane_signal.get("steer_norm", 0.0)),
        "lane_state": lane_state,
        "lane_quality": lane_quality,
        "lane_stable": bool(lane_signal.get("stable_forward", False)),
        "speed_limit_active": float(now) < float(state.get("speed_limit_until", 0.0)),
        "straight_active": float(now) < float(state.get("straight_until", 0.0)),
        "lost_sec": 0.0 if state.get("lost_since") is None else float(now) - float(state["lost_since"]),
        "final_stop_pending": bool(state.get("final_stop_pending", False)),
        "final_stop_in_sec": max(0.0, float(state.get("final_stop_at", math.inf)) - float(now)) if bool(state.get("final_stop_pending", False)) else 0.0,
    }




