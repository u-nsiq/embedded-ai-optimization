"""Verbose integrated runtime v2.

lane 추천값과 state machine 최종 명령을 분리해서 보여준다.
현장에서 "왜 이렇게 움직였는지"를 볼 때 이 파일을 실행한다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import time

from config import (
    CAMERA,
    HORN,
    LANE_DECODE,
    LANE_MODEL,
    LANE_POSTPROCESS,
    MOTOR,
    ROOT_DIR,
    RUNTIME,
    REDLINE_EVENT,
    SIGN_MODEL,
    SIGN_TRIGGER,
    STATE,
)
from lane.pipeline import init_lane_pipeline, run_lane_pipeline
from sign.detector import detect_signs, init_sign_detector
from sign.trigger import compact_sign_debug, init_sign_trigger_state, update_sign_trigger
from redline.detector import compact_redline_debug, detect_redline, init_redline_state, update_redline_event_state
from state_machine import init_state_machine, update_state_machine
from utils.camera import LiveCamera
from utils.horn import HornBuzzer
from utils.motor import DifferentialMotor, command_from_steer
from utils.timing import sleep_to_target


def _event_names(events):
    return [str(e.get("name")) for e in events or []]


def main():
    camera = LiveCamera(CAMERA)
    motor = DifferentialMotor(MOTOR)
    horn = HornBuzzer(HORN)
    try:
        first_frame = camera.read_bgr()
        lane = init_lane_pipeline(ROOT_DIR, LANE_MODEL, LANE_DECODE, LANE_POSTPROCESS, first_frame=first_frame)
        sign_detector = init_sign_detector(ROOT_DIR, SIGN_MODEL) if bool(SIGN_MODEL.get("enabled", True)) else None
        sign_state = init_sign_trigger_state(SIGN_TRIGGER)
        redline_state = init_redline_state(REDLINE_EVENT)
        sm_state = init_state_machine(time.perf_counter())

        print("[main_log:v2] integrated lane + sign + state-machine log")
        print(
            f"[main_log:v2] lane={lane['selected_model']} post={lane['selected_postprocess']} "
            f"sign={SIGN_MODEL.get('selected')} motors={MOTOR['enable_motors']} target_fps={RUNTIME['target_fps']}"
        )

        frame_idx = 0
        t_start = time.perf_counter()
        last_sign_debug = []
        last_sign_count = 0
        last_sign_ms = 0.0
        last_redline_observation = {"active": False, "contours": [], "best": None}
        last_redline_update = {"events": [], "history_hits": 0, "required_hits": 0}

        while True:
            t0 = time.perf_counter()
            frame = first_frame if frame_idx == 0 else camera.read_bgr()
            now = time.perf_counter()

            lane_result = run_lane_pipeline(lane, frame)
            signal = lane_result["lane_signal"]
            sign_events = []
            redline_events = []

            if sign_detector is not None and frame_idx % max(1, int(RUNTIME["sign_every_frames"])) == 0:
                detections, sign_timing = detect_signs(sign_detector, frame)
                sign_out = update_sign_trigger(sign_state, detections, SIGN_TRIGGER, now)
                sign_events = sign_out["events"]
                last_sign_debug = sign_out["debug"]
                last_sign_count = len(detections)
                last_sign_ms = float(sign_timing.get("total_ms", 0.0))

            do_redline = frame_idx % max(1, int(RUNTIME.get("redline_every_frames", 1))) == 0
            if do_redline:
                last_redline_observation = detect_redline(frame, REDLINE_EVENT)
            last_redline_update = update_redline_event_state(
                redline_state, last_redline_observation, REDLINE_EVENT, now, is_new_observation=do_redline
            )
            redline_events = last_redline_update["events"]

            control = update_state_machine(
                sm_state,
                signal,
                sign_events + redline_events,
                now,
                STATE,
                use_lane_speed_scale=bool(RUNTIME.get("use_lane_speed_scale", True)),
            )
            command = command_from_steer(
                control["steer_norm"],
                MOTOR,
                reason=control["phase"],
                speed_scale=control["speed_scale"],
                motor_mode=control.get("motor_mode", "normal"),
                pivot_pwm=control.get("pivot_pwm", None),
            )
            motor.apply(command)
            horn.apply(control["horn_on"])

            sleep_to_target(t0, float(RUNTIME["target_fps"]))
            frame_idx += 1

            if frame_idx % int(RUNTIME["log_every_frames"]) == 0:
                fps = frame_idx / max(1e-6, time.perf_counter() - t_start)
                loop_ms = (time.perf_counter() - t0) * 1000.0
                sign_msg = compact_sign_debug(last_sign_debug, max_rows=int(RUNTIME.get("sign_log_max_rows", 4)))
                redline_msg = compact_redline_debug(last_redline_observation, last_redline_update)
                print(
                    f"[{frame_idx:05d}] phase={control['phase']:<12} race={control['race_phase']:<20} "
                    f"reason={control['reason']} event={control['accepted_event']} "
                    f"lane={signal['lane_state']:<16} lane_steer={signal['steer_norm']:+.3f} "
                    f"cmd_steer={control['steer_norm']:+.3f} speed={control['speed_scale']:.2f} "
                    f"motor={control.get('motor_mode', 'normal')} "
                    f"q={signal['quality']:.2f} stable={int(signal['stable_forward'])} "
                    f"lost={control['lost_sec']:.1f}s final={int(control['final_stop_pending'])}:{control['final_stop_in_sec']:.1f}s effects="
                    f"S20:{int(control['speed_limit_active'])}/STR:{int(control['straight_active'])}/H:{int(control['horn_on'])} "
                    f"signs={last_sign_count} yolo={last_sign_ms:.1f}ms events={_event_names(sign_events + redline_events)} "
                    f"L={command.left:+.2f} R={command.right:+.2f} loop={loop_ms:.1f}ms fps={fps:.2f}"
                )
                if sign_msg:
                    print(f"          sign: {sign_msg}")
                if redline_msg:
                    print(f"          {redline_msg}")
    except KeyboardInterrupt:
        print("\n[main_log:v2] interrupted")
    finally:
        horn.close()
        motor.close()
        camera.close()


if __name__ == "__main__":
    main()

