"""Final drive runtime v2.

시연용 main이다. 화면 overlay나 frame 저장 없이 lane + sign + state machine만 실행한다.
현장 디버그가 필요하면 main_log.py를 먼저 사용한다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
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
from sign.trigger import init_sign_trigger_state, update_sign_trigger
from redline.detector import detect_redline, init_redline_state, update_redline_event_state
from state_machine import init_state_machine, update_state_machine
from utils.camera import LiveCamera
from utils.horn import HornBuzzer
from utils.motor import DifferentialMotor, command_from_steer
from utils.timing import sleep_to_target


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

        print(
            f"[main:v2] lane={lane['selected_model']} post={lane['selected_postprocess']} "
            f"sign={SIGN_MODEL.get('selected')} motors={MOTOR['enable_motors']} target_fps={RUNTIME['target_fps']}"
        )

        frame_idx = 0
        last_redline_observation = {"active": False, "contours": [], "best": None}
        while True:
            t0 = time.perf_counter()
            frame = first_frame if frame_idx == 0 else camera.read_bgr()
            now = time.perf_counter()

            lane_result = run_lane_pipeline(lane, frame)
            sign_events = []
            redline_events = []

            if sign_detector is not None and frame_idx % max(1, int(RUNTIME["sign_every_frames"])) == 0:
                detections, _ = detect_signs(sign_detector, frame)
                sign_out = update_sign_trigger(sign_state, detections, SIGN_TRIGGER, now)
                sign_events = sign_out["events"]

            do_redline = frame_idx % max(1, int(RUNTIME.get("redline_every_frames", 1))) == 0
            if do_redline:
                last_redline_observation = detect_redline(frame, REDLINE_EVENT)
            redline_out = update_redline_event_state(
                redline_state, last_redline_observation, REDLINE_EVENT, now, is_new_observation=do_redline
            )
            redline_events = redline_out["events"]

            control = update_state_machine(
                sm_state,
                lane_result["lane_signal"],
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
    except KeyboardInterrupt:
        print("\n[main:v2] interrupted")
    finally:
        horn.close()
        motor.close()
        camera.close()


if __name__ == "__main__":
    main()

