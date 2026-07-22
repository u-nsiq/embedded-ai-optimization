"""Lane-only runtime.

표지판/신호등/state machine 없이 lane model만 테스트한다.
현장에서 lane postprocess와 motor 값을 잡을 때 이 파일부터 실행한다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import time

from config import CAMERA, LANE_DECODE, LANE_MODEL, LANE_POSTPROCESS, MOTOR, ROOT_DIR, RUNTIME
from lane.pipeline import init_lane_pipeline, run_lane_pipeline
from utils.camera import LiveCamera
from utils.motor import DifferentialMotor, command_from_steer
from utils.timing import sleep_to_target


def main():
    camera = LiveCamera(CAMERA)
    motor = DifferentialMotor(MOTOR)
    try:
        first_frame = camera.read_bgr()
        lane = init_lane_pipeline(ROOT_DIR, LANE_MODEL, LANE_DECODE, LANE_POSTPROCESS, first_frame=first_frame)
        print(
            f"[lane_only:v2] model={lane['selected_model']} post={lane['selected_postprocess']} "
            f"motors={MOTOR['enable_motors']} target_fps={RUNTIME['target_fps']}"
        )

        frame_idx = 0
        t_start = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            frame = first_frame if frame_idx == 0 else camera.read_bgr()
            result = run_lane_pipeline(lane, frame)
            signal = result["lane_signal"]
            debug = signal.get("debug", {})

            speed_scale = signal["speed_scale"] if bool(RUNTIME.get("use_lane_speed_scale", True)) else 1.0
            command = command_from_steer(signal["steer_norm"], MOTOR, reason=signal["lane_state"], speed_scale=speed_scale)
            motor.apply(command)

            sleep_to_target(t0, float(RUNTIME["target_fps"]))
            frame_idx += 1

            if frame_idx % int(RUNTIME["log_every_frames"]) == 0:
                fps = frame_idx / max(1e-6, time.perf_counter() - t_start)
                loop_ms = (time.perf_counter() - t0) * 1000.0
                slope = float(debug.get(
                    "weighted_slope",
                    debug.get(
                        "center_slope",
                        debug.get("local_slope", debug.get("measured_heading", debug.get("single_heading", debug.get("smoothed_heading", 0.0)))),
                    ),
                ))
                push = float(debug.get("push_term", 0.0))
                risk = float(debug.get("departure_risk", 0.0))
                print(
                    f"[{frame_idx:05d}] post={signal['postprocess']} state={signal['lane_state']:<16} "
                    f"lanes={signal['visible_lane_count']} feat={signal['feature_count']} "
                    f"steer={signal['steer_norm']:+.3f} raw={signal['raw_steer']:+.3f} "
                    f"slope={slope:+.3f} push={push:+.3f} risk={risk:.2f} "
                    f"q={signal['quality']:.2f} spd={speed_scale:.2f} "
                    f"L={command.left:+.2f} R={command.right:+.2f} loop={loop_ms:.1f}ms fps={fps:.2f}"
                )
    except KeyboardInterrupt:
        print("\n[lane_only:v2] interrupted")
    finally:
        motor.close()
        camera.close()


if __name__ == "__main__":
    main()

