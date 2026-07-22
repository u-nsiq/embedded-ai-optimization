"""Lane-only drive with realtime overlay + capture."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import csv
from datetime import datetime
import time

import cv2

from config import CAMERA, LANE_DECODE, LANE_MODEL, LANE_POSTPROCESS, MOTOR, ROOT_DIR, RUNTIME
from lane.pipeline import init_lane_pipeline, run_lane_pipeline
from debug.main_overlay import draw_lane_overlay
from utils.camera import LiveCamera
from utils.motor import DifferentialMotor, command_from_steer
from utils.timing import sleep_to_target


def parse_args():
    parser = argparse.ArgumentParser(description="Lane drive overlay debugger")
    parser.add_argument("--save-every", type=int, default=0, help="N frame마다 overlay jpg 저장. 0이면 자동 저장 안 함.")
    parser.add_argument("--output-dir", type=str, default="outputs/lane_overlay_debug", help="저장 폴더")
    parser.add_argument("--no-window", action="store_true", help="OpenCV 창 없이 저장/로그만 수행")
    return parser.parse_args()


def make_run_dir(base_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def open_csv(run_dir):
    path = run_dir / "lane_overlay_log.csv"
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame", "time_sec", "postprocess", "lane_state", "visible_lane_count",
            "feature_count", "steer_norm", "raw_steer", "speed_scale", "quality",
            "stable_forward", "reason", "left_pwm", "right_pwm", "loop_ms", "fps",
        ],
    )
    writer.writeheader()
    return f, writer, path


def write_csv_row(writer, frame_idx, elapsed, signal, command, loop_ms, fps):
    debug = signal.get("debug", {})
    writer.writerow({
        "frame": frame_idx,
        "time_sec": f"{elapsed:.3f}",
        "postprocess": signal["postprocess"],
        "lane_state": signal["lane_state"],
        "visible_lane_count": signal["visible_lane_count"],
        "feature_count": signal["feature_count"],
        "steer_norm": f"{signal['steer_norm']:.6f}",
        "raw_steer": f"{signal['raw_steer']:.6f}",
        "speed_scale": f"{signal['speed_scale']:.6f}",
        "quality": f"{signal['quality']:.6f}",
        "stable_forward": int(bool(signal["stable_forward"])),
        "reason": debug.get("reason", ""),
        "left_pwm": f"{command.left:.6f}",
        "right_pwm": f"{command.right:.6f}",
        "loop_ms": f"{loop_ms:.3f}",
        "fps": f"{fps:.3f}",
    })


def save_overlay(run_dir, frame_idx, vis, reason):
    path = run_dir / f"frame_{frame_idx:05d}_{reason}.jpg"
    cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return path


def main():
    args = parse_args()
    run_dir = make_run_dir(args.output_dir)
    csv_file, csv_writer, csv_path = open_csv(run_dir)

    camera = LiveCamera(CAMERA)
    motor = DifferentialMotor(MOTOR)
    try:
        first_frame = camera.read_bgr()
        lane = init_lane_pipeline(ROOT_DIR, LANE_MODEL, LANE_DECODE, LANE_POSTPROCESS, first_frame=first_frame)
        print(
            f"[lane_overlay_drive:v2] model={lane['selected_model']} post={lane['selected_postprocess']} "
            f"motors={MOTOR['enable_motors']} target_fps={RUNTIME['target_fps']}"
        )
        print(f"[lane_overlay_drive:v2] output_dir={run_dir}")
        print(f"[lane_overlay_drive:v2] csv={csv_path}")

        frame_idx = 0
        t_start = time.perf_counter()
        saved_count = 0
        while True:
            t0 = time.perf_counter()
            frame = first_frame if frame_idx == 0 else camera.read_bgr()
            lane_result = run_lane_pipeline(lane, frame)
            signal = lane_result["lane_signal"]

            speed_scale = signal["speed_scale"] if bool(RUNTIME.get("use_lane_speed_scale", True)) else 1.0
            command = command_from_steer(
                signal["steer_norm"],
                MOTOR,
                reason=f"lane_overlay:{signal['lane_state']}",
                speed_scale=speed_scale,
            )
            motor.apply(command)

            loop_ms = (time.perf_counter() - t0) * 1000.0
            frame_idx += 1
            elapsed = time.perf_counter() - t_start
            fps = frame_idx / max(1e-6, elapsed)

            vis = draw_lane_overlay(frame, lane_result, fps, loop_ms, lane["selected_model"])
            cv2.putText(
                vis,
                f"MOTOR L={command.left:+.2f} R={command.right:+.2f} speed_scale={speed_scale:.2f}",
                (18, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            write_csv_row(csv_writer, frame_idx, elapsed, signal, command, loop_ms, fps)
            if frame_idx % 10 == 0:
                csv_file.flush()

            if args.save_every > 0 and frame_idx % int(args.save_every) == 0:
                save_overlay(run_dir, frame_idx, vis, "auto")
                saved_count += 1

            if not args.no_window:
                cv2.imshow(str(RUNTIME.get("overlay_window_name", "team3_lane_overlay_drive")), vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    path = save_overlay(run_dir, frame_idx, vis, "manual")
                    saved_count += 1
                    print(f"[save] {path}")

            if frame_idx % int(RUNTIME["log_every_frames"]) == 0:
                debug = signal.get("debug", {})
                slope = float(debug.get(
                    "weighted_slope",
                    debug.get(
                        "center_slope",
                        debug.get("local_slope", debug.get("measured_heading", debug.get("single_heading", debug.get("smoothed_heading", 0.0)))),
                    ),
                ))
                print(
                    f"[{frame_idx:05d}] post={signal['postprocess']} state={signal['lane_state']:<16} "
                    f"lanes={signal['visible_lane_count']} feat={signal['feature_count']} "
                    f"steer={signal['steer_norm']:+.3f} slope={slope:+.3f} "
                    f"q={signal['quality']:.2f} spd={speed_scale:.2f} "
                    f"L={command.left:+.2f} R={command.right:+.2f} "
                    f"loop={loop_ms:.1f}ms fps={fps:.2f} saved={saved_count}"
                )

            sleep_to_target(t0, float(RUNTIME["target_fps"]))
    except KeyboardInterrupt:
        print("\n[lane_overlay_drive:v2] interrupted")
    finally:
        csv_file.flush()
        csv_file.close()
        motor.close()
        camera.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        print(f"[lane_overlay_drive:v2] saved outputs in: {run_dir}")


if __name__ == "__main__":
    main()


