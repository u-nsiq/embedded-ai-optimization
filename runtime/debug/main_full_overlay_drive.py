"""Full integrated drive overlay debug.

main.py/main_log.py와 같은 전체 주행 루프를 돌리면서
lane + sign + redline + state machine 결과를 한 화면에 겹쳐 보여준다.

최종 시연용이 아니라 현장 튜닝용이다.
- 실제 모터가 켜진다. config의 MOTOR["enable_motors"] 값을 따른다.
- overlay/opencv 때문에 main.py보다 느리다.
- q: 종료, s: 현재 화면 저장
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import cv2

from config import (
    CAMERA,
    HORN,
    LANE_DECODE,
    LANE_MODEL,
    LANE_POSTPROCESS,
    MOTOR,
    REDLINE_EVENT,
    ROOT_DIR,
    RUNTIME,
    SIGN_MODEL,
    SIGN_TRIGGER,
    STATE,
)
from debug.main_overlay import draw_lane_overlay
from debug.main_redline_overlay import draw_redline_overlay
from debug.main_sign_overlay import _draw_detections, _draw_fire_banner, _draw_roi_guides, _draw_text_box
from lane.pipeline import init_lane_pipeline, run_lane_pipeline
from redline.detector import compact_redline_debug, detect_redline, init_redline_state, update_redline_event_state
from sign.detector import detect_signs, init_sign_detector
from sign.trigger import compact_sign_debug, init_sign_trigger_state, update_sign_trigger
from state_machine import init_state_machine, update_state_machine
from utils.camera import LiveCamera
from utils.horn import HornBuzzer
from utils.motor import DifferentialMotor, command_from_steer
from utils.timing import sleep_to_target


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-every", type=int, default=0, help="N frame마다 overlay 저장. 0이면 자동 저장 안 함.")
    parser.add_argument("--no-window", action="store_true", help="cv2 창 없이 저장/log만 수행.")
    parser.add_argument("--max-sign-rows", type=int, default=5, help="overlay 상단에 표시할 sign trigger row 개수.")
    return parser.parse_args()


def _event_names(events):
    return [str(e.get("name")) for e in events or []]


def _short(text, max_len=120):
    text = str(text or "")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _draw_state_overlay(
    vis,
    lane_result,
    control,
    command,
    sign_debug,
    sign_events,
    redline_observation,
    redline_update,
    frame_idx,
    loop_ms,
    fps,
):
    signal = lane_result["lane_signal"]
    sign_msg = compact_sign_debug(sign_debug, max_rows=5)
    red_msg = compact_redline_debug(redline_observation, redline_update)
    lines = [
        f"FULL DRIVE OVERLAY  q:quit s:save frame={frame_idx}",
        f"phase={control['phase']} race={control['race_phase']} reason={control['reason']}",
        f"event={control['accepted_event']} events={_event_names(sign_events + redline_update.get('events', []))}",
        f"lane={signal['lane_state']} steer_lane={signal['steer_norm']:+.3f} steer_cmd={control['steer_norm']:+.3f}",
        f"speed={control['speed_scale']:.2f} q={signal['quality']:.2f} stable={int(signal['stable_forward'])} lost={control['lost_sec']:.1f}s",
        f"motor={control.get('motor_mode', 'normal')} L={command.left:+.2f} R={command.right:+.2f} loop={loop_ms:.1f}ms fps={fps:.2f}",
        _short(f"sign: {sign_msg}", 135),
        _short(red_msg, 135),
    ]
    _draw_text_box(vis, lines, x=18, y=28)
    return vis


def _draw_redline_on_existing(vis, observation, update):
    # debug.main_redline_overlay의 draw_redline_overlay는 frame copy를 새로 만들기 때문에,
    # full overlay에서는 필요한 선/박스만 현재 vis 위에 직접 그린다.
    h, w = vis.shape[:2]
    y0 = observation.get("roi_y0")
    if y0 is None:
        y0 = int(float(REDLINE_EVENT.get("bottom_roi_y0", 0.70)) * h)
    cv2.line(vis, (0, int(y0)), (w, int(y0)), (0, 255, 255), 2)
    cv2.putText(vis, "redline ROI", (12, max(24, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    band = REDLINE_EVENT.get("central_band", [0.35, 0.65])
    x0 = int(float(band[0]) * w)
    x1 = int(float(band[1]) * w)
    cv2.rectangle(vis, (x0, int(y0)), (x1, h - 1), (255, 120, 0), 1)

    for item in observation.get("contours", []) or []:
        bx1, by1, bx2, by2 = [int(round(v)) for v in item.get("box_xyxy", [0, 0, 0, 0])]
        color = (0, 255, 0) if update.get("events") else (0, 0, 255)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            vis,
            f"red w={float(item.get('width_ratio', 0.0)):.2f}",
            (bx1, max(18, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    if update.get("events"):
        cv2.putText(vis, "FIRE: final_redline", (20, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 255, 0), 3)


def _append_csv(csv_path, row, write_header):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = _parse_args()
    save_every = max(0, int(args.save_every))
    out_dir = Path(ROOT_DIR) / "outputs" / "full_overlay_drive" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "log.csv"
    wrote_header = False

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

        print("[full_overlay_drive] integrated lane + sign + redline + state + motor")
        print(
            f"[full_overlay_drive] lane={lane['selected_model']} post={lane['selected_postprocess']} "
            f"sign={SIGN_MODEL.get('selected')} motors={MOTOR['enable_motors']} target_fps={RUNTIME['target_fps']}"
        )
        print(f"[full_overlay_drive] output={out_dir}")

        frame_idx = 0
        t_start = time.perf_counter()
        last_detections = []
        last_sign_debug = []
        last_sign_timing = {"total_ms": 0.0}
        last_red_obs = {"active": False, "contours": [], "best": None}
        last_red_update = {"events": [], "history_hits": 0, "required_hits": int(REDLINE_EVENT.get("required_hits", 2))}

        while True:
            t0 = time.perf_counter()
            frame = first_frame if frame_idx == 0 else camera.read_bgr()
            now = time.perf_counter()

            lane_result = run_lane_pipeline(lane, frame)
            signal = lane_result["lane_signal"]

            # sign event는 one-shot이다.
            # YOLO를 실행한 frame에서 새로 fire된 event만 state machine에 넘긴다.
            sign_events = []
            if sign_detector is not None and frame_idx % max(1, int(RUNTIME["sign_every_frames"])) == 0:
                last_detections, last_sign_timing = detect_signs(sign_detector, frame)
                sign_out = update_sign_trigger(sign_state, last_detections, SIGN_TRIGGER, now)
                last_sign_debug = sign_out["debug"]
                sign_events = sign_out["events"]

            do_redline = frame_idx % max(1, int(RUNTIME.get("redline_every_frames", 1))) == 0
            if do_redline:
                last_red_obs = detect_redline(frame, REDLINE_EVENT)
            last_red_update = update_redline_event_state(
                redline_state, last_red_obs, REDLINE_EVENT, now, is_new_observation=do_redline
            )
            redline_events = last_red_update["events"]

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

            loop_before_sleep_ms = (time.perf_counter() - t0) * 1000.0
            frame_idx += 1
            fps = frame_idx / max(1e-6, time.perf_counter() - t_start)

            vis = draw_lane_overlay(frame, lane_result, fps, loop_before_sleep_ms, lane["selected_model"])
            _draw_roi_guides(vis, SIGN_TRIGGER)
            _draw_detections(vis, last_detections, last_sign_debug)
            _draw_fire_banner(vis, sign_events + redline_events)
            _draw_redline_on_existing(vis, last_red_obs, last_red_update)
            _draw_state_overlay(
                vis,
                lane_result,
                control,
                command,
                last_sign_debug,
                sign_events,
                last_red_obs,
                last_red_update,
                frame_idx,
                loop_before_sleep_ms,
                fps,
            )

            row = {
                "frame": frame_idx,
                "time_sec": f"{time.perf_counter() - t_start:.3f}",
                "phase": control["phase"],
                "race_phase": control["race_phase"],
                "reason": control["reason"],
                "accepted_event": control["accepted_event"],
                "events": "|".join(_event_names(sign_events + redline_events)),
                "lane_state": signal["lane_state"],
                "visible_lane_count": signal["visible_lane_count"],
                "feature_count": signal["feature_count"],
                "lane_steer": f"{signal['steer_norm']:.5f}",
                "cmd_steer": f"{control['steer_norm']:.5f}",
                "speed_scale": f"{control['speed_scale']:.5f}",
                "quality": f"{signal['quality']:.5f}",
                "stable_forward": int(signal["stable_forward"]),
                "left_pwm": f"{command.left:.5f}",
                "right_pwm": f"{command.right:.5f}",
                "sign_count": len(last_detections),
                "sign_ms": f"{float(last_sign_timing.get('total_ms', 0.0)):.3f}",
                "redline_active": int(bool(last_red_obs.get("active", False))),
                "loop_ms": f"{loop_before_sleep_ms:.3f}",
                "fps": f"{fps:.3f}",
            }
            _append_csv(csv_path, row, write_header=not wrote_header)
            wrote_header = True

            if save_every > 0 and frame_idx % save_every == 0:
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}_auto.jpg"), vis)

            if not args.no_window:
                cv2.imshow(str(RUNTIME.get("overlay_window_name", "team3_full_overlay_drive")), vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    out_path = out_dir / f"frame_{frame_idx:05d}_manual.jpg"
                    cv2.imwrite(str(out_path), vis)
                    print(f"[full_overlay_drive] saved {out_path}")

            sleep_to_target(t0, float(RUNTIME["target_fps"]))
    except KeyboardInterrupt:
        print("\n[full_overlay_drive] interrupted")
    finally:
        horn.close()
        motor.close()
        camera.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
