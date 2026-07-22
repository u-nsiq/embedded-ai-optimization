"""Redline overlay debug.

모터/state machine 없이 카메라 + HSV redline detector만 실행한다.
빨간 종료선이 어느 위치/크기에서 final_redline 이벤트가 되는지 확인할 때 사용한다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import time

import cv2

from config import CAMERA, REDLINE_EVENT, ROOT_DIR, RUNTIME
from redline.detector import compact_redline_debug, detect_redline, init_redline_state, update_redline_event_state
from utils.camera import LiveCamera
from utils.timing import sleep_to_target


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=1, help="redline detector를 몇 frame마다 실행할지")
    parser.add_argument("--save-every", type=int, default=0, help="N frame마다 overlay 저장. 0이면 자동 저장 안 함")
    return parser.parse_args()


def _draw_text_box(img, lines, x=18, y=28):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.60
    thick = 2
    line_h = 24
    max_w = 0
    for text in lines:
        (w, _), _ = cv2.getTextSize(str(text), font, scale, thick)
        max_w = max(max_w, w)
    h = line_h * len(lines) + 14
    cv2.rectangle(img, (x - 8, y - 22), (x + max_w + 12, y - 22 + h), (0, 0, 0), -1)
    cv2.rectangle(img, (x - 8, y - 22), (x + max_w + 12, y - 22 + h), (80, 80, 80), 1)
    for i, text in enumerate(lines):
        cv2.putText(img, str(text), (x, y + i * line_h), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_redline_overlay(frame, observation, update, frame_idx, fps, every):
    vis = frame.copy()
    h, w = vis.shape[:2]
    y0 = observation.get("roi_y0")
    if y0 is None:
        y0 = int(float(REDLINE_EVENT.get("bottom_roi_y0", 0.70)) * h)
    cv2.line(vis, (0, int(y0)), (w, int(y0)), (0, 255, 255), 2)
    cv2.putText(vis, "redline ROI", (12, max(24, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    band = REDLINE_EVENT.get("central_band", [0.35, 0.65])
    x0 = int(float(band[0]) * w)
    x1 = int(float(band[1]) * w)
    cv2.rectangle(vis, (x0, int(y0)), (x1, h - 1), (255, 120, 0), 1)

    for item in observation.get("contours", []) or []:
        bx1, by1, bx2, by2 = [int(round(v)) for v in item.get("box_xyxy", [0, 0, 0, 0])]
        color = (0, 255, 0) if update.get("events") else (0, 0, 255)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 3)
        cv2.putText(
            vis,
            f"w={float(item.get('width_ratio', 0.0)):.2f} area={float(item.get('area', 0.0)):.0f}",
            (bx1, max(18, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if update.get("events"):
        cv2.rectangle(vis, (0, h - 52), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, "FIRE: final_redline", (20, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

    lines = [
        "redline overlay  q:quit s:save",
        f"every={every} frame={frame_idx} fps={fps:.2f}",
        compact_redline_debug(observation, update),
    ]
    _draw_text_box(vis, lines)
    return vis


def main():
    args = _parse_args()
    every = max(1, int(args.every))
    save_every = max(0, int(args.save_every))
    out_dir = Path(ROOT_DIR) / "outputs" / "redline_overlay_debug" / time.strftime("%Y%m%d_%H%M%S")
    if save_every > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    camera = LiveCamera(CAMERA)
    state = init_redline_state(REDLINE_EVENT)
    print(f"[redline_overlay:v2] every={every}")
    print("[redline_overlay:v2] motors/state_machine are not initialized")

    frame_idx = 0
    t_start = time.perf_counter()
    last_obs = {"active": False, "contours": [], "best": None, "mask": None}
    last_update = {"events": [], "history_hits": 0, "required_hits": int(REDLINE_EVENT.get("required_hits", 2))}
    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.read_bgr()
            now = time.perf_counter()
            if frame_idx % every == 0:
                last_obs = detect_redline(frame, REDLINE_EVENT)
                last_update = update_redline_event_state(state, last_obs, REDLINE_EVENT, now, is_new_observation=True)
                if last_update.get("events"):
                    print(f"[redline_event] {last_update['events']}")
            frame_idx += 1
            fps = frame_idx / max(1e-6, time.perf_counter() - t_start)
            vis = draw_redline_overlay(frame, last_obs, last_update, frame_idx, fps, every)
            cv2.imshow(str(RUNTIME.get("overlay_window_name", "team3_redline_overlay")), vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"frame_{frame_idx:05d}_manual.jpg"
                cv2.imwrite(str(out_path), vis)
                print(f"[redline_overlay:v2] saved {out_path}")
            if save_every > 0 and frame_idx % save_every == 0:
                out_path = out_dir / f"frame_{frame_idx:05d}_auto.jpg"
                cv2.imwrite(str(out_path), vis)
            sleep_to_target(t0, float(RUNTIME["target_fps"]))
    except KeyboardInterrupt:
        print("\n[redline_overlay:v2] interrupted")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
