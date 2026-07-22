"""Sign event trigger overlay debug.

모터와 상태머신은 전혀 실행하지 않는다.
카메라 + YOLO + sign trigger만 실행해서, 표지판이 어느 거리/위치에서
event로 발화하는지 화면으로 확인하는 디버그 스크립트다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import time
from pathlib import Path

import cv2

from config import CAMERA, ROOT_DIR, RUNTIME, SIGN_MODEL, SIGN_TRIGGER
from sign.detector import detect_signs, init_sign_detector
from sign.trigger import compact_sign_debug, init_sign_trigger_state, update_sign_trigger
from utils.camera import LiveCamera
from utils.timing import sleep_to_target


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=1, help="YOLO를 몇 frame마다 실행할지. 이벤트 타이밍 확인은 1 권장.")
    parser.add_argument("--save-every", type=int, default=0, help="N frame마다 overlay를 자동 저장. 0이면 자동 저장 안 함.")
    parser.add_argument("--max-debug-rows", type=int, default=8, help="상단에 표시할 trigger debug row 개수.")
    return parser.parse_args()


def _status_color(status, reason=""):
    reason = str(reason or "")
    if status == "fire":
        return (0, 255, 0)
    if status == "wait_hits":
        return (0, 210, 255)
    if status == "cooldown":
        return (255, 255, 0)
    if status == "reject":
        if reason.startswith("too_far"):
            return (255, 150, 0)
        if reason.startswith("outside_roi"):
            return (0, 0, 255)
        if reason.startswith("low_conf"):
            return (120, 120, 120)
        return (180, 80, 255)
    return (230, 230, 230)


def _display_name(row):
    class_name = str(row.get("class_name", ""))
    event_name = str(row.get("event_name") or class_name)
    return f"{event_name}({class_name})" if event_name != class_name else class_name


def _match_debug_row(det, debug_rows):
    class_name = str(det.get("class_name", ""))
    candidates = [r for r in debug_rows if str(r.get("class_name", "")) == class_name and r.get("conf") is not None]
    if not candidates:
        return None
    det_conf = float(det.get("confidence", 0.0))
    det_size = float(det.get("box_size", 0.0))
    return min(candidates, key=lambda r: abs(float(r.get("conf") or 0.0) - det_conf) + abs(float(r.get("box_size") or 0.0) - det_size))


def _draw_text_box(img, lines, x=18, y=28):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thick = 2
    line_h = 23
    max_w = 0
    for text in lines:
        (w, _), _ = cv2.getTextSize(str(text), font, scale, thick)
        max_w = max(max_w, w)
    h = line_h * len(lines) + 14
    cv2.rectangle(img, (x - 8, y - 22), (x + max_w + 12, y - 22 + h), (0, 0, 0), -1)
    cv2.rectangle(img, (x - 8, y - 22), (x + max_w + 12, y - 22 + h), (80, 80, 80), 1)
    for i, text in enumerate(lines):
        cv2.putText(img, str(text), (x, y + i * line_h), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _draw_roi_guides(img, cfg):
    h, w = img.shape[:2]
    colors = {"default": (80, 80, 220), "near_sign": (0, 180, 255), "near_light": (255, 180, 0), "early_sign": (180, 255, 0)}

    def draw_roi(name, roi):
        x1 = int(float(roi.get("x_min", 0.0)) * w)
        x2 = int(float(roi.get("x_max", 1.0)) * w)
        y1 = int(float(roi.get("y_min", 0.0)) * h)
        y2 = int(float(roi.get("y_max", 1.0)) * h)
        color = colors.get(name, (160, 160, 160))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, f"ROI {name}", (x1 + 5, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    default_roi = cfg.get("default", {}).get("roi")
    if default_roi:
        draw_roi("default", default_roi)
    for name, group in cfg.get("groups", {}).items():
        if "roi" in group:
            base = dict(default_roi or {})
            base.update(group.get("roi", {}))
            draw_roi(name, base)


def _draw_detections(img, detections, debug_rows):
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.get("box_xyxy", [0, 0, 0, 0])]
        row = _match_debug_row(det, debug_rows)
        status = row.get("status") if row else "det"
        reason = row.get("reason") if row else ""
        color = _status_color(status, reason)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if status == "fire" else 2)
        label = _display_name(row) if row else str(det.get("class_name", "?"))
        text1 = f"{label} {float(det.get('confidence', 0.0)):.2f} size={float(det.get('box_size', 0.0)):.2f}"
        text2 = f"{status} hits={row.get('hit_streak')}/{row.get('required_hits')} {reason}" if row else "detected"
        cv2.putText(img, text1, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(img, text2, (x1, min(img.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def _draw_fire_banner(img, events):
    if not events:
        return
    names = ", ".join(str(e.get("name")) for e in events)
    cv2.rectangle(img, (0, img.shape[0] - 52), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, f"FIRE: {names}", (20, img.shape[0] - 17), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)


def draw_overlay(frame, detections, trigger_out, timing, frame_idx, fps, every, max_debug_rows):
    vis = frame.copy()
    debug_rows = trigger_out.get("debug", [])
    events = trigger_out.get("events", [])
    _draw_roi_guides(vis, SIGN_TRIGGER)
    _draw_detections(vis, detections, debug_rows)
    _draw_fire_banner(vis, events)
    compact = compact_sign_debug(debug_rows, max_rows=max_debug_rows)
    if len(compact) > 120:
        compact = compact[:117] + "..."
    lines = [
        "sign trigger overlay  q:quit s:save",
        f"model={SIGN_MODEL.get('selected')} every={every} frame={frame_idx}",
        f"det={len(detections)} events={len(events)} yolo={timing.get('total_ms', 0.0):.1f}ms fps={fps:.2f}",
        compact or "no trigger candidates",
    ]
    _draw_text_box(vis, lines)
    return vis


def main():
    args = _parse_args()
    every = max(1, int(args.every))
    save_every = max(0, int(args.save_every))
    out_dir = Path(ROOT_DIR) / "outputs" / "sign_overlay_debug" / time.strftime("%Y%m%d_%H%M%S")
    if save_every > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    camera = LiveCamera(CAMERA)
    detector = init_sign_detector(ROOT_DIR, SIGN_MODEL)
    trigger_state = init_sign_trigger_state(SIGN_TRIGGER)
    print(f"[sign_overlay:v2] model={detector.selected_model if detector else 'disabled'} every={every}")
    print("[sign_overlay:v2] motors/state_machine are not initialized")

    frame_idx = 0
    t_start = time.perf_counter()
    last_detections = []
    last_trigger_out = {"events": [], "debug": []}
    last_timing = {"total_ms": 0.0}
    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.read_bgr()
            now = time.perf_counter()
            if frame_idx % every == 0:
                last_detections, last_timing = detect_signs(detector, frame)
                last_trigger_out = update_sign_trigger(trigger_state, last_detections, SIGN_TRIGGER, now)
                if last_trigger_out.get("events"):
                    print(f"[sign_event] {last_trigger_out['events']}")
            frame_idx += 1
            fps = frame_idx / max(1e-6, time.perf_counter() - t_start)
            vis = draw_overlay(frame, last_detections, last_trigger_out, last_timing, frame_idx, fps, every, int(args.max_debug_rows))
            cv2.imshow(str(RUNTIME.get("overlay_window_name", "team3_sign_overlay")), vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"frame_{frame_idx:05d}_manual.jpg"
                cv2.imwrite(str(out_path), vis)
                print(f"[sign_overlay:v2] saved {out_path}")
            if save_every > 0 and frame_idx % save_every == 0:
                out_path = out_dir / f"frame_{frame_idx:05d}_auto.jpg"
                cv2.imwrite(str(out_path), vis)
            sleep_to_target(t0, float(RUNTIME["target_fps"]))
    except KeyboardInterrupt:
        print("\n[sign_overlay:v2] interrupted")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
