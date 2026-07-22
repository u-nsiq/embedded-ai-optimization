"""Sign trigger debug runtime.

Lane/state machine 없이 카메라 + YOLO + event trigger만 확인한다.
현장에서 표지판을 카메라 앞에 두고 "왜 이벤트가 안 터졌는지"를 볼 때 사용한다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import time
from collections import Counter, defaultdict

from config import CAMERA, ROOT_DIR, RUNTIME, SIGN_MODEL, SIGN_TRIGGER
from sign.detector import detect_signs, init_sign_detector
from sign.trigger import compact_sign_debug, init_sign_trigger_state, update_sign_trigger
from utils.camera import LiveCamera
from utils.timing import sleep_to_target


def _event_label(row):
    event_name = str(row.get("event_name") or row.get("class_name"))
    class_name = str(row.get("class_name"))
    if event_name != class_name:
        return f"{event_name}({class_name})"
    return class_name


def _reason_head(reason):
    return str(reason or "").split(" ", 1)[0]


def _new_summary():
    return {
        "frames": 0,
        "sign_frames": 0,
        "fires": Counter(),
        "wait_hits": Counter(),
        "cooldowns": Counter(),
        "rejects": Counter(),
        "max_size": defaultdict(float),
    }


def _update_summary(summary, events, debug_rows):
    for event in events:
        summary["fires"][str(event.get("name"))] += 1

    for row in debug_rows:
        label = _event_label(row)
        if row.get("conf") is not None:
            summary["max_size"][label] = max(summary["max_size"][label], float(row.get("box_size") or 0.0))

        status = row.get("status")
        if status == "wait_hits":
            summary["wait_hits"][label] += 1
        elif status == "cooldown":
            summary["cooldowns"][label] += 1
        elif status == "reject" and row.get("conf") is not None:
            summary["rejects"][f"{label}:{_reason_head(row.get('reason'))}"] += 1


def _print_summary(summary):
    print("\n[sign_log:v2] summary")
    print(f"  frames={summary['frames']} sign_frames={summary['sign_frames']}")

    if summary["fires"]:
        print("  fire:", ", ".join(f"{k}x{v}" for k, v in summary["fires"].most_common()))
    else:
        print("  fire: none")

    if summary["wait_hits"]:
        print("  wait_hits:", ", ".join(f"{k}x{v}" for k, v in summary["wait_hits"].most_common(6)))
    if summary["cooldowns"]:
        print("  cooldown:", ", ".join(f"{k}x{v}" for k, v in summary["cooldowns"].most_common(6)))
    if summary["rejects"]:
        print("  rejected:", ", ".join(f"{k}x{v}" for k, v in summary["rejects"].most_common(8)))
    if summary["max_size"]:
        print("  max_size:", ", ".join(f"{k}={v:.2f}" for k, v in sorted(summary["max_size"].items())))


def main():
    camera = LiveCamera(CAMERA)
    detector = init_sign_detector(ROOT_DIR, SIGN_MODEL)
    trigger_state = init_sign_trigger_state(SIGN_TRIGGER)
    summary = _new_summary()
    print(f"[sign_log:v2] model={detector.selected_model if detector else 'disabled'}")
    print("[sign_log:v2] Ctrl+C to stop")

    frame_idx = 0
    t_start = time.perf_counter()
    last_debug = []
    try:
        while True:
            t0 = time.perf_counter()
            frame = camera.read_bgr()
            now = time.perf_counter()

            if frame_idx % int(RUNTIME["sign_every_frames"]) == 0:
                detections, timing = detect_signs(detector, frame)
                out = update_sign_trigger(trigger_state, detections, SIGN_TRIGGER, now)
                last_debug = out["debug"]
                summary["sign_frames"] += 1
                _update_summary(summary, out["events"], last_debug)
                if out["events"]:
                    print(f"[sign_event] {out['events']}")
            else:
                detections = []
                timing = {"total_ms": 0.0}

            sleep_to_target(t0, float(RUNTIME["target_fps"]))
            frame_idx += 1
            summary["frames"] = frame_idx

            if frame_idx % int(RUNTIME["log_every_frames"]) == 0:
                fps = frame_idx / max(1e-6, time.perf_counter() - t_start)
                msg = compact_sign_debug(last_debug, max_rows=int(RUNTIME.get("sign_log_max_rows", 4)))
                print(
                    f"[{frame_idx:05d}] det={len(detections)} yolo={timing['total_ms']:.1f}ms "
                    f"fps={fps:.2f} {msg}"
                )
    except KeyboardInterrupt:
        print("\n[sign_log:v2] interrupted")
    finally:
        _print_summary(summary)
        camera.close()


if __name__ == "__main__":
    main()
