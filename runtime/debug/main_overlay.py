"""Lane overlay viewer for team3_drive_v2.

바퀴를 굴리지 않고 lane model/decoder/postprocess가 무엇을 보고 있는지만 확인한다.
후처리 후보별 debug dict를 최대한 공통 방식으로 그린다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import time

import cv2
import numpy as np

from config import CAMERA, LANE_DECODE, LANE_MODEL, LANE_POSTPROCESS, ROOT_DIR, RUNTIME
from lane.geometry import HALF_W, IMAGE_CENTER_X, RAW_H, RAW_W
from lane.postprocess import selected_postprocess_cfg
from lane.pipeline import init_lane_pipeline, run_lane_pipeline
from utils.camera import LiveCamera
from utils.timing import sleep_to_target


def _pt(x, y):
    return int(round(float(x))), int(round(float(y)))


def _draw_text_box(img, lines, x=18, y=28):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
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


def _draw_reference_geometry(img, cfg):
    for key, label, color in [
        ("near_y_ratio", "near", (0, 255, 255)),
        ("mid_y_ratio", "mid", (255, 255, 0)),
        ("far_y_ratio", "far", (255, 0, 255)),
        ("top_y_ratio", "top", (255, 0, 255)),
        ("primary_y_ratio", "primary", (255, 255, 0)),
        ("bottom_y_ratio", "bottom", (0, 255, 255)),
        ("lookahead_y_ratio", "look", (0, 165, 255)),
    ]:
        if key not in cfg:
            continue
        y = float(cfg[key]) * RAW_H
        cv2.line(img, _pt(0, y), _pt(RAW_W, y), color, 1)
        cv2.putText(img, label, _pt(12, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.line(img, _pt(IMAGE_CENTER_X, 0), _pt(IMAGE_CENTER_X, RAW_H), (170, 170, 170), 2)
    if "safe_distance_ratio" in cfg:
        safe = float(cfg["safe_distance_ratio"]) * HALF_W
        cv2.line(img, _pt(IMAGE_CENTER_X - safe, 0), _pt(IMAGE_CENTER_X - safe, RAW_H), (80, 80, 220), 1)
        cv2.line(img, _pt(IMAGE_CENTER_X + safe, 0), _pt(IMAGE_CENTER_X + safe, RAW_H), (80, 80, 220), 1)


def _draw_decoded_lanes(img, lanes):
    colors = [(0, 255, 0), (0, 180, 255), (255, 120, 0), (255, 0, 200), (200, 200, 200)]
    for idx, lane in enumerate(lanes or []):
        pts = np.asarray(lane.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            continue
        valid = (
            np.isfinite(pts[:, 0])
            & np.isfinite(pts[:, 1])
            & (pts[:, 0] >= -RAW_W * 0.25)
            & (pts[:, 0] <= RAW_W * 1.25)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] <= RAW_H)
        )
        pts = pts[valid]
        if len(pts) == 0:
            continue
        color = colors[idx % len(colors)]
        pts_i = np.round(pts).astype(np.int32)
        if len(pts_i) >= 2:
            cv2.polylines(img, [pts_i.reshape(-1, 1, 2)], False, color, 3)
        for x, y in pts_i[::2]:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)
        conf = float(lane.get("conf", 0.0))
        x0, y0 = pts_i[min(len(pts_i) - 1, max(0, len(pts_i) // 2))]
        cv2.putText(img, f"L{idx} {conf:.2f}", (int(x0) + 6, int(y0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_feature_samples(img, debug, cfg):
    features = debug.get("features", []) or []
    weights = debug.get("weights", []) or []

    # 후보마다 이름은 다를 수 있지만 의미는 같다.
    # slope_push/08g: x_near, x_mid, x_far
    # inside_soft:    x_bottom, x_primary, x_top
    y_bottom = float(cfg.get("near_y_ratio", cfg.get("bottom_y_ratio", 0.82))) * RAW_H
    y_mid = float(cfg.get("mid_y_ratio", cfg.get("primary_y_ratio", 0.72))) * RAW_H
    y_top = float(cfg.get("far_y_ratio", cfg.get("top_y_ratio", 0.62))) * RAW_H

    for idx, feat in enumerate(features):
        weight = float(weights[idx]) if idx < len(weights) else float(feat.get("quality", 0.0))
        color = (0, 255, 255) if weight >= 0.45 else (120, 120, 255)
        x_bottom = feat.get("x_near", feat.get("x_bottom"))
        x_mid = feat.get("x_mid", feat.get("x_primary"))
        x_top = feat.get("x_far", feat.get("x_top"))
        if x_bottom is None or x_mid is None or x_top is None:
            continue
        pts = [
            _pt(x_bottom, y_bottom),
            _pt(x_mid, y_mid),
            _pt(x_top, y_top),
        ]
        cv2.polylines(img, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2)
        for p in pts:
            cv2.circle(img, p, 7, color, 2)
        cv2.putText(img, f"F{idx} q={weight:.2f} h={float(feat['heading']):+.2f}",
                    (pts[1][0] + 8, pts[1][1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def _draw_stable_center_debug(img, debug, cfg):
    if "centers" not in debug:
        return
    near_y = float(cfg["near_y_ratio"]) * RAW_H
    mid_y = float(cfg["mid_y_ratio"]) * RAW_H
    far_y = float(cfg["far_y_ratio"]) * RAW_H
    ys = [near_y, mid_y, far_y]
    centers = debug.get("centers", [])
    fit = debug.get("center_fit", centers)
    if len(centers) == 3:
        pts = [_pt(x, y) for x, y in zip(centers, ys)]
        cv2.polylines(img, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 3)
        for p in pts:
            cv2.circle(img, p, 8, (0, 255, 255), -1)
    if len(fit) == 3:
        pts = [_pt(x, y) for x, y in zip(fit, ys)]
        cv2.polylines(img, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], False, (0, 200, 255), 2)
    if "lookahead_x" in debug:
        p = _pt(debug["lookahead_x"], debug.get("lookahead_y", float(cfg["lookahead_y_ratio"]) * RAW_H))
        cv2.circle(img, p, 11, (0, 120, 255), 3)
        cv2.putText(img, "look", (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2, cv2.LINE_AA)


def _draw_steer_arrow(img, signal):
    steer = float(signal.get("steer_norm", 0.0))
    start = _pt(IMAGE_CENTER_X, RAW_H - 35)
    end = _pt(IMAGE_CENTER_X + steer * HALF_W * 0.65, RAW_H - 190)
    color = (0, 255, 0) if abs(steer) < 0.25 else (0, 165, 255)
    if abs(steer) > 0.50:
        color = (0, 0, 255)
    cv2.arrowedLine(img, start, end, color, 5, tipLength=0.18)
    cv2.putText(img, f"steer {steer:+.2f}", (start[0] + 16, start[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_lane_overlay(frame, lane_result, fps, loop_ms, selected_model):
    vis = frame.copy()
    signal = lane_result["lane_signal"]
    debug = signal.get("debug", {})
    cfg = selected_postprocess_cfg(LANE_POSTPROCESS)

    _draw_reference_geometry(vis, cfg)
    _draw_decoded_lanes(vis, lane_result["lanes"])
    _draw_feature_samples(vis, debug, cfg)
    _draw_stable_center_debug(vis, debug, cfg)
    _draw_steer_arrow(vis, signal)

    slope = debug.get(
        "weighted_slope",
        debug.get(
            "center_slope",
            debug.get("local_slope", debug.get("measured_heading", debug.get("single_heading", debug.get("smoothed_heading", 0.0)))),
        ),
    )
    lines = [
        f"model={selected_model} post={signal['postprocess']} q:quit",
        f"state={signal['lane_state']} lanes={signal['visible_lane_count']} features={signal['feature_count']}",
        f"steer={signal['steer_norm']:+.3f} raw={signal['raw_steer']:+.3f} speed={signal['speed_scale']:.2f}",
        f"slope={float(slope):+.3f} quality={signal['quality']:.2f} stable={int(signal['stable_forward'])}",
        f"reason={debug.get('reason', '')} loop={loop_ms:.1f}ms fps={fps:.2f}",
    ]
    _draw_text_box(vis, lines)
    return vis


def main():
    camera = LiveCamera(CAMERA)
    try:
        first_frame = camera.read_bgr()
        lane = init_lane_pipeline(ROOT_DIR, LANE_MODEL, LANE_DECODE, LANE_POSTPROCESS, first_frame=first_frame)
        print(
            f"[main_overlay:v2] lane={lane['selected_model']} post={lane['selected_postprocess']} "
            f"target_fps={RUNTIME['target_fps']}"
        )
        print("[main_overlay:v2] motors are not initialized in this script")

        frame_idx = 0
        t_start = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            frame = first_frame if frame_idx == 0 else camera.read_bgr()
            lane_result = run_lane_pipeline(lane, frame)
            loop_ms = (time.perf_counter() - t0) * 1000.0
            frame_idx += 1
            fps = frame_idx / max(1e-6, time.perf_counter() - t_start)

            vis = draw_lane_overlay(frame, lane_result, fps, loop_ms, lane["selected_model"])
            cv2.imshow(str(RUNTIME.get("overlay_window_name", "team3_lane_overlay")), vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            sleep_to_target(t0, float(RUNTIME["target_fps"]))
    except KeyboardInterrupt:
        print("\n[main_overlay:v2] interrupted")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


