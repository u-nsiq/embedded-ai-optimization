from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class CameraGeometry:
    raw_width: int = 1296
    raw_height: int = 972
    cut_height: int = 445
    model_width: int = 800
    model_height: int = 320
    num_points: int = 72
    max_lanes: int = 4

    @property
    def visible_height(self) -> int:
        return self.raw_height - self.cut_height

    @classmethod
    def from_config(cls, config: dict | None = None) -> "CameraGeometry":
        config = config or {}
        fields = {name: config.get(name, getattr(cls(), name)) for name in cls.__dataclass_fields__}
        return cls(**fields)


def geometry_dict(geom: CameraGeometry) -> dict:
    return asdict(geom) | {"visible_height": geom.visible_height}


def windows_long_path(path: Path | str) -> str:
    path_obj = Path(path)
    if os.name == "nt" and not path_obj.is_absolute():
        path_obj = path_obj.resolve()
    s = str(path_obj)
    if os.name == "nt" and not s.startswith("\\\\?\\") and len(s) >= 240:
        return "\\\\?\\" + s
    return s


def path_exists(path: Path | str) -> bool:
    return os.path.exists(windows_long_path(path))


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        os.makedirs(windows_long_path(path), exist_ok=True)


def remove_tree(path: Path) -> None:
    target = windows_long_path(path)
    if not os.path.exists(target):
        return
    for root, dirs, files in os.walk(target, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                os.remove(file_path)
            except FileNotFoundError:
                try:
                    os.remove(windows_long_path(file_path))
                except (FileNotFoundError, OSError):
                    pass
            except OSError:
                try:
                    os.remove(windows_long_path(file_path))
                except (FileNotFoundError, OSError):
                    pass
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)
            except FileNotFoundError:
                pass
            except OSError:
                try:
                    os.rmdir(windows_long_path(dir_path))
                except OSError:
                    pass
    try:
        os.rmdir(target)
    except FileNotFoundError:
        pass
    except OSError:
        try:
            shutil.rmtree(target, ignore_errors=True)
        except OSError:
            pass


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    ensure_dirs(path.parent)
    with open(windows_long_path(path), "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_input_path(path_str: str, input_cfg: dict) -> Path | None:
    if not path_str:
        return None

    s = path_str.replace("\\", "/")
    for src_prefix, dst_prefix in input_cfg.get("path_prefix_map", {}).items():
        src = src_prefix.replace("\\", "/").rstrip("/")
        if s.startswith(src):
            rel = s[len(src):].lstrip("/")
            return Path(dst_prefix) / rel

    path = Path(path_str)
    if path.is_absolute():
        return path

    raw_root = input_cfg.get("raw_root")
    if raw_root:
        return Path(raw_root) / path
    return path


def copy_image(src: Path, dst: Path) -> None:
    ensure_dirs(dst.parent)
    shutil.copy2(windows_long_path(src), windows_long_path(dst))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(windows_long_path(path)) as im:
        return np.array(im.convert("RGB"))


def save_rgb(path: Path, image_rgb: np.ndarray, quality: int = 95) -> None:
    ensure_dirs(path.parent)
    Image.fromarray(image_rgb.astype(np.uint8)).save(windows_long_path(path), quality=quality)


def save_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dirs(path.parent)
    Image.fromarray(mask.astype(np.uint8)).save(windows_long_path(path))


def crop_road(rgb: np.ndarray, geom: CameraGeometry) -> np.ndarray:
    return rgb[geom.cut_height:, :, :]


def hsv_yellow_mask(
    rgb: np.ndarray,
    lower: Sequence[int],
    upper: Sequence[int],
    morph_kernel: int = 5,
) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    if morph_kernel and morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def yellow_mask_raw(rgb: np.ndarray, hsv_cfg: dict, geom: CameraGeometry, crop_before_hsv: bool = True) -> np.ndarray:
    if crop_before_hsv:
        crop = crop_road(rgb, geom)
        mask_crop = hsv_yellow_mask(
            crop,
            lower=hsv_cfg["lower"],
            upper=hsv_cfg["upper"],
            morph_kernel=int(hsv_cfg.get("morph_kernel", 5)),
        )
        full = np.zeros((geom.raw_height, geom.raw_width), dtype=np.uint8)
        full[geom.cut_height:, :] = mask_crop
        return full
    return hsv_yellow_mask(
        rgb,
        lower=hsv_cfg["lower"],
        upper=hsv_cfg["upper"],
        morph_kernel=int(hsv_cfg.get("morph_kernel", 5)),
    )


def row_runs(row_mask: np.ndarray, min_width: int = 4) -> list[tuple[int, int, float]]:
    xs = np.where(row_mask > 0)[0]
    if xs.size == 0:
        return []
    breaks = np.where(np.diff(xs) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, xs.size - 1]
    runs = []
    for s_idx, e_idx in zip(starts, ends):
        x1 = int(xs[s_idx])
        x2 = int(xs[e_idx])
        if x2 - x1 + 1 >= min_width:
            runs.append((x1, x2, (x1 + x2) / 2.0))
    return runs


def sort_lane_bottom_to_top(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted([(float(x), float(y)) for x, y in points], key=lambda p: -p[1])


def smooth_lane_polyfit(
    points: Sequence[tuple[float, float]],
    geom: CameraGeometry,
    degree: int = 2,
    y_step: int = 8,
    min_points: int = 8,
) -> list[tuple[float, float]]:
    pts = sort_lane_bottom_to_top(points)
    if len(pts) < min_points:
        return pts
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    fit_degree = min(degree, max(1, len(np.unique(ys)) - 1))
    try:
        coeff = np.polyfit(ys, xs, deg=fit_degree)
        y_min = max(float(ys.min()), float(geom.cut_height))
        y_max = min(float(ys.max()), float(geom.raw_height - 1))
        sample_ys = np.arange(y_max, y_min - 1, -y_step, dtype=np.float32)
        smooth_xs = np.polyval(coeff, sample_ys)
        keep = (smooth_xs >= 0) & (smooth_xs < geom.raw_width)
        return [(float(x), float(y)) for x, y in zip(smooth_xs[keep], sample_ys[keep])]
    except Exception:
        return pts


def lane_y_span(points: Sequence[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    ys = np.array([p[1] for p in points], dtype=np.float32)
    return float(ys.max() - ys.min())


def lane_smoothness(points: Sequence[tuple[float, float]]) -> float:
    pts = sort_lane_bottom_to_top(points)
    if len(pts) < 4:
        return float("inf")
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    return float(np.mean(np.abs(np.diff(xs, n=2))))


def lane_mask_support(mask_raw: np.ndarray, lanes: Sequence[Sequence[tuple[float, float]]], radius: int = 4) -> float:
    total = 0
    supported = 0
    h, w = mask_raw.shape[:2]
    for lane in lanes:
        for x, y in lane:
            xi = int(round(x))
            yi = int(round(y))
            total += 1
            if xi < 0 or yi < 0 or xi >= w or yi >= h:
                continue
            x0 = max(0, xi - radius)
            x1 = min(w, xi + radius + 1)
            y0 = max(0, yi - radius)
            y1 = min(h, yi + radius + 1)
            if np.any(mask_raw[y0:y1, x0:x1] > 0):
                supported += 1
    return float(supported / total) if total else 0.0


def summarize_lanes(lanes: Sequence[Sequence[tuple[float, float]]], base_quality: dict[str, float] | None = None) -> dict[str, float]:
    q = dict(base_quality or {})
    q["num_lanes"] = float(len(lanes))
    q["point_total"] = float(sum(len(lane) for lane in lanes))
    q["median_points_per_lane"] = float(np.median([len(lane) for lane in lanes])) if lanes else 0.0
    q["median_y_span"] = float(np.median([lane_y_span(lane) for lane in lanes])) if lanes else 0.0
    smooth_values = [lane_smoothness(lane) for lane in lanes if len(lane) >= 4]
    q["median_smoothness"] = float(np.median(smooth_values)) if smooth_values else float("inf")
    return q


def extract_lane_points_rowwise(
    mask_raw: np.ndarray,
    geom: CameraGeometry,
    y_step: int = 8,
    min_run_width: int = 4,
    min_points: int = 8,
    center_x: float | None = None,
) -> tuple[list[list[tuple[float, float]]], dict[str, float]]:
    if center_x is None:
        center_x = geom.raw_width / 2.0
    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    both_rows = one_sided_rows = empty_rows = 0

    for y in range(geom.raw_height - 1, geom.cut_height - 1, -y_step):
        runs = row_runs(mask_raw[y], min_width=min_run_width)
        if not runs:
            empty_rows += 1
            continue
        left_candidates = [r for r in runs if r[2] < center_x]
        right_candidates = [r for r in runs if r[2] >= center_x]
        left = max(left_candidates, key=lambda r: r[2]) if left_candidates else None
        right = min(right_candidates, key=lambda r: r[2]) if right_candidates else None
        if left is not None:
            left_points.append((left[2], float(y)))
        if right is not None:
            right_points.append((right[2], float(y)))
        if left is not None and right is not None:
            both_rows += 1
        elif left is not None or right is not None:
            one_sided_rows += 1

    lanes = []
    if len(left_points) >= min_points:
        lanes.append(left_points)
    if len(right_points) >= min_points:
        lanes.append(right_points)
    return lanes, summarize_lanes(lanes, {
        "both_rows": float(both_rows),
        "one_sided_rows": float(one_sided_rows),
        "empty_rows": float(empty_rows),
        "left_points": float(len(left_points)),
        "right_points": float(len(right_points)),
    })


def extract_lane_points_rowwise_poly(mask_raw: np.ndarray, geom: CameraGeometry, degree: int = 2, **params):
    lanes, q = extract_lane_points_rowwise(mask_raw, geom=geom, **params)
    y_step = int(params.get("y_step", 8))
    min_points = int(params.get("min_points", 8))
    smoothed = [smooth_lane_polyfit(lane, geom=geom, degree=degree, y_step=y_step, min_points=min_points) for lane in lanes]
    smoothed = [lane for lane in smoothed if len(lane) >= min_points]
    return smoothed, summarize_lanes(smoothed, q)


def _points_from_component(labels: np.ndarray, component_id: int, geom: CameraGeometry, y_step: int, min_run_width: int) -> list[tuple[float, float]]:
    points = []
    component_mask = (labels == component_id).astype(np.uint8)
    for y in range(geom.raw_height - 1, geom.cut_height - 1, -y_step):
        runs = row_runs(component_mask[y], min_width=min_run_width)
        if not runs:
            continue
        run = max(runs, key=lambda r: r[1] - r[0])
        points.append((float(run[2]), float(y)))
    return points


def extract_lane_points_components(
    mask_raw: np.ndarray,
    geom: CameraGeometry,
    y_step: int = 6,
    min_run_width: int = 4,
    min_points: int = 8,
    min_area: int = 450,
    min_height: int = 120,
    degree: int = 2,
) -> tuple[list[list[tuple[float, float]]], dict[str, float]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_raw > 0).astype(np.uint8), 8)
    center_x = geom.raw_width / 2.0
    comps = []
    for cid in range(1, num_labels):
        x, y, w, h, area = stats[cid]
        if area < min_area or h < min_height or y + h < geom.cut_height:
            continue
        cx, cy = centroids[cid]
        score = float(h * 3 + np.sqrt(area) - abs(cx - center_x) * 0.02)
        comps.append({"cid": cid, "cx": float(cx), "area": int(area), "height": int(h), "score": score})

    selected = []
    left_candidates = [c for c in comps if c["cx"] < center_x]
    right_candidates = [c for c in comps if c["cx"] >= center_x]
    if left_candidates:
        selected.append(max(left_candidates, key=lambda c: c["score"]))
    if right_candidates:
        selected.append(max(right_candidates, key=lambda c: c["score"]))

    lanes = []
    for comp in selected:
        pts = _points_from_component(labels, comp["cid"], geom, y_step, min_run_width)
        if len(pts) >= min_points:
            lanes.append(smooth_lane_polyfit(pts, geom=geom, degree=degree, y_step=y_step, min_points=min_points))
    lanes = [lane for lane in lanes if len(lane) >= min_points]
    return lanes, summarize_lanes(lanes, {
        "component_candidates": float(len(comps)),
        "selected_components": float(len(selected)),
        "selected_area_total": float(sum(c["area"] for c in selected)),
    })


def extract_lanes_by_method(mask_raw: np.ndarray, geom: CameraGeometry, method: str, **params):
    if method == "rowwise":
        return extract_lane_points_rowwise(mask_raw, geom=geom, **params)
    if method == "rowwise_poly":
        return extract_lane_points_rowwise_poly(mask_raw, geom=geom, **params)
    if method == "component_poly":
        return extract_lane_points_components(mask_raw, geom=geom, **params)
    raise ValueError(f"unknown lane extraction method: {method}")


def draw_lane_mask(shape_hw: tuple[int, int], lanes: Sequence[Sequence[tuple[float, float]]], thickness: int = 10) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for idx, lane in enumerate(lanes, start=1):
        if len(lane) < 2:
            continue
        pts = np.array([[round(x), round(y)] for x, y in lane], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=idx, thickness=thickness)
    return mask


def write_lines_txt(path: Path, lanes: Sequence[Sequence[tuple[float, float]]]) -> None:
    ensure_dirs(path.parent)
    lines = []
    for lane in lanes:
        parts = []
        for x, y in lane:
            if x >= 0 and y >= 0:
                parts.append(f"{x:.2f} {y:.2f}")
        if parts:
            lines.append(" ".join(parts))
    with open(windows_long_path(path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def overlay_lanes(rgb: np.ndarray, lanes: Sequence[Sequence[tuple[float, float]]], mask: np.ndarray | None = None) -> np.ndarray:
    out = rgb.copy()
    if mask is not None:
        color = np.zeros_like(out)
        color[:, :, 0] = (mask > 0) * 255
        out = cv2.addWeighted(out, 0.75, color, 0.25, 0)
    colors = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (255, 0, 255)]
    for idx, lane in enumerate(lanes):
        if len(lane) < 2:
            continue
        pts = np.array([[round(x), round(y)] for x, y in lane], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=colors[idx % len(colors)], thickness=4)
    return out
