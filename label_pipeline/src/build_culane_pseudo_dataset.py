from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import culane_builder_common as cb


DEFAULT_REVIEW_PRIORITY_POLICY = {
    "straight": "normal",
    "left_curve": "normal",
    "right_curve": "normal",
    "off_center_left": "normal",
    "off_center_right": "normal",
    "intersection_approach": "high",
    "intersection_left": "high",
    "intersection_right": "high",
    "failure_cases": "hard_case",
}


def scene_rows(rows: list[dict[str, str]], label_column: str, max_per_scene: int | None = None) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(label_column, "unknown"), []).append(row)

    selected = []
    for _, items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: (r.get("timestamp", ""), r.get("frame_id", ""), r.get("raw_path", "")))
        if max_per_scene is not None and len(items) > max_per_scene:
            idxs = [round(i * (len(items) - 1) / (max_per_scene - 1)) for i in range(max_per_scene)] if max_per_scene > 1 else [0]
            items = [items[i] for i in idxs]
        selected.extend(items)
    return selected


def reject_reasons_from_quality(quality: dict[str, float], rules: dict) -> list[str]:
    num_lanes = int(quality.get("num_lanes", 0))
    point_total = int(quality.get("point_total", 0))
    median_y_span = float(quality.get("median_y_span", 0.0))
    mask_support = float(quality.get("mask_support", 0.0))
    crossing_risk = float(quality.get("crossing_risk", 0.0))
    small_gap_risk = float(quality.get("small_gap_risk", 0.0))

    reasons = []
    if num_lanes < int(rules.get("min_lanes", 1)):
        reasons.append("no_lane")
    if point_total < int(rules.get("min_point_total", 12)):
        reasons.append("low_point_total")
    if median_y_span < float(rules.get("min_y_span", 80.0)):
        reasons.append("short_y_span")
    if mask_support < float(rules.get("min_mask_support", 0.0)):
        reasons.append("low_mask_support")
    if bool(rules.get("reject_crossing", False)) and crossing_risk >= 1.0:
        reasons.append("crossing_risk")
    if bool(rules.get("reject_small_gap", False)) and small_gap_risk >= 1.0:
        reasons.append("small_gap_risk")
    return reasons


def status_from_quality(quality: dict[str, float], rules: dict) -> str:
    return "reject" if reject_reasons_from_quality(quality, rules) else "usable"


def lane_exist_flags(num_lanes: int, max_lanes: int) -> list[int]:
    return [1 if i < num_lanes else 0 for i in range(max_lanes)]


def list_line(record: dict, max_lanes: int, include_mask: bool) -> str:
    img = "/" + record["dataset_image"]
    flags = " ".join(map(str, lane_exist_flags(int(record.get("num_lanes", 0)), max_lanes)))
    if include_mask:
        return f"{img} /{record['dataset_mask']} {flags}"
    return img


def write_list(path: Path, lines: list[str]) -> None:
    cb.ensure_dirs(path.parent)
    with open(cb.windows_long_path(path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def build_dataset(config: dict, max_per_scene: int | None = None, clean: bool = False) -> dict:
    project_root = Path(config.get("project_root", ".")).resolve()
    input_cfg = config["input"]
    output_cfg = config["output"]
    build_cfg = config.get("build", {})
    geom = cb.CameraGeometry.from_config(config.get("camera", {}))

    manifest_path = Path(input_cfg["manifest_csv"])
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    output_root = Path(output_cfg["dataset_root"])
    if not output_root.is_absolute():
        output_root = project_root / output_root

    label_column = input_cfg.get("label_column", "label")
    raw_path_column = input_cfg.get("raw_path_column", "raw_path")
    review_priority_policy = config.get(
        "review_priority_policy",
        config.get("scene_policy", DEFAULT_REVIEW_PRIORITY_POLICY),
    )
    max_lanes = int(config.get("camera", {}).get("max_lanes", geom.max_lanes))

    rows = cb.read_manifest(manifest_path)
    max_per_scene = max_per_scene if max_per_scene is not None else build_cfg.get("max_per_scene")
    rows = scene_rows(rows, label_column=label_column, max_per_scene=max_per_scene)

    if clean and cb.path_exists(output_root):
        cb.remove_tree(output_root)

    image_base = Path(output_cfg.get("image_base", "driver_map"))
    mask_base = Path(output_cfg.get("mask_base", "laneseg_label_w16")) / image_base
    review_root = output_root / "_review_overlays"
    cb.ensure_dirs(output_root / "list", review_root)

    hsv_cfg = config["hsv"]
    extraction_cfg = config["lane_extraction"]
    method = extraction_cfg["method"]
    method_params = {k: v for k, v in extraction_cfg.items() if k != "method"}
    quality_gate = config.get("quality_gate", {})
    review_cfg = config.get("review", {})
    max_review_overlays = int(review_cfg.get("max_overlays", 300))
    save_usable_every = int(review_cfg.get("save_usable_every", 5))
    crop_before_hsv = bool(config.get("preprocess", {}).get("crop_before_hsv", True))

    records = []
    overlay_count = 0

    for row_idx, row in enumerate(rows):
        scene = row.get(label_column, "unknown")
        review_priority = review_priority_policy.get(scene, "normal")
        raw_path = cb.resolve_input_path(row.get(raw_path_column, ""), input_cfg)
        if raw_path is None or not cb.path_exists(raw_path):
            records.append({
                "scene": scene,
                "review_priority": review_priority,
                "status": "missing_raw",
                "raw_path": row.get(raw_path_column, ""),
            })
            continue

        rgb = cb.load_rgb(raw_path)
        mask = cb.yellow_mask_raw(rgb, hsv_cfg, geom=geom, crop_before_hsv=crop_before_hsv)
        lanes, quality = cb.extract_lanes_by_method(mask, geom=geom, method=method, **method_params)
        lanes = sorted(lanes, key=lambda lane: sum(p[0] for p in lane) / max(1, len(lane)) if lane else 1e9)
        quality["mask_support"] = cb.lane_mask_support(mask, lanes)
        quality = {k: (int(v) if isinstance(v, float) and v.is_integer() else float(v)) for k, v in quality.items()}
        reject_reasons = reject_reasons_from_quality(quality, quality_gate)
        status = "reject" if reject_reasons else "usable"

        rel_img = image_base / status / scene / raw_path.name
        img_dst = output_root / rel_img
        lines_dst = img_dst.with_suffix(".lines.txt")
        mask_dst_rel = mask_base / status / scene / raw_path.with_suffix(".png").name
        mask_dst = output_root / mask_dst_rel

        if status != "reject":
            cb.copy_image(raw_path, img_dst)
            cb.write_lines_txt(lines_dst, lanes)
            lane_mask = cb.draw_lane_mask((geom.raw_height, geom.raw_width), lanes, thickness=int(config.get("mask_thickness", 10)))
            cb.save_mask(mask_dst, lane_mask)
            should_save_overlay = (
                review_priority != "normal"
                or (save_usable_every > 0 and row_idx % save_usable_every == 0)
            )
            if overlay_count < max_review_overlays and should_save_overlay:
                overlay = cb.overlay_lanes(rgb, lanes, mask=mask)
                cb.save_rgb(review_root / review_priority / status / scene / raw_path.name, overlay)
                overlay_count += 1

        records.append({
            "scene": scene,
            "review_priority": review_priority,
            "status": status,
            "reject_reasons": ";".join(reject_reasons),
            "raw_path": str(raw_path),
            "dataset_image": rel_img.as_posix() if status != "reject" else "",
            "dataset_mask": mask_dst_rel.as_posix() if status != "reject" else "",
            **quality,
        })

    accepted = sorted([r for r in records if r.get("status") == "usable"], key=lambda r: r["dataset_image"])
    train_ratio = float(build_cfg.get("train_ratio", 0.8))
    split_idx = max(1, int(len(accepted) * train_ratio)) if accepted else 0
    train_records = accepted[:split_idx]
    val_records = accepted[split_idx:]
    if accepted and not val_records:
        val_records = accepted[-1:]
        train_records = accepted[:-1]

    list_dir = output_root / "list"
    write_list(list_dir / "train_gt.txt", [list_line(r, max_lanes, include_mask=True) for r in train_records])
    write_list(list_dir / "val.txt", [list_line(r, max_lanes, include_mask=False) for r in val_records])
    write_list(list_dir / "test.txt", [list_line(r, max_lanes, include_mask=False) for r in val_records])

    split_dir = list_dir / "test_split"
    cb.ensure_dirs(split_dir)
    write_list(split_dir / "test0_normal.txt", [list_line(r, max_lanes, include_mask=False) for r in val_records])
    for idx, name in enumerate(["crowd", "hlight", "shadow", "noline", "arrow", "curve", "cross", "night"], start=1):
        write_list(split_dir / f"test{idx}_{name}.txt", [])

    manifest_out = output_root / "build_manifest.csv"
    fieldnames = sorted({k for r in records for k in r.keys()})
    with open(cb.windows_long_path(manifest_out), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    status_counts: dict[str, int] = {}
    scene_status_counts: dict[str, int] = {}
    review_priority_counts: dict[str, int] = {}
    for r in records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        key = f"{r.get('scene', 'unknown')}::{r['status']}"
        scene_status_counts[key] = scene_status_counts.get(key, 0) + 1
        review_priority = r.get("review_priority", "unknown")
        review_priority_counts[review_priority] = review_priority_counts.get(review_priority, 0) + 1

    summary = {
        "name": config.get("name", "culane_pseudo_dataset"),
        "output_root": str(output_root),
        "input_rows": len(rows),
        "status_counts": status_counts,
        "scene_status_counts": scene_status_counts,
        "review_priority_counts": review_priority_counts,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "manifest": str(manifest_out),
        "geometry": cb.geometry_dict(geom),
        "hsv": hsv_cfg,
        "lane_extraction": extraction_cfg,
        "quality_gate": quality_gate,
    }
    cb.write_json(output_root / "build_summary.json", summary)
    report_path = Path(config.get("reports_dir", output_root / "_reports"))
    if not report_path.is_absolute():
        report_path = project_root / report_path
    cb.write_json(report_path / f"{config.get('name', 'culane_pseudo_dataset')}_build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CULane-style pseudo-label dataset from manifest images.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-per-scene", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    config = cb.read_json(args.config)
    summary = build_dataset(config, max_per_scene=args.max_per_scene, clean=args.clean)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
