from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path("<workspace-root>")
DEFAULT_BLACKLIST = (
    PROJECT_ROOT
    / "30_pipelines"
    / "culane_pseudo_dataset_builder"
    / "inputs"
    / "blacklists"
    / "train_exclude_rules.csv"
)


def windows_long_path(path: Path | str) -> str:
    path_obj = Path(path)
    if os.name == "nt" and not path_obj.is_absolute():
        path_obj = path_obj.resolve()
    s = str(path_obj)
    if os.name == "nt" and not s.startswith("\\\\?\\") and len(s) >= 240:
        return "\\\\?\\" + s
    return s


def strip_windows_long_prefix(path_str: str) -> str:
    return path_str[4:] if path_str.startswith("\\\\?\\") else path_str


def image_size(path: Path) -> str | None:
    try:
        with Image.open(windows_long_path(path)) as im:
            return f"{im.width}x{im.height}"
    except Exception:
        return None


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(windows_long_path(root)):
        for name in filenames:
            path = Path(strip_windows_long_prefix(str(Path(dirpath) / name)))
            if path.suffix.lower() in IMAGE_EXTS:
                files.append(path)
    return sorted(files)


def add_image_rows(
    rows: list[dict[str, str]],
    skipped: list[dict[str, str]],
    dataset: str,
    split_role: str,
    source_group: str,
    root: Path,
    label: str,
    project_root: Path,
) -> None:
    for path in iter_images(root):
        size = image_size(path)
        if size is None:
            skipped.append({
                "dataset": dataset,
                "split_role": split_role,
                "source_group": source_group,
                "label": label,
                "raw_path": str(path),
                "reason": "unreadable_image",
            })
            continue
        rows.append({
            "dataset": dataset,
            "split_role": split_role,
            "source_group": source_group,
            "label": label,
            "raw_path": str(path),
            "file_name": path.name,
            "relative_path": path.relative_to(project_root).as_posix(),
            "width_height": size,
        })


def add_label_dir_rows(
    rows: list[dict[str, str]],
    skipped: list[dict[str, str]],
    dataset: str,
    split_role: str,
    source_group: str,
    root: Path,
    label_prefix: str,
    project_root: Path,
    exclude_labels: set[str] | None = None,
) -> None:
    exclude_labels = exclude_labels or set()
    if not root.exists():
        return
    label_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    for label_dir in label_dirs:
        label_name = label_dir.name
        if label_name in exclude_labels:
            continue
        add_image_rows(
            rows=rows,
            skipped=skipped,
            dataset=dataset,
            split_role=split_role,
            source_group=source_group,
            root=label_dir,
            label=f"{label_prefix}_{label_name}",
            project_root=project_root,
        )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split_role",
        "source_group",
        "label",
        "raw_path",
        "file_name",
        "relative_path",
        "width_height",
    ]
    with open(windows_long_path(path), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_time_from_name(file_name: str) -> str:
    parts = file_name.split("_")
    return parts[1] if len(parts) > 1 else ""


def read_exclude_rules(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(windows_long_path(path), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def match_exclude_rule(row: dict[str, str], rules: list[dict[str, str]]) -> dict[str, str] | None:
    file_time = image_time_from_name(row["file_name"])
    for rule in rules:
        if row["label"] != rule.get("label", ""):
            continue
        if rule.get("start_time", "") <= file_time <= rule.get("end_time", ""):
            return rule
    return None


def apply_train_exclude_rules(
    rows: list[dict[str, str]],
    rules: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        rule = match_exclude_rule(row, rules)
        if rule is None:
            kept.append(row)
            continue
        out = dict(row)
        out["exclude_reason"] = rule.get("reason", "manual_exclude")
        out["exclude_start_time"] = rule.get("start_time", "")
        out["exclude_end_time"] = rule.get("end_time", "")
        excluded.append(out)
    return kept, excluded


def summarize(rows: list[dict[str, str]]) -> dict:
    by_dataset: dict[str, int] = {}
    by_group: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_size: dict[str, int] = {}
    for row in rows:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        by_group[row["source_group"]] = by_group.get(row["source_group"], 0) + 1
        by_label[row["label"]] = by_label.get(row["label"], 0) + 1
        by_size[row["width_height"]] = by_size.get(row["width_height"], 0) + 1
    return {
        "total": len(rows),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_group": dict(sorted(by_group.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_size": dict(sorted(by_size.items())),
    }


def build_manifests(
    project_root: Path,
    exclude_rules: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]], list[dict[str, str]]]:
    field1_raw = project_root / "10_experiments" / "02_field1_map_lab" / "data" / "20260430_map_lab" / "raw"
    field2_raw = project_root / "10_experiments" / "04_field2_map_lab" / "field2_experiment" / "data" / "20260501_field2" / "raw"

    skipped: list[dict[str, str]] = []
    field1_train: list[dict[str, str]] = []
    field2_train: list[dict[str, str]] = []
    holdout_eval: list[dict[str, str]] = []

    add_label_dir_rows(
        field1_train,
        skipped,
        dataset="field1",
        split_role="train_candidate",
        source_group="raw_lane",
        root=field1_raw / "lane",
        label_prefix="field1_lane",
        project_root=project_root,
    )
    add_label_dir_rows(
        field1_train,
        skipped,
        dataset="field1",
        split_role="train_candidate",
        source_group="raw_signs",
        root=field1_raw / "signs",
        label_prefix="field1_sign",
        project_root=project_root,
        exclude_labels={"background"},
    )
    add_image_rows(
        holdout_eval,
        skipped,
        dataset="field1",
        split_role="holdout_eval",
        source_group="raw_signs_background",
        root=field1_raw / "signs" / "background",
        label="field1_sign_background",
        project_root=project_root,
    )

    add_label_dir_rows(
        field2_train,
        skipped,
        dataset="field2",
        split_role="train_candidate",
        source_group="raw_lane_drive_capture",
        root=field2_raw / "01_lane_drive_capture",
        label_prefix="field2_lane",
        project_root=project_root,
    )
    add_label_dir_rows(
        field2_train,
        skipped,
        dataset="field2",
        split_role="train_candidate",
        source_group="raw_lane_preview",
        root=field2_raw / "03_lane_preview",
        label_prefix="field2_preview",
        project_root=project_root,
    )
    add_label_dir_rows(
        field2_train,
        skipped,
        dataset="field2",
        split_role="train_candidate",
        source_group="raw_sign_capture_drive",
        root=field2_raw / "06_sign_capture_drive",
        label_prefix="field2_sign",
        project_root=project_root,
    )
    add_label_dir_rows(
        field2_train,
        skipped,
        dataset="field2",
        split_role="train_candidate",
        source_group="raw_traffic_capture",
        root=field2_raw / "07_traffic_capture",
        label_prefix="field2_traffic",
        project_root=project_root,
        exclude_labels={"background"},
    )
    add_image_rows(
        holdout_eval,
        skipped,
        dataset="field2",
        split_role="holdout_eval",
        source_group="raw_traffic_background",
        root=field2_raw / "07_traffic_capture" / "background",
        label="field2_traffic_background",
        project_root=project_root,
    )
    add_label_dir_rows(
        field2_train,
        skipped,
        dataset="field2",
        split_role="train_candidate",
        source_group="raw_redline_capture_debug",
        root=field2_raw / "08_redline_capture_debug",
        label_prefix="field2_redline",
        project_root=project_root,
    )

    field1_train, field1_excluded = apply_train_exclude_rules(field1_train, exclude_rules)
    field2_train, field2_excluded = apply_train_exclude_rules(field2_train, exclude_rules)
    excluded = field1_excluded + field2_excluded
    combined_train = field1_train + field2_train
    return {
        "field1_train_candidate_manifest.csv": field1_train,
        "field2_train_candidate_manifest.csv": field2_train,
        "field1_field2_train_candidate_manifest.csv": combined_train,
        "holdout_eval_candidate_manifest.csv": holdout_eval,
    }, skipped, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/holdout candidate manifests for CULane pseudo dataset builds.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "30_pipelines" / "culane_pseudo_dataset_builder" / "inputs" / "manifests",
    )
    parser.add_argument("--exclude-rules", type=Path, default=DEFAULT_BLACKLIST)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    exclude_rules = read_exclude_rules(args.exclude_rules.resolve())
    manifests, skipped, excluded = build_manifests(project_root, exclude_rules)

    for name, rows in manifests.items():
        write_manifest(output_dir / name, rows)

    summary = {
        name: summarize(rows)
        for name, rows in manifests.items()
    }
    summary["notes"] = [
        "Only 1296x972 raw images are included.",
        "field1 raw/signs/background is held out from training.",
        "field2 raw/07_traffic_capture/background is held out from training.",
        "field1 preprocess/debug images are intentionally excluded because the pipeline uses raw camera geometry.",
        "Manual train exclude rules are applied before writing train manifests.",
    ]
    summary["exclude_rules"] = exclude_rules
    summary["excluded"] = excluded
    summary["skipped"] = skipped

    summary_path = output_dir / "candidate_manifest_summary.json"
    with open(windows_long_path(summary_path), "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
