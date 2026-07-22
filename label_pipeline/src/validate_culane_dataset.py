from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import culane_builder_common as cb


def validate_dataset(dataset_root: Path, geom: cb.CameraGeometry) -> dict:
    dataset_root = dataset_root.resolve()
    list_dir = dataset_root / "list"
    issues = []
    counts = {}

    for name in ["train_gt.txt", "val.txt", "test.txt"]:
        path = list_dir / name
        if not path.exists():
            issues.append(f"missing list file: {path}")
            counts[name] = 0
            continue

        rows = [line.strip().split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[name] = len(rows)

        for row_idx, row in enumerate(rows):
            img_rel = row[0].lstrip("/")
            img_path = dataset_root / img_rel
            lines_path = img_path.with_suffix(".lines.txt")
            if not cb.path_exists(img_path):
                issues.append(f"{name}:{row_idx} missing image {img_rel}")
                continue
            if not cb.path_exists(lines_path):
                issues.append(f"{name}:{row_idx} missing lines {lines_path.relative_to(dataset_root)}")
            else:
                with open(cb.windows_long_path(lines_path), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    issues.append(f"{name}:{row_idx} empty lines {lines_path.relative_to(dataset_root)}")

            try:
                with Image.open(cb.windows_long_path(img_path)) as im:
                    size = im.size
                if size != (geom.raw_width, geom.raw_height):
                    issues.append(f"{name}:{row_idx} unexpected image size {size} {img_rel}")
            except Exception as exc:
                issues.append(f"{name}:{row_idx} unreadable image {img_rel}: {exc!r}")

            if name == "train_gt.txt":
                if len(row) < 2 + geom.max_lanes:
                    issues.append(f"{name}:{row_idx} malformed train row: {' '.join(row)}")
                    continue
                mask_rel = row[1].lstrip("/")
                mask_path = dataset_root / mask_rel
                if not cb.path_exists(mask_path):
                    issues.append(f"{name}:{row_idx} missing mask {mask_rel}")
                else:
                    try:
                        with Image.open(cb.windows_long_path(mask_path)) as im:
                            mask_size = im.size
                        if mask_size != (geom.raw_width, geom.raw_height):
                            issues.append(f"{name}:{row_idx} unexpected mask size {mask_size} {mask_rel}")
                    except Exception as exc:
                        issues.append(f"{name}:{row_idx} unreadable mask {mask_rel}: {exc!r}")

    summary = {
        "dataset_root": str(dataset_root),
        "geometry": cb.geometry_dict(geom),
        "counts": counts,
        "issue_count": len(issues),
        "issues_sample": issues[:100],
    }
    cb.write_json(dataset_root / "validation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated CULane-style pseudo-label dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Optional builder config for camera geometry.")
    args = parser.parse_args()

    if args.config:
        config = cb.read_json(args.config)
        geom = cb.CameraGeometry.from_config(config.get("camera", {}))
    else:
        geom = cb.CameraGeometry()

    print(json.dumps(validate_dataset(args.dataset_root, geom), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
