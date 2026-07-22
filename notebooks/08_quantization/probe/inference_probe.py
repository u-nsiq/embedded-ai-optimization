#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent

RAW_W = 1296
RAW_H = 972
CUT_HEIGHT = 445
IMG_W = 800
IMG_H = 320
NUM_PRIORS = 192
OUTPUT_DIM = 78


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows, fieldnames=None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def cmd(args, timeout=1.0):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=timeout, text=True).strip()
    except Exception:
        return ""


def parse_temp(text):
    try:
        return float(text.split("=")[1].split("'")[0])
    except Exception:
        return math.nan


def parse_clock(text):
    try:
        return float(text.split("=")[1]) / 1_000_000.0
    except Exception:
        return math.nan


def read_cpu_times():
    try:
        vals = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
        return sum(vals), idle
    except Exception:
        return math.nan, math.nan


def cpu_percent(prev, cur):
    if prev is None:
        return math.nan
    total = cur[0] - prev[0]
    idle = cur[1] - prev[1]
    if not math.isfinite(total) or total <= 0:
        return math.nan
    return max(0.0, min(100.0, 100.0 * (1.0 - idle / total)))


def mem_available_mb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return math.nan


def telemetry(prev_cpu):
    cur = read_cpu_times()
    return {
        "temp_c": parse_temp(cmd(["vcgencmd", "measure_temp"])),
        "clock_arm_mhz": parse_clock(cmd(["vcgencmd", "measure_clock", "arm"])),
        "throttled": cmd(["vcgencmd", "get_throttled"]),
        "cpu_percent": cpu_percent(prev_cpu, cur),
        "mem_available_mb": mem_available_mb(),
    }, cur


def read_records(max_frames):
    with (ROOT / "samples" / "records_manifest.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = rows[: int(max_frames)] if max_frames else rows
    if not rows:
        raise RuntimeError("No sample rows.")
    return rows


def imread_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def preprocess_bgr(bgr):
    crop = bgr[int(CUT_HEIGHT):, :, :]
    resized = cv2.resize(crop, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0


def make_session(model_path: Path, threads: int):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def run_raw(sess, input_name, output_name, tensor):
    out = sess.run([output_name], {input_name: tensor})[0]
    if out.shape != (1, NUM_PRIORS, OUTPUT_DIM):
        raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
    return out


def finite(values):
    return np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)


def mean(values):
    arr = finite(values)
    return float(arr.mean()) if len(arr) else math.nan


def pct(values, p):
    arr = finite(values)
    return float(np.percentile(arr, p)) if len(arr) else math.nan


def model_info(name):
    manifest = read_json(ROOT / "models" / "models_manifest.json")
    by_name = {m["name"]: m for m in manifest["models"]}
    if name not in by_name:
        raise KeyError(f"Unknown model: {name}")
    return by_name[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--warmup-runs", type=int, default=5)
    ap.add_argument("--telemetry-every", type=int, default=10)
    args = ap.parse_args()

    model = model_info(args.model)
    model_path = ROOT / model["onnx_rel"]
    records = read_records(args.max_frames)

    run_dir = ROOT / "outputs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}_t{args.threads}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", {
        "model": args.model,
        "threads": args.threads,
        "max_frames": args.max_frames,
        "warmup_runs": args.warmup_runs,
        "telemetry_every": args.telemetry_every,
        "model_info": model,
        "measurement": "ONNX inference only. Images are loaded/preprocessed before timed inference loop.",
    })

    print(f"[probe] loading {len(records)} images and preprocessing tensors")
    t_prep0 = time.perf_counter()
    tensors = []
    for row in records:
        bgr = imread_bgr(ROOT / row["image_rel"])
        tensors.append(preprocess_bgr(bgr))
    prep_total_s = time.perf_counter() - t_prep0

    sess, input_name, output_name = make_session(model_path, args.threads)
    for _ in range(args.warmup_runs):
        run_raw(sess, input_name, output_name, tensors[0])

    frame_rows = []
    tel_rows = []
    prev_cpu = read_cpu_times()
    print(f"[probe] model={args.model} threads={args.threads} frames={len(tensors)}")
    wall_loop_start = time.perf_counter()
    inference_total_ms = 0.0
    latest_tel = {}

    for i, tensor in enumerate(tensors):
        if i % max(1, args.telemetry_every) == 0:
            tel, prev_cpu = telemetry(prev_cpu)
            tel.update({"frame_index": i, "model": args.model, "threads": args.threads})
            tel_rows.append(tel)
            latest_tel = tel

        t0 = time.perf_counter()
        out = run_raw(sess, input_name, output_name, tensor)
        t1 = time.perf_counter()
        inf_ms = (t1 - t0) * 1000.0
        inference_total_ms += inf_ms
        frame_rows.append({
            "frame_index": i,
            "key": records[i]["key"],
            "model": args.model,
            "threads": args.threads,
            "inference_ms": inf_ms,
            "fps_from_inference": 1000.0 / inf_ms if inf_ms > 0 else math.nan,
            "raw_min": float(out.min()),
            "raw_max": float(out.max()),
            "raw_mean": float(out.mean()),
        })
        if i % 50 == 0:
            temp = latest_tel.get("temp_c", math.nan)
            cpu = latest_tel.get("cpu_percent", math.nan)
            clock = latest_tel.get("clock_arm_mhz", math.nan)
            throttled = latest_tel.get("throttled", "") or "unknown"
            print(
                f"[{i:04d}] inf={inf_ms:.1f}ms fps={1000.0/inf_ms:.2f} "
                f"temp={temp:.1f}C cpu={cpu:.1f}% clock={clock:.0f}MHz throttled={throttled}"
            )

    wall_loop_total_s = time.perf_counter() - wall_loop_start
    inf_values = [r["inference_ms"] for r in frame_rows]
    summary = {
        "model": args.model,
        "threads": args.threads,
        "frames": len(frame_rows),
        "preprocess_total_s_not_in_inference_loop": prep_total_s,
        "wall_loop_total_s": wall_loop_total_s,
        "inference_total_s": inference_total_ms / 1000.0,
        "non_inference_loop_overhead_s": wall_loop_total_s - (inference_total_ms / 1000.0),
        "inference_mean_ms": mean(inf_values),
        "inference_p50_ms": pct(inf_values, 50),
        "inference_p90_ms": pct(inf_values, 90),
        "inference_p95_ms": pct(inf_values, 95),
        "inference_max_ms": pct(inf_values, 100),
        "fps_from_inference_mean": 1000.0 / mean(inf_values),
        "temp_max_c": pct([t.get("temp_c", math.nan) for t in tel_rows], 100),
        "temp_mean_c": mean([t.get("temp_c", math.nan) for t in tel_rows]),
        "cpu_percent_mean": mean([t.get("cpu_percent", math.nan) for t in tel_rows]),
        "cpu_percent_max": pct([t.get("cpu_percent", math.nan) for t in tel_rows], 100),
        "clock_arm_mean_mhz": mean([t.get("clock_arm_mhz", math.nan) for t in tel_rows]),
        "mem_available_min_mb": pct([t.get("mem_available_mb", math.nan) for t in tel_rows], 0),
        "throttled_values": " | ".join(sorted({t["throttled"] for t in tel_rows if t.get("throttled")})),
    }
    write_csv(run_dir / "frames.csv", frame_rows)
    write_csv(run_dir / "telemetry.csv", tel_rows)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", [summary])
    print("[probe] written:", run_dir)
    print("[summary]")
    print(f"  model={args.model} threads={args.threads} frames={len(frame_rows)}")
    print(f"  wall_loop_total={summary['wall_loop_total_s']:.2f}s inference_total={summary['inference_total_s']:.2f}s non_inference_overhead={summary['non_inference_loop_overhead_s']:.2f}s")
    print(f"  inference_mean={summary['inference_mean_ms']:.1f}ms p95={summary['inference_p95_ms']:.1f}ms max={summary['inference_max_ms']:.1f}ms fps_mean={summary['fps_from_inference_mean']:.2f}")
    print(f"  temp_max={summary['temp_max_c']:.1f}C cpu_mean={summary['cpu_percent_mean']:.1f}% cpu_max={summary['cpu_percent_max']:.1f}% clock_mean={summary['clock_arm_mean_mhz']:.0f}MHz")
    print(f"  throttled_values={summary['throttled_values']}")


if __name__ == "__main__":
    main()
