#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2

from lane_runtime_fp32 import (
    HAS_SCIPY,
    RAW_H,
    RAW_W,
    decode_raw_to_lanes,
    init_drive_memory,
    load_package_contracts,
    make_ort_session,
    run_onnx_raw,
    update_drive,
    update_driving_params,
    warmup_ort_session,
)
from motor_adapter import DifferentialMotor, command_from_steer


ROOT = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_config_hash(cfg):
    text = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def run_cmd(args, timeout=0.4):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return out.strip()
    except Exception:
        return ""


def parse_temp(text):
    match = re.search(r"temp=([0-9.]+)", text or "")
    return float(match.group(1)) if match else None


def parse_clock_mhz(text):
    match = re.search(r"frequency\(\d+\)=([0-9]+)", text or "")
    return float(match.group(1)) / 1_000_000.0 if match else None


def parse_mem_available_mb():
    try:
        data = Path("/proc/meminfo").read_text(encoding="utf-8")
    except Exception:
        return None
    match = re.search(r"MemAvailable:\s+([0-9]+)\s+kB", data)
    return float(match.group(1)) / 1024.0 if match else None


def read_cpu_times():
    try:
        first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return None
    parts = first.split()
    if not parts or parts[0] != "cpu":
        return None
    vals = [float(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
    total = sum(vals)
    return total, idle


class CpuUsageMeter:
    def __init__(self):
        self.prev = read_cpu_times()

    def sample(self):
        cur = read_cpu_times()
        if cur is None or self.prev is None:
            self.prev = cur
            return None
        total_delta = cur[0] - self.prev[0]
        idle_delta = cur[1] - self.prev[1]
        self.prev = cur
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


class SystemTelemetry:
    def __init__(self, every_n_frames=1):
        self.every_n_frames = max(1, int(every_n_frames))
        self.cpu = CpuUsageMeter()
        self.last = {
            "temp_c": None,
            "arm_clock_mhz": None,
            "throttled": "",
            "cpu_percent": None,
            "mem_available_mb": None,
        }

    def sample(self, frame_idx):
        if frame_idx % self.every_n_frames != 0:
            return self.last
        temp = parse_temp(run_cmd(["vcgencmd", "measure_temp"]))
        clock = parse_clock_mhz(run_cmd(["vcgencmd", "measure_clock", "arm"]))
        throttled = run_cmd(["vcgencmd", "get_throttled"])
        cpu_percent = self.cpu.sample()
        mem = parse_mem_available_mb()
        self.last = {
            "temp_c": temp,
            "arm_clock_mhz": clock,
            "throttled": throttled,
            "cpu_percent": cpu_percent,
            "mem_available_mb": mem,
        }
        return self.last


class LiveCamera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = str(cfg.get("backend", "picamera2"))
        self.width = int(cfg.get("width", RAW_W))
        self.height = int(cfg.get("height", RAW_H))
        self.force_resize = bool(cfg.get("force_resize_to_raw", True))
        self.color_mode = str(cfg.get("color_mode", "bgr")).lower()
        self.rotate_180 = bool(cfg.get("rotate_180", False))
        self.flip_horizontal = bool(cfg.get("flip_horizontal", False))
        self.flip_vertical = bool(cfg.get("flip_vertical", False))
        self.picam2 = None
        self.cap = None
        if self.backend == "picamera2":
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            cam_cfg = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(cam_cfg)
            self.picam2.start()
            time.sleep(0.5)
            print(f"[camera] picamera2 {self.width}x{self.height}")
        elif self.backend == "opencv":
            index = int(cfg.get("opencv_index", 0))
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open OpenCV camera index {index}")
            print(f"[camera] opencv index={index} {self.width}x{self.height}")
        else:
            raise ValueError(f"Unsupported camera backend: {self.backend}")

    def _frame_to_bgr(self, frame):
        if self.color_mode in {"bgr", "bgr888", "as_is", "as-is", "none"}:
            return frame.copy()
        if self.color_mode in {"rgb", "rgb888", "rgb_to_bgr"}:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raise ValueError(f"Unsupported camera color_mode: {self.color_mode}")

    def _apply_geometry(self, bgr):
        if self.rotate_180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        if self.flip_horizontal:
            bgr = cv2.flip(bgr, 1)
        if self.flip_vertical:
            bgr = cv2.flip(bgr, 0)
        return bgr

    def read_bgr(self):
        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            bgr = self._frame_to_bgr(frame)
        else:
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                raise RuntimeError("OpenCV camera read failed")
        bgr = self._apply_geometry(bgr)
        if self.force_resize and (bgr.shape[0] != RAW_H or bgr.shape[1] != RAW_W):
            bgr = cv2.resize(bgr, (RAW_W, RAW_H), interpolation=cv2.INTER_LINEAR)
        return bgr

    def close(self):
        if self.picam2 is not None:
            self.picam2.stop()
        if self.cap is not None:
            self.cap.release()


def open_log_writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", newline="")
    fields = [
        "frame_idx",
        "t_sec",
        "capture_ms",
        "inference_ms",
        "decode_ms",
        "steering_ms",
        "motor_ms",
        "telemetry_ms",
        "pipeline_ms",
        "fps_inst",
        "fps_ema",
        "lane_count",
        "raw_mode",
        "effective_mode",
        "lost_frames",
        "steer_norm",
        "turn_bias",
        "motor_left",
        "motor_right",
        "motor_reason",
        "temp_c",
        "arm_clock_mhz",
        "throttled",
        "cpu_percent",
        "mem_available_mb",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


def fmt_opt(value, digits=3):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Minimal FP32 lane runtime stress probe for Raspberry Pi.")
    parser.add_argument("--config", type=Path, default=ROOT / "config_minimal_drive.json")
    args = parser.parse_args()

    cfg = read_json(args.config)
    run_cfg = cfg.get("run", {})
    camera_cfg = cfg.get("camera", {})
    onnx_cfg = cfg.get("onnx", {})
    motor_cfg = cfg.get("motor", {})
    telemetry_cfg = cfg.get("telemetry", {})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = str(run_cfg.get("name", "runtime_probe"))
    cfg_hash = stable_config_hash(cfg)
    out_root = ROOT / str(run_cfg.get("output_dir", "probe_runs"))
    run_dir = out_root / f"{timestamp}_{name}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config_used.json", cfg)

    print("[run]", run_dir)
    print("[decoder] scipy spline:", HAS_SCIPY)
    print("[debug] overlay/save/window disabled by design")

    decode_contract, driving_contract, _model_info = load_package_contracts(ROOT)
    driving_contract = update_driving_params(driving_contract, cfg.get("drive_postprocess_overrides", {}))

    session, input_name, output_name = make_ort_session(ROOT, int(onnx_cfg.get("threads", 4)))
    camera = LiveCamera(camera_cfg)
    motor = DifferentialMotor(motor_cfg)
    memory = init_drive_memory()
    telemetry = SystemTelemetry(telemetry_cfg.get("every_n_frames", 5))

    first_frame = camera.read_bgr()
    warmup_ort_session(session, input_name, output_name, first_frame, runs=int(onnx_cfg.get("warmup_runs", 3)))

    log_f, writer = open_log_writer(run_dir / "runtime_log.csv")
    duration_sec = float(run_cfg.get("duration_sec", 180.0))
    print_every = max(1, int(run_cfg.get("print_every_n_frames", 10)))
    frame_idx = 0
    start = time.perf_counter()
    fps_ema = None
    throttle_seen = set()

    try:
        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start
            if duration_sec > 0 and elapsed >= duration_sec:
                break

            t0 = time.perf_counter()
            bgr = camera.read_bgr()
            t1 = time.perf_counter()
            raw = run_onnx_raw(session, input_name, output_name, bgr)
            t2 = time.perf_counter()
            lanes = decode_raw_to_lanes(raw, decode_contract)
            t3 = time.perf_counter()
            row = update_drive(lanes, memory, driving_contract)
            t4 = time.perf_counter()
            command = command_from_steer(row["steer_norm"], motor_cfg, reason=row["effective_mode"])
            motor.apply(command)
            t5 = time.perf_counter()
            sysrow = telemetry.sample(frame_idx)
            t6 = time.perf_counter()

            pipeline_ms = (t6 - t0) * 1000.0
            fps_inst = 1000.0 / max(1e-6, pipeline_ms)
            fps_ema = fps_inst if fps_ema is None else 0.2 * fps_inst + 0.8 * fps_ema
            throttled = sysrow.get("throttled") or ""
            if throttled and throttled != "throttled=0x0":
                throttle_seen.add(throttled)

            writer.writerow({
                "frame_idx": frame_idx,
                "t_sec": f"{elapsed:.4f}",
                "capture_ms": f"{(t1 - t0) * 1000.0:.3f}",
                "inference_ms": f"{(t2 - t1) * 1000.0:.3f}",
                "decode_ms": f"{(t3 - t2) * 1000.0:.3f}",
                "steering_ms": f"{(t4 - t3) * 1000.0:.3f}",
                "motor_ms": f"{(t5 - t4) * 1000.0:.3f}",
                "telemetry_ms": f"{(t6 - t5) * 1000.0:.3f}",
                "pipeline_ms": f"{pipeline_ms:.3f}",
                "fps_inst": f"{fps_inst:.3f}",
                "fps_ema": f"{fps_ema:.3f}",
                "lane_count": len(lanes),
                "raw_mode": row.get("raw_mode", ""),
                "effective_mode": row.get("effective_mode", ""),
                "lost_frames": row.get("lost_frames", 0),
                "steer_norm": f"{row.get('steer_norm', 0.0):.6f}",
                "turn_bias": f"{row.get('turn_bias', 0.0):.6f}",
                "motor_left": f"{command.left:.6f}",
                "motor_right": f"{command.right:.6f}",
                "motor_reason": command.reason,
                "temp_c": fmt_opt(sysrow.get("temp_c")),
                "arm_clock_mhz": fmt_opt(sysrow.get("arm_clock_mhz"), 1),
                "throttled": throttled,
                "cpu_percent": fmt_opt(sysrow.get("cpu_percent"), 2),
                "mem_available_mb": fmt_opt(sysrow.get("mem_available_mb"), 1),
            })
            if frame_idx % print_every == 0:
                print(
                    f"[{frame_idx:05d}] {row['effective_mode']:<20} "
                    f"lanes={len(lanes)} steer={row['steer_norm']:+.3f} "
                    f"L={command.left:+.2f} R={command.right:+.2f} "
                    f"{pipeline_ms:.1f}ms fps={fps_ema:.2f} "
                    f"temp={fmt_opt(sysrow.get('temp_c')) or '?'} "
                    f"thr={throttled or '?'}"
                )
            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        motor.close()
        camera.close()
        log_f.close()
        cv2.destroyAllWindows()
        summary = {
            "frames": frame_idx,
            "duration_sec": time.perf_counter() - start,
            "config_hash": cfg_hash,
            "run_dir": str(run_dir),
            "throttle_seen": sorted(throttle_seen),
            "note": "Minimal runtime probe: no overlay window, no image/video saving, no Jupyter.",
        }
        write_json(run_dir / "summary.json", summary)
        print("[run] done")
        print("[run] frames:", frame_idx)
        print("[run] log:", run_dir / "runtime_log.csv")
        if throttle_seen:
            print("[warn] throttling/undervoltage observed:", sorted(throttle_seen))


if __name__ == "__main__":
    main()
