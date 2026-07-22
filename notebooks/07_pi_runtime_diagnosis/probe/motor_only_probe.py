from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from motor_adapter import DifferentialMotor, MotorCommand


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run_text(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def parse_temp(text):
    try:
        return float(text.split("=")[1].split("'")[0])
    except Exception:
        return float("nan")


def parse_clock_mhz(text):
    try:
        return float(text.split("=")[1]) / 1_000_000.0
    except Exception:
        return float("nan")


def read_mem_available_mb():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def read_cpu_times():
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            parts = f.readline().split()[1:]
        vals = [int(x) for x in parts]
        idle = vals[3] + vals[4]
        total = sum(vals)
        return idle, total
    except Exception:
        return None


def cpu_percent(prev, cur):
    if prev is None or cur is None:
        return float("nan")
    idle_delta = cur[0] - prev[0]
    total_delta = cur[1] - prev[1]
    if total_delta <= 0:
        return float("nan")
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def telemetry(prev_cpu):
    cur_cpu = read_cpu_times()
    return {
        "temp_c": parse_temp(run_text(["vcgencmd", "measure_temp"])),
        "arm_clock_mhz": parse_clock_mhz(run_text(["vcgencmd", "measure_clock", "arm"])),
        "throttled": run_text(["vcgencmd", "get_throttled"]),
        "cpu_percent": cpu_percent(prev_cpu, cur_cpu),
        "mem_available_mb": read_mem_available_mb(),
    }, cur_cpu


def main():
    cfg_path = Path("config_motor_only_probe.json")
    cfg = read_json(cfg_path)
    run_cfg = cfg.get("run", {})
    motor_cfg = cfg.get("motor", {})
    command_cfg = cfg.get("command", {})
    tele_cfg = cfg.get("telemetry", {})

    name = str(run_cfg.get("name", "motor_only_probe"))
    duration_sec = float(run_cfg.get("duration_sec", 90.0))
    interval_sec = max(0.02, float(run_cfg.get("interval_sec", 0.1)))
    print_every = int(run_cfg.get("print_every_n", 10))
    output_root = Path(str(run_cfg.get("output_dir", "probe_runs")))
    every_n = max(1, int(tele_cfg.get("every_n_steps", 5)))

    left = float(command_cfg.get("left", motor_cfg.get("base_speed", 0.3)))
    right = float(command_cfg.get("right", motor_cfg.get("base_speed", 0.3)))
    reason = str(command_cfg.get("reason", "motor_only_constant"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config_used.json", cfg)

    fields = [
        "step_idx", "t_sec", "apply_ms", "sleep_ms", "loop_ms",
        "motor_left", "motor_right", "motor_reason",
        "temp_c", "arm_clock_mhz", "throttled", "cpu_percent", "mem_available_mb",
    ]

    motor = DifferentialMotor(motor_cfg)
    command = MotorCommand(left=left, right=right, reason=reason)
    prev_cpu = read_cpu_times()
    throttle_seen = set()
    steps = 0
    t0 = time.perf_counter()

    print(f"[motor-only] run_dir={run_dir}")
    print(f"[motor-only] duration={duration_sec}s command L={left:+.3f} R={right:+.3f}")
    print("[motor-only] Ctrl+C stops motors and closes the log.")

    try:
        with open(run_dir / "runtime_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            while True:
                loop_start = time.perf_counter()
                t_sec = loop_start - t0
                if duration_sec > 0 and t_sec >= duration_sec:
                    break

                t_apply0 = time.perf_counter()
                motor.apply(command)
                t_apply1 = time.perf_counter()

                tele = {"temp_c": "", "arm_clock_mhz": "", "throttled": "", "cpu_percent": "", "mem_available_mb": ""}
                if steps % every_n == 0:
                    tele, prev_cpu = telemetry(prev_cpu)
                    if tele.get("throttled"):
                        throttle_seen.add(str(tele["throttled"]))

                elapsed = time.perf_counter() - loop_start
                sleep_sec = max(0.0, interval_sec - elapsed)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                loop_end = time.perf_counter()

                row = {
                    "step_idx": steps,
                    "t_sec": f"{t_sec:.4f}",
                    "apply_ms": f"{(t_apply1 - t_apply0) * 1000.0:.3f}",
                    "sleep_ms": f"{sleep_sec * 1000.0:.3f}",
                    "loop_ms": f"{(loop_end - loop_start) * 1000.0:.3f}",
                    "motor_left": f"{left:.6f}",
                    "motor_right": f"{right:.6f}",
                    "motor_reason": reason,
                    **tele,
                }
                writer.writerow(row)
                if print_every > 0 and steps % print_every == 0:
                    print(f"[{steps:05d}] L={left:+.2f} R={right:+.2f} temp={tele.get('temp_c','')} throttled={tele.get('throttled','')}")
                steps += 1
    except KeyboardInterrupt:
        print("\n[motor-only] interrupted by user.")
    finally:
        motor.stop()
        motor.close()

    summary = {
        "steps": steps,
        "duration_sec": time.perf_counter() - t0,
        "run_dir": str(run_dir.resolve()),
        "throttle_seen": sorted(throttle_seen),
        "note": "Motor-only probe: no camera, no ONNX inference, no image/video saving.",
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
