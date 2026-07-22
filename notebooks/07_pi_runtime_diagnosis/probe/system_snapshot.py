#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_cmd(args, timeout=2.0):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return {"ok": True, "cmd": args, "output": out.strip(), "error": ""}
    except Exception as exc:
        return {"ok": False, "cmd": args, "output": "", "error": repr(exc)}


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except Exception as exc:
        return f"ERROR: {exc!r}"


def module_version(name):
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"missing: {exc!r}"


def collect_snapshot():
    commands = {
        "uname": ["uname", "-a"],
        "lscpu": ["lscpu"],
        "free_h": ["free", "-h"],
        "df_h_root": ["df", "-h", "/"],
        "vcgencmd_version": ["vcgencmd", "version"],
        "vcgencmd_measure_temp": ["vcgencmd", "measure_temp"],
        "vcgencmd_measure_clock_arm": ["vcgencmd", "measure_clock", "arm"],
        "vcgencmd_get_throttled": ["vcgencmd", "get_throttled"],
        "vcgencmd_get_config_int": ["vcgencmd", "get_config", "int"],
        "libcamera_list": ["libcamera-hello", "--list-cameras"],
    }
    command_results = {}
    for key, cmd in commands.items():
        timeout = 5.0 if key == "libcamera_list" else 2.0
        command_results[key] = run_cmd(cmd, timeout=timeout)

    snapshot = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_epoch": time.time(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "device_tree_model": read_text("/proc/device-tree/model"),
        "proc_cpuinfo": read_text("/proc/cpuinfo"),
        "proc_meminfo": read_text("/proc/meminfo"),
        "packages": {
            "numpy": module_version("numpy"),
            "cv2": module_version("cv2"),
            "onnxruntime": module_version("onnxruntime"),
            "scipy": module_version("scipy"),
            "picamera2": module_version("picamera2"),
            "gpiozero": module_version("gpiozero"),
        },
        "commands": command_results,
        "interpretation_notes": [
            "vcgencmd get_throttled == throttled=0x0 means no current or historical throttling flags.",
            "Non-zero get_throttled can indicate undervoltage, frequency capping, throttling, or historical occurrence.",
            "This snapshot is a baseline before runtime probes; runtime_probe.py logs temp/clock/throttled during driving.",
        ],
    }
    return snapshot


def write_outputs(snapshot):
    out_dir = ROOT / "system_snapshots" / datetime.now().strftime("%Y%m%d_%H%M%S_system_snapshot")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "system_snapshot.json"
    txt_path = out_dir / "system_snapshot.txt"
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# System Snapshot")
    lines.append("")
    lines.append(f"created_at: {snapshot['created_at']}")
    lines.append(f"python: {snapshot['python_executable']}")
    lines.append(f"machine: {snapshot['platform']['machine']}")
    lines.append(f"platform: {snapshot['platform']['platform']}")
    lines.append(f"device_tree_model: {snapshot['device_tree_model']}")
    lines.append("")
    lines.append("## Package Versions")
    for key, value in snapshot["packages"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Key Commands")
    for key in [
        "uname",
        "lscpu",
        "free_h",
        "df_h_root",
        "vcgencmd_measure_temp",
        "vcgencmd_measure_clock_arm",
        "vcgencmd_get_throttled",
        "libcamera_list",
    ]:
        result = snapshot["commands"].get(key, {})
        lines.append(f"### {key}")
        if result.get("ok"):
            lines.append(result.get("output", ""))
        else:
            lines.append(f"ERROR: {result.get('error', '')}")
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return out_dir, json_path, txt_path


def main():
    snapshot = collect_snapshot()
    out_dir, json_path, txt_path = write_outputs(snapshot)
    print("[system_snapshot] written:", out_dir)
    print("[system_snapshot] json:", json_path)
    print("[system_snapshot] txt:", txt_path)
    throttled = snapshot["commands"].get("vcgencmd_get_throttled", {}).get("output", "")
    print("[system_snapshot] throttled:", throttled or "unknown")
    temp = snapshot["commands"].get("vcgencmd_measure_temp", {}).get("output", "")
    print("[system_snapshot] temp:", temp or "unknown")


if __name__ == "__main__":
    main()
