from __future__ import annotations

from pathlib import Path

from .decoder import decode_raw_to_lanes
from .postprocess import init_memory, selected_postprocess_name, update_lane_postprocess
from .runtime import make_ort_session, run_onnx_raw, warmup_ort_session


def resolve_path(root_dir, path_text):
    p = Path(path_text)
    return p if p.is_absolute() else Path(root_dir) / p


def resolve_selected_model(root_dir, cfg):
    selected = str(cfg["selected"])
    entry = cfg["models"][selected]
    model_path = resolve_path(root_dir, entry["onnx_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    external = entry.get("external_data_path")
    if external:
        external_path = resolve_path(root_dir, external)
        if not external_path.exists():
            raise FileNotFoundError(external_path)
    return selected, model_path


def init_lane_pipeline(root_dir, lane_model_cfg, decode_cfg, postprocess_cfg, first_frame=None):
    selected, model_path = resolve_selected_model(root_dir, lane_model_cfg)
    session, input_name, output_name = make_ort_session(model_path, threads=lane_model_cfg["threads"])
    if first_frame is not None:
        warmup_ort_session(session, input_name, output_name, first_frame, runs=lane_model_cfg["warmup_runs"])
    return {
        "selected_model": selected,
        "selected_postprocess": selected_postprocess_name(postprocess_cfg),
        "session": session,
        "input_name": input_name,
        "output_name": output_name,
        "decode_cfg": decode_cfg,
        "postprocess_cfg": postprocess_cfg,
        "memory": init_memory(postprocess_cfg),
    }


def run_lane_pipeline(pipe, bgr):
    raw = run_onnx_raw(pipe["session"], pipe["input_name"], pipe["output_name"], bgr)
    lanes = decode_raw_to_lanes(raw, pipe["decode_cfg"])
    signal = update_lane_postprocess(lanes, pipe["memory"], pipe["postprocess_cfg"])
    return {
        "lanes": lanes,
        "lane_signal": signal,
        "steer_norm": signal["steer_norm"],
        "lane_state": signal["lane_state"],
    }
