from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import CUT_HEIGHT, IMG_H, IMG_W, NUM_PRIORS, OUTPUT_DIM, RAW_H, RAW_W


def fs_path(path):
    return str(Path(path))


def make_ort_session(model_path, threads=2):
    """ONNX Runtime 세션을 만든다.

    threads는 Pi에서 성능/전력 균형을 정하는 값이다.
    lane-only 기본은 2 thread다.
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        sess_options.intra_op_num_threads = int(threads)
    session = ort.InferenceSession(
        fs_path(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name


def preprocess_bgr_for_model(bgr):
    """학습/ONNX/Pi 검증과 같은 입력 전처리.

    카메라 frame은 BGR 1296x972로 맞춘다.
    이후 cut_height=445 아래 영역만 crop하고 800x320으로 resize한 뒤 /255 한다.
    """
    if bgr.shape[0] != RAW_H or bgr.shape[1] != RAW_W:
        bgr = cv2.resize(bgr, (RAW_W, RAW_H), interpolation=cv2.INTER_LINEAR)
    crop = bgr[int(CUT_HEIGHT):, :, :]
    resized = cv2.resize(crop, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
    if tensor.shape != (1, 3, IMG_H, IMG_W):
        raise AssertionError(tensor.shape)
    return tensor


def run_onnx_raw(session, input_name, output_name, bgr):
    inp = preprocess_bgr_for_model(bgr)
    out = session.run([output_name], {input_name: inp})[0]
    if out.shape != (1, NUM_PRIORS, OUTPUT_DIM):
        raise AssertionError(out.shape)
    return out[0].astype(np.float32)


def warmup_ort_session(session, input_name, output_name, bgr, runs=3):
    inp = preprocess_bgr_for_model(bgr)
    for _ in range(int(runs)):
        _ = session.run([output_name], {input_name: inp})[0]
