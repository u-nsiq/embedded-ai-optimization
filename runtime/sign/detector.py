from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class SignDetector:
    """YOLO ONNX detector runtime object.

    detector는 이벤트 판단을 하지 않는다.
    단지 frame에서 bbox, class, confidence, box_size를 계산해서 반환한다.
    """

    session: ort.InferenceSession
    input_name: str
    input_size: int
    class_names: list[str]
    conf_threshold: float
    iou_threshold: float
    max_det: int
    selected_model: str
    model_path: str


def resolve_path(root_dir, path_text):
    path = Path(path_text)
    return path if path.is_absolute() else Path(root_dir) / path


def init_sign_detector(root_dir, model_cfg):
    """8-class sign/traffic YOLO ONNX detector를 초기화한다."""
    if not bool(model_cfg.get("enabled", True)):
        return None

    selected = str(model_cfg["selected"])
    model_path = resolve_path(root_dir, model_cfg["models"][selected]["onnx_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Sign ONNX model is missing: {model_path}\n"
            "Colab 학습 결과 ONNX를 models/sign/에 넣고 config.py의 SIGN_MODEL 경로를 맞춰야 한다."
        )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    threads = int(model_cfg.get("threads", 1))
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    return SignDetector(
        session=session,
        input_name=session.get_inputs()[0].name,
        input_size=int(model_cfg["input_size"]),
        class_names=list(model_cfg["classes"]),
        conf_threshold=float(model_cfg["conf_threshold"]),
        iou_threshold=float(model_cfg["iou_threshold"]),
        max_det=int(model_cfg["max_det"]),
        selected_model=selected,
        model_path=str(model_path),
    )


def letterbox_rgb(rgb, input_size, color=(114, 114, 114)):
    """Ultralytics YOLO export와 같은 square letterbox 전처리."""
    h, w = rgb.shape[:2]
    size = int(input_size)
    scale = min(size / float(w), size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_w = size - new_w
    pad_h = size - new_h
    pad_left = int(round(pad_w / 2 - 0.1))
    pad_right = int(round(pad_w / 2 + 0.1))
    pad_top = int(round(pad_h / 2 - 0.1))
    pad_bottom = int(round(pad_h / 2 + 0.1))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return padded, {"orig_w": w, "orig_h": h, "scale": scale, "pad_left": pad_left, "pad_top": pad_top}


def preprocess_bgr_for_yolo(bgr, input_size):
    """BGR camera frame -> RGB letterbox tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded, meta = letterbox_rgb(rgb, input_size)
    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)
    return tensor, meta


def normalize_yolo_output(raw_output, class_count):
    """Ultralytics detect ONNX output을 [N, 4 + class_count] 형태로 맞춘다."""
    pred = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
    pred = np.asarray(pred)
    if pred.ndim == 3:
        pred = pred[0]

    # 일반 export: [4 + nc, anchors] 또는 [anchors, 4 + nc]
    expected = 4 + int(class_count)
    if pred.shape[0] == expected:
        pred = pred.T
    elif pred.shape[-1] == expected:
        pass
    else:
        raise ValueError(f"unexpected YOLO output shape: {pred.shape}, expected channel {expected}")
    return pred.astype(np.float32)


def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def rescale_boxes_to_original(xyxy, meta):
    boxes = xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] -= float(meta["pad_left"])
    boxes[:, [1, 3]] -= float(meta["pad_top"])
    boxes[:, :4] /= float(meta["scale"])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, float(meta["orig_w"] - 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, float(meta["orig_h"] - 1))
    return boxes


def nms_numpy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        order = order[1:][inter / union <= float(iou_threshold)]
    return keep


def postprocess_yolo(raw_output, meta, class_names, conf_threshold, iou_threshold, max_det):
    pred = normalize_yolo_output(raw_output, len(class_names))
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:4 + len(class_names)]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    mask = scores >= float(conf_threshold)
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return []

    boxes_xyxy = rescale_boxes_to_original(xywh_to_xyxy(boxes_xywh), meta)
    detections = []
    image_w = max(1.0, float(meta["orig_w"]))
    image_h = max(1.0, float(meta["orig_h"]))
    image_area = image_w * image_h

    for class_id in sorted(set(class_ids.tolist())):
        idx = np.where(class_ids == class_id)[0]
        for k in nms_numpy(boxes_xyxy[idx], scores[idx], iou_threshold):
            i = idx[k]
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            w_norm = float(w / image_w)
            h_norm = float(h / image_h)
            detections.append({
                "class_id": int(class_ids[i]),
                "class_name": class_names[int(class_ids[i])],
                "confidence": float(scores[i]),
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_w": float(w),
                "bbox_h": float(h),
                "area_ratio": float((w * h) / image_area),
                "width_ratio": w_norm,
                "height_ratio": h_norm,
                "box_size": float(max(w_norm, h_norm)),
                "aspect_hw": float(h / max(1.0, w)),
                "center_x_norm": float(((x1 + x2) * 0.5) / image_w),
                "center_y_norm": float(((y1 + y2) * 0.5) / image_h),
                "top_y_norm": float(y1 / image_h),
                "bottom_y_norm": float(y2 / image_h),
            })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections[:int(max_det)]


def detect_signs(detector, bgr):
    """frame 하나에서 sign/traffic detections를 반환한다."""
    if detector is None:
        return [], {"total_ms": 0.0, "preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}

    t0 = time.perf_counter()
    tensor, meta = preprocess_bgr_for_yolo(bgr, detector.input_size)
    t1 = time.perf_counter()
    raw = detector.session.run(None, {detector.input_name: tensor})
    t2 = time.perf_counter()
    detections = postprocess_yolo(
        raw,
        meta,
        detector.class_names,
        detector.conf_threshold,
        detector.iou_threshold,
        detector.max_det,
    )
    t3 = time.perf_counter()
    return detections, {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "inference_ms": (t2 - t1) * 1000.0,
        "postprocess_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
    }
