# ============================================================
# 2. Lane model
# ============================================================
# selected만 바꾸면 lane ONNX 후보를 갈아낄 수 있다.
#
# 추천 테스트 순서:
#   1) ptq_backbone_neck: Pi 속도 이득이 있고, 품질도 상대적으로 무난했던 후보
#   2) fp32: 정확도 기준 모델. 느리지만 비교용으로 중요
#   3) ptq_fast: 빠른 후보. 주행 품질이 나쁘면 제외
#
# threads:
#   lane ONNX Runtime CPU thread 수. Pi 5에서는 2를 기본값으로 둔다.
#   3은 조금 빠를 수 있지만 전력/열 여유가 줄 수 있다.
LANE_MODEL = {
    "selected": "qat_layer4",
    "threads": 3,
    "warmup_runs": 3,
    "models": {
        "fp32": {
            "onnx_path": "models/lane/lane_field12_v1_fp32.onnx",
            # 이 ONNX 내부 external-data 이름이 MapLane...onnx.data로 박혀 있다.
            # models/lane 안에 같은 이름의 링크/파일이 반드시 있어야 한다.
            "external_data_path": "models/lane/MapLane_LocalFit_Field12_v1_best_opset17_static_b1.onnx.data",
        },
        "ptq_fast": {
            "onnx_path": "models/lane/lane_quant_ptq11_static_qdq_u8s8.onnx",
        },
        "ptq_backbone": {
            "onnx_path": "models/lane/lane_quant_ptq11b_backbone_all_qdq_u8s8.onnx",
        },
        "ptq_backbone_neck": {
            "onnx_path": "models/lane/lane_quant_ptq11b_backbone_neck_qdq_u8s8.onnx",
        },
        "qat_layer4": {
            "onnx_path": "models/lane/lane_quant_qat15_layer4_only_static_qdq_u8s8.onnx",
        },
        "qat_full": {
            "onnx_path": "models/lane/lane_quant_qat15_full_model_static_qdq_u8s8.onnx",
        },
    },
}


# ============================================================
# 3. Lane decoder
# ============================================================
# 12/07에서 공식 CLRNet overlap NMS를 numpy로 재현한 값이다.
# 보통은 건드리지 않는다.
#
# conf_threshold:
#   lane 후보 신뢰도 하한. 기준 0.35.
#   lane이 너무 안 나오면 0.30으로 낮추고, 잡음 lane이 많으면 0.40으로 올린다.
#
# nms_topk:
#   후처리에 넘길 최대 lane 수. 후처리 후보가 여러 lane을 쓸 수 있으므로 4까지 열어둔다.
LANE_DECODE = {
    "conf_threshold": 0.35,
    "nms_thres": 70.0,
    "nms_topk": 4,
}
