# runtime

최종 시연(2026-05-28)에서 실제로 주행한 Raspberry Pi 5 런타임 코드. 설정값까지 시연 당일 상태 그대로다.

## 모듈 구성

| 경로 | 역할 |
| --- | --- |
| `main.py` | 멀티레이트 주행 루프. lane 추론 / sign 탐지 / redline 검출 / 모터 제어를 서로 다른 주기로 돌린다 |
| `state_machine.py` | 이벤트 수용과 상태 전이 (PENDING → TURN / STOPPED / START_IGNORE_REDLINE 등) |
| `config.py`, `configs/` | 주행 파라미터. 시연 당일 값 그대로 (`lane.py`, `lane_postprocess.py`, `sign.py`, `redline.py`, `hardware.py`, `runtime.py`, `state.py`) |
| `lane/` | ONNX 추론 → CLRNet 방식 디코딩 → 후처리 → `steer_norm` 변환 파이프라인 |
| `lane/postprocesses/` | 조향 후처리 후보 4종 (`fixed_base`, `slope_push`, `inside_soft`, `stable_center_tangent`). 시연은 `fixed_base` 사용 |
| `sign/` | YOLO11n 8-class 표지판·신호등 탐지(`detector.py`)와 detection→event 변환(`trigger.py`) |
| `redline/` | OpenCV HSV 기반 종료선(빨간 정지선) 검출 |
| `utils/` | camera / motor / horn / timing |
| `debug/` | 오버레이·로그 확인용 엔트리포인트 (`main_lane_only.py`, `main_full_overlay_drive.py` 등). 모터 없이 인식 파이프라인만 확인할 때 사용 |

## 시연 당일 구성

- Lane 모델: CLRKDNet(ResNet-18) QAT-lite INT8, `qat_layer4` (layer4만 fake-quant 후 static 양자화)
- ONNX Runtime CPU threads: 3
- 조향 후처리: `fixed_base` (양쪽 lane이면 center+slope, 한쪽 lane이면 slope만)
- 표지판·신호등: YOLO11n 8-class
