# 17 Pi Quantized Lane Runtime Prune 결과 분석
작성일: 2026-05-13

## 목적
12번에서 만든 FP32 CLRKDNet 주행 모델은 Pi에서 정확도/주행 가능성은 확인했지만, 14번 진단에서 FP32 ONNX 추론이 너무 느리고 CPU/열 여유가 작다는 문제가 확인되었다. 17번은 주행 로직 전체가 아니라 **ONNX lane 모델 추론만** Pi에서 측정하여, 이후 실제 통합 주행 스크립트에 넣을 양자화 후보를 줄이는 단계다.

## 해석 전제
- 사용자가 각 `.sh`를 재부팅 후 일일이 실행했다. 따라서 결과 폴더명 시간이 가까워 보이더라도 동시 실행으로 간주하지 않는다.
- 각 `.sh`는 실행 직전 `system_snapshot.py`를 실행했고, 그 다음 500장 field3 샘플에 대해 **순수 ONNX inference 시간만** 측정했다.
- 17번 결과는 decoder/steering/camera/motor를 포함하지 않는다. 14번 FP32 probe와 비교할 때는 `inference_ms` 기준으로 보는 것이 맞다.
- 모델의 주행 행동 보존 여부는 16번 `behavior_gate_summary.csv`를 함께 참고한다. 17번은 속도/열/CPU 가지치기이고, 16번은 주행 행동 변화량 gate다.

## 14번 FP32 기준선
| model | threads | frames | mean inference ms | FPS | CPU mean % | temp max C | throttled |
|---|---:|---:|---:|---:|---:|---:|---|
| FP32 | 1 | 125 | 712.3 | 1.40 | 28.5 | 79.0 | `throttled=0x0` |
| FP32 | 2 | 192 | 458.5 | 2.18 | 52.5 | 86.2 | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0008` |
| FP32 | 3 | 234 | 375.9 | 2.66 | 77.0 | 86.2 | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0006 | throttled=0xe0008` |
| FP32 | 4 | 241 | 361.7 | 2.76 | 98.0 | 87.3 | `throttled=0x0 | throttled=0x80000 | throttled=0xe0000 | throttled=0xe0006 | throttled=0xe0008` |

요약하면 FP32는 thread 2에서 약 458ms, thread 3에서 약 376ms, thread 4에서도 약 357ms다. thread 4는 CPU를 거의 100% 사용했고, thermal/clock 관련 throttling 이력이 남았다. 즉 실제 통합 주행에서 traffic/sign/red-line까지 같이 넣기에는 여유가 너무 작다.

## 17번 양자화 후보 Pi 추론 결과
| model | t | mean ms | p95 ms | FPS | speedup vs FP32 same t | start temp | max temp | CPU mean % | start throttled | run throttled | 16 behavior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `qat15_layer4_only_static_qdq_u8s8` | 2 | 111.4 | 115.9 | 8.98 | 4.12x | 45.5 | 77.4 | 51.5 | `throttled=0x0` | `throttled=0x0` | False |
| `qat15_full_model_static_qdq_u8s8` | 2 | 111.8 | 116.6 | 8.94 | 4.10x | 53.8 | 84.0 | 49.8 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008` | False |
| `ptq11_static_qdq_u8s8` | 2 | 116.4 | 136.9 | 8.59 | 3.94x | 55.4 | 85.1 | 49.9 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0008` | False |
| `ptq11b_backbone_neck_qdq_u8s8` | 2 | 162.3 | 190.2 | 6.16 | 2.82x | 56.5 | 85.1 | 49.9 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0000 | throttled=0xe0008` | True |
| `ptq11b_backbone_all_qdq_u8s8` | 2 | 181.3 | 204.2 | 5.52 | 2.53x | 63.1 | 85.6 | 49.9 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0006 | throttled=0xe0008` | True |
| `ptq11b_backbone_all_qoperator_u8s8` | 2 | 197.4 | 222.6 | 5.07 | 2.32x | 55.4 | 84.5 | 50.0 | `throttled=0x0` | `throttled=0x0 | throttled=0x80008 | throttled=0xe0006 | throttled=0xe0008` | True |
| `ptq11b_backbone_layer4_qdq_u8s8` | 2 | 358.8 | 415.1 | 2.79 | 1.28x | 49.9 | 85.6 | 50.1 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0008` | True |
| `qat15_full_model_static_qdq_u8s8` | 3 | 86.5 | 93.5 | 11.56 | 4.34x | 52.7 | 85.1 | 74.5 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0000 | throttled=0xe0008` | False |
| `qat15_layer4_only_static_qdq_u8s8` | 3 | 86.9 | 96.5 | 11.51 | 4.33x | 52.7 | 85.1 | 74.8 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0006 | throttled=0xe0008` | False |
| `ptq11_static_qdq_u8s8` | 3 | 102.4 | 124.1 | 9.77 | 3.67x | 61.5 | 86.7 | 74.6 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0006 | throttled=0xe0008` | False |
| `ptq11b_backbone_neck_qdq_u8s8` | 3 | 135.2 | 161.6 | 7.40 | 2.78x | 53.2 | 85.6 | 74.6 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0000 | throttled=0xe0006 | throttled=0xe0008` | True |
| `ptq11b_backbone_all_qdq_u8s8` | 3 | 142.4 | 164.9 | 7.02 | 2.64x | 57.1 | 85.6 | 74.8 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0000 | throttled=0xe0008` | True |
| `ptq11b_backbone_all_qoperator_u8s8` | 3 | 179.5 | 202.5 | 5.57 | 2.09x | 57.6 | 86.7 | 75.1 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0x80008 | throttled=0xe0000 | throttled=0xe0006 | throttled=0xe0008` | True |
| `ptq11b_backbone_layer4_qdq_u8s8` | 3 | 297.2 | 337.9 | 3.36 | 1.26x | 57.6 | 86.7 | 74.9 | `throttled=0x0` | `throttled=0x0 | throttled=0x80000 | throttled=0xe0000 | throttled=0xe0006 | throttled=0xe0008` | True |

## 속도 기준 해석
- `qat15_layer4_only_static_qdq_u8s8`, `qat15_full_model_static_qdq_u8s8`, `ptq11_static_qdq_u8s8`는 Pi에서 가장 빠른 그룹이다. thread 3 기준 약 86~102ms로 FP32 thread 3 대비 약 3.7~4.3배 빠르다. 다만 16번 behavior gate에서는 주행 행동 보존이 좋지 않았으므로, **속도 최우선 risky 후보**로만 남긴다.
- `ptq11b_backbone_neck_qdq_u8s8`, `ptq11b_backbone_all_qdq_u8s8`는 16번 behavior gate를 통과했고, Pi에서도 thread 3 기준 약 135~142ms다. FP32 thread 3 대비 약 2.6~2.8배 빠르다. 따라서 **속도와 행동 보존 균형 후보**로 남긴다.
- `ptq11b_backbone_all_qoperator_u8s8`는 behavior gate는 통과했지만, QDQ 계열보다 느리고 뚜렷한 장점이 없다. **후순위 후보**로 둔다.
- `ptq11b_backbone_layer4_qdq_u8s8`는 행동 보존은 가장 안정적인 편이지만 속도 이득이 작다. thread 3 기준 FP32 대비 약 1.26배뿐이라, “Pi CPU 여유 확보”라는 17번 목적에는 맞지 않는다. **주요 후보에서 제외**하는 방향이 합리적이다.

## pruning 제안
최종 주행 스크립트에서 모델 파일만 갈아끼울 수 있게 만드는 것이 맞다. 다만 모든 모델을 계속 들고 가면 실험 관리가 흐려지므로 아래처럼 나누는 것이 좋다.

### 계속 들고 갈 모델
- `ptq11b_backbone_neck_qdq_u8s8`: 균형형 1순위.
- `ptq11b_backbone_all_qdq_u8s8`: 균형형 2순위.
- `ptq11_static_qdq_u8s8`: 빠른 risky 후보.
- `qat15_layer4_only_static_qdq_u8s8`: 빠른 risky 후보.
- `qat15_full_model_static_qdq_u8s8`: 빠른 risky 후보.

### 기본 후보에서 빼도 되는 모델
- `ptq11b_backbone_layer4_qdq_u8s8`: 속도 이득이 작아서 최적화 목적에는 약함.
- `ptq11b_backbone_all_qoperator_u8s8`: QDQ 균형 후보보다 느리고, 특별한 이점이 아직 없음. 완전 삭제보다는 archive/fallback으로 보관 정도가 적당함.

## thread 해석
- thread 2는 CPU 약 50%를 쓰고, thread 3은 CPU 약 75%를 쓴다.
- thread 3이 대체로 더 빠르지만 max temp와 throttling 이력이 더 자주 올라간다.
- traffic/sign/red-line을 같은 Pi에서 같이 돌릴 예정이면 thread 2가 시스템 여유 측면에서 더 안전하다. lane만 최대 FPS로 돌릴 때는 thread 3이 유리하다.

## thermal / throttling 해석
- `throttled=0x0`은 해당 시점에 undervoltage/thermal/frequency cap 이력이 없다는 뜻이다.
- `0x80000`, `0x80008`, `0xe0008`, `0xe0006` 등이 보이면 soft temp limit, frequency cap, throttling 이력이 남았다는 뜻이다.
- 17번에서도 많은 run이 80C 중반까지 올라갔다. 양자화로 latency는 크게 개선되지만, Pi 5가 열적으로 아주 넉넉해지는 것은 아니다. 그래도 FP32 thread 4처럼 CPU 100%를 계속 쓰는 상황에서는 확실히 벗어난다.

## 결론
17번 결과는 “양자화가 Pi에서 의미 있는 속도 이득을 주는가?”라는 질문에 대해 **그렇다**고 답한다. FP32는 2~3fps 수준이었고, 양자화 후보는 모델에 따라 5~11fps 수준까지 올라간다. 이제 주행 모델 쪽은 `model_name`만 바꿔 끼울 수 있는 구조로 만들고, lane 모델 선택은 이후 실제 통합 주행에서 비교하면 된다.

즉, 이 단계에서 주행 모델 튜닝을 더 파고들기보다, traffic light / red line / sign event 로직을 통합하는 다음 단계로 넘어가도 된다. 단, 통합 스크립트는 모델 경로와 ORT thread 수를 config로 바꿀 수 있어야 한다.
