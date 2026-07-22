# 2026-05-12 FP32 Pi Runtime Diagnosis Summary

## 목적

12번 실험 폴더에서 구축한 CLRKDNet FP32 ONNX 주행 모델이 Raspberry Pi 5에서 실제 시험 주행용으로 버틸 수 있는지 확인했다. 이번 14번은 주행 후처리 파라미터 튜닝이 아니라, Pi 런타임 안정성의 원인을 분리하는 실험이다.

확인하려던 질문은 다음과 같다.

- FP32 추론 자체가 Pi CPU에 과한가?
- low voltage warning은 모터 자체 문제인가, 아니면 추론 부하와 결합될 때 생기는가?
- `threads=4`가 정말 최선인가?
- sign / traffic / red line 추가 전에 주행 모델만으로도 시스템 여유가 있는가?

## 실험 구성

Pi에 `pi_runtime_probe`를 배포하고 다음 로그를 수집했다.

- `system_snapshot`: Pi 보드/OS/Python/패키지/온도/클럭/throttled 상태 기록
- `motor_only_probe`: 카메라와 ONNX 없이 모터만 L/R=0.4로 90초 구동
- `motor_off_probe`: 모터 없이 카메라 + FP32 ONNX + decoder + steering만 180초 실행
- `minimal_drive`: 저장/overlay/window 없이 실제 모터 주행
- `threads sweep`: motor off 상태에서 ONNX Runtime threads=1/2/3/4 비교

로그는 로컬 `results_from_pi/pi_runtime_probe_logs_20260512_122540`로 회수했고, Pi의 `probe_runs`, `system_snapshots`는 삭제했다.

## 하드웨어/환경

- Board: Raspberry Pi 5 Model B Rev 1.0
- CPU: Cortex-A76 4 cores, max 2.4GHz
- RAM: 4GB
- OS: Linux 6.12.47 aarch64
- Python: `/home/pi/AI_CAR/env/bin/python3`
- ONNX Runtime: 1.25.1
- OpenCV: 4.13.0
- SciPy: 1.17.1

## 핵심 결과

### 1. 모터 단독은 정상

`motor_only_probe` 결과:

| 항목 | 결과 |
|---|---:|
| duration | 90 sec |
| motor command | L=0.4, R=0.4 |
| throttled | `0x0` only |
| temp | 50.5 -> 53.2 C |
| CPU mean | 1.3% |

모터만으로는 low voltage나 thermal throttling이 발생하지 않았다. 따라서 모터 자체만으로 Pi 전원이 무너지는 상황은 아니다.

### 2. FP32 추론만으로 thermal throttling 재현

`motor_off_probe` 결과:

| 항목 | 결과 |
|---|---:|
| duration | 180 sec |
| pipeline mean | 371 ms |
| inference mean | 360 ms |
| CPU mean | 약 99% |
| temp | 64.2 -> 85.1 C |
| throttled | soft temp / frequency cap / throttle 발생 |

저장, overlay, window, Jupyter 없이도 FP32 ONNX inference만으로 Pi가 거의 풀로드에 들어가고 열 제한을 받았다.

### 3. 실제 minimal drive에서는 under-voltage 이력 발생

`minimal_drive` 결과:

| 항목 | 결과 |
|---|---:|
| duration | 161 sec |
| frames | 420 |
| pipeline mean | 382 ms |
| inference mean | 371 ms |
| FPS mean | 2.65 |
| temp max | 87.8 C |
| throttled | `0xf0000`, `0xf0006`, `0xf0008` 포함 |

`0xf0000` 계열에는 `under_voltage_occurred`가 포함된다. 즉 VNC low voltage warning은 로그와 일치한다.

해석:

- motor only: 저전압 없음
- FP32 inference only: 열 throttling 발생
- FP32 inference + motor: 열 throttling + under-voltage 이력 발생

따라서 low voltage는 모터 단독 문제가 아니라, CPU 풀로드 상태에서 모터 구동까지 겹치며 전원 여유가 줄어드는 상황으로 보는 게 타당하다.

## Threads Sweep

motor off 상태에서 ONNX Runtime threads만 바꿔 90초씩 비교했다.

| threads | FPS mean | pipeline mean | inference mean | CPU mean | temp range | throttling |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.39 | 721 ms | 712 ms | 28.5% | 68 -> 79 C | 없음 |
| 2 | 2.15 | 469 ms | 458 ms | 52.5% | 68 -> 86 C | soft temp |
| 3 | 2.62 | 386 ms | 376 ms | 77.0% | 72 -> 86 C | thermal throttle |
| 4 | 2.71 | 374 ms | 362 ms | 98.0% | 72 -> 87 C | thermal throttle |

판단:

- `threads=1`: 안정적이지만 너무 느림
- `threads=2`: CPU 여유는 있으나 주행 반응성이 떨어질 가능성이 큼
- `threads=4`: 가장 빠르지만 CPU 98%로 시스템 여유가 거의 없음
- `threads=3`: 속도 손해가 작고 CPU 여유가 생겨 FP32 임시 주행 후보로 가장 균형적

단, `threads=3`도 결국 thermal throttle을 막지는 못한다.

## 결론

현재 FP32 ONNX 모델은 Pi 5에서 실행은 가능하지만, 장시간 안정 주행용으로는 여유가 부족하다.

정리하면:

```text
motor only       -> 정상, 저전압 없음
FP32 inference   -> CPU 풀로드, 열 throttling 발생
FP32 + motor     -> thermal throttling + under-voltage 이력 발생
threads=3        -> FP32 임시 후보 중 가장 균형적
threads=4        -> fastest지만 CPU/열/전원 안정성 최악
```

즉 다음 단계에서 sign, traffic light, red line event를 추가하기 전에, lane model runtime을 먼저 안정화해야 한다.

## 다음 작업 후보

우선순위는 다음 순서가 적절하다.

1. 14번 결과를 기준으로 FP32 baseline은 `threads=3` 후보로 보존
2. 팬/방열/전원 공급 개선 후 같은 probe 재실행
3. 모델 경량화 재검토
   - 일반 PTQ는 12번 11/11b에서 의미 보존 실패
   - QAT 또는 더 작은 backbone/입력 해상도 축소 검토 필요
4. 실제 시험 코드에서는 저장/overlay/window 없이 minimal runtime 유지
5. sign / traffic / red line은 lane runtime 안정화 이후 통합

## 현재 의사결정

주행 후처리 파라미터 튜닝은 잠시 중단한다. 13번 실험에서 FP32 모델이 맵 가장자리 주행을 어느 정도 수행함은 확인했다. 지금 병목은 조향 수식보다 Pi 런타임 안정성이다.

다음 실험은 `threads=3` 또는 경량화 모델을 기준으로 동일한 probe 체계로 비교하는 방식으로 진행한다.
