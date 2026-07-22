# notebooks

프로젝트에서 수행한 실험 노트북. 폴더 번호는 실제 진행 순서다.

| 폴더 | 내용 |
| --- | --- |
| `01_lane_model_feasibility` | CULane 사전학습 CLRKDNet의 ONNX 변환과 출력 확인, Pi latency 벤치, 프로젝트 맵 이미지에서의 domain gap 확인 |
| `02_pseudo_label_exploration` | CULane 레이블 형식 분석, HSV 레이블 스윕, lane 추출 방식 비교, reject gate 스윕, builder 검증 |
| `03_lane_finetuning_v1` | 1차 fine-tuning 학습과 검증, 주행 후처리 관점에서의 한계 확인 |
| `04_lane_precision_tuning` | 추가 수집 데이터 기반 정밀 튜닝 시도와 데이터셋 비교 |
| `05_local_fit_rebuild` | GT 레이블 정책 재구성(Local Fit), 최종 데이터셋 구축, 재학습, ONNX export parity |
| `06_lane_decode_and_steering` | lane decode 계약과 조향 후처리 후보 실험 |
| `07_pi_runtime_diagnosis` | Pi 실주행 latency 계측, motor·inference 부하 분리 진단 |
| `08_quantization` | PTQ 후보와 선택 경계, QAT-lite 학습·변환, 주행 행동 게이트, Pi 확인 |
| `09_sign_detection` | 표지판 데이터 정리, Roboflow 라벨링 준비, YOLO11n 학습·선정, 8-class 재학습 |
| `10_traffic_light_and_redline` | 신호등·종료선 HSV/contour 검출기 설계와 event trigger |
| `11_event_contracts` | detection을 주행 event로 바꾸는 정책과 상태 기계 계약 |
