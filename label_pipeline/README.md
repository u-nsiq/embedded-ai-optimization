# label_pipeline

현장에서 수집한 맵 이미지로 CLRKDNet fine-tuning용 CULane 형식 pseudo-label 데이터셋을 만드는 파이프라인.

## 처리 흐름

```text
이미지 manifest (CSV)
→ HSV 마스크
→ connected component 기반 lane 추출 (component_poly)
→ quality gate
→ CULane 형식 출력 (이미지 + .lines.txt + segmentation mask + list)
```

## 구성

| 파일 | 역할 |
| --- | --- |
| `src/create_candidate_manifests.py` | 수집 이미지에서 train / holdout 후보 manifest 생성 |
| `src/build_culane_pseudo_dataset.py` | manifest와 config로 데이터셋 생성 |
| `src/validate_culane_dataset.py` | 생성된 데이터셋의 형식·수량 검증 |
| `src/culane_builder_common.py` | HSV 마스크·lane 추출·quality gate 공통 로직 |
| `configs/*.json` | 대상 manifest, 출력 경로, 아래 결정값을 묶은 실행 설정 |

학습 후보는 1296x972 raw 이미지 9,425장(1·2차 현장 수집), fine-tuning에 넣지 않는 holdout 734장은 별도 manifest로 관리했다.

## 핵심 결정값

```json
{
  "hsv": {
    "lower": [22, 90, 110],
    "upper": [38, 255, 255],
    "morph_kernel": 5
  },
  "lane_extraction": {
    "method": "component_poly",
    "y_step": 6,
    "min_run_width": 4,
    "min_points": 8,
    "min_area": 450,
    "min_height": 120,
    "degree": 2
  },
  "quality_gate": {
    "min_lanes": 1,
    "min_point_total": 12,
    "min_y_span": 80,
    "min_mask_support": 0.2,
    "reject_crossing": false,
    "reject_small_gap": false
  }
}
```

이 gate는 좋은 label을 보증하는 기준이 아니라, 확실한 불량 pseudo-label을 자동으로 걸러내는 최소 기준이다. 결정값이 정해진 과정은 [notebooks/02_pseudo_label_exploration](../notebooks/02_pseudo_label_exploration/)에 있다.
