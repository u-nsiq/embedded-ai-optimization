# Week 1 — Why Optimize & Pruning

> 주제: 임베디드 시스템에서의 AI 최적화 필요성 + Pruning 개요

---

## Why Optimize? — 임베디드 환경의 제약 조건

딥러닝 모델은 수백만 개의 파라미터로 구성되어 있다. 서버나 데스크탑은 이를 충분히 감당하지만, 임베디드 디바이스는 그렇지 않다. 크게 세 가지 제약이 문제가 된다.

### Memory Constraint

임베디드 디바이스는 메모리가 제한적이라, 대형 모델을 그대로 올리기 어렵다. 모델의 크기를 줄이거나 메모리 사용을 최소화하는 방법이 필요하다.

### Power Constraint

배터리로 구동되거나 전력 공급이 제한된 환경(IoT, Edge Device)에서는 전력 소모 자체가 병목이 된다. 연산량을 줄이는 것이 곧 전력 절감으로 이어진다.

### Latency Requirement

자율주행 차량의 도로 상황 인식처럼, 실시간으로 데이터를 처리해야 하는 경우가 많다. 매우 빠른 응답 시간이 요구되므로, 모델의 추론(Inference) 속도가 곧 시스템의 가용성과 직결된다.

---

## AI Optimization 개요

### Hardware vs Software Optimization

AI 최적화는 크게 두 방향으로 나뉜다.
- **Hardware Optimization**: 칩 설계, 가속기(NPU, GPU) 등 하드웨어 단에서 접근
- **Software Optimization**: 알고리즘 혹은 시스템 레벨에서 접근 -> 이 강의에서 다루는 주제

Software Optimization은 다시 Algorithm Optimization과 System Optimization으로 나뉜다.

### Algorithm Optimization 3종 요약

| 기법                                | 핵심 아이디어                                              |
| --------------------------------- | ---------------------------------------------------- |
| **Pruning(가지치기)**                 | 중요도 낮은 weight/layer 제거 → 모델 경량화                      |
| **Quantization(양자화)**             | weight/activation을 더 낮은 비트로 표현 (e.g. Float32 → Int8) |
| **Knowledge Distillation(지식 증류)** | 큰 Teacher Model의 지식을 작은 Student Model에 전이            |

**Quantization**은 메모리 절약 + 정수 연산 특성상 연산 속도도 빠르다.

**Knowledge Distillation**은 Student Model이 혼자 학습할 때보다 더 좋은 일반화 성능을 얻는다는 점이 핵심이다.

### System Optimization: Scheduling


알고리즘 모듈들이 최소한의 자원만 사용하면서도 시스템이 안정적으로 동작하도록 스케줄을 짜는 것이다.

임베디드처럼 자원이 빠듯한 환경에서는 프로세서 사용 최적화, 메모리·전력 소비 최소화, 적절한 실행 타이밍 설정이 필수다.


---

## Pruning 기초

### Pruning이란?

딥러닝 모델의 weight 중 중요도가 낮은 연결(connection)을 제거하여 파라미터 수를 줄이는 방법이다. 중요도가 낮은 weight connection을 끊어도 모델 출력에 미치는 영향은 미미하다는 관찰에서 출발한다.


### Pruning의 장점

1. **Training Operations 감소** → 연산 속도 가속화
2. **Training Memory 절약** → 더 크거나 복잡한 모델 실험 가능
3. **Sparsity로 Inference 가속** → 0인 값은 곱셈 연산에서 제외 가능
4. **Inference 메모리 감소** → 메모리 효율 상승

---

## Pruning 설계의 4가지 축

Pruning을 적용할 때는 4가지 질문에 답해야 한다.

### What — Granularity (무엇을 제거할 것인가)

**Granularity** = 세분성. 즉, '제거 단위의 세밀함'을 의미.

- **Unstructured Pruning**: 개별 weight나 activation 단위로 독립적으로 제거. 비규칙적(Irregular).
- **Structured Pruning**: Channel, Kernel, Layer처럼 구조적 단위로 묶어서 제거. 규칙적(Regular)이라 하드웨어 친화적.
- **Local** vs **Global**: Local은 레이어별로 따로 Pruning 비율 설정, Global은 전체 네트워크에서 통합적으로 판단.

### How — Importance Criterion (어떻게 판단할 것인가)

weight나 구조 단위의 중요도를 어떤 기준으로 평가할 것인가.

- **Magnitude-based**: weight의 절댓값(L1) 또는 크기(L2)로 중요도 측정. 가장 단순하고 널리 쓰임.
- **Gradient-based**: 학습 gradient 정보를 활용해 중요도 판단.
- **Learned**: 중요도 자체를 학습으로 획득.
- **Information-based / Hessian-based / Bayesian**: 손실 함수 변화량, Hessian 행렬, 확률적 접근 등 보다 정교한 기준.

### When — Training 시점 (언제 제거할 것인가)

1. **Before Training**: 기존 Pruning 모델의 구조(connectivity)를 활용해 처음부터 트레이닝.
2. **During Training**: 트레이닝 도중 Pruning을 수행하고, 이후 Fine-tuning까지 진행.
3. **After Training**: 트레이닝 완료 후 Pruning → 바로 Inference 단계로 이행.

### How Much — One-shot vs Iterative (얼마나, 몇 번 제거할 것인가)

- **One-shot**: 한 번에 원하는 비율만큼 제거.
- **Iterative**: 조금씩 반복적으로 제거. 성능 회복을 위한 Fine-tuning을 중간중간 삽입할 수 있어 정밀도 손실이 적다.
- 개별적으로 볼지, 전체를 통합적으로 고려할지도 설계 선택지다.

4가지 축 중 **What(Granularity)** 과 **How(Importance Criterion)** 는 Pruning 방법론의 핵심이다. 이어지는 두 섹션에서 각각을 구체적인 기법 단위로 살펴본다.

---

## Granularity별 Pruning 기법

Pruning을 실제로 적용하려면 '무엇을' 제거할지 먼저 결정해야 한다. Granularity에 따라 제거 단위가 달라지고, 이 선택이 곧 하드웨어 친화성과 성능 트레이드오프를 결정한다. 불규칙한(Irregular) 방식에서 규칙적인(Regular) 방식 순으로 살펴본다.

### Fine-grained Pruning (Unstructured)

세밀한 수준에서 개별 weight나 neuron을 독립적으로 평가해 선택적으로 제거한다. 제거된 위치는 0으로 채워지고, 결과 행렬은 **Sparse Matrix** 형태가 된다. 0이 아닌 요소만 연산에 참여하므로 효율이 높아진다.

- **장점**: 큰 성능 저하 없이 모델 크기·연산량 감소. 메모리·연산 자원이 제한된 환경에 유리.
- **단점**: 최적 성능을 위해 복잡한 Fine-tuning 필요. 추가 최적화 단계가 요구될 수 있음.


### Pattern-based Pruning

규칙적이거나 반복되는 패턴을 식별하여 해당 패턴 단위로 Pruning한다. 패턴이 규칙적이므로 하드웨어(GPU, TPU)에서 최적화가 쉽고 연산 효율도 높다.

- **장점**: 하드웨어 친화적. 모델의 구조적 일관성을 유지하면서 성능 저하 방어 가능.
- **단점**: 초기 패턴 식별·설정이 복잡. 반복 패턴이 없으면 효과적이지 않음. 잘못된 패턴 선택 시 성능 크게 저하.


### Vector-level Pruning

하나의 neuron 또는 출력 노드에 연결된 모든 weight를 하나의 벡터로 보고, 벡터 단위로 중요도를 평가해 Pruning한다. CNN에서도 필터를 벡터 단위로 Pruning할 수 있다.

- **장점**: 연산 효율 증가, 모델 경량화, 하드웨어 최적화.
- **단점**: 성능 저하 위험. 중요도 평가 기준을 정확하게 설정해야 함.


### Kernel-level Pruning

CNN(Convolutional Neural Networks)에서 필터(커널) 단위로 Pruning하는 기법이다. Convolution 연산을 통해 Feature Map을 생성하는데, 학습 중 거의 사용되지 않거나 출력 변화에 기여하지 않는 커널을 가지치기 대상으로 간주한다. 제거된 커널이 생성하던 Feature Map은 더 이상 사용되지 않는다.

- **장점**: 파라미터 수 감소, Convolution 연산량 감소, 하드웨어 친화적.
- **단점**: 잘못된 커널 제거 시 성능 저하. 최적의 커널 선택이 어려움.


### Channel-level Pruning

합성곱(Convolutional) 레이어의 출력 채널(Feature Map) 단위로 Pruning하는 방법이다. 학습 중 기여도가 낮은 채널을 제거한다.

- **장점**: 네트워크 전체 파라미터 수 감소. CNN 연산량 감소, Inference 속도 향상. GPU 같은 하드웨어 가속기에서 최적화된 연산 가능.
- **단점**: 중요한 채널이 잘못 제거되면 성능 저하. 여러 레이어 간 상호작용을 고려해야 해서 복잡.


---

## Importance Criterion

무엇을 제거할지 단위를 정했다면, 이제 '어떻게' 중요도를 판단할 것인지가 문제다. 어떤 기준(Criterion)을 쓰느냐가 곧 어떤 weight를 살리고 어떤 것을 버릴지를 결정한다.

### Magnitude-based Pruning

weight의 크기(magnitude)로 중요도를 측정하는 가장 기본적인 방법이다. 절댓값이 작은 weight는 출력에 영향이 적다고 판단해 제거한다.

**L1-norm (Element-wise)**

각 요소의 절댓값을 그대로 중요도로 사용한다.

$$\text{importance} = |w|$$

**L1-norm (Row-wise)**

행 전체의 절댓값 합을 중요도로 사용한다.

$$\text{importance} = \sum_j |w_j|$$

**L2-norm (Row-wise)**

행 전체의 제곱합의 루트를 중요도로 사용한다.

$$\text{importance} = \sqrt{\sum_j w_j^2}$$

중요도가 기준값(Threshold)보다 낮은 weight는 0으로 설정하거나 제거한다.


### Scaling-based Pruning

각 weight나 neuron에 곱해지는 **Scaling Factor**로 해당 요소의 중요도를 나타내는 방법이다. 트레이닝 과정에서 Scaling Factor도 함께 학습되며, 이후 Factor 값이 작은 채널이나 neuron을 Pruning 대상으로 선택한다.

