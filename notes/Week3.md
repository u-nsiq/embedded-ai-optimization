# Week 3 — Quantization & Knowledge Distillation

> 주제: 양자화 예시·실습 / 지식 증류 개요 및 종류

---

## 양자화 종류 정리

Quantization을 설계할 때는 세 가지 질문에 답해야 한다.

- **언제** 양자화할 것인가? (When to quantize?)
- **얼마나** 양자화할 것인가? (How much to quantize?)
- **어떻게** 양자화할 것인가? (How to quantize?)

### 언제: 사후 / 훈련 중 / 동적

| 비교 항목     | 사후 양자화 (PTQ)    | 훈련 중 양자화 (QAT)          | 동적 양자화 (Dynamic)          |
| --------- | --------------- | ----------------------- | ------------------------- |
| 적용 시점     | 훈련 후 적용         | 훈련 과정에서 양자화 반영          | 추론 시 실시간 양자화              |
| 정확도       | 성능 손실 발생 가능     | 손실 가장 적음, 원래 모델과 거의 유사  | 손실 적으나 일부 발생 가능           |
| 훈련 시간     | 추가 훈련 없음        | 양자화 고려해 훈련 → 시간 더 오래 걸림 | 기존 모델 그대로, 바로 적용          |
| 양자화 적용 범위 | 가중치만 또는 가중치+활성화 | 가중치+활성화 모두              | 가중치는 동적 양자화, 활성화는 FP32 유지 |
| 적용 모델     | CNN, RNN 등 대부분  | 대부분, 특히 성능이 중요한 모델      | 주로 RNN(LSTM, GRU)         |

### 얼마나: 동적 범위 / 완전 정수 / FP16

Week 2에서 다룬 PTQ 세 가지(Dynamic Range, Full Integer, Float16 Quantization)와 대응된다.

### 어떻게: 균등 vs 비균등

**균등 양자화(Uniform Quantization)** 는 값의 범위를 균일한 간격으로 나눠 매핑하는 방식이다.

고정된 스케일링 팩터(Scaling Factor)를 사용해 부동소수점 수를 정수로 변환한다.

CNN처럼 값이 고르게 분포하는 경우에 적합하다.

**비균등 양자화(Non-uniform Quantization)** 는 범위별로 다른 스케일을 할당하는 방식이다.

구간마다 간격이 달라서, 특정 값 범위에 더 높은 정밀도를 집중할 수 있다.

값이 균등하게 분포하지 않고 특정 구간에 몰려 있는 경우에 유리하다.

---

## 양자화 예시: 완전 정수 양자화

완전 정수 양자화(Full Integer Quantization)를 사후 양자화로 적용하는 경우를 구체적으로 살펴본다.

중간 레이어의 출력(활성화 값)과 입력까지 모두 8비트 정수로 변환한다.

8비트 정수는 2의 보수 표현으로 −128 ~ 127, 총 256개의 값을 표현할 수 있다.

### 스케일링 값(Scale) 계산

부동소수점 값의 범위가 $[-2.5, 2.5]$라고 가정하면 스케일링 값은 다음과 같이 계산된다.

$$\text{scale} = \frac{2.5 - (-2.5)}{127 - (-128)} = \frac{5}{255} \approx 0.0196$$

### 양자화 공식

$$\text{정수 값} = \text{round}\left(\frac{\text{부동소수점 값}}{\text{scale}}\right) + \text{zero\\_point}$$

**`zero_point`** 는 부동소수점에서 0.0에 해당하는 정수 값이다.

$[-2.5, 2.5]$ 범위는 중간값이 0이므로, 이 예시에서 `zero_point` = 0이다.

### 가중치 양자화 예제

가중치 $[0.75,\ {-1.25},\ 2.5,\ 0.125]$를 양자화한다. `zero_point`는 0이므로 생략한다.

$$0.75 \rightarrow \text{round}(0.75 / 0.0196) = \text{round}(38.27) = 38$$

$$-1.25 \rightarrow \text{round}(-1.25 / 0.0196) = \text{round}(-63.77) = -64$$

$$2.5 \rightarrow \text{round}(2.5 / 0.0196) = \text{round}(127.55) = 128 \rightarrow \text{클램핑} \Rightarrow 127$$

8비트 정수의 최댓값은 127이므로, 128은 127로 클램핑된다.

$$0.125 \rightarrow \text{round}(0.125 / 0.0196) = \text{round}(6.38) = 6$$

결론적으로 부동소수점 가중치 $[0.75,\ {-1.25},\ 2.5,\ 0.125]$는 정수 $[38,\ {-64},\ 127,\ 6]$으로 변환된다.

### 복원 공식

양자화된 정수를 다시 부동소수점으로 복원할 때는 스케일링 값을 곱하면 된다.

$$\text{부동소수점 값} = (\text{정수 값} - \text{zero\\_point}) \times \text{scale}$$

복원된 가중치는 원래의 $[0.75,\ {-1.25},\ 2.5,\ 0.125]$와 거의 일치한다.

이 과정을 통해 메모리와 연산 효율성을 크게 향상시키면서도 원본 값을 근사 복원할 수 있다.

---

## 지식 증류(Knowledge Distillation) 개요

### 지식 증류란?

**지식 증류(Knowledge Distillation)** 는 크고 성능 좋은 모델(Teacher Model)이 학습한 지식을 작은 모델(Student Model)로 전달하는 기법이다.

목표는 Student Model이 Teacher Model에 버금가는 정확도를 내도록 만드는 것이다.

Student Model은 더 적은 계산과 메모리 자원으로 효율적인 추론이 가능해서, 리소스가 제한된 환경(모바일, IoT, 임베디드 시스템)에서의 배포에 유용하다.

### 지식 증류의 종류

지식 증류는 **어떤 것을 증류하는가**, **어떻게 증류하는가** 두 축으로 분류할 수 있다.

| 분류 기준 | 방식 |
|---|---|
| 어떤 것을 | 로짓 기반, 피처 기반, 어텐션 기반, 회귀 기반, 관계 기반, 표현 간소화, 합성 모델 |
| 어떻게 | 오프라인, 온라인, 셀프 증류 |

---

## 로짓 기반 지식 증류 (Logit-based Distillation)

### Logit이란?

신경망의 최종 출력층에서 **Softmax 함수가 적용되기 전의 값**이다.

각 클래스에 대한 모델의 자신감(확신도)을 나타내며, 입력 → 로짓 → Softmax → 확률 분포 순서로 처리된다.

### Soft Targets vs Hard Labels

**Hard Labels** 는 정답 클래스가 1, 나머지가 0인 일반적인 레이블이다.

**Soft Targets** 는 Teacher Model의 로짓에 Softmax를 적용한 부드러운 확률 분포다.

Soft Targets에는 클래스 간 유사성 정보가 담겨 있다.

예를 들어 자동차 이미지를 분류할 때, Hard Label은 자동차: 1, 사람: 0이지만, Soft Targets은 자동차: 높은 확률, 사람: 약간의 확률처럼 클래스 간 관계를 반영한다.

### 온도(Temperature Scaling)

Softmax에 온도 매개변수 $T$를 추가하면 확률 분포가 더 부드러워진다.

$$p_i = \frac{\exp\!\left(\dfrac{z_i}{T}\right)}{\displaystyle\sum_j \exp\!\left(\dfrac{z_j}{T}\right)}$$

$T$ 값이 높을수록 확률 분포가 평탄해져서, 클래스 간 관계를 더 세밀하게 학습할 수 있다.

### 계산 예시

자동차인지 사람인지 구별하는 모델이 있다고 가정한다.

**Teacher 모델** (로짓: 자동차=5, 사람=1, $T=1$):

$$P(\text{자동차}) = \frac{\exp(5)}{\exp(5) + \exp(1)} \approx 0.982$$

**Student 모델** (로짓: 자동차=3, 사람=2, $T=1$):

$$P(\text{자동차}) = \frac{\exp(3)}{\exp(3) + \exp(2)} \approx 0.731$$

두 모델의 Soft Targets이 다르다. Student는 이 차이를 줄이도록 학습한다.

### 손실 함수

로짓 기반 증류의 최종 손실 함수는 두 가지 손실의 가중합으로 정의된다.

$$L = \alpha \cdot L_{\text{Distillation}} + (1 - \alpha) \cdot L_{\text{Student}}$$

- **$L_{\text{Distillation}}$**: Teacher와 Student의 Soft Targets 간 차이를 최소화하는 손실. KL 발산(KL Divergence)을 주로 사용한다.
- **$L_{\text{Student}}$**: Student의 최종 출력과 실제 정답 레이블(Hard Labels) 간 차이를 줄이는 손실. 크로스 엔트로피 손실(Cross-Entropy Loss)을 주로 사용한다.
- **$\alpha$**: 두 손실 간의 가중치를 조절하는 하이퍼파라미터다.

### 장점

- **클래스 간 관계 학습**: Soft Targets에는 클래스 간 유사성·차이 정보가 담겨 있어, 단순 정답 예측을 넘어 클래스 관계를 이해할 수 있다.
- **모델 압축**: 더 작은 Student Model이 Teacher의 성능을 따라가므로 메모리·연산 자원이 절약된다.
- **추론 효율성**: 모바일·임베디드처럼 제한된 환경에서도 빠르고 효율적인 추론이 가능하다.

---

## 피처 기반 지식 증류 (Feature-based Distillation)

### 개념

로짓 기반이 최종 출력(로짓)을 전달하는 반면, 피처 기반 증류는 **중간 레이어의 Feature Map**을 전달한다.

Feature Map은 각 레이어에서 학습된 중간 출력으로, 입력 데이터의 중요한 특성 정보를 담고 있다.

Teacher는 Feature Map을 추출해 전달하고, Student는 유사한 Feature Map을 출력하도록 학습한다.

이를 통해 최종 결과만 모방하는 것보다 더 깊은 정보를 전달받을 수 있다.

### 학습 과정

1. **피처 맵 추출**: Teacher의 특정 레이어에서 Feature Map을 추출한다.
2. **피처 맵 정렬**: Teacher와 Student의 Feature Map 크기가 다를 수 있으므로, 비교 가능하도록 정렬(Alignment)이 필요하다.
3. **손실 함수 정의**: Feature Map 간 차이를 최소화하는 손실을 정의한다.

$$L_{\text{feature}} = \| \text{Feature}_{\text{teacher}} - \text{Feature}_{\text{student}} \|_2^2$$

Student는 이 Feature Loss와 Student Loss를 함께 최소화하도록 학습한다.

### 장점

- **중간 표현 학습**: 중간 계층에서 학습된 중요 패턴을 Student가 학습할 수 있어, 복잡한 입력 데이터 처리에 유리하다.
- **모델 성능 향상**: 최종 결과만 학습하는 것보다 중간 표현을 함께 학습하는 쪽이 더 나은 성능을 낼 가능성이 크다.
- **효율적 모델 압축**: 작은 Student가 중간 레이어의 피처까지 학습하므로 크기 축소 시 성능 저하가 최소화된다.

---

## 어텐션 기반 지식 증류 (Attention-based Distillation)

### 개념

Teacher 모델의 **어텐션 메커니즘(Attention Mechanism)** 을 Student가 학습하도록 하는 방식이다.

어텐션 메커니즘은 딥러닝 모델이 입력 데이터에서 중요한 부분에 집중하는 방법이다.

Transformer 같은 어텐션 기반 모델에서 주로 활용된다.

### 어텐션 맵(Attention Map)

어텐션 메커니즘이 계산한 가중치 맵으로, 입력의 어떤 부분이 중요하다고 판단되었는지를 시각적으로 표현한다.

- NLP 모델에서는 문장의 각 단어 간 연관성을 나타낸다.
- 이미지 모델에서는 특정 이미지 영역에 집중하는 어텐션 맵이 생성된다.

### 학습 원리

1. Teacher는 여러 계층에서 어텐션 메커니즘을 사용해 입력의 중요 부분에 주목한다.
2. Student는 Teacher의 어텐션 맵을 모방하도록 학습한다.
3. 어텐션 맵 간 유사성을 측정하는 손실 함수를 정의한다.
4. Student Loss도 함께 학습해 실제 데이터 예측 정확도를 유지한다.

### 손실 함수

어텐션 맵 간 차이를 줄이는 손실로 주로 두 가지가 사용된다.

**L2 손실** (어텐션 맵 간 거리):

$$L_{\text{attention}} = \| A_{\text{teacher}} - A_{\text{student}} \|_2^2$$

**KL 발산** (어텐션 맵 간 확률 분포 유사성):

$$L_{\text{attention}} = \text{KL}(A_{\text{teacher}} \| A_{\text{student}})$$

### 장점

- **어텐션 정보 전달**: Teacher가 입력 데이터에서 중요하게 본 패턴을 Student가 더 잘 학습하게 된다.
- **일반화 성능 향상**: 새로운 데이터에 대한 예측 정확도가 개선된다.
- **학습 효율성**: 중요 정보를 빠르게 습득해 작은 Student도 복잡한 작업 처리에 유리해진다.

---

## 관계 기반 지식 증류 (Relational Knowledge Distillation)

### 개념

Teacher가 학습한 **데이터 포인트 간의 관계(Instance Relations)** 를 Student에게 전달하는 방식이다.

개별 예측 값을 모방하는 것이 아니라, 데이터 간 유사성과 구조적 관계를 학습한다.

같은 클래스는 가깝게, 다른 클래스는 멀게 학습한다.

### 학습 과정

1. **관계 추출**: Teacher는 인스턴스 간 유사성·거리 학습 후, 이를 바탕으로 데이터를 구조적으로 이해한다.
2. **관계 모방**: Student는 Teacher가 학습한 인스턴스 관계를 최대한 재현한다.
3. **손실 함수 정의**: 두 모델 간의 관계 차이를 최소화한다.

$$L_{\text{relation}} = \sum_{i,j} \| d_{\text{teacher}}(x_i, x_j) - d_{\text{student}}(x_i, x_j) \|_2^2$$

### 장점

- **데이터 간 상호작용 학습**: 단순 예측 값이 아니라 상호작용과 구조적 관계를 학습해 데이터를 더 깊이 이해한다.
- **일반화 성능 향상**: 교사가 학습한 관계를 Student가 모방해 새로운 데이터에 더 좋은 예측이 가능하다.
- **더 나은 분류 성능**: 비슷한 클래스의 인스턴스는 가깝게, 다른 클래스는 멀게 학습해 분류 성능이 향상된다.

---

## 오프라인 지식 증류 (Offline Distillation)

### 개념

지식 증류의 한 방식으로, **Teacher와 Student가 별도의 독립된 훈련 과정을 거치는** 방법이다.

Teacher는 사전에 미리 학습(Pre-trained)되어 있고, 그 결과를 바탕으로 Student가 학습한다.

두 모델은 동시에 학습되지 않으며, Teacher가 고정된 상태에서 Student가 학습한다.

### 학습 과정

**1단계 — 교사 모델 학습:**
대형 Teacher 모델을 충분히 학습한다.

각 입력에 대해 로짓을 생성하고 Softmax를 적용해 확률 분포를 출력한다.

학습이 완료되면 Teacher 모델의 지식(예측 결과)을 저장한다.

**2단계 — 학생 모델 학습:**
Student는 두 가지 신호를 동시에 학습한다.

- **Soft Targets**: Teacher의 부드러운 확률 분포
- **Hard Labels**: 실제 정답 레이블

Student는 증류 손실(Distillation Loss)과 학생 손실(Student Loss)을 함께 최소화한다.

$$L = \alpha \cdot L_{\text{Distillation}} + (1 - \alpha) \cdot L_{\text{Student}}$$

### 장점

- **독립적인 학습 단계**: Teacher 변경 없이 여러 Student 모델을 훈련할 수 있는 유연성이 있다.
- **시간 효율성**: Student 학습 속도가 빠르다. Teacher의 복잡한 학습 과정에 추가 시간이 소요되지 않는다.
- **성능 유지**: 더 작은 Student도 Teacher의 Soft Targets를 학습해 비슷한 성능을 유지할 수 있다.

### 단점

- **교사 모델 고정**: Teacher가 사전 학습되어 고정된 상태이므로, Student 학습 중에 Teacher를 개선할 수 없다.
- **동시 학습 불가**: Teacher 학습 → Student 학습의 두 단계가 분리되어 있어, 두 모델이 서로 영향을 주고받으며 함께 발전할 수 없다.
