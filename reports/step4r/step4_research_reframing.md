# Step 4 이후 연구 방향 재정의 (Research Reframing Memo)

- 문서 작성일: 2026-05-08
- 작성 대상: AICRC_v2 main pipeline의 Step 4 이후 단계 전반
- 선행 고정 문서:
  - `reports/step1/manifest/manifest_summary.md`
  - `reports/step2/step25_final_synthesis.md`
  - `reports/step3/step3_output_schema_uncertainty_policy.md`
- 본 문서로 인해 main pipeline에서 제외되는 기존 산출물:
  - `reports/step4/` — LR baseline 결과
  - `reports/step5/` ~ `reports/step7/` — rule/template caption 정책·템플릿·생성 결과
  - `reports/step8/` — template 출력에 대한 human review 계획·샘플·패키지

  위 산출물은 git history와 reference artifact로 **보존**하되, 논문 main claim을 뒷받침하는 근거로 **인용하지 않는다**.

본 문서는 새 코드를 추가하지 않으며, 데이터·모델·캡션·평가 어느 것도 새로 만들지 않는다. 본 문서는 Step 4 이후 단계의 *연구 프레이밍, 이론 기반, 산출물 정의, 평가 방식*을 재고정하는 정책 문서이다.

본 연구의 main pipeline은 sensor-to-text generation이 **아니다**. main pipeline은 **uncertainty-aware sensor-to-schema-to-caption pipeline**이다. text는 schema의 표현층(presentation layer)이며 학습 target이 아니다. 또한 *개인 의미 모형*은 본 석사 논문 범위에서 명시적으로 **제외**한다.

---

## 0. 본 문서의 위치

본 문서는 다음 두 가지를 명시한다:

1. **무엇이 main pipeline에서 빠지는가** — 기존 Step 4 LR baseline, Step 5~7 rule/template caption, Step 8 human review.
2. **무엇이 새 main pipeline의 정의인가** — uncertainty-aware sensor-to-schema-to-caption pipeline. 이론 기반 + 학습된 시스템 + 자동 검증 layer로 구성된다.

본 문서는 Step 4R 기존 계획서(`reports/step4r/step4r_pilot_modeling_plan.md`)를 확장한다. 기존 4R 계획의 sklearn 모델 비교는 본 문서의 §5.1 (4R-A)로 흡수되고, 그 위에 §5.2 (4R-B), §5.3 (4R-C)이 추가된다.

---

## 1. 교수님 피드백 요약

논문은 *모델 구현 결과*가 아니라 *이론 기반 연구*에 근거해야 한다는 피드백이 있었다.

핵심 요지:

- 단일 분류기를 학습해 accuracy를 올리는 것은 석사 학위 논문의 main contribution이 되기 어렵다.
- 본 연구가 다루는 wrist IMU 신호는 squat의 핵심 동작(무릎 각도, 골반 위치 등)을 *직접* 측정하지 않는다. 분류 정확도의 상한이 물리적으로 존재하며, 이 상한을 *모델로 돌파하려는 시도*는 학술적 기여로 약하다.
- 대신, 손목 IMU로 *관측 가능한 것과 관측 불가능한 것의 경계*를 정의하고, 그 경계 위에서 시스템이 자신의 불확실성을 정직하게 표현하도록 설계하는 것이 contribution이 된다.
- 따라서 모델 비교 실험은 그 자체로 main contribution이 될 수 없으며, 정보이론적·관측이론적 프레이밍을 통해 *왜 그러한 schema 설계가 필요한가*를 설명할 수 있어야 한다.

본 문서는 이 피드백을 반영해 연구 프레이밍을 재정의한다.

---

## 2. 기존 pipeline의 한계

| 단계 | 산출물 | 한계 |
|---|---|---|
| Step 4 | LR baseline (test macro F1 = 0.288) | 학습된 모델이 LR 1개뿐. 모델 비교 부재. 단일 argmax 출력의 한계가 분류 결과로만 보고됨. |
| Step 5~6 | caption policy / template 설계 | 출력은 closed-vocabulary template filling. sensor signal이 caption 표현으로 *학습된 적 없음*. |
| Step 7 | template 캡션 생성기 | rule 기반. 새로운 입력에 대해 어떠한 *학습된 일반화*도 수행하지 않는다. |
| Step 8 | human review 계획·샘플·패키지 | 평가 대상이 학습된 sensor-to-text 시스템이 아니라 *룰의 출력*이라 review 결과가 시스템의 generalization 근거가 되지 않는다. |

종합하면, 기존 pipeline은 다음 세 가지를 동시에 충족하지 못한다:

1. **학습된 sensor-to-X 모델이 main contribution을 뒷받침해야 한다.** 현재는 LR 1개 + 룰 템플릿이라 "학습된 시스템"이 약하다.
2. **불확실성 표현이 임시방편이 아니라 이론적으로 동기화되어야 한다.** 현재 schema는 정책 문서로 정의되어 있으나, 정보이론적·관측이론적 근거 서술이 빈약하다.
3. **평가가 학습된 시스템의 generalization을 측정해야 한다.** 현재 Step 8은 룰 출력의 가독성을 측정하는 데 가깝다.

---

## 3. 새 연구 질문

본 연구의 main research question을 다음과 같이 재정의한다.

> **wrist IMU만으로 관측 가능한 squat의 불확실성 구조를 모델링하고, 그 불확실성을 schema로 표현한 뒤, 한국어 caption으로 정직하게 전달하는 시스템을 어떻게 설계할 수 있는가?**

하위 연구 질문:

- **RQ1.** wrist IMU의 관측 가능성 한계로 인해 어떤 클래스가 *원리적으로* 분리되지 않는가?
- **RQ2.** 분류기의 posterior entropy, attention entropy, 그리고 confusion pattern은 RQ1의 관측 한계를 어떻게 *정량적으로* 반영하는가?
- **RQ3.** 이러한 불확실성 신호를 closed-vocabulary uncertainty schema에 매핑한 뒤, schema-grounded LLM caption으로 정직하게 한국어로 전달할 수 있는가?

본 연구는 *sensor-to-text generation* 자체를 학습하는 시스템을 만들지 않는다. 또한 *개인 의미 모형*은 명시적으로 본 논문의 범위 밖이며, future work로 분리한다.

---

## 4. 이론 기반

본 절은 본 연구가 의존하는 이론 기반을 명시한다. 본 절의 항목은 모두 §5의 모델·§6의 caption layer·§7의 평가 layer에서 *직접 사용되는* 양이며, 장식적 인용이 아니다.

### 4.1 정보 이론 — predictive entropy, uncertainty, ambiguity

분류기가 출력하는 class posterior $p(y \mid x)$에 대해, 본 연구는 다음 양을 uncertainty 측정에 사용한다.

- **Predictive entropy**: $H[y \mid x] = -\sum_c p(c \mid x)\,\log p(c \mid x)$. 높을수록 분류기가 어떤 클래스도 강하게 지지하지 않음을 의미한다.
- **Top-1 / top-2 margin**: $p_{(1)}(x) - p_{(2)}(x)$. argmax 분리가 얼마나 좁게 결정되었는지를 보여준다.
- **Class-set posterior mass**: 미리 정의한 ambiguity group(예: $\{C1, C5, C6\}$, $\{C3, C4, C2\}$)의 합산 posterior. 단일 argmax가 약하더라도 *그룹으로서의 확신*이 충분할 수 있는지를 측정한다.

위 세 양은 서로 다른 종류의 불확실성을 분리한다.

- predictive entropy → *aleatoric / epistemic 결합* uncertainty
- margin → *경계 근접도*
- class-set posterior mass → *허용 가능한 모호함* (Step 3 schema의 `class_set` 출력에 직접 매핑)

본 연구는 위 세 양을 모두 schema 출력에 반영하고, 임의로 한 양만 사용하지 않는다.

### 4.2 손목 IMU의 관측 가능성 한계

squat은 대퇴·하퇴·골반·체간의 다관절 운동이다. wrist IMU가 직접 측정하는 것은 손목의 선형 가속도(`acc_x/y/z`)와 손목의 각속도(`gyro_x/y/z`)뿐이며, 손목은 squat의 *주 운동 사슬*에 포함되지 않는다. 자세에 따라 wrist 신호의 해석이 달라진다는 사실은 Step 2.5에서 자세별 anchor agreement가 0.044 ~ 0.884로 크게 다름이 확인되었다는 데서 정량적으로 드러난다 (특히 HW 자세에서 acc-bottom과 gyro-peak가 다른 물리 이벤트를 측정한다).

따라서 wrist IMU만으로는 다음을 *원리적으로 직접 측정할 수 없다*:

- knee valgus / varus
- hip-knee-ankle 정렬
- 골반 후방 경사 / 전방 경사
- 무릎 깊이의 절대값 (acc_z 적분은 drift로 인해 신뢰 불가)

본 연구는 이 한계를 *모델 한계*가 아니라 *센서 한계*로 다룬다. 이는 어떤 모델 아키텍처를 가져와도 해소되지 않는다. schema는 이 한계를 인정하고 표현해야 하며, caption은 측정되지 않은 양을 측정한 것처럼 표현해서는 안 된다.

### 4.3 Attention 기반 시계열 근거 분석

raw IMU sequence 모델(§5.2 4R-B)은 attention weight를 통해 *어느 timestep이 분류 결정에 기여했는가*를 노출한다. 본 연구는 이를 두 가지 용도로 사용한다.

- **Attention entropy**: timestep 분포에 대한 entropy $-\sum_t \alpha_t \log \alpha_t$. 낮을수록 모델이 좁은 시간 영역(예: bottom 근처)에 근거하고 있음을 뜻한다. 높을수록 분류 근거가 시계열 전반에 흩어져 있음을 뜻하며, 이는 *어떤 단일 동작 단서로도 분류가 정해지지 않았음*을 시사한다.
- **Anchor 일치도와의 비교**: Step 2.5에서 정의한 자세별 anchor(SA/CA는 ensemble bottom, HW는 acc-only bottom)와 attention peak 위치의 일치도를 측정한다. 일치도가 높으면 모델이 데이터에 존재하는 물리적 단서를 사용하고 있다는 근거가 되고, 낮으면 분류기가 *다른 신호*에 의존하거나 *시계열 패턴이 부재*함을 시사한다.

attention 분석은 분류 자체가 아니라 분류 *근거의 지형*을 보고하기 위한 도구이다. 따라서 attention entropy와 posterior entropy는 항상 함께 보고된다.

### 4.4 Step 2.5 empirical evidence — uncertainty schema의 데이터 근거

본 연구의 uncertainty schema는 임의로 정의되지 않았다. Step 2.5(`reports/step2/step25_final_synthesis.md`)에서 도출된 다음 empirical evidence가 schema 설계의 *데이터 근거*이다.

| 클래스 | 분리도 (참조 LR) | 본 연구의 schema 처리 |
|---|---|---|
| **C2** | recall 0.66 ~ 0.77로 가장 높음. argmax 분리가 데이터 안에 실제로 존재. | 단일 라벨 confident 출력의 후보. 단언 가능. |
| **C1 / C5 / C6** | 그룹 내부 혼동 약 27%. 손목 IMU만으로 *물리적으로* 구분이 어려운 클래스 군. | 단일 라벨 출력 금지. `class_set ⊆ {C1, C5, C6}` 형태의 그룹 출력 + within-group ambiguity flag. |
| **C3 / C4** | pair 혼동 약 11%에 더해, C3 → C2로 50%, C4 → C2로 41% 흡수. | pair ambiguity와 함께 *C2 흡수 가능성*을 동시에 인정하는 hedge 출력 (`{C3, C4, C2}` 형태). |

위 표는 RQ1에 대한 *데이터 답변*이며, RQ2(분류기 신호가 이 한계를 어떻게 정량적으로 반영하는가)와 RQ3(이를 어떻게 한국어로 정직하게 표현할 것인가)의 출발점이다.

본 연구는 위 한계를 *모델로 돌파하려 하지 않는다*. 새 모델은 위 한계를 *더 잘 정량화*하고 *더 정직하게 표현*하는 데 사용된다.

---

## 5. 새 Step 4 설계

기존 Step 4R 계획서는 sklearn feature-based 모델 비교에 한정되어 있었다. 본 문서는 이를 4R-A로 흡수하고, 그 위에 raw sequence model과 contrastive alignment를 추가한다.

모든 후보 모델은 동일한 split(`v1_36_8_8`, train 36 / val 8 / test 8 participant), 동일한 평가 layer(분류 / calibration / schema 행동 / 근거 분석)에서 비교한다.

### 5.1 4R-A — HistGradientBoosting (feature-based ceiling)

- **목적**: handcrafted feature 위에서의 분류·calibration *상한선*을 정량화한다.
- **입력**: Step 4 modeling dataset의 main 5 raw 또는 zscore feature + posture one-hot.
- **모델**: HistGradientBoostingClassifier (multinomial, balanced class weight).
- **선택 이유**:
  - tabular feature에서 LR/SVM/RF/MLP 중 가장 안정적으로 강한 성능을 보이는 family이며, sklearn 안에서 calibration도 비교적 안정적이다.
  - 본 연구의 main contribution이 feature-based 모델 자체에 있지 않으므로, 같은 family 안에서 *상한선 한 점*을 깔끔하게 보여주면 충분하다.
  - SVM/RF/MLP는 Step 4R 기존 계획에서 후보로 두되, HistGB가 feature-based ceiling으로 채택되면 main pipeline에서는 이 한 모델만 보고하고 나머지는 ablation으로 부속한다.
- **산출물**: posterior, calibration 지표, schema 출력, ambiguity 행동.
- **역할**: §5.2 (raw sequence) 대비, *feature ceiling 위/아래 어디에 있는가*를 보여주는 reference.

### 5.2 4R-B — BiGRU + Attention (raw IMU sensor-to-schema)

- **목적**: handcrafted feature를 거치지 않고 raw IMU sequence에서 직접 학습된 sensor-to-schema 모델을 구성한다. 본 연구의 *학습된 시스템* 본체.
- **입력**: 길이 128, 6채널(`acc_x/y/z, gyro_x/y/z`)의 IMU sequence + posture one-hot (sequence-level concat 또는 conditioning vector).
- **아키텍처 개요** (확정 아님, 설계 의도만 고정):
  - 1D 입력 → BiGRU encoder → temporal attention pooling → posture-conditioned head → 6-class softmax.
  - attention weight $\alpha_t$는 (1) posterior 산출의 가중 pooling으로 사용되고, (2) attention entropy로 별도 보고된다.
- **regularization**:
  - 9275 sample(train ≈ 6,412)은 raw deep model에 충분치 않을 위험이 있으므로, dropout / weight decay / early stopping을 명시적으로 사용한다.
  - augmentation(jittering, scaling, time warping)은 ablation 대상이다.
  - pretrained IMU encoder 활용은 별도 검토 대상이며 본 단계에서는 from-scratch로 우선 시도한다.
- **선택 이유**:
  - sensor-to-schema의 *학습* 본체로 해석 가능한 후보.
  - attention을 통해 §4.3의 시계열 근거 분석이 가능하다.
  - LSTM 대비 BiGRU는 더 작고 학습 안정성이 좋아 9275 규모에서 적절하다.
  - Transformer는 본 단계에서 제외(데이터 규모 대비 과대). future work로 분리.
- **출력**: posterior, attention weight, posture-conditioned representation. 이 representation은 §5.3과 공유될 수 있다.

### 5.3 4R-C — Contrastive IMU-Text Alignment (optional extension)

- **위치**: optional. 4R-A, 4R-B 결과가 §5.5 성공 기준을 충족했을 때만 진행한다.
- **목적**: IMU representation과 schema-derived text representation을 같은 공간에 정렬해, 표현 학습 차원에서도 본 시스템의 representation이 의미 있는지를 점검한다.
- **방법 개요**:
  - IMU encoder(4R-B의 backbone 재사용)와 text encoder(사전학습된 한국어 LM의 frozen encoder)를 contrastive loss로 정렬.
  - positive pair = (IMU, 해당 sample의 schema-derived caption). negative = batch 내 다른 sample.
  - 평가: sample 단위 retrieval (IMU → caption), 동일 class set 내 retrieval 정확도.
- **주의**:
  - 본 단계는 *분류 성능 향상*을 위한 것이 **아니다**. representation의 *의미적 정합성*을 검사하는 부속 실험이다.
  - 4R-C가 실패해도 4R-A / 4R-B 결과는 부정되지 않는다.
  - 4R-C는 논문 main chapter가 아니라 ablation/extension chapter로 다룬다.

### 5.4 평가 layer (네 모델 공통)

§4의 이론 기반과 직접 매핑된다.

1. **분류 성능** — accuracy, macro F1, per-class precision/recall, confusion matrix, ambiguity group 행동 (C1/C5/C6 within-group, C3/C4 pair, C3/C4 → C2 absorption).
2. **Calibration** — log loss, Brier (multi-class), ECE (15-bin), reliability diagram, top-1 prob 분포, top1-top2 margin 분포.
3. **Schema 행동** — class_set size 분포, no_call 비율, caption_confidence_level 분포, ambiguity flag 분포, anchor-driven vs low-confidence no_call의 분리.
4. **근거 분석** — predictive entropy, attention entropy(4R-B 한정), anchor-attention 일치도, ambiguity group 안에서 entropy가 *얼마나 높아지는가*.

본 연구는 (1)만 보고하지 않는다. (1)~(4)를 *함께* 보고하는 것이 main contribution의 일부이다.

### 5.5 모델 채택 기준

- 4R-A → 4R-B 비교에서, 4R-B가 (1) 분류 성능에서 미세 개선이라도 보이고 (2) calibration 또는 (4) 근거 분석 중 하나에서 명확한 개선을 보일 때, 4R-B를 main으로 채택한다.
- 4R-B가 분류 성능은 같거나 낮지만 (4) 근거 분석이 §4의 이론 기반과 *명확히 정합*할 경우에도 main으로 채택할 수 있다 (이때 contribution은 분류 성능이 아니라 근거 분석으로 옮겨진다).
- 위 둘 다 실패하면 4R-A를 main으로 채택하고, 4R-B 결과는 *raw sequence DL의 데이터-규모 한계*로 ablation 챕터에 보고한다.

---

## 6. Step 5 ~ 7 — schema-grounded LLM caption

기존 Step 5~7의 rule/template caption은 main pipeline에서 제외된다. 새 Step 5~7은 다음과 같이 재정의된다.

### 6.1 입력

- 분류기 출력: `class_set`, `class_posterior`, `caption_confidence_level`, `uncertainty_flags`, `no_call`, `posture`, `anchor_reliability`, posterior entropy, attention entropy(4R-B 채택 시).
- 위 입력은 모두 Step 3 schema와 §5 모델 출력에서 *그대로 가져온다*. caption layer는 새로운 분류·판단을 하지 않는다.

### 6.2 LLM의 역할

- **LLM은 표현층(presentation layer)이지 판단자(classifier / judge)가 아니다.** LLM은 schema에 적힌 사실만 한국어 문장으로 풀어낸다.
- **LLM은 새로운 클래스를 추론하지 않는다.** schema가 `class_set = {C1, C5, C6}`이면 LLM은 그 그룹 모호도를 표현하며, 셋 중 하나를 LLM이 자체적으로 좁히지 않는다.
- **LLM은 측정되지 않은 양을 표현하지 않는다.** §4.2의 관측 한계 목록(knee valgus, 정렬, 골반 경사 등)은 caption 어휘에서 금지된다.

### 6.3 prompt 구조 원칙

- system prompt: 본 시스템의 관측 한계, schema 어휘, 금지 표현을 명시한다.
- user prompt: schema 출력의 closed-vocabulary 표현만 들어간다 (자연어 자유 입력 없음).
- 출력 제약: caption은 schema-faithful이어야 하며, schema에 없는 클래스/측정/판단을 도입하면 invalid로 간주한다.

prompt 본문, 어휘 표, 금지 표현 목록의 구체 작성은 본 문서가 결정하지 않는다 — Step 5~7 재실행 시 별도 정책 문서로 잠근다.

### 6.4 평가

§7로 이관된다.

---

## 7. Step 8 — automatic schema-caption validation

기존 Step 8의 human review(`reports/step8/step8_*`)는 main pipeline에서 제외된다. 새 Step 8은 다음과 같이 재정의된다.

### 7.1 schema-faithfulness 자동 검증

caption이 schema를 *왜곡 없이* 표현했는지를 자동으로 검사한다. 검사 항목:

- caption 안에서 추론되는 class set이 schema의 `class_set`과 동일한가.
- caption의 confidence 표현이 schema의 `caption_confidence_level`과 일치하는가.
- caption이 §4.2 관측 한계 목록에 속하는 어떤 표현도 사용하지 않는가 (어휘 차단 리스트).
- caption이 schema의 `uncertainty_flags`를 누락하지 않는가.
- `no_call=true`인 경우 caption이 자료 없음을 *과장 없이* 표현했는가.

### 7.2 검증 방식

검증은 다음 두 layer를 모두 통과해야 한다.

1. **Rule-based vocabulary check** (필수). 금지 어휘 / 필수 hedge 표현 / closed vocabulary 위반 여부를 결정적으로 검사한다.
2. **LLM judge** (보조). closed-vocabulary 라벨로만 출력하도록 강제한다. LLM judge가 *판단자*가 되는 위험을 차단하기 위해, judge의 출력은 caption 수정 신호가 아니라 metric 산출에만 사용한다.

### 7.3 metric

- schema-caption fidelity rate (전체 / class / posture / confidence level 별)
- forbidden vocabulary occurrence rate
- ambiguity preservation rate (schema에 ambiguity flag가 있을 때 caption이 이를 표현했는가)
- no-call message integrity (no_call일 때 caption이 자료 없음을 정확히 표현했는가)

### 7.4 human review의 위치

- main pipeline 평가에서 **제외**한다.
- 별도의 사용성·가독성 검토는 future work로 분리하며, 본 논문의 main claim 근거로 사용하지 않는다.

---

## 8. 논문 contribution 후보

본 연구의 main contribution은 다음 셋으로 정리된다. 셋은 서로 독립이 아니라, 8.1이 8.2의 표현 공간을 정의하고, 8.2가 8.3의 어휘 선택 근거가 된다.

### 8.1 wrist IMU의 *관측 가능성*과 *불확실성*을 구분하는 schema 설계

- *관측 가능성 한계*(sensor가 원리적으로 측정하지 못하는 양)와 *분류 불확실성*(모델 또는 데이터의 불확실성)을 분리해 표현하는 closed-vocabulary schema.
- 두 종류의 한계가 caption에서 서로 다른 방식으로 다루어진다 — 관측 한계는 *어휘 자체에서 금지*되고, 불확실성은 *명시적 hedge*로 표현된다.
- 이는 wrist IMU 기반 운동 평가에서 흔히 발생하는 *과장된 의학적 표현*(예: "knee valgus가 관찰됨")을 구조적으로 차단한다.

### 8.2 Posterior entropy + attention entropy + confusion pattern 기반 uncertainty 분석

- 단일 entropy가 아니라 세 종류의 신호를 *함께* 보고한다:
  - **posterior entropy** — 분류 결과의 모호도
  - **attention entropy** — 시계열 근거의 분산도 (4R-B에서)
  - **confusion pattern** — Step 2.5에서 정의된 ambiguity group과의 일치도
- 세 신호가 §4.4 *Step 2.5 empirical evidence*와 정합하는지를 정량적으로 보인다.
- 이를 통해 uncertainty schema가 임의로 정의된 정책이 아니라 *데이터와 모델 모두로부터 측정되는 양*에 매핑됨을 보인다.

### 8.3 Uncertainty-aware Korean caption generation

- schema-grounded LLM caption layer로, 분류기 출력 → 자연어 출력의 변환을 *학습 없이* 보장한다.
- LLM은 표현층이며, caption은 schema에 *충실*하면서 한국어로 자연스럽게 읽혀야 한다.
- 본 contribution은 자유로운 sensor-to-text generation이 아니라, *불확실성을 정직하게 표현하는 한국어 출력 시스템*에 한정된다.

---

## 9. 본 문서가 명시적으로 결정하지 않는 사항

본 문서는 정책 문서이며, 다음은 *명시적으로 미결정*이다. 후속 단계에서 별도로 잠근다.

- 4R-A, 4R-B, 4R-C 각 모델의 hyperparameter 후보.
- 4R-B의 BiGRU layer 수, hidden size, attention 형태(additive vs scaled-dot)의 최종 선택.
- LLM caption layer의 prompt 본문, 어휘 표, 금지 표현 목록.
- §7 schema-faithfulness 자동 검증의 LLM judge 모델 선택.
- 논문 chapter 구조 (Modeling 챕터 안에 4R-A/4R-B/4R-C가 어떻게 배치되는지, schema 챕터와 caption 챕터의 분리 여부).
- 지도교수와의 chapter 구조 합의 일자.

---

## 10. 보존 정책

기존 Step 4 LR baseline, Step 5~7 rule/template caption, Step 8 human review 산출물은 다음 정책으로 보존한다.

- git에서 삭제하지 않는다.
- 논문 main contribution의 근거로 인용하지 않는다.
- 필요한 경우 *연구 진행 과정의 기록*으로만 인용 가능하다 (예: pivoting 동기, ablation 비교의 reference point).
- 새 main pipeline 산출물은 `reports/step4r/`, `reports/step5_v2/`, `reports/step6_v2/`, `reports/step7_v2/`, `reports/step8_v2/` 같이 *명시적으로 분리된 경로*로 적재한다 (구체 경로는 후속 정책 문서에서 잠근다).

---

## 11. 본 문서가 변경되는 조건

다음 중 하나가 발생하면 본 문서를 갱신한다 (덮어쓰지 않고 갱신 일자와 변경 내역을 추가).

- 4R-B 학습이 데이터 규모 한계로 *반복적으로* 실패해 raw sequence DL 자체를 main에서 빼야 할 때.
- 지도교수와의 chapter 구조 합의 결과가 §3 또는 §8과 충돌할 때.
- §4.2 관측 한계 목록이 추가/삭제될 때.
- §6의 LLM 표현층 정책이 *자유로운 sensor-to-text generation*으로 확대될 때 (현재 명시적으로 제외).
- *개인 의미 모형*이 본 논문 범위로 다시 들어올 때 (현재 명시적으로 제외).

---

*본 문서는 2026-05-08에 작성되었다. 새 코드, 새 모델, 새 데이터, 새 캡션은 본 문서로 인해 생성되지 않는다. Step 1~3 산출물은 수정되지 않는다.*
