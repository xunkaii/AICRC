# Step 7_v3 — Caption Generation Prototype 결과 (clinical pool)

- 생성 스크립트: `scripts/generate_step7_v3_captions.py`
- mode: `full`
- provider: `mock`
- model: `mock`
- temperature: `0.0` / max_tokens: `1024` / sleep: `0.5s`

---

## 1. 본 단계의 위치

Step 7_v3 는 Step 6_v3 가 commit 한 clinical pool 기반 prompt / output schema / 정책 위에서 한국어 caption 을 생성하는 분기이다. v2 (sensor observability 우선) 와 *병렬* 로 운영되며, v2 baseline 산출물 (reports/step7_v2/) 은 본 분기로 인해 수정되지 않는다. Step 8_v2 automatic schema-caption validation 의 *입력* 으로 v2 와 함께 비교 metric 을 생성한다.

---

## 2. 실행 설정

| 항목 | 값 |
|---|---|
| mode | full |
| provider | mock |
| model | mock |
| temperature | 0.0 |
| max_tokens | 1024 |
| inter-call sleep | 0.5s |
| max_retries | 3 (then fallback) |
| seed | 42 |

---

## 3. 종합 결과

| 지표 | 값 |
|---|---:|
| 총 sample | 9275 |
| 첫 시도 통과 | 9275 |
| retry 후 통과 | 0 |
| fallback | 0 |
| schema_faithfulness_rate | 1.0000 |
| fallback_rate | 0.0000 |
| mean_n_retries | 0.0000 |

---

## 4. confidence level별 caption 수

| level | count | rate |
|---|---:|---:|
| confident | 1502 | 0.1619 |
| hedged | 7312 | 0.7884 |
| low | 370 | 0.0399 |
| no_call | 91 | 0.0098 |

## 5. ambiguity group별 caption 수

| group | count | rate |
|---|---:|---:|
| confident_C2 | 1538 | 0.1658 |
| no_call | 91 | 0.0098 |
| pair_c3_c4 | 1498 | 0.1615 |
| pair_plus_c2_absorption | 1558 | 0.1680 |
| within_group_c1_c5_c6 | 4590 | 0.4949 |

## 6. no_call caption 수: 91 / 9275 (0.0098)

---

## 7. 위반 발생 집계

| 검사 | invalid count |
|---|---:|
| forbidden expression (책임/처방) | 0 |
| attention term leakage | 0 |
| class_set narrowing | 0 |
| used_pool_entries 형식 오류 | 0 |
| fallback | 0 |

---

## 8. 대표 caption 예시 10개

**EP29_C1_HW_rep01** (HW, confident_C2, confident)

> 손을 허리에 둔 자세에서 세 가지 중 가장 안정적인 조건입니다. 무릎 굴곡이 목표 각도에 도달하기 전에 동작이 마무리되는 패턴이 보입니다. 비교적 일관되게 나타납니다. 손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다.

**EP52_C3_HW_rep05** (HW, pair_c3_c4, hedged)

> 손을 허리에 둔 상태에서 고관절과 체간의 협응 패턴이 두드러집니다. 요추 굴곡이 과도해지면서 척추 중립이 무너지는 경향이 보입니다 동시에 좌측에 무게가 더 실리는 패턴이 신호에 나타납니다. 복합 결함 두 가지가 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오.

**EP01_C4_CA_rep07** (CA, pair_plus_c2_absorption, hedged)

> 팔을 교차한 자세에서 팔을 앞으로 뻗은 자세보다 몸의 균형 잡기가 수월합니다. 고관절 굴곡 범위가 충분히 확보되지 않는 패턴이 나타납니다 동시에 무릎이 내측으로 쏠리는 패턴이 고관절 내회전과 함께 나타납니다 동시에 좌측에 무게가 더 실리는 패턴이 신호에 나타납니다. 깊이 부족 신호와 무릎 관련 비대칭 신호가 함께 나타납니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다.

**EP19_C5_SA_rep09** (SA, within_group_c1_c5_c6, hedged)

> 팔을 앞으로 뻗은 상태에서 척추 중립 유지에 대한 요구가 높아집니다. 우측 무릎이 내측으로 쏠리는 패턴이 하강 구간에서 나타납니다 동시에 고관절 내전·내회전 패턴이 양측에서 함께 나타납니다. 정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오.

**EP51_C1_CA_rep04** (CA, within_group_c1_c5_c6, hedged)

> 팔을 교차한 상태에서 어깨 긴장이 동작 패턴에 영향을 줄 수 있습니다. 발목·무릎·고관절의 협응이 안정적으로 이루어지는 패턴이 보입니다 함께 우측에 무게가 더 실리는 패턴이 신호에 나타납니다 함께 고관절 내전·내회전 패턴이 양측에서 함께 나타납니다. 정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정.

**EP31_C5_CA_rep07** (CA, within_group_c1_c5_c6, hedged)

> 팔을 교차한 자세에서 체간 안정화 패턴이 변화하는 경향이 보입니다. 발목·무릎·고관절의 협응이 안정적으로 이루어지는 패턴이 보입니다 동시에 우측 무릎이 내측으로 쏠리는 패턴이 하강 구간에서 나타납니다 동시에 발목 가동성 제한이 양측 무릎 정렬에 복합적으로 영향을 주는 경향이 보입니다. 정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 본 설명은 손목 IMU 신호 기반의 추정이며, 무릎 각도·골반 운동·발목 가동 범위는 직접 측정되지 않습니다.

**EP34_C6_CA_rep09** (CA, within_group_c1_c5_c6, low)

> 팔을 교차한 상태에서 어깨 긴장이 동작 패턴에 영향을 줄 수 있습니다. 우측 무릎이 내측으로 쏠리는 패턴이 하강 구간에서 나타납니다 동시에 발목 가동성 제한이 양측 무릎 정렬에 복합적으로 영향을 주는 경향이 보입니다. 정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다. 후보군 수준으로만 해석하는 것이 적절합니다. 손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다.

**EP25_C5_HW_rep10** (HW, within_group_c1_c5_c6, hedged)

> 손을 허리에 둔 자세에서 세 가지 중 가장 안정적인 조건입니다. 발목·무릎·고관절의 협응이 안정적으로 이루어지는 패턴이 보입니다 한편 우측 무릎이 내측으로 쏠리는 패턴이 하강 구간에서 나타납니다 한편 양측 무릎이 동시에 내측으로 쏠리는 패턴이 나타납니다. 정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정.

**EP22_C5_CA_rep05** (CA, within_group_c1_c5_c6, hedged)

> 팔을 교차한 자세에서 체간 안정화 패턴이 변화하는 경향이 보입니다. 발목·무릎·고관절의 협응이 안정적으로 이루어지는 패턴이 보입니다 동시에 우측 무릎이 내측으로 쏠리는 패턴이 하강 구간에서 나타납니다 동시에 고관절 내전·내회전 패턴이 양측에서 함께 나타납니다. 정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서는 무릎 각도나 골반 움직임을 직접 측정하지 않습니다.

**EP30_C6_HW_rep07** (HW, within_group_c1_c5_c6, hedged)

> 허리 거치 자세에서 골반 움직임이 IMU 신호에 직접적으로 반영됩니다. 우측에 무게가 더 실리는 패턴이 신호에 나타납니다 한편 고관절 내전·내회전 패턴이 양측에서 함께 나타납니다. 정상 패턴과 무릎 관련 비대칭 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정.

---

## 9. fallback 또는 invalid 예시

fallback 발생 없음.

---

## 10. Step 8_v2 automatic validation 으로 넘기는 산출물

- `data/step7_v3/step7_v3_captions.csv` (full mode 결과)
- `data/step7_v3/step7_v3_run_log.csv` (per-attempt log)
- `data/step7_v3/step7_v3_golden_captions.csv` (golden cases)
- `data/step7_v3/step7_v3_golden_run_log.csv`
- `reports/step7_v3/step7_v3_caption_generation_summary.csv`
- `reports/step7_v3/step7_v3_golden_validation_summary.csv`
- `reports/step7_v3/step7_v3_caption_generation_results.md` (본 보고서)

---

*본 보고서는 자동 생성된다. v2 산출물 (reports/step7_v2/, data/step7_v2/) 은 본 분기로 인해 수정되지 않는다.*