# Step 7_v2 — Caption Generation Prototype 결과

- 생성 스크립트: `scripts/generate_step7_v2_captions.py`
- mode: `full`
- provider: `mock`
- model: `mock`
- temperature: `0.0` / max_tokens: `512` / sleep: `0.5s`

---

## 1. 본 단계의 위치

Step 7_v2는 Step 6_v2가 commit한 prompt / output schema / golden cases / validation checklist 위에서 schema-grounded Korean caption을 생성하는 prototype 단계이다. Step 8_v2 automatic schema-caption validation의 *입력*이 된다. human review는 main evaluation에 포함하지 않는다 (`reports/step4_research_reframing.md` §7).

---

## 2. 실행 설정

| 항목 | 값 |
|---|---|
| mode | full |
| provider | mock |
| model | mock |
| temperature | 0.0 |
| max_tokens | 512 |
| inter-call sleep | 0.5s |
| max_retries | 3 (then fallback) |
| seed | 42 |

---

## 3. 종합 결과

| 지표 | 값 |
|---|---:|
| 총 sample | 9275 |
| 첫 시도 통과 | 9273 |
| retry 후 통과 | 0 |
| fallback | 2 |
| schema_faithfulness_rate | 0.9998 |
| fallback_rate | 0.0002 |
| mean_n_retries | 0.0006 |

---

## 4. confidence level별 caption 수

| level | count | rate |
|---|---:|---:|
| confident | 1652 | 0.1781 |
| hedged | 7200 | 0.7763 |
| low | 364 | 0.0392 |
| no_call | 59 | 0.0064 |

## 5. ambiguity group별 caption 수

| group | count | rate |
|---|---:|---:|
| confident_C2 | 1694 | 0.1826 |
| no_call | 59 | 0.0064 |
| pair_c3_c4 | 1530 | 0.1650 |
| pair_plus_c2_absorption | 1556 | 0.1678 |
| within_group_c1_c5_c6 | 4436 | 0.4783 |

## 6. no_call caption 수: 59 / 9275 (0.0064)

---

## 7. 위반 발생 집계

| 검사 | invalid count |
|---|---:|
| forbidden expression | 0 |
| attention term leakage | 0 |
| class_set narrowing | 2 |
| unsupported biomechanical claim | 0 |
| fallback | 2 |

---

## 8. 대표 caption 예시 10개

**EP29_C1_HW_rep01** (HW, within_group_c1_c5_c6, hedged)

> 손을 허리에 둔 자세에서 측정된 손목 IMU 패턴은, 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

**EP52_C3_HW_rep05** (HW, pair_c3_c4, hedged)

> 손을 허리에 둔 자세에서 측정된 손목 IMU 패턴은 복합 오류 후보 두 가지가 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

**EP01_C4_CA_rep07** (CA, pair_plus_c2_absorption, hedged)

> 팔을 교차한 자세에서 측정된 손목 IMU 패턴은 깊이 부족 계열 신호와 복합 오류 후보가 함께 나타날 가능성이 있습니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

**EP19_C5_SA_rep09** (SA, pair_plus_c2_absorption, hedged)

> 팔을 앞으로 둔 자세에서 측정된 손목 IMU 패턴은 깊이 부족 계열 신호와 복합 오류 후보가 함께 나타날 가능성이 있습니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

**EP51_C1_CA_rep04** (CA, within_group_c1_c5_c6, hedged)

> 팔을 교차한 자세에서 측정된 손목 IMU 패턴은, 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

**EP31_C5_CA_rep07** (CA, pair_plus_c2_absorption, hedged)

> 팔을 교차한 자세에서 측정된 손목 IMU 패턴은 깊이 부족 계열 신호와 복합 오류 후보가 함께 나타날 가능성이 있습니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

**EP34_C6_CA_rep09** (CA, within_group_c1_c5_c6, low)

> 팔을 교차한 자세에서, 동작 기준점이 불안정하여 설명 신뢰도가 낮은 상태입니다. 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타날 가능성이 있으나, 후보군 수준으로만 해석하는 것이 적절합니다.

**EP25_C5_HW_rep10** (HW, confident_C2, confident)

> 손을 허리에 둔 자세에서 기록된 손목 IMU 신호는 깊이 부족 계열로 해석 가능한 패턴이 비교적 일관되게 나타납니다. 손목 센서 기준의 추정이며, 정확한 자세 평가는 영상 또는 전문가 평가와 함께 해석해 주십시오.

**EP22_C5_CA_rep05** (CA, within_group_c1_c5_c6, hedged)

> 팔을 교차한 자세에서 측정된 손목 IMU 패턴은, 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

**EP30_C6_HW_rep07** (HW, within_group_c1_c5_c6, hedged)

> 손을 허리에 둔 자세에서 측정된 손목 IMU 패턴은, 정상 패턴과 무릎 관련 오류 후보 패턴이 함께 나타나는 경계 신호로 보입니다. 단일 유형으로 좁히기 어렵습니다. 손목 센서 기준의 추정입니다.

---

## 9. fallback 또는 invalid 예시

- `EP03_C1_SA_rep09`: errors=`fallback_used|class_set_narrowing_within_group`
- `EP01_C1_HW_rep01`: errors=`fallback_used|class_set_narrowing_within_group`

---

## 10. Step 8_v2 automatic validation으로 넘기는 산출물

- `data/step7_v2/step7_v2_captions.csv` (full mode 결과)
- `data/step7_v2/step7_v2_run_log.csv` (per-attempt log)
- `data/step7_v2/step7_v2_golden_captions.csv` (golden cases)
- `data/step7_v2/step7_v2_golden_run_log.csv`
- `reports/step7_v2/step7_v2_caption_generation_summary.csv`
- `reports/step7_v2/step7_v2_golden_validation_summary.csv`
- `reports/step7_v2/step7_v2_caption_generation_results.md` (본 보고서)

---

*본 보고서는 자동 생성된다. 기존 Step 1~6_v2 산출물은 수정되지 않는다.*