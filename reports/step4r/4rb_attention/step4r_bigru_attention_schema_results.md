# Step 4R-B 후처리 2/3 — Calibrated Schema Output 결과

- 생성 스크립트: `scripts/generate_step4r_attention_schema_outputs.py`
- 입력: `data/step4r/4rb_attention/step4r_bigru_attention_logits_calibrated.npz` (probs_calibrated 사용)
- temperature 적용: T = 1.957900
- 본 스크립트는 기존 `scripts/generate_step4_schema_outputs.py`를 수정하지 않으며, 동일 logic을 새 스크립트에 재구현했다.

---

## 1. 본 단계의 위치

calibrated posterior(`probs_calibrated`)를 Step 3 §3 ~ §9 출력 스키마로 변환한다. schema 결정에는 calibrated posterior만 사용하며, raw posterior는 비교용으로만 보존한다.

**LLM은 판단자가 아니라 표현층이다.** 본 schema가 caption layer (Step 5~7 재정의 후속) 의 *입력*이며, caption은 schema에 충실하게 한국어로 표현될 뿐 새로운 분류·판단을 하지 않는다 (`reports/step4_research_reframing.md` §6 참조).

---

## 2. Operational thresholds (val split 자체 산출)

Step 3 §6/§8은 임계값을 commit하지 않으므로, 본 단계는 BiGRU calibrated posterior에 대해 val split만 사용해 Step 4 calibration과 동일한 절차로 임계값을 자체 산출했다. 이 값들은 **temporary operational thresholds**이며, 후속 단계(예: human review로 재calibration)에서 갱신될 수 있다.

| threshold | 산출 방식 | 값 | 표시 |
|---|---|---:|---|
| confident_C2_threshold | smallest t s.t. precision(p_C2 ≥ t on val) ≥ 0.60 | 0.2700 | temporary operational |
| non_trivial_C2_threshold | percentile 50 of p_C2 over val (pred ∈ C3/C4) | 0.0456 | temporary operational |
| within_group_threshold | percentile 25 of min(p_C1,p_C5,p_C6) over val (pred ∈ C1/C5/C6) | 0.0461 | temporary operational |
| anchor_suppression_threshold | Step 4 heuristic (boundary-aligned bin cut) | 0.5000 | reused from Step 4 |
| anchor_no_call_threshold | Step 4 heuristic (boundary-aligned bin cut) | 0.2500 | reused from Step 4 |

---

## 3. split별 no_call 및 caption_confidence_level 분포

### 3.1 train split (n = 6412)

| level | count | rate |
|---|---:|---:|
| confident | 977 | 0.1524 |
| hedged | 5135 | 0.8008 |
| low | 238 | 0.0371 |
| no_call | 62 | 0.0097 |
| **(no_call total)** | **62** | **0.0097** |

### 3.2 val split (n = 1436)

| level | count | rate |
|---|---:|---:|
| confident | 244 | 0.1699 |
| hedged | 1115 | 0.7765 |
| low | 65 | 0.0453 |
| no_call | 12 | 0.0084 |
| **(no_call total)** | **12** | **0.0084** |

### 3.3 test split (n = 1427)

| level | count | rate |
|---|---:|---:|
| confident | 281 | 0.1969 |
| hedged | 1062 | 0.7442 |
| low | 67 | 0.0470 |
| no_call | 17 | 0.0119 |
| **(no_call total)** | **17** | **0.0119** |

---

## 4. class_set 크기 분포 (전체)

| size | count | rate |
|---|---:|---:|
| 0 | 91 | 0.0098 |
| 1 | 1538 | 0.1658 |
| 2 | 2201 | 0.2373 |
| 3 | 5445 | 0.5871 |

---

## 5. ambiguity flag 발생률 (전체)

| flag | rows containing it | rate |
|---|---:|---:|
| `anchor_unreliable` | 441 | 0.0475 |
| `confident_C2` | 1549 | 0.1670 |
| `low_confidence_no_class_set` | 59 | 0.0064 |
| `pair_ambiguity_c3_c4` | 1498 | 0.1615 |
| `pair_plus_c2_absorption` | 1579 | 0.1702 |
| `posture_unknown` | 0 | 0.0000 |
| `within_group_ambiguity_c1_c5_c6` | 4590 | 0.4949 |

---

## 6. top1 argmax와 class_set의 차이

non-no_call 행 중에서 top1 argmax가 실제 class_set에 *포함되지 않는* 비율. schema rule이 argmax 외 다른 class를 어떻게 추가/대체하는지 보는 지표.

| split | non-no_call n | top1 not in class_set | rate |
|---|---:|---:|---:|
| train | 6350 | 0 | 0.0000 |
| val | 1424 | 0 | 0.0000 |
| test | 1410 | 0 | 0.0000 |
| all | 9184 | 0 | 0.0000 |

---

## 7. Step 4R-A / 4R-B와의 연결 해석

- **분류 본체 차원** — 4R-A HGB가 LR과 동등한 분류 ceiling이었던 반면 4R-B BiGRU+Attention은 test macro F1을 ~+0.21 끌어올렸다 (4R-B results §3). 본 schema는 그 calibrated posterior 위에서 작동한다.
- **Schema 행동 차원** — Step 2.5 §7의 ambiguity 패턴(C2 단언 가능 / C1·C5·C6 그룹 모호 / C3·C4의 C2 흡수)이 본 schema의 ambiguity_group 분포로 반영되어야 한다 (§5의 flag 발생률). Step 4R-B가 분류는 잘하지만 ambiguity 구조가 무너지면(예: 거의 모두 confident_C2 단일 emission) Step 3 정책이 의도한 *정직한 모호함 표현*이 사라진다 — 이를 §3 / §5에서 점검한다.
- **Calibration 차원** — calibration 단계 1/3에서 fitted T로 over-confidence를 보정했다. 그 효과가 hedged/low 비율 증가, confident 비율 감소, no_call 비율 감소(low_confidence 경로가 raw에서보다 적게 발생)로 반영되는지를 §3 분포에서 확인한다.

---

## 8. 본 단계에서 명시적으로 결정하지 않는 사항

- LLM caption layer prompt 본문, 어휘표, 금지 표현 목록은 본 단계의 범위 밖이다.
- temporary operational threshold의 commit (실험 후 별도 단계에서 잠금).
- human review 재개 (`reports/step4_research_reframing.md` §7에 의해 main pipeline에서 제외 중).

---

## 9. 산출물 목록

- `data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv`
- `reports/step4r/4rb_attention/step4r_bigru_attention_schema_summary.csv` (long format: split, metric, value)
- `reports/step4r/4rb_attention/step4r_bigru_attention_schema_results.md` (본 보고서)

---

*본 보고서는 자동 생성된다. 기존 Step 1 ~ 4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 `step4r_bigru_attention_schema_outputs_*` 등 다른 모델의 schema 출력 파일은 영향 없음.*
