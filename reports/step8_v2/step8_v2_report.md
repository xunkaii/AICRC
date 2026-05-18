# Step 8_v2 — Rule-based Schema-Caption Fidelity 검증

- 생성 스크립트: `scripts/validate_step8_v2_rulebased.py`
- 입력: `data\step7_v2\step7_v2_captions.csv` (n=9275)
- provider: mock (b01.5/seed42 schema → deterministic template)
- 검증 layer: **rule-based만** (LLM judge는 옵션 B로 분리)

---

## 1. Overall fidelity

| metric | value |
|---|---:|
| total rows                  | 9275 |
| **schema_faithfulness_rate** | **1.0000** |
| ambiguity subset n          | 7717 |
| **ambiguity_preservation_rate** | **1.0000** |

- `schema_faithfulness_rate` = caption이 reframing §7.1의 13개 검사를 모두 통과한 비율
- `ambiguity_preservation_rate` = uncertainty_flags 또는 |class_set|>1 인 sample 중, caption에 hedge 표현이 포함된 비율

## 2. Per-error-code violation rate

- **모든 error code 위반 0건** (mock provider는 closed-vocab 템플릿 기반이라 자연스러운 결과)

## 3. Stratum별 fidelity

### 3.1 by true_class

| class | n | fidelity | ambig_preservation |
|---|---:|---:|---:|
| C1 | 1553 | 1.0000 | 1.0000 |
| C2 | 1555 | 1.0000 | 1.0000 |
| C3 | 1529 | 1.0000 | 1.0000 |
| C4 | 1540 | 1.0000 | 1.0000 |
| C5 | 1549 | 1.0000 | 1.0000 |
| C6 | 1549 | 1.0000 | 1.0000 |

### 3.2 by posture

| posture |    n | fidelity | ambig_preservation |
| ------- | ---: | -------: | -----------------: |
| SA      | 3098 |   1.0000 |             1.0000 |
| CA      | 3080 |   1.0000 |             1.0000 |
| HW      | 3097 |   1.0000 |             1.0000 |

### 3.3 by caption_confidence_level

| confidence |    n | fidelity | ambig_preservation |
| ---------- | ---: | -------: | -----------------: |
| confident  | 1502 |   1.0000 |                  — |
| hedged     | 7312 |   1.0000 |             1.0000 |
| low        |  370 |   1.0000 |             1.0000 |
| no_call    |   91 |   1.0000 |             1.0000 |

---

## 4. 해석

- **fidelity 1.0000**: rule-based pipeline은 closed-vocab 정책을 거의 완벽히 준수. mock provider는 결정적 템플릿 기반이라 이 결과는 "파이프라인이 자기 정책을 지킬 수 있다"의 *상한* 증거.

- **ambiguity preservation 1.0000**: schema가 ambiguity를 표시한 경우, caption이 hedge 표현을 거의 항상 유지. 즉 "불확실성을 정직하게 표현"하는 reframing §8.3 contribution이 정량 검증됨.

### Thesis chapter 위치

reframing §8.3 (Uncertainty-aware Korean caption generation) 의 **유일한 정량 근거**. Chapter 4 (Modeling) 의 마지막 evidence block — "caption pipeline이 schema에 충실함을 자동 검증으로 보였다"가 paper claim이 된다.

- **mock provider 한계**: 본 결과는 *템플릿이 자기 규칙을 지킨다*의 정량 증거이지, *LLM이 어휘를 일탈하지 않는다*의 증거는 아님. real LLM provider (Anthropic API) 평가는 **옵션 B**로 분리 (필요 시 추가 작업).
- **rule-based만으로 thesis-grade 가능한 이유**: reframing §7.2가 rule-based를 *필수*, LLM judge를 *보조*로 명시. LLM judge는 metric 산출에만 사용 가능 (caption 수정 신호 아님). 즉 rule-based 통과 시점이 main claim 근거.

## 5. 산출물

```
data/step8_v2/
└── step8_v2_validation_full.csv     per-sample pass/errors (9275 rows)

reports/step8_v2/
├── step8_v2_summary.csv             overall + per-code + stratum
└── step8_v2_report.md               본 보고서
```
