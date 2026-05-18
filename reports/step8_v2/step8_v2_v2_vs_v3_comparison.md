# Step 8_v2 — v2 / v3 Caption Pipeline 비교 검증

- 생성 스크립트: `scripts/validate_step8_v2_v3_compare.py`
- v2 input : `data/step7_v2/step7_v2_captions.csv` (n=9275)
- v3 input : `data/step7_v3/step7_v3_captions.csv` (n=9275)
- 검증 layer: rule-based
- 기존 `step8_v2_report.md` / `step8_v2_summary.csv` 는 본 스크립트로 수정되지 않음 (별도 baseline 보존)

---

## 1. Overall fidelity

| 검증 행렬                              |    n | pass |   fidelity |
| ---------------------------------- | ---: | ---: | ---------: |
| **(A) v2 captions × v2 validator** | 9275 | 9275 | **1.0000** |
| **(B) v3 captions × v3 validator** | 9275 | 9275 | **1.0000** |
| (C) v3 captions × v2 validator     | 9275 |  312 |     0.0336 |

**핵심 해석:**
- A vs B: 각 pipeline 이 *자기 정책* 에 대해 거의 동등하게 충실함 (1.0000 vs 1.0000).
- C: v3 captions 를 *v2 의 strict 정책* 에 대해 평가하면 fidelity 가 0.0336 로 떨어짐 — clinical vocabulary 도입으로 v2 정책 위반 96.6% 발생.
- (C) 가 클수록 v2 → v3 trade-off 폭이 큼. 정책 완화 효과 = (1 - fid_c).

## 2. Diversity (표현 다양성)

| 지표                    |    v2 |    v3 |         Δ |
| --------------------- | ----: | ----: | --------: |
| unique caption_ko     |    28 |  5514 | **+5486** |
| caption 길이 mean (자)   | 107.6 | 214.2 |    +106.5 |
| caption 길이 median (자) |   109 |   227 |      +118 |
| caption 길이 max (자)    |   119 |   294 |      +175 |

- v3 가 **196.9배** 더 다양한 caption 을 생성 — POSTURE_POOL × JOINT_POOL hash permutation 의 효과.

## 3. Clinical vocabulary 등장 비율

| 어휘             |    v2 |     v3 |           Δ |
| -------------- | ----: | -----: | ----------: |
| 좌측             | 0.00% | 32.95% | **+32.95%** |
| 우측             | 0.00% | 49.23% | **+49.23%** |
| 양측             | 0.00% | 48.65% | **+48.65%** |
| 고관절            | 0.00% | 78.59% | **+78.59%** |
| 골반             | 0.00% | 60.39% | **+60.39%** |
| 무릎 정렬          | 0.00% | 37.81% | **+37.81%** |
| 내회전            | 0.00% | 43.90% | **+43.90%** |
| 내전             | 0.00% | 16.67% | **+16.67%** |
| 발목 가동          | 0.00% | 47.26% | **+47.26%** |
| 요추             | 0.00% | 16.35% | **+16.35%** |
| 흉추             | 0.00% |  3.86% |  **+3.86%** |
| 척추 중립          | 0.00% | 27.45% | **+27.45%** |
| knee valgus    | 0.00% |  0.00% |  **+0.00%** |
| posterior tilt | 0.00% |  0.00% |  **+0.00%** |

- v2 에서 0% 인 항목 (e.g., 좌측/우측/양측/무릎 정렬/고관절) 이 v3 에서 의도적으로 도입됨 → 정책 분기 효과 확인.

## 4. Stratum별 fidelity (A vs B)

### 4.1 by true class

| class | n (v2/v3) | v2 fidelity | v3 fidelity |
|---|---:|---:|---:|
| C1 | 1553 / 1553 | 1.0000 | 1.0000 |
| C2 | 1555 / 1555 | 1.0000 | 1.0000 |
| C3 | 1529 / 1529 | 1.0000 | 1.0000 |
| C4 | 1540 / 1540 | 1.0000 | 1.0000 |
| C5 | 1549 / 1549 | 1.0000 | 1.0000 |
| C6 | 1549 / 1549 | 1.0000 | 1.0000 |

### 4.2 by posture

| posture | n (v2/v3) | v2 fidelity | v3 fidelity |
|---|---:|---:|---:|
| SA | 3098 / 3098 | 1.0000 | 1.0000 |
| CA | 3080 / 3080 | 1.0000 | 1.0000 |
| HW | 3097 / 3097 | 1.0000 | 1.0000 |

### 4.3 by caption_confidence_level

| level | n (v2/v3) | v2 fidelity | v3 fidelity |
|---|---:|---:|---:|
| confident | 1502 / 1502 | 1.0000 | 1.0000 |
| hedged | 7312 / 7312 | 1.0000 | 1.0000 |
| low | 370 / 370 | 1.0000 | 1.0000 |
| no_call | 91 / 91 | 1.0000 | 1.0000 |

## 5. Error code violation rate (3-way)

| error code                         | A: v2/v2 | B: v3/v3 | C: v3/v2 (strict) |
| ---------------------------------- | -------: | -------: | ----------------: |
| `biomech_claim`                    |   0.0000 |   0.0000 |            0.5694 |
| `class_set_narrowing_pair`         |   0.0000 |   0.0000 |            0.3295 |
| `class_set_narrowing_pair_plus_c2` |   0.0000 |   0.0000 |            0.1680 |
| `class_set_narrowing_within_group` |   0.0000 |   0.0000 |            0.4949 |
| `direction_token`                  |   0.0000 |   0.0000 |            0.4865 |
| `posture_phrase_missing`           |   0.0000 |   0.0000 |            0.7457 |

---

## 6. Trade-off 해석 (thesis chapter 자료)

본 비교는 reframing §8.3 의 *uncertainty-aware caption layer* 가 **두 끝점**을 가짐을 정량 증거로 제시한다:

- **v2 보수 (sensor observability 우선)**: clinical 어휘 0%, schema 충실도 1.0000. 측정 가능성 한계 안에서만 표현하므로 "안전" 하나, C4/C5/C6 구분 불가 + 표현 단조.
- **v3 완화 (clinical richness 우선)**: clinical 어휘 의도적 도입, schema 충실도 1.0000. caption 다양성 196.9배 증가 + 좌/우 비대칭 표현 가능. 단 v2 strict 정책 기준으로는 96.6% 위반 (1.0000 → 0.0336).

→ 본 trade-off 자체가 paper chapter 1 개 거리. 실제 임상/현장 환경에 따라 어느 끝점을 선택할지 *사용자/연구자가 명시적으로 결정* 할 수 있는 framework 가 갖춰짐.

**한계:**
- 본 결과는 mock provider 기반. real LLM 평가 시 v3 fidelity 가 약간 떨어질 수 있음 (anthropic golden 15-case 에서는 1.0000 유지 — 일관 가능성 높음).
- (C) 는 v3 가 v2 의 used_pool_entries 필드를 "extra_fields" 로 보는 등의 schema 불일치도 포함 → 순수 어휘 위반 외 형식 위반 일부 포함됨.

## 7. 산출물

```
data/step8_v2/
├── step8_v2_v2_validation.csv        (A) per-sample, n=9275
├── step8_v2_v3_validation.csv        (B) per-sample, n=9275
└── step8_v2_v3_under_v2policy.csv    (C) per-sample, n=9275

reports/step8_v2/
├── step8_v2_v2_vs_v3_summary.csv     side-by-side metrics
└── step8_v2_v2_vs_v3_comparison.md   본 보고서

기존 보존 (본 스크립트로 수정되지 않음):
├── step8_v2_report.md                v2 baseline (이전 run)
└── step8_v2_summary.csv              v2 baseline summary
```
