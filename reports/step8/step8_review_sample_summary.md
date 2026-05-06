# Step 8-1 — Review Sample Extraction Summary

이 문서는 Step 8 평가 계획(`reports/step8/step8_caption_evaluation_plan.md`)에
따라 Step 7 caption prototype CSV를 read-only로 읽어 human review용
stratified sample CSV를 *생성한* 결과를 요약한다. 본 단계는 sample
*생성*만 수행하며, 실제 human review / raw vs zscore 최종 branch 선택
/ final wording 확정 / caption 재생성 / 새 모델 학습 / threshold
recalibration / schema output 재생성 / 새 physical feature 계산은
*수행하지 않는다*. data/step7 caption CSV는 read-only로 참고하였다.

본 문서는 다음에 종속된다 (충돌 시 종속 문서가 우선):

- `reports/step8/step8_caption_evaluation_plan.md` — Step 8 평가 설계
- `reports/step7/step7_caption_generation_prototype.md` — Step 7 / 7-1 결과
- `reports/step6/step6_caption_template_design.md` — caption template 정책
- `reports/step5/step5_caption_policy_design.md` — caption policy
- `data/step7/step7_captions_raw.csv`, `data/step7/step7_captions_zscore.csv` — caption prototype (read-only 입력)

---

## 1. Purpose

| 항목 | 내용 |
|---|---|
| Step 8-1 목적 | Step 8 sampling 전략 / review schema에 따라 review sample CSV 3종을 생성 |
| 본 단계가 *수행하는* 것 | 입력 검증 / 자동 사전점검 지표 산출 / branch별 review sample 생성 / paired review sample 생성 / 보고서 작성 |
| 본 단계가 *수행하지 않는* 것 | human review 실행, reviewer 입력 채우기, raw vs zscore 최종 branch 선택, final wording 확정, caption 재생성 |
| reviewer 입력 컬럼 | 모두 *빈 칸*으로 출력됨 (D1~D6 rating / branch_preference / issue_tags / free_text_comment / reviewer_decision) |
| 결정론성 | `random_state = 42` — 동일 입력에서 sample이 동일 |

---

## 2. Inputs and outputs

| 구분 | 경로 | 모드 |
|---|---|---|
| 입력 | `data/step7/step7_captions_raw.csv` | read-only |
| 입력 | `data/step7/step7_captions_zscore.csv` | read-only |
| 출력 | `data/step8/step8_review_sample_raw.csv` | write (utf-8-sig) |
| 출력 | `data/step8/step8_review_sample_zscore.csv` | write (utf-8-sig) |
| 출력 | `data/step8/step8_review_sample_paired.csv` | write (utf-8-sig) |
| 보조 | `data/step8/step8_metrics.json` | write (utf-8) — 본 보고서의 표 산출 근거 |
| 스크립트 | `scripts/build_step8_review_sample.py` | new |
| 보고서 | `reports/step8/step8_review_sample_summary.md` | 본 문서 |

실행 명령:

```
python scripts/build_step8_review_sample.py
```

---

## 3. Input validation results

| 검증 항목 | raw | zscore |
|---|---|---|
| row count = 9275 | OK | OK |
| sample_id unique | OK | OK |
| split count train / val / test = 6412 / 1436 / 1427 | OK | OK |
| `caption_validation_pass` 모두 True | OK (9275 / 9275) | OK (9275 / 9275) |
| `caption_validation_issues` 모두 `[]` | OK | OK |
| `no_call ↔ class_set == []` 일관성 | OK | OK |
| raw / zscore sample_id set 동일 | OK | OK |

위반 발생 시 `ValueError`로 즉시 실패하도록 설계됨. 본 실행에서는 모든
검증을 통과했다.

---

## 4. Automatic pre-screening metrics

Step 8 §6에 따른 자동 지표 (read-only 집계).

### 4.1 row / unique caption / R13 / validation issue

| 지표 | raw | zscore |
|---|---:|---:|
| row count (M1) | 9275 | 9275 |
| unique caption count (M2) | 65 | 67 |
| validation issue count (M9) | 0 | 0 |
| R13 duplicate / technical wording hits (M10, unique caption 단위) | 0 | 0 |

### 4.2 template_id distribution (M3)

| template_id | raw | zscore |
|---|---:|---:|
| T-CONF-1 | 723 | 725 |
| T-HEDGE-1 | 5881 | 5854 |
| T-HEDGE-3 | 23 | 29 |
| T-LOW-2 | 230 | 254 |
| T-NC-ANCHOR | 25 | 22 |
| T-NC-LOWCONF | 2393 | 2391 |
| (T-CONF-2 / T-HEDGE-2 / T-LOW-1 / T-NC-POSTURE) | 0 | 0 |

### 4.3 caption_confidence_level distribution (M4)

| level | raw | zscore |
|---|---:|---:|
| confident | 723 | 725 |
| hedged | 5904 | 5883 |
| low | 230 | 254 |
| no_call | 2418 | 2413 |

### 4.4 class_set_prediction distribution (M5)

| class_set | raw | zscore |
|---|---:|---:|
| `["C2"]` | 746 | 754 |
| `["C1","C5","C6"]` | 2876 | 2842 |
| `["C1","C5"]` | 10 | 15 |
| `["C1","C6"]` | 309 | 310 |
| `["C5","C6"]` | 294 | 325 |
| `["C3","C4"]` | 1417 | 1386 |
| `["C3","C4","C2"]` | 1205 | 1230 |
| `[]` (no_call) | 2418 | 2413 |

### 4.5 no_call_reason distribution (M6)

| no_call_reason | raw | zscore |
|---|---:|---:|
| `posture_unknown` | 0 | 0 |
| `anchor_driven` | 25 | 22 |
| `low_confidence_no_class_set` | 2393 | 2391 |
| no_call 합계 | 2418 | 2413 |

Step 4 §10의 anchor-driven 분해(25 / 22)와 정확히 일치 — Step 7
generator가 schema output을 재해석하지 않았음을 그대로 보존한다.

### 4.6 caption length distribution (M7)

| 통계 | raw | zscore |
|---|---:|---:|
| min | 38 | 38 |
| p25 | 56 | 56 |
| median | 65 | 65 |
| p75 | 69 | 69 |
| max | 79 | 79 |
| mean | 60.6 | 60.6 |

> 글자 수 단위. outlier는 없다. raw와 zscore의 전체 분포가 거의 동일.

### 4.7 duplicate caption frequency Top 10 (M8)

raw branch Top 10 (caption — row count):

| caption_ko | row count |
|---|---:|
| 현재 신호만으로는 동작 유형을 안정적으로 좁히기 어렵습니다. 무리해서 판단하지 않고 보류합니다. | 2393 |
| 기준 시점 단서를 신뢰하기 어려워, 해당 단서에 의존하는 동작 유형 판단을 보류합니다. | 25 |
| (T-HEDGE-1 다양한 wording — 자세 / class_set / feature phrase 조합으로 다수의 unique caption) | 다양 |

> Top-2가 no_call template (T-NC-LOWCONF / T-NC-ANCHOR)으로 row 수의
> 대부분을 차지한다 (no_call 2418 + anchor-driven 25 = 2443 ≈ 26%).
> Top 20 전체는 `data/step8/step8_metrics.json`의
> `branch_metrics.{raw,zscore}.duplicate_caption_top20`에서 확인 가능.

### 4.8 raw vs zscore equality (M11 / M12)

| 지표 | 값 |
|---|---:|
| joined row count (sample_id 기준 inner) | 9275 |
| same caption row count | 8036 |
| different caption row count | **1239** |
| equality rate (M11) | **0.8664** (≈ 86.6%) |
| differing sample count (M12) | 1239 |

### 4.9 raw vs zscore caption 차이 유형 (전체 9275행)

| `branch_difference_type` | row count | 비율 |
|---|---:|---:|
| `same_caption` | 8036 | 86.64% |
| `different_class_set` | 701 | 7.56% |
| `multiple_differences` | 528 | 5.69% |
| `different_feature_phrase` | 10 | 0.11% |
| `different_confidence_level` | 0 | 0.00% |
| `different_no_call` | 0 | 0.00% |
| `different_caption_same_schema` | 0 | 0.00% |

해석:
- 87%는 두 branch가 *동일* caption — Step 4 §6 / Step 7 §12 분포가
  거의 동일하다는 점과 일관.
- 차이 1239건의 대부분은 `class_set_prediction` 차이(701) 또는
  multiple-differences(528).
- `different_no_call` 0 / `different_confidence_level` 0 — branch가
  동일 sample에 대해 *no_call 여부*와 *confidence level*은 *모두 동일*.
  이는 Step 4 generator가 branch 간 anchor-driven no_call 분해(25 vs 22)에서
  *서로 다른 sample_id에 대해* no_call을 발화한다는 사실과 정합한다 — 즉,
  같은 sample_id에서 branch 간 level이 바뀌는 일은 *발생하지 않는다*.

---

## 5. Review sample construction

### 5.1 sample 수 / unique caption coverage

| 항목 | raw sample | zscore sample | paired sample |
|---|---:|---:|---:|
| target max size | 150 | 150 | 150 |
| 실제 row 수 | 140 | 142 | 150 |
| unique caption 커버 | 65 / 65 (100%) | 67 / 67 (100%) | — |
| 결정론 seed | 42 | 42 | 42 (+ same-sample sub-seed 43) |

> sample 수가 정확히 150에 도달하지 *않은* 이유: Phase 1 unique
> caption(65 / 67) 후 Phase 2 quota 합산이 max_size 이하에서
> 종료되었기 때문이다. unique caption 커버리지 달성이 우선이라는
> Step 8 §7.2 우선순위 #1을 그대로 따랐다.

### 5.2 raw branch sample coverage (n=140)

| 축 | 분포 |
|---|---|
| template_id | T-HEDGE-1 / T-LOW-2 / T-NC-LOWCONF / T-NC-ANCHOR / T-CONF-1 / T-HEDGE-3 (발화된 6개 모두 포함) |
| caption_confidence_level | hedged 90 / no_call 25 / low 17 / confident 8 |
| class_set_prediction | `["C3","C4"]` 27 / `["C3","C4","C2"]` 26 / `[]` 25 / `["C1","C5","C6"]` 24 / `["C5","C6"]` 12 / `["C1","C6"]` 11 / `["C2"]` 11 / `["C1","C5"]` 4 (8개 whitelist 모두 포함) |
| no_call_reason | low_confidence_no_class_set 15 / anchor_driven 10 / posture_unknown 0 (데이터에 없음) |
| posture | CA 65 / SA 45 / HW 30 |
| split | train 84 / test 34 / val 22 |
| anchor_unreliable=True row 수 | 40 |
| `["C3","C4","C2"]` row 수 | 26 |
| `["C3","C4"]` (no C2) row 수 | 27 |
| test split row 수 | 34 |

### 5.3 zscore branch sample coverage (n=142)

| 축 | 분포 |
|---|---|
| template_id | T-HEDGE-1 / T-NC-LOWCONF / T-LOW-2 / T-NC-ANCHOR / T-CONF-1 / T-HEDGE-3 (발화된 6개 모두 포함) |
| caption_confidence_level | hedged 89 / no_call 29 / low 17 / confident 7 |
| class_set_prediction | `["C3","C4","C2"]` 32 / `[]` 29 / `["C1","C5","C6"]` 24 / `["C3","C4"]` 24 / `["C1","C6"]` 11 / `["C5","C6"]` 10 / `["C2"]` 9 / `["C1","C5"]` 3 (8개 whitelist 모두 포함) |
| no_call_reason | low_confidence_no_class_set 19 / anchor_driven 10 / posture_unknown 0 |
| posture | CA 59 / SA 53 / HW 30 |
| split | train 90 / test 33 / val 19 |
| anchor_unreliable=True row 수 | 42 |
| `["C3","C4","C2"]` row 수 | 32 |
| `["C3","C4"]` (no C2) row 수 | 24 |
| test split row 수 | 33 |

### 5.4 paired review sample (n=150)

| `branch_difference_type` | sample row count |
|---|---:|
| `different_class_set` | 71 |
| `multiple_differences` | 49 |
| `same_caption` | 30 |
| `different_feature_phrase` | 0 (전체 9275 중 10건만 존재 — 본 sample의 random subset에서는 미선택) |
| 합계 | 150 |

> sampling 정책: differing 우선 120 + same 우선 30 (Step 8 §7).
> differing 9275 중 1239건 / same 8036건 풀에서 random_state=42로 추출.

### 5.5 sampling 우선순위 충족도

| 우선순위 (Step 8 §7.2) | raw | zscore |
|---|---|---|
| #1 unique caption 대표 | 65/65 (100%) | 67/67 (100%) |
| #2 high-frequency template | T-HEDGE-1 / T-NC-LOWCONF / T-CONF-1 모두 포함 | 동일 |
| #3 rare template | T-HEDGE-3 / T-NC-ANCHOR 포함 | 동일 |
| #4 no_call 사유 | low_confidence_no_class_set + anchor_driven 모두 포함 | 동일 |
| #5 `["C3","C4","C2"]` 충분 포함 | 26 rows | 32 rows |
| #6 `["C3","C4"]` (no C2) 충분 포함 | 27 rows | 24 rows |
| #7 posture 균형 | SA / CA / HW 모두 포함 | 동일 |
| #8 split 균형 + test 우선 | train / val / test 모두 포함, test 34 / 33 | 동일 |
| #9 paired (raw vs zscore differing) | paired CSV에서 differing 120 우선 추출 | 동일 |

> `posture_unknown` no_call 사유는 본 데이터에 *존재하지 않으므로* (Step
> 4 §10 / Step 7 §9), sample에도 0건이며 이는 정상이다.

---

## 6. Review CSV schema

### 6.1 branch sample 컬럼 (raw / zscore CSV 공통, 32개)

#### 6.1.1 carry / meta (22 컬럼)

| 컬럼 | 출처 / 의미 |
|---|---|
| `review_id` | 신규 — `{branch}-{4자리 idx}` |
| `branch` | `raw` 또는 `zscore` |
| `sample_id` | Step 7 caption CSV |
| `split` | Step 7 caption CSV |
| `posture` | Step 7 caption CSV |
| `class_id_for_audit_only` | Step 7 `class_id` 재명명 — *audit 전용* (§6.3 참고) |
| `caption_ko` | Step 7 caption CSV — 평가 대상 |
| `template_id` | Step 7 caption CSV |
| `class_set_prediction` | Step 7 caption CSV |
| `uncertainty_flags` | Step 7 caption CSV |
| `caption_confidence_level` | Step 7 caption CSV |
| `no_call` | Step 7 caption CSV |
| `no_call_reason` | Step 7 caption CSV |
| `main_feature_phrase` | Step 7 caption CSV |
| `uncertainty_phrase` | Step 7 caption CSV |
| `anchor_reliability_bin` | 신규 derive — `high (≥0.75)` / `mid_high (≥0.50)` / `mid_low (≥0.25)` / `low (<0.25)` |
| `anchor_unreliable` | 신규 derive — `uncertainty_flags`에 `anchor_unreliable` 포함 여부 |
| `feature_phrase_source` | Step 7 caption CSV |
| `feature_phrase_fallback_used` | Step 7 caption CSV |
| `caption_validation_pass` | Step 7 caption CSV (모두 True) |
| `caption_length_chars` | 신규 derive — `caption_ko` 글자 수 |
| `review_priority_tag` | 신규 derive — JSON list (§6.2) |

#### 6.1.2 reviewer 입력 (10 컬럼, 모두 빈 칸으로 출력)

| 컬럼 | 타입 | 값 / 범위 |
|---|---|---|
| `rating_policy_compliance` (D1) | int | 1~5 |
| `rating_schema_faithfulness` (D2) | int | 1~5 |
| `rating_naturalness` (D3) | int | 1~5 |
| `rating_non_overclaiming` (D4) | int | 1~5 |
| `rating_usefulness` (D5) | int | 1~5 |
| `rating_feature_phrase_appropriateness` (D6) | int | 1~5 |
| `branch_preference` (D7) | enum | `raw` / `zscore` / `tie` / `cannot_judge` |
| `issue_tags` | JSON list | Step 8 §9 closed vocabulary (T1~T18) |
| `free_text_comment` | string | reviewer 자유 기술 |
| `reviewer_decision` | enum | `accept` / `accept_with_minor_edit` / `revise_template` / `revise_policy` / `reject` |

### 6.2 review_priority_tag closed vocabulary

| tag | 부여 조건 |
|---|---|
| `unique_caption` | Phase 1에서 unique caption 대표로 선택된 row |
| `high_frequency_template` | template_id ∈ {T-HEDGE-1, T-NC-LOWCONF, T-CONF-1} |
| `rare_template` | template_id ∈ {T-HEDGE-3, T-NC-ANCHOR, T-LOW-1, T-LOW-2} |
| `no_call` | `no_call == True` |
| `anchor_driven_no_call` | `no_call_reason == "anchor_driven"` |
| `c3c4c2` | `class_set_prediction == ["C3","C4","C2"]` |
| `c3c4_without_c2` | `class_set_prediction == ["C3","C4"]` |
| `anchor_unreliable` | `uncertainty_flags`에 `anchor_unreliable` 포함 |
| `branch_different` | 같은 sample_id에서 raw vs zscore caption_ko가 다름 (§4.8 differing 1239건) |
| `test_split` | `split == "test"` |

복수 tag가 동시에 부여될 수 있다 (JSON list로 저장).

### 6.3 paired sample 컬럼 (24 컬럼)

| 그룹 | 컬럼 |
|---|---|
| meta | `review_id`, `sample_id`, `split`, `posture`, `class_id_for_audit_only` |
| raw 측 | `raw_caption_ko`, `raw_template_id`, `raw_class_set_prediction`, `raw_caption_confidence_level`, `raw_no_call`, `raw_no_call_reason`, `raw_main_feature_phrase` |
| zscore 측 | `zscore_caption_ko`, `zscore_template_id`, `zscore_class_set_prediction`, `zscore_caption_confidence_level`, `zscore_no_call`, `zscore_no_call_reason`, `zscore_main_feature_phrase` |
| 비교 derive | `captions_equal`, `branch_difference_type` |
| reviewer 입력 (빈 칸) | `reviewer_branch_preference`, `paired_issue_tags`, `paired_free_text_comment` |

### 6.4 `class_id_for_audit_only` 사용 주의 (blinded-first 원칙)

| 항목 | 정책 |
|---|---|
| 기본 review mode | **blinded** — reviewer는 `class_id_for_audit_only` 컬럼을 *보지 않거나 mask*한다 |
| audit mode 사용 시점 | blinded review 결과 확정 *후*, class 일치 점검에 한정 |
| 컬럼 carry 이유 | 후속 audit-mode 분석을 위해 *값은 carry*하되, 컬럼명에 `_for_audit_only`를 명시하여 caption 평가 시 사용하지 *않도록* 표시 |
| reviewer 지침 | "정답 class를 맞히는지보다 schema output을 과장 없이 잘 표현하는지 평가" (Step 8 §12 I1) |
| caption 생성 근거 | `class_id`는 caption 생성 입력이 *아님* — Step 7 generator는 `class_id`를 사용하지 않았다 (Step 6 §3 / §3.1) |

본 단계에서는 컬럼을 *값과 함께* carry한다. blinded review 운영(컬럼
mask) 또는 audit review 운영(컬럼 노출)은 reviewer 환경에서 결정한다.

---

## 7. Limitations

| # | 한계 | 영향 / 완화 |
|---|---|---|
| L1 | sampling은 review용이며 *전체 성능 추정*이 아니다 | sample size(140 / 142 / 150)는 모집단 분포 추정이 아닌 review 품질 점검 용 |
| L2 | reviewer 점수는 *아직 비어 있다* | 본 단계는 sample 생성만 수행 — Step 8-2 / 9에서 채움 |
| L3 | raw vs zscore branch preference는 *아직 결정하지 않는다* | D7 reviewer 입력 누적 후 후속 단계에서 결정 (Step 5 §9 #7 / Step 6 §12 C9 정책 유지) |
| L4 | unique caption 중심 sampling은 *빈도 가중* 사용자 경험과 다를 수 있다 | branch sample은 unique caption 100% 커버를 우선 — 빈도 가중 review가 필요하면 별도 sample을 후속에서 추가 가능 |
| L5 | `class_id_for_audit_only`는 audit mode에서만 해석해야 한다 | §6.4 blinded-first 원칙 준수 |
| L6 | `posture_unknown` no_call 사유는 본 데이터에 0건 — sample에도 0건 | Step 4 §10 / Step 7 §9와 정합. 후속 데이터에서 자세 라벨 결손이 발생하면 sampling 시 별도 quota 필요 |
| L7 | `T-CONF-2` / `T-HEDGE-2` / `T-LOW-1` / `T-NC-POSTURE` 4개 template은 본 데이터에서 발화 0건 | sample에도 등장하지 않음 — 후보 template은 *조건부 발화*가 정상 (Step 7 §8 / §13 O3) |
| L8 | paired sample에서 `different_feature_phrase`는 0건 추출됨 (전체 9275 중 10건만 존재) | feature phrase 차이 review가 필요하면 explicit quota를 후속에 추가 가능 |

---

## 8. Next step

| 단계 | 내용 | 비고 |
|---|---|---|
| Step 8-2 (or pilot human review) | reviewer가 sample CSV의 `rating_*` / `branch_preference` / `issue_tags` / `free_text_comment` / `reviewer_decision`을 채움 | blinded mode 우선 (§6.4) |
| Step 9 (or 그 이후) | review 결과 분석 / wording refinement / branch decision 후보 검토 | Step 8 §11 acceptance gate (A1~A9) 평가 |
| 본 단계 비-결정 | raw vs zscore final branch / final wording / threshold final commit / model retraining | 본 단계 범위 외 |

본 단계 sample CSV 3종은 *prototype review용 입력*이며, 어떤 final
deployment commit도 포함하지 않는다.

---

*본 문서는 Step 8-1 review sample extraction summary로 작성되었다. 실제
human review / reviewer 입력 채우기 / raw vs zscore 최종 branch 선택 /
final wording 확정 / caption 재생성 / 새 모델 학습 / threshold
recalibration / schema output 재생성 / 새 physical feature 계산은
*수행하지 않았다*. Step 1 / Step 2 / Step 3 / Step 4 / Step 5 / Step 6 /
Step 7 / Step 8 산출물은 수정되지 않았다. data/step7 caption CSV는
read-only 참고로만 사용했다. sample CSV의 reviewer 입력 컬럼은 모두
빈 칸으로 출력되었다.*
