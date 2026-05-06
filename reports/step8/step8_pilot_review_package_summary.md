# Step 8-2 — Pilot Review Package Summary

이 문서는 Step 8-1 review sample CSV를 **blinded review용 패키지**로
변환한 결과를 요약한다. 본 단계는 *패키지 준비*만 수행하며, 실제 human
review / reviewer rating 입력 / caption 수정 / final branch 선택 / final
wording 확정은 *수행하지 않는다*.

본 문서는 다음에 종속된다 (충돌 시 종속 문서가 우선):

- `reports/step8/step8_caption_evaluation_plan.md` — Step 8 평가 설계
- `reports/step8/step8_review_sample_summary.md` — Step 8-1 sample
- `reports/step8/step8_pilot_review_instructions.md` — Step 8-2 reviewer 지침
- `reports/step7/step7_caption_generation_prototype.md` — Step 7 / 7-1 결과

---

## 1. Purpose

| 항목 | 내용 |
|---|---|
| Step 8-2 목적 | Step 8-1 sample(raw 140 / zscore 142 / paired 150)을 blinded review에 사용 가능한 형태로 변환 |
| 본 단계가 *수행하는* 것 | 입력 검증 / blinded CSV 3종 작성 (`class_id_for_audit_only` 제거) / audit key CSV 3종 작성 (별도 보관) / reviewer 지침 / 본 요약 |
| 본 단계가 *수행하지 않는* 것 | reviewer rating 입력 / caption 수정 / caption 재생성 / 새 모델 학습 / threshold recalibration / schema output 재생성 / 새 physical feature 계산 / raw vs zscore 최종 branch 확정 / final wording 확정 |
| reviewer 입력 컬럼 | 모두 *빈 칸*으로 carry — 빈 칸 검증을 disk roundtrip까지 확인 |
| 결정론성 | 본 단계는 sampling을 수행하지 않으며, 입력 row 그대로 cols만 필터링하므로 결정론적 |

---

## 2. Inputs and outputs

| 구분 | 경로 | 모드 | 비고 |
|---|---|---|---|
| 입력 | `data/step8/step8_review_sample_raw.csv` | read-only | 140 rows |
| 입력 | `data/step8/step8_review_sample_zscore.csv` | read-only | 142 rows |
| 입력 | `data/step8/step8_review_sample_paired.csv` | read-only | 150 rows |
| 출력 (blinded) | `data/step8/step8_pilot_review_raw_blinded.csv` | write (utf-8-sig) | reviewer 배포용 |
| 출력 (blinded) | `data/step8/step8_pilot_review_zscore_blinded.csv` | write (utf-8-sig) | reviewer 배포용 |
| 출력 (blinded) | `data/step8/step8_pilot_review_paired_blinded.csv` | write (utf-8-sig) | reviewer 배포용 |
| 출력 (audit) | `data/step8/step8_pilot_review_audit_key_raw.csv` | write (utf-8-sig) | **운영자 전용** |
| 출력 (audit) | `data/step8/step8_pilot_review_audit_key_zscore.csv` | write (utf-8-sig) | **운영자 전용** |
| 출력 (audit) | `data/step8/step8_pilot_review_audit_key_paired.csv` | write (utf-8-sig) | **운영자 전용** |
| 스크립트 | `scripts/build_step8_pilot_review_pack.py` | new | |
| 보고서 | `reports/step8/step8_pilot_review_instructions.md` | new | reviewer 지침 |
| 보고서 | `reports/step8/step8_pilot_review_package_summary.md` | new | 본 문서 |

실행 명령:

```
python scripts/build_step8_pilot_review_pack.py
```

---

## 3. Input validation results

| 검증 항목 | raw input | zscore input | paired input |
|---|---|---|---|
| row count | 140 (= expected) | 142 (= expected) | 150 (= expected) |
| 필수 컬럼 존재 | OK (32 cols) | OK (32 cols) | OK (24 cols) |
| `review_id` unique | OK | OK | OK |
| `sample_id` 결손 없음 | OK | OK | OK |
| `caption_validation_pass` 모두 True | OK | OK | (해당 컬럼 없음) |
| reviewer 입력 컬럼이 빈 칸 | OK (10 cols) | OK (10 cols) | OK (3 cols) |
| paired `captions_equal` ↔ `branch_difference_type` 일관성 | — | — | OK (불일치 0 rows) |

위반 발생 시 `ValueError`로 즉시 실패하도록 설계됨. 본 실행에서는 모든
검증을 통과했다.

---

## 4. Blinded package results

### 4.1 row / col

| 파일 | input cols | blinded output cols | reviewer-facing 파일에서 제외된 컬럼 |
|---|---:|---:|---|
| `step8_pilot_review_raw_blinded.csv` (140 rows) | 32 | 30 | `class_id_for_audit_only`, `caption_validation_pass` |
| `step8_pilot_review_zscore_blinded.csv` (142 rows) | 32 | 30 | `class_id_for_audit_only`, `caption_validation_pass` |
| `step8_pilot_review_paired_blinded.csv` (150 rows) | 24 | 23 | `class_id_for_audit_only` |

설명:

- **raw / zscore**: input 32 cols → blinded output 30 cols. reviewer-facing blinded 파일에서는 `class_id_for_audit_only`와 `caption_validation_pass` 두 컬럼을 제외했다.
  - `class_id_for_audit_only`는 §5의 audit key 파일로 *분리*했다 (blinded-first 원칙).
  - `caption_validation_pass`는 본 단계 §3 입력 검증에서 *모든 row가 True임을 이미 확인*했으므로 reviewer-facing 파일에서는 제외했다 (reviewer가 평가할 추가 정보가 아님).
- **paired**: input 24 cols → blinded output 23 cols. `class_id_for_audit_only` 1개만 audit key로 분리했고, paired CSV는 `caption_validation_pass`를 carry하지 않으므로 추가 제외 컬럼은 없다.
- **blinded-first 원칙은 유지**된다 — reviewer는 정답 class를 보지 않는다.
- **reviewer 입력 컬럼은 모두 빈 칸으로 유지**된다 (§4.2에서 disk roundtrip까지 재확인).

### 4.2 reviewer 입력 컬럼이 비어 있음

| 파일 | 비어 있어야 할 컬럼 | 비어 있는지 (disk roundtrip 재확인) |
|---|---|---|
| raw blinded | `rating_policy_compliance`, `rating_schema_faithfulness`, `rating_naturalness`, `rating_non_overclaiming`, `rating_usefulness`, `rating_feature_phrase_appropriateness`, `branch_preference`, `issue_tags`, `free_text_comment`, `reviewer_decision` (10개) | **모두 빈 칸 (True)** |
| zscore blinded | (raw와 동일 10개) | **모두 빈 칸 (True)** |
| paired blinded | `reviewer_branch_preference`, `paired_issue_tags`, `paired_free_text_comment` (3개) | **모두 빈 칸 (True)** |

스크립트는 (a) DataFrame에서 reviewer 컬럼을 빈 문자열로 강제 설정하고,
(b) 저장 *후* 디스크에서 다시 읽어 비어 있음을 재확인한다. 어느 한
컬럼이라도 비어 있지 않으면 `ValueError`로 실패하여 디스크에 일관되지
않은 패키지를 남기지 *않는다*.

### 4.3 컬럼 순서 (Step 8-2 §2~§3에 정의된 순서 그대로)

raw / zscore blinded:

```
review_id, branch, sample_id, split, posture,
caption_ko, template_id, class_set_prediction, uncertainty_flags,
caption_confidence_level, no_call, no_call_reason,
main_feature_phrase, uncertainty_phrase,
anchor_reliability_bin, anchor_unreliable,
feature_phrase_source, feature_phrase_fallback_used,
caption_length_chars, review_priority_tag,
rating_policy_compliance, rating_schema_faithfulness, rating_naturalness,
rating_non_overclaiming, rating_usefulness,
rating_feature_phrase_appropriateness,
branch_preference, issue_tags, free_text_comment, reviewer_decision
```

paired blinded:

```
review_id, sample_id, split, posture,
raw_caption_ko, zscore_caption_ko,
raw_template_id, zscore_template_id,
raw_class_set_prediction, zscore_class_set_prediction,
raw_caption_confidence_level, zscore_caption_confidence_level,
raw_no_call, zscore_no_call,
raw_no_call_reason, zscore_no_call_reason,
raw_main_feature_phrase, zscore_main_feature_phrase,
captions_equal, branch_difference_type,
reviewer_branch_preference, paired_issue_tags, paired_free_text_comment
```

---

## 5. Audit key files

### 5.1 row / col

| 파일 | row count | col count | 컬럼 |
|---|---:|---:|---|
| `step8_pilot_review_audit_key_raw.csv` | 140 | 4 | `review_id`, `branch`, `sample_id`, `class_id_for_audit_only` |
| `step8_pilot_review_audit_key_zscore.csv` | 142 | 4 | `review_id`, `branch`, `sample_id`, `class_id_for_audit_only` |
| `step8_pilot_review_audit_key_paired.csv` | 150 | 3 | `review_id`, `sample_id`, `class_id_for_audit_only` |

### 5.2 사용 정책

| 항목 | 정책 |
|---|---|
| 배포 대상 | **reviewer에게 배포하지 않음** — 운영자 전용 |
| 사용 시점 | blinded review 완료 *후* audit-mode 분석에 한정 |
| 분리 보관 권장 | blinded CSV와 *다른* 폴더 / 다른 권한으로 보관 |
| caption 생성 근거 여부 | `class_id`는 caption 생성 입력이 *아님* (Step 7 generator는 사용하지 않음 — Step 6 §3 / §3.1) |
| blinded-first 원칙 | reviewer 평가 단계에서는 정답 class를 *보지 않는다* — D2 (schema faithfulness)와 class correctness가 섞이지 않도록 |

audit key는 `review_id` 기준으로 blinded review 결과 CSV와 join 가능하다.

---

## 6. Reviewer instruction summary

`reports/step8/step8_pilot_review_instructions.md`의 핵심 항목 요약.
상세는 해당 문서 참고.

| 항목 | 요약 |
|---|---|
| Rating dimensions | D1~D6 (1~5점 Likert) — `rating_policy_compliance` 등 6개 컬럼 |
| Automatic fail rules | F1~F7 — 발견 즉시 D1 = 1, critical tag 부착 |
| Issue tags | T1~T18 closed vocabulary, JSON list 입력 |
| Critical tags | T1 (`policy_violation`) / T10 (`missing_C2_in_C3C4C2`) / T12 (`anchor_phrase_violation`) — 1건이라도 발생 시 D1 = 1 강제 |
| reviewer_decision | `accept` / `accept_with_minor_edit` / `revise_template` / `revise_policy` / `reject` 중 1택 |
| Paired branch preference | `raw` / `zscore` / `tie` / `cannot_judge` — 본 단계에서 *final 선택 아님* |
| What not to do | audit key 열람 / 정답 class 추정 / carry 컬럼 수정 / `caption_ko` 직접 수정 / branch 최종 선택 / 모델·threshold·schema·feature 변경 |

---

## 7. Limitations

| # | 한계 | 영향 / 완화 |
|---|---|---|
| L1 | review가 아직 *수행되지 않았다* | 본 단계는 패키지 준비만 — Step 8-3 / 9에서 review 결과 누적 |
| L2 | rating은 모두 *비어 있다* | reviewer가 채울 때까지 평균 / 분포 산출 불가 |
| L3 | branch preference는 아직 *모른다* | D7 누적 후 후속 단계에서 검토 (Step 5 §9 #7 / Step 6 §12 C9 정책 유지) |
| L4 | audit key 파일은 존재하지만 blinded review 완료 *전*에는 사용해서는 안 된다 | 운영자가 별도 보관 / 권한 분리 |
| L5 | final wording / final branch decision은 본 단계 결과로 결정되지 않는다 | 본 단계 범위 외 |
| L6 | reviewer 인원 / inter-rater agreement 설계는 본 문서에 포함되지 않음 | 후속 운영 단계에서 결정 (pilot 결과 반영) |
| L7 | `posture_unknown` no_call 사유는 본 데이터에 0건 — sample / blinded CSV에도 0건 | Step 4 §10 / Step 7 §9와 정합. 후속 데이터에서 자세 라벨 결손이 발생하면 별도 sample 필요 |

---

## 8. Next step

| 단계 | 내용 | 비고 |
|---|---|---|
| Step 8-3 (or pilot review completion) | reviewer가 blinded CSV의 rating / issue_tags / decision (paired는 branch_preference)을 *직접 채움* | 본 문서 / `step8_pilot_review_instructions.md` 지침을 따름 |
| Step 9 (또는 그 이후) | 채워진 review CSV를 read-only로 읽어 D1~D6 평균 / critical issue rate / branch preference / acceptance gate (Step 8 §11 A1~A9) 분석 | `scripts/analyze_step8_pilot_review.py` 같은 분석 스크립트 작성 가능 |
| audit-mode 분석 | blinded review 완료 *후* audit key를 join하여 class correctness 점검 (참고용) | caption faithfulness와 분리 |
| 본 단계 비-결정 | raw vs zscore final branch / final wording / threshold final commit / model retraining | 본 단계 범위 외 |

---

*본 문서는 Step 8-2 pilot review *package summary*로 작성되었다. 실제
human review / reviewer rating 입력 / caption 수정 / caption 재생성 /
새 모델 학습 / threshold recalibration / schema output 재생성 / 새
physical feature 계산 / raw vs zscore 최종 branch 확정 / final wording
확정은 *수행하지 않았다*. Step 1 / Step 2 / Step 3 / Step 4 / Step 5 /
Step 6 / Step 7 / Step 8 / Step 8-1 산출물은 수정되지 않았다. 입력 CSV는
read-only로 사용했고, blinded CSV에서 `class_id_for_audit_only`는
제거되었으며 reviewer 입력 컬럼은 모두 빈 칸으로 출력되었다.*
