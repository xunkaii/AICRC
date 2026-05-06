# Step 8-2 — Pilot Human Review Instructions

이 문서는 Step 8 평가 계획(`reports/step8/step8_caption_evaluation_plan.md`)
및 Step 8-1 review sample(`reports/step8/step8_review_sample_summary.md`)에
근거한 **pilot human review reviewer 지침**이다. 본 단계는 reviewer가
blinded review CSV를 *직접 채우는* 단계이며, 본 문서 자체는 review
*지침*만 정의한다.

---

## 1. Purpose

| 항목 | 내용 |
|---|---|
| 목적 | reviewer가 Step 8-1 sample을 blinded mode에서 읽고 caption 품질을 평가하도록 가이드 |
| 평가 관점 | caption이 *정답 class를 맞히는지*보다, *schema output을 과장 없이* 잘 표현하는지 |
| no_call 해석 | 실패가 아니라 *보수적 판단 보류* — class를 추정하지 않은 것이 정상 |
| `class_set` 해석 | `["C3","C4","C2"]` 등은 *일부러* 단일 class로 좁히지 않은 것 — 모호성 유지가 정책 |
| raw vs zscore | **최종 선택 전 비교 중** — 본 단계에서 final branch를 선택하지 않음 |
| reviewer 입력 | rating / issue_tags / free_text_comment / reviewer_decision (paired는 branch_preference) — 지금까지 빈 칸으로 출력됨 |

---

## 2. Files to review

| 파일 | 내용 | 비고 |
|---|---|---|
| `data/step8/step8_pilot_review_raw_blinded.csv` | raw branch 140 rows | reviewer가 채움 |
| `data/step8/step8_pilot_review_zscore_blinded.csv` | zscore branch 142 rows | reviewer가 채움 |
| `data/step8/step8_pilot_review_paired_blinded.csv` | raw vs zscore paired 150 rows | reviewer가 채움 |
| `data/step8/step8_pilot_review_audit_key_*.csv` (3개) | **reviewer에게 제공하지 않음** — audit mode 분석 전용 | 운영자만 보관 |

원칙:
- blinded CSV에는 `class_id_for_audit_only`가 *제거*되어 있다 — reviewer는 정답 class를 보지 못한다.
- audit key 파일은 별도 폴더 / 별도 권한으로 보관해 reviewer 노출을 차단한다.
- blinded review가 끝난 *후*에만 audit key를 사용한다 (운영자 분석 단계).

---

## 3. Rating dimensions (D1~D6)

각 row에 대해 다음 6개 컬럼을 1~5점으로 채운다.

| 컬럼 | 차원 | 평가 질문 |
|---|---|---|
| `rating_policy_compliance` | D1 | Step 5 / Step 6 정책을 어기지 않는가 (no_call에서 class 추정 / `["C3","C4","C2"]`에서 C2 누락 / `anchor_unreliable`에서 억제 표현 누락 등) |
| `rating_schema_faithfulness` | D2 | `class_set_prediction` / `uncertainty_flags` / `no_call_reason`이 caption에 *과장 / 재해석 없이* 반영되는가 |
| `rating_naturalness` | D3 | 한국어 문장이 자연스러운가 (분석 보고서처럼 딱딱하지 않고, 반복 / 중복 표현이 없는가) |
| `rating_non_overclaiming` | D4 | confidence level에 맞게 조심스럽게 표현하는가 (`confident`에서도 과도 단정 금지, `low` / `no_call`에서 무리한 판단 금지) |
| `rating_usefulness` | D5 | 사용자 / 연구자가 이해할 수 있는 정보를 주는가 (너무 일반적 / 빈 문장처럼 느껴지지 않는가) |
| `rating_feature_phrase_appropriateness` | D6 | `main_feature_phrase`가 caption 맥락과 맞는가, `anchor_unreliable`일 때 적절히 억제되었는가, heuristic을 measurement처럼 표현하지 않는가 |

### 3.1 점수 기준 (D1~D6 공통)

| 점수 | 기준 |
|---|---|
| 5 | 문제 없음 / 매우 적절 |
| 4 | 사소한 표현 문제만 있음 |
| 3 | 의미는 맞지만 어색하거나 정보성이 약함 |
| 2 | 정책 또는 schema 충실성에 *가까운* 문제가 있음 |
| 1 | 명백한 정책 위반 또는 misleading caption |

---

## 4. Automatic fail rules

다음 조건 중 *하나라도* 발견되면 즉시 D1 = 1로 기록하고 `policy_violation`
또는 §5의 critical tag를 `issue_tags`에 추가한다. `free_text_comment`에
어떤 위반인지 한 줄로 기록한다.

| # | 조건 |
|---|---|
| F1 | `no_call == True`인 caption_ko에 어떤 class 추정 표현이 있음 |
| F2 | `class_set_prediction == ["C3","C4","C2"]`인 caption_ko에 C2가 빠져 있음 |
| F3 | `class_set_prediction == ["C3","C4"]`인 caption_ko에 C2가 들어 있음 |
| F4 | `anchor_unreliable == True`인데 caption에 depth / recovery / bottom-transition 직접 표현 |
| F5 | `lateral_proxy_gyro` / `knee valgus` / `무릎 각도` / `무릎 외반` 등 §6 §7 금지 표현 |
| F6 | `posterior 확률` / `threshold` 후보 이름 / `pred_argmax_debug` 등이 노출됨 |
| F7 | "확실히" / "100%" / "정답" / "모델 실패" / "데이터 오류" 같은 표현이 있음 |

---

## 5. Issue tags (closed vocabulary)

`issue_tags` 컬럼에 JSON list로 입력한다.

| # | tag | 의미 | criticality |
|---|---|---|---|
| T1 | `policy_violation` | Step 5 / Step 6 정책 위반 | **critical** |
| T2 | `schema_mismatch` | schema output을 caption이 잘못 표현 | high |
| T3 | `no_call_too_vague` | no_call 사유 정보가 부족 | medium |
| T4 | `no_call_too_strong` | no_call이 시스템 결함처럼 들림 | high |
| T5 | `overclaiming` | confidence level 대비 과도한 단정 | high |
| T6 | `too_technical` | 분석 용어 / 변수명 노출 | medium |
| T7 | `too_generic` | 너무 일반적이라 정보가 빈약 | medium |
| T8 | `awkward_korean` | 한국어 자연성 부족 | low |
| T9 | `repeated_wording` | 중복 / 반복 표현 | low |
| T10 | `missing_C2_in_C3C4C2` | `["C3","C4","C2"]` caption에서 C2 누락 | **critical** |
| T11 | `class_set_not_preserved` | class_set 멤버가 누락 / 치환됨 | high |
| T12 | `anchor_phrase_violation` | `anchor_unreliable`에서 depth / recovery / bottom-transition 직접 표현 | **critical** |
| T13 | `feature_phrase_mismatch` | feature phrase가 맥락 / level / 자세와 맞지 않음 | medium |
| T14 | `useful` | 정보성이 명확함 (긍정) | — |
| T15 | `acceptable` | 사소한 문제 외에는 받아들일 만함 (긍정) | — |
| T16 | `needs_minor_wording` | 작은 wording 다듬기만 필요 | low |
| T17 | `needs_template_revision` | template 자체 재설계 필요 | high |
| T18 | `needs_policy_revision` | Step 5 / Step 6 정책 자체 재검토 필요 | high |

복수 tag 가능. JSON list 형태로 입력 — 예:

```
["awkward_korean","needs_minor_wording"]
```

**critical** tag (T1 / T10 / T12) 1건이라도 입력되면 D1 = 1로 강제한다.

---

## 6. reviewer_decision

각 row의 최종 결정. 다음 중 *하나*를 입력한다.

| 값 | 의미 |
|---|---|
| `accept` | 그대로 사용 가능 |
| `accept_with_minor_edit` | 사소한 wording 다듬기 후 사용 |
| `revise_template` | 같은 sample도 template 재설계 후 다시 보아야 함 |
| `revise_policy` | Step 5 / Step 6 정책 자체 재검토 필요 |
| `reject` | 현 상태로 사용 불가 — critical issue 동반 가능 |

---

## 7. Paired review instructions

`step8_pilot_review_paired_blinded.csv`에서는 같은 `sample_id`에 대한
raw caption(`raw_caption_ko`)과 zscore caption(`zscore_caption_ko`)을
*나란히* 비교한다.

| 컬럼 | 입력 |
|---|---|
| `reviewer_branch_preference` | `raw` / `zscore` / `tie` / `cannot_judge` 중 하나 |
| `paired_issue_tags` | §5 vocabulary에서 (선택, JSON list) |
| `paired_free_text_comment` | 어느 caption이 어떤 점에서 더 적절한지 한 줄 |

| `reviewer_branch_preference` 값 | 의미 |
|---|---|
| `raw` | raw caption이 더 적절 |
| `zscore` | zscore caption이 더 적절 |
| `tie` | 차이 미미 / 동등 |
| `cannot_judge` | 판단 보류 |

주의:
- 본 단계에서 final branch를 *선택하지 않는다*. branch_preference는
  **후속 분석용 의견 누적**이다 (Step 5 §9 #7 / Step 6 §12 C9 정책 유지).
- `branch_difference_type` 컬럼에는 두 caption이 어느 차원에서 다른지
  자동 분류된 라벨(`same_caption` / `different_class_set` /
  `different_confidence_level` / `different_no_call` /
  `different_feature_phrase` / `different_caption_same_schema` /
  `multiple_differences`)이 들어 있다 — 비교 시 참고.

---

## 8. Review order (권장)

| 단계 | 내용 |
|---|---|
| 1 | raw blinded CSV 20~30행으로 scoring 기준 *연습* (calibration) |
| 2 | zscore blinded CSV 20~30행 검토 — 같은 기준 적용 일관성 확인 |
| 3 | paired blinded CSV에서 raw / zscore 차이를 비교하며 D7 입력 |
| 4 | 전체 pilot sample (raw 140 / zscore 142 / paired 150) 채우기 |
| 5 | `free_text_comment`에 *반복되는* 문제(어떤 wording이 반복적으로 어색한지 등)를 한 줄로 기록 |

---

## 9. What not to do

| # | 금지 행동 | 사유 |
|---|---|---|
| N1 | `class_id_for_audit_only`를 보고 평가 | blinded CSV에는 *제거되어 있음*. audit key 파일을 reviewer에게 노출하지 않는다 |
| N2 | 정답 class를 맞혔는지를 *주된* 기준으로 삼기 | Step 5 §9 #1 — single argmax는 caption 답이 아님 |
| N3 | reviewer 입력 컬럼 *외*의 컬럼 수정 | meta / caption / template_id / class_set 등은 *carry*용 — 변경 금지 |
| N4 | `caption_ko`를 직접 수정 | caption 수정은 본 단계 범위 외 — 필요 시 `reviewer_decision = revise_template` 또는 `revise_policy`로 표시 |
| N5 | raw / zscore 최종 선택 | Step 5 §9 #7 / Step 6 §12 C9 정책 유지 |
| N6 | 새 모델 학습 / threshold recalibration / schema output 재생성 / 새 feature 계산 | 본 단계 범위 외 |
| N7 | audit key 파일 열람 | blinded review 완료 *전*에는 운영자만 보관 |

---

## 10. Completion criteria

pilot review 완료 기준 (reviewer 측 체크리스트).

| # | 기준 | 대상 |
|---|---|---|
| 1 | `rating_policy_compliance` ~ `rating_feature_phrase_appropriateness` 6개 컬럼이 *모든 row*에 채워짐 | raw / zscore branch CSV |
| 2 | 필요한 row에 `issue_tags`가 JSON list로 입력됨 | raw / zscore / paired |
| 3 | `reviewer_decision`이 *모든 row*에 입력됨 | raw / zscore branch CSV |
| 4 | paired CSV의 `reviewer_branch_preference`가 *모든 row*에 입력됨 | paired CSV |
| 5 | critical issue (T1 / T10 / T12)가 있는 경우 `free_text_comment`에 사유 한 줄 | 해당 row |
| 6 | `caption_ko` / meta / template / class_set 등 carry 컬럼 *수정 없음* | 모든 CSV |

본 문서는 reviewer 지침이며, 실제 review 수행 / rating 입력 / final
branch 선택은 *본 단계 범위 외*이다.

---

*본 문서는 Step 8-2 pilot human review *지침*으로 작성되었다. 실제 human
review / reviewer rating 입력 / caption 수정 / caption 재생성 / 새 모델
학습 / threshold recalibration / schema output 재생성 / 새 physical
feature 계산 / raw vs zscore 최종 branch 확정 / final wording 확정은
*수행하지 않았다*. Step 1 / Step 2 / Step 3 / Step 4 / Step 5 / Step 6 /
Step 7 / Step 8 / Step 8-1 산출물은 수정되지 않았다.*
