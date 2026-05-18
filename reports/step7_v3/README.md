# Step 7_v3 — Caption Generation (clinical pool 적용)

본 디렉토리는 Step 7_v3 caption generation 결과를 담을 위치이다. **현 시점에서는 결과 파일이 없으며**, 정책 문서 commit만 완료된 상태이다.

---

## 1. 본 분기의 위치

| 항목 | 값 |
|---|---|
| 대상 | Step 7_v3 caption pipeline (v2와 병렬 운영) |
| 정책 | `reports/step5_v3/step5_v3_caption_policy.md` (v3 전용) |
| pool | `reports/step5_v3/step5_v3_pool_design.md` (SA 5 / CA 5 / HW 3, J-C1~C6) |
| 금지 표현 | `reports/step5_v3/step5_v3_forbidden_expressions.md` (v3 축소판) |
| system prompt | `reports/step6_v3/step6_v3_final_system_prompt.md` |
| v2 보존 위치 | `reports/step7_v2/` (수정 금지) |

---

## 2. 현재 상태 (2026-05-18)

- [x] 정책 문서 commit (step5_v3 × 3, step6_v3 × 1)
- [ ] 교수님 검토 (정책 + pool entry 적합성)
- [ ] 생성 스크립트 작성 (`scripts/generate_step7_v3_captions.py`)
- [ ] mock provider golden run (baseline 비교용)
- [ ] anthropic provider full run (9,275 sample)
- [ ] Step 8_v2 validation 적용 (v2 vs v3 비교 metric)

---

## 3. 예상 산출물 (생성 후)

```
reports/step7_v3/
  README.md                                  (본 문서)
  step7_v3_caption_generation_results.md     (full run report)
  step7_v3_caption_generation_summary.csv    (요약 metric)
  step7_v3_golden_validation_summary.csv     (golden test 결과)
  step7_v3_vs_v2_comparison.md               (v2 baseline 대비 비교)

data/step7_v3/
  step7_v3_captions.csv                      (full mode caption)
  step7_v3_run_log.csv                       (per-attempt log)
  step7_v3_golden_captions.csv               (golden cases)
  step7_v3_golden_run_log.csv
```

---

## 4. v2 → v3 마이그레이션 노트 (스크립트 작성 시)

`scripts/generate_step7_v2_captions.py`를 *복사*해서 `generate_step7_v3_captions.py`로 분기. 다음 위치만 변경:

| 위치                         | v2                                | v3                                                              |
| -------------------------- | --------------------------------- | --------------------------------------------------------------- |
| SYS_PROMPT_MD              | `step6_v2_final_system_prompt.md` | `step6_v3_final_system_prompt.md`                               |
| OUTPUT_DATA_DIR            | `data/step7_v2/`                  | `data/step7_v3/`                                                |
| OUTPUT_REPORT_DIR          | `reports/step7_v2/`               | `reports/step7_v3/`                                             |
| POSTURE_VOCAB              | 3개 fixed phrase                   | POSTURE_POOL (13개, hash 선택)                                     |
| CLASS_VOCAB                | 6개 통일 phrase                      | JOINT_POOL (19개, hash 선택)                                       |
| OUTPUT_REQUIRED_FIELDS     | 5개                                | 6개 (`used_pool_entries` 추가)                                     |
| FORBIDDEN_EXACT            | 18+                               | 축소판 (책임/처방/모델내부만)                                               |
| FORBIDDEN_DIRECTION_TOKENS | active                            | **비활성**                                                         |
| BIOMECH_TOKENS             | active                            | **비활성**                                                         |
| build_user_payload         | constraint_template 주입            | constraint_template + POSTURE_POOL + JOINT_POOL + hash index 주입 |

v2 스크립트는 *수정하지 않고*, v3 스크립트는 신규 파일로 작성한다 (CLAUDE.md 원칙).

---

## 5. 비교 자료 (Step 8_v2 validation 통합)

Step 8_v2 validator는 v2와 v3 *양쪽* caption CSV를 입력받아 다음 metric을 비교:

| metric | v2 baseline | v3 (예상) |
|---|---|---|
| schema_faithfulness_rate | 0.9998 (mock) | TBD |
| forbidden vocabulary occurrence | 0건 | TBD (책임/처방 카테고리만 검사) |
| posture phrase consistency | 100% | TBD |
| class_set narrowing rate | 0% | TBD |
| **caption 평균 길이** | TBD | TBD (v3가 더 길 것으로 예상) |
| **unique caption count** | low (vocab 통일로 인해) | high (pool permutation으로 인해) |
| **clinical vocabulary density** | 0 | TBD |

---

## 6. 본 디렉토리가 변경되는 시점

- 교수님 검토 결과에 따른 pool 조정
- 생성 스크립트 작성 완료 (`scripts/generate_step7_v3_captions.py`)
- mock golden run 완료
- anthropic full run 완료
- Step 8_v2 validation 통합 완료

---

*본 README는 placeholder이며, 후속 단계에서 결과 보고서로 갱신된다. step7_v2는 본 분기와 무관하게 그대로 보존된다.*
