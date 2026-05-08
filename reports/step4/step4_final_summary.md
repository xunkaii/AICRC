# Step 4 — Final Summary

이 문서는 Step 4의 최종 요약이다. 새 분석을 수행하지 않으며, 모델 재학습,
threshold recalibration, schema output 재생성, caption template 작성을
포함하지 않는다. 본 문서는 Step 4 단계에서 이미 산출된 보고서·스크립트·
데이터의 상태를 정리하고, 무엇을 commit하지 않은 채 다음 단계로 넘기는지를
명시한다.

---

## 1. Step 4 목적

- Step 3가 정의한 출력 스키마(`class_posterior`, `class_set_prediction`,
  `uncertainty_flags`, `caption_confidence_level`, `no_call`)를 실제 모델
  출력으로 채우는 것이 목적이었다.
- 단순 accuracy 최대화는 목적이 아니었다. Step 3 스키마와의 호환성과
  ambiguity 정책의 적용 가능성이 우선되었다.
- caption template wording은 Step 4의 범위 밖이었다 — Step 5의 caption
  policy 단계에서 다룬다.

---

## 2. 생성된 산출물 목록

### data

| 경로 | 내용 |
|---|---|
| `data/step4/step4_modeling_dataset.csv` | manifest + main raw/zscore feature + posture one-hot + anchor 정보 결합 데이터셋 |
| `data/step4/step4_predictions_raw.csv` | raw 조건 baseline의 rep 단위 class posterior |
| `data/step4/step4_predictions_zscore.csv` | zscore 조건 baseline의 rep 단위 class posterior |
| `data/step4/step4_schema_outputs_raw.csv` | raw 결과를 Step 3 §3 emission 형태로 채운 출력 |
| `data/step4/step4_schema_outputs_zscore.csv` | zscore 결과를 Step 3 §3 emission 형태로 채운 출력 |
| `data/step4/step4_model_raw.joblib` | raw 조건 학습된 LR 모델 |
| `data/step4/step4_model_zscore.joblib` | zscore 조건 학습된 LR 모델 |

### reports

| 경로 | 내용 |
|---|---|
| `reports/step4/step4_feature_model_decision_memo.md` | feature 분류 / baseline 모델 결정 메모 |
| `reports/step4/step4_modeling_calibration_plan.md` | 실행 계획서 (입력·출력·검증 잠금) |
| `reports/step4/step4_modeling_calibration_results.md` | raw/zscore 평가 메트릭 정리 |
| `reports/step4/step4_threshold_calibration.md` | val 기반 threshold 후보값 + 근거 |
| `reports/step4/step4_final_summary.md` | 본 문서 (최종 요약) |

### scripts

| 경로 | 역할 |
|---|---|
| `scripts/build_step4_modeling_dataset.py` | 입력 CSV 결합 + posture-train zscore 적용 + 검증 후 데이터셋 산출 |
| `scripts/train_step4_baseline.py` | multinomial LR 두 조건 학습 + predictions/모델 산출 |
| `scripts/generate_step4_schema_outputs.py` | predictions + threshold 후보 → Step 3 schema 출력 산출 |
| `scripts/validate_step4_schema_outputs.py` | schema 출력 CSV의 §9 검증 전용 스크립트 |

---

## 3. Modeling dataset 생성 결과

| 항목 | 값 |
|---|---|
| row 수 | 9275 |
| split (train / val / test) | 6412 / 1436 / 1427 |
| main raw feature | 5개 (`motion_range_acc_z`, `depth_proxy`, `motion_range_gyro_mag`, `bottom_stability_acc`, `bottom_transition_delta_acc_z`) |
| main zscore feature | 5개 (각 raw에 `_zscore` 접미) |
| posture one-hot | `posture_SA`, `posture_CA`, `posture_HW` |
| anchor 정보 | `anchor_reliability`, `anchor_type` |
| baseline 입력 제외 | `participant_zscore` 계열, `bottom_recovery_slope_acc_z` (hold-out), `lateral_proxy_gyro` (heuristic) |
| raw/zscore feature finite check | 통과 (NaN/±inf 0) |
| anchor_reliability 범위 검증 | 통과 ([0, 1]) |
| anchor consistency (feature_bank vs bottom_event_audit) | 통과 (4 geometry 컬럼 9275행 일치) |

---

## 4. Baseline 모델

| 항목 | 값 |
|---|---|
| 모델 | multinomial logistic regression (다항) |
| penalty | L2 |
| class_weight | balanced |
| solver | lbfgs (기본 multi_class='auto'와 결합 시 다항 동작) |
| 입력 (raw 조건) | main 5 raw + posture 3 one-hot |
| 입력 (zscore 조건) | main 5 zscore + posture 3 one-hot |
| 출력 | `class_posterior` (`p_C1..p_C6`, 적절한 분포) |
| 분할 | `split_version = v1_36_8_8` |
| `pred_argmax_debug` | 디버깅용. 최종 schema output 아님 |

---

## 5. raw vs zscore baseline 성능 요약

| 조건 | split | n | accuracy | macro F1 | weighted F1 |
|---|---|---:|---:|---:|---:|
| raw | train | 6412 | 0.2993 | 0.2742 | 0.2743 |
| raw | val | 1436 | 0.2695 | 0.2255 | 0.2256 |
| raw | test | 1427 | 0.3210 | 0.2843 | 0.2841 |
| zscore | train | 6412 | 0.3015 | 0.2791 | 0.2792 |
| zscore | val | 1436 | 0.2758 | 0.2337 | 0.2338 |
| zscore | test | 1427 | 0.3203 | 0.2885 | 0.2884 |

**해석**:
- 두 조건 모두 chance(1/6 ≈ 0.167) 위에 위치하지만, 강한 단일 분류기는 아니다.
- zscore가 val/test의 macro F1과 calibration 쪽에서 약간 우세하다.
- 그러나 차이가 작아 raw 조건을 폐기하지 않는다.
- 이 결과는 single argmax 출력이 아니라 class-set + uncertainty 출력이
  필요하다는 Step 3의 판단을 데이터로 지지한다.

---

## 6. Posterior calibration 요약

| 조건 | split | log loss | Brier (multi-class) | ECE (15-bin) |
|---|---|---:|---:|---:|
| raw | train | 1.6518 | 0.7836 | 0.0116 |
| raw | val | 1.6389 | 0.7834 | 0.0724 |
| raw | test | 1.6173 | 0.7746 | 0.0320 |
| zscore | train | 1.6549 | 0.7838 | 0.0131 |
| zscore | val | 1.6268 | 0.7791 | 0.0651 |
| zscore | test | 1.6141 | 0.7734 | 0.0260 |

**해석**:
- zscore가 val / test calibration에서 약간 우세하다.
- 차이가 작아 최종 normalization commit은 아직 하지 않는다.
- val ECE는 두 조건 모두 train/test보다 높음 — split 간 분포 미세 변동을
  반영하지만, threshold calibration이 val에서 이루어졌으므로 본 결과는
  보고용으로만 둔다.

---

## 7. Ambiguity-group 결과 요약 (test split)

| 조건 | C2 recall | C1/C5/C6 internal | C3/C4 pair | C3 → C2 absorb | C4 → C2 absorb |
|---|---:|---:|---:|---:|---:|
| raw | 0.8159 | 0.2462 | 0.2199 | 0.3277 | 0.2723 |
| zscore | 0.8159 | 0.2238 | 0.2008 | 0.3361 | 0.2681 |

**해석**:
- **C2는 비교적 강하게 잡힌다** (recall ~0.82) — Step 3 §4의 `[C2]`
  단독 confident 분기가 정당화된다.
- **C1/C5/C6는 그룹 ambiguity가 필요하다** (그룹 내 혼동 ~22~25%) —
  Step 3 §4의 `[C1, C5, C6]` 또는 그 부분집합 정책 유지.
- **C3/C4는 pair ambiguity뿐 아니라 C2 absorption hedge가 필요하다**
  (C3 → C2 ~33%, C4 → C2 ~27%, pair 내부 혼동 ~20%) — Step 3 §4의
  `[C3, C4, C2]` 헤지의 필수성을 baseline 차원에서 재확인.
- 본 결과는 Step 3 class_set_prediction 정책 변경 근거가 되지 않는다.

---

## 8. Threshold 후보 요약 (val 기준, commit 아님)

| threshold                      |    raw | zscore |
| ------------------------------ | -----: | -----: |
| `confident_C2_threshold`       | 0.3800 | 0.3900 |
| `non_trivial_C2_threshold`     | 0.1123 | 0.1059 |
| `within_group_threshold`       | 0.1650 | 0.1646 |
| `anchor_suppression_threshold` | 0.5000 | 0.5000 |
| `anchor_no_call_threshold`     | 0.2500 | 0.2500 |

**해석**:
- 모든 임계값은 **val split 기준** 후보값이다 (test 누수 없음).
- 임계값은 아직 commit하지 않는다.
- `anchor_no_call_threshold < anchor_suppression_threshold` 관계는 두
  조건 모두에서 만족된다 (Step 3 §6 위반 없음).
- 두 조건 사이의 후보값은 거의 동일 — 정규화 선택이 임계값 calibration에
  미치는 영향은 작다.

---

## 9. Schema output 분포 요약

### 9.1 caption_confidence_level

| 조건     | confident | hedged | low | no_call |
| ------ | --------: | -----: | --: | ------: |
| raw    |       723 |   5904 | 230 |    2418 |
| zscore |       725 |   5883 | 254 |    2413 |

### 9.2 class_set_prediction

| 조건 | `["C2"]` | `["C1","C5","C6"]` | `["C1","C5"]` | `["C1","C6"]` | `["C5","C6"]` | `["C3","C4"]` | `["C3","C4","C2"]` | `[]` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 746 | 2876 | 10 | 309 | 294 | 1417 | 1205 | 2418 |
| zscore | 754 | 2842 | 15 | 310 | 325 | 1386 | 1230 | 2413 |

**해석**:
- 대부분의 출력은 **hedged** — Step 3 §7 ambiguity 어휘가 활발히 동작.
- **confident는 제한적**이며 모두 `[C2]` 중심이다.
- **no_call은 약 26%** 수준이며, 대부분 `low_confidence_no_class_set`
  경로에서 발생한다.
- **`anchor_unreliable`은 즉시 전체 no_call로 이어지지 않고 confidence
  downgrade로 작동**했다 — Step 3 §6의 두 단계 임계값(suppression /
  no_call) 분리 설계가 의도대로 동작함을 확인.

---

## 10. Validator 결과

`scripts/validate_step4_schema_outputs.py` 실행 — **raw 10/10, zscore 10/10,
전체 20/20 통과**.

| # | 검증 항목 | raw | zscore |
|---|---|---|---|
| 1 | row count (=9275) | OK | OK |
| 2 | required columns (19개) | OK | OK |
| 3 | class_posterior distribution (NaN/inf/음수 없음, 합=1) | OK | OK |
| 4 | class_set_prediction whitelist (8개 집합) | OK | OK |
| 5 | uncertainty_flags vocabulary (7개 closed vocab + 중복 없음) | OK | OK |
| 6 | caption_confidence_level enum (4개) | OK | OK |
| 7 | no_call consistency (양방향, level 일치) | OK | OK |
| 8 | no_call reason closure | OK | OK |
| 9 | anchor threshold relation (no_call < suppression) | OK | OK |
| 10 | track separation (forbidden token 없음) | OK | OK |

**no_call reason closure 분해**:

| 조건     | 총 no_call | low_confidence_no_class_set | anchor-driven | posture_unknown |
| ------ | --------: | --------------------------: | ------------: | --------------: |
| raw    |      2418 |                        2393 |            25 |               0 |
| zscore |      2413 |                        2391 |            22 |               0 |

`posture_unknown`이 0인 이유는 입력 데이터에 invalid posture가 없기
때문이다 (manifest의 `posture_canonical`은 모두 SA/CA/HW). anchor-driven
no_call(25 / 22)은 모두 `anchor_reliability < 0.25` AND `class_set ∈
{[C2], [C3,C4,C2]}` 케이스로, generator의 anchor 우선 source 정책과
정확히 일치한다.

---

## 11. 현재 해석

- zscore가 약간 우세하지만, raw도 유지한다 — 차이가 작고 두 조건의
  threshold 후보값도 거의 같다.
- 최종 normalization은 commit하지 않는다.
- 단일 argmax 출력은 여전히 부적절하다 — C2 recall은 높지만 C1/C5/C6와
  C3/C4의 ambiguity 패턴이 명확히 남아 있다.
- Step 3의 class-set / uncertainty / no_call 정책은 baseline 결과로도
  정당화된다.
- 현재 feature set은 class signal을 일부 담지만, C1~C6의 단일 분리를
  강하게 해결하지 못한다 — 이는 Step 2.5의 reference separability 결과와
  같은 방향이며, baseline의 한계로 받아들인다.
- Step 5는 caption wording으로 바로 가기 전에 **caption policy를 먼저
  설계**해야 한다 — 어떤 confidence level / class_set / uncertainty
  flag 조합에 어떤 표현 범위를 허용할지의 정책이 wording보다 선행한다.

---

## 12. 아직 commit하지 않는 결정

| 결정 항목                                   | 상태                                   |
| --------------------------------------- | ------------------------------------ |
| final normalization (raw vs zscore)     | 미commit (두 조건 모두 후속 단계 후보로 유지)       |
| final threshold values                  | 미commit (val 기반 후보값만 보고)             |
| final model architecture                | 미commit (baseline은 비교 기준이며 최종 약속 아님) |
| caption wording                         | 미commit (Step 4 범위 외)                |
| performance success criterion           | 미commit (Step 3 §11에 의해 범위 외)        |
| primary deployed branch (raw or zscore) | 미commit                              |

---

## 13. 다음 단계

### Option A — Checkpoint commit

- 현재 Step 4 산출물을 Git commit한다.
- 목적: rollback 가능한 checkpoint 확보.
- 산출물이 `data/step4/`, `reports/step4/`, `scripts/`로 명확히 분리되어
  있어 commit 단위가 깔끔하다.

### Option B — Step 5 caption policy design 진입

- Step 5는 caption template 문장 작성 전에 **policy를 먼저 설계**한다.
  policy가 정해지지 않은 채 wording부터 쓰는 것은 Step 3의 closed
  vocabulary / no_call / heuristic 분리 정책과 충돌할 수 있다.
- Step 5 policy 설계가 다루어야 할 항목:
  - confidence level별 caption 허용 범위 (`confident` / `hedged` /
    `low` / `no_call`에서 어떤 표현이 허용/금지되는가)
  - no_call message policy (no_call 메시지의 형태 / 어조 / 정보량 한계)
  - heuristic 표현 제한 (Step 3 §5에 따른 `lateral_proxy_gyro` 등의
    표현 금지 — knee valgus 측정으로 표현 금지)
  - `anchor_unreliable` 시 suppress할 표현 (depth/recovery 언어 억제 —
    Step 3 §6, §8)
  - class_set별 caption stance (예: `[C3, C4, C2]` 헤지에서 C2 흡수
    가능성을 어떻게 언어화할 것인가)

### 추천 순서

1. 본 final summary 작성 (이 문서).
2. Step 4 산출물 git commit (Option A).
3. Step 5 caption policy design 시작 (Option B).

---

*본 문서는 Step 4 final summary로 작성되었다. 새 feature는 계산되지
않았고, 모델은 재학습되지 않았으며, threshold는 recalibrate되지 않았고,
schema output은 재생성되지 않았으며, caption template은 작성되지 않았다.
Step 1 / Step 2 / Step 3 산출물은 수정되지 않았다.*
