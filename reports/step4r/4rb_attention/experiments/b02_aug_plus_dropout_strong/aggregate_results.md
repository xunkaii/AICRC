# 4R-B 집계 — exp_id=`b02_aug_plus_dropout_strong` (seeds [42, 43, 44])

- 생성 스크립트: `scripts/aggregate_step4r_experiment.py`
- per-seed 산출물은 `experiments/{exp_id}/seed{N}/` 참조.

---

## 1. test 분류·calibration (seed별 + mean±std)

ECE는 raw posterior(temperature scaling 적용 전)와 calibrated posterior(T 적용 후) 두 단계를 모두 보고한다. macro F1과 accuracy는 monotonic T 변환에서 보존되므로 단일 컬럼.

| seed | T | test macro F1 | test ECE (raw) | test ECE (calibrated) |
|---|---:|---:|---:|---:|
| 42 | 1.0831 | 0.4958 | 0.0383 | 0.0361 |
| 43 | 2.0317 | 0.4954 | 0.1275 | 0.0646 |
| 44 | 2.4387 | 0.5227 | 0.1707 | 0.0354 |
| **mean±std** | **1.8512±0.6956** | **0.5046±0.0156** | **0.1121±0.0675** | **0.0453±0.0167** |

---

## 2. all-row schema 분포 (mean±std)

| level | mean | std |
|---|---:|---:|
| confident | 0.1821 | 0.0257 |
| hedged | 0.7635 | 0.0266 |
| low | 0.0389 | 0.0015 |
| no_call | 0.0155 | 0.0117 |

---

## 3. schema collapse 체크 (vs legacy 단일-seed 4R-B)

기준 baseline (legacy 단일 seed): confident=0.1670, hedged=0.7521
본 실험 mean: confident=0.1821 (Δ=+1.51%p), hedged=0.7635 (Δ=+1.14%p)

기준: ±10%p 이내. confident와 hedged 둘 다 통과해야 schema 보존.

**판정: ✅ schema 분포 보존** — confident Δ=+1.51%p, hedged Δ=+1.14%p (둘 다 ±10%p 이내)

---

## 4. ambiguity group 분포 (mean±std, all split)

| group | mean | std |
|---|---:|---:|
| confident_C2 | 0.1867 | 0.0264 |
| within_group_c1_c5_c6 | 0.4752 | 0.0038 |
| pair_c3_c4 | 0.1607 | 0.0077 |
| pair_plus_c2_absorption | 0.1619 | 0.0190 |
| no_call | 0.0155 | 0.0117 |
| uncategorized | 0.0000 | 0.0000 |

---

## 5. per-class F1 (test, mean±std)

| class | mean F1 | std | mean recall | std |
|---|---:|---:|---:|---:|
| C1 | 0.5941 | 0.0108 | 0.6596 | 0.0489 |
| C2 | 0.6655 | 0.0210 | 0.7727 | 0.0387 |
| C3 | 0.4869 | 0.0944 | 0.4762 | 0.1717 |
| C4 | 0.5000 | 0.0561 | 0.5220 | 0.1192 |
| C5 | 0.3775 | 0.0671 | 0.3375 | 0.0763 |
| C6 | 0.4036 | 0.0120 | 0.3431 | 0.0229 |

---

## 6. ambiguity 지표 (test, mean±std)

| 지표 | mean | std |
|---|---:|---:|
| amb_c2_recall | 0.7727 | 0.0387 |
| amb_c1_c5_c6_internal | 0.3855 | 0.0186 |
| amb_c3_c4_pair | 0.2791 | 0.0311 |
| amb_c3_to_c2_absorb | 0.2311 | 0.0333 |
| amb_c4_to_c2_absorb | 0.0922 | 0.0137 |

---

## 7. median seed 선택 (downstream caption layer용)

test macroF1 기준 median seed = **42** (macroF1=0.4958, T=1.0831)

Step 5_v2~7_v2 (caption layer) 재실행 시 본 median seed의 schema CSV를 입력으로 사용한다.

---

*aggregate report. 본 실험의 per-seed 산출물은 수정되지 않는다.*