# 4R-B 집계 — exp_id=`b00_baseline_3seed` (seeds [42, 43, 44])

- 생성 스크립트: `scripts/aggregate_step4r_experiment.py`
- per-seed 산출물은 `experiments/{exp_id}/seed{N}/` 참조.

---

## 1. test 분류·calibration (seed별 + mean±std)

ECE는 raw posterior(temperature scaling 적용 전)와 calibrated posterior(T 적용 후) 두 단계를 모두 보고한다. macro F1과 accuracy는 monotonic T 변환에서 보존되므로 단일 컬럼.

| seed | T | test macro F1 | test ECE (raw) | test ECE (calibrated) |
|---|---:|---:|---:|---:|
| 42 | 1.9579 | 0.5184 | 0.1340 | 0.0540 |
| 43 | 3.4728 | 0.4720 | 0.2727 | 0.0271 |
| 44 | 1.5989 | 0.5150 | 0.0965 | 0.0504 |
| **mean±std** | **2.3432±0.9946** | **0.5018±0.0259** | **0.1677±0.0928** | **0.0438±0.0146** |

---

## 2. all-row schema 분포 (mean±std)

| level | mean | std |
|---|---:|---:|
| confident | 0.1604 | 0.0049 |
| hedged | 0.7904 | 0.0066 |
| low | 0.0396 | 0.0010 |
| no_call | 0.0095 | 0.0027 |

---

## 3. schema collapse 체크 (vs legacy 단일-seed 4R-B)

기준 baseline (legacy 단일 seed): confident=0.1670, hedged=0.7521
본 실험 mean: confident=0.1604 (Δ=-0.66%p), hedged=0.7904 (Δ=+3.83%p)

기준: ±10%p 이내. confident와 hedged 둘 다 통과해야 schema 보존.

**판정: ✅ schema 분포 보존** — confident Δ=-0.66%p, hedged Δ=+3.83%p (둘 다 ±10%p 이내)

---

## 4. ambiguity group 분포 (mean±std, all split)

| group | mean | std |
|---|---:|---:|
| confident_C2 | 0.1643 | 0.0048 |
| within_group_c1_c5_c6 | 0.4856 | 0.0159 |
| pair_c3_c4 | 0.1727 | 0.0134 |
| pair_plus_c2_absorption | 0.1679 | 0.0157 |
| no_call | 0.0095 | 0.0027 |
| uncategorized | 0.0000 | 0.0000 |

---

## 5. per-class F1 (test, mean±std)

| class | mean F1 | std | mean recall | std |
|---|---:|---:|---:|---:|
| C1 | 0.5894 | 0.0503 | 0.5837 | 0.0794 |
| C2 | 0.6088 | 0.0412 | 0.6890 | 0.0678 |
| C3 | 0.4825 | 0.0569 | 0.4832 | 0.0983 |
| C4 | 0.5004 | 0.0323 | 0.5035 | 0.0567 |
| C5 | 0.3849 | 0.0518 | 0.3782 | 0.0912 |
| C6 | 0.4449 | 0.0512 | 0.4069 | 0.0898 |

---

## 6. ambiguity 지표 (test, mean±std)

| 지표 | mean | std |
|---|---:|---:|
| amb_c2_recall | 0.6890 | 0.0678 |
| amb_c1_c5_c6_internal | 0.3823 | 0.0526 |
| amb_c3_c4_pair | 0.2784 | 0.0044 |
| amb_c3_to_c2_absorb | 0.2255 | 0.0437 |
| amb_c4_to_c2_absorb | 0.1163 | 0.0542 |

---

## 7. median seed 선택 (downstream caption layer용)

test macroF1 기준 median seed = **44** (macroF1=0.5150, T=1.5989)

Step 5_v2~7_v2 (caption layer) 재실행 시 본 median seed의 schema CSV를 입력으로 사용한다.

---

*aggregate report. 본 실험의 per-seed 산출물은 수정되지 않는다.*