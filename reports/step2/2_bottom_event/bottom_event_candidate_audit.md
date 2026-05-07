# Step 2.5 — Bottom-Event Candidate Audit

Goal: check whether a single "bottom" event is a stable candidate
anchor across reps. **No anchor decision is finalized here.**

## Method

- Bottom search range on the original signal: [25%, 75%) of n_rows.
- `acc_bottom_idx` = argmin of acc_z (smoothed, k=9) in range.
- `gyro_bottom_idx` = argmax of |gyro| (smoothed, k=9) in range.
- `ensemble_bottom_idx` = floor((acc + gyro) / 2).
- `bottom_agreement_ratio` = |acc - gyro| / n_rows.
- `bottom_agree_default` = ratio ≤ **0.1** (initial threshold; not final).

## Overall

- Reps audited: **9275**
- `bottom_agree_default` rate (threshold = 0.1): **0.5804** (5383 / 9275)
- Reps with `bottom_candidate_warning`: 6252

## Per (class × posture) agree rate at threshold = 0.1

| class | SA | CA | HW |
|---|---|---|---|
| C1 | 0.736 | 0.753 | 0.193 |
| C2 | 0.793 | 0.832 | 0.044 |
| C3 | 0.862 | 0.835 | 0.099 |
| C4 | 0.884 | 0.843 | 0.044 |
| C5 | 0.815 | 0.735 | 0.202 |
| C6 | 0.786 | 0.771 | 0.225 |

- Cells (class × posture) total : 18 / 18
- Cells with agree_rate ≥ 70% : 12
- Cells with agree_rate ≥ 50% : 12
- Min cell agree_rate : 0.044  (C2 × HW)
- Max cell agree_rate : 0.884  (C4 × SA)

## Threshold sensitivity

Full table: `reports/bottom_agreement_threshold_sensitivity.csv`.
Threshold = 0.10 is an initial value. Bottom-anchor decisions, if
any, should re-read this table after Step 2.5 finalizes.

## Bottom-region class-difference signal (cross-reference)

- 11 / 18 (posture × channel) cells show higher mean η² in the bottom region than full-timeline.
- Bottom-region difference signal: **predominant**.

## Preliminary verdict

- Cells with agree ≥ 70% : 12 / 18  (threshold for GO majority: 10)
- Bottom-region η² stronger than full-range: 11 / 18 cells

- Closest classification: **GO**

Reasoning (heuristic, not a final decision):
- GO requires both a majority of cells with agree ≥ 70% AND a
  predominant bottom-region η² signal.
- WEAK-GO covers in-between cases (50–70% agreement on most cells,
  or partial bottom-region η² signal).
- NO-GO is when most cells have agree < 50%, or the bottom-region
  η² signal is at or below full-range mean.

This script does NOT commit a bottom anchor; it only reports the
first-pass evidence so the next step can decide.
