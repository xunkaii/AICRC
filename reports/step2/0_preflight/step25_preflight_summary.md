# Step 2.5 — Preflight Sanity Summary

- Manifest: `data\manifest_split.csv`
- Loaded sample rows: **9275**
- Bottom-audit records produced: **9275**

## 1. Split sizes

| split | participants | samples |
|---|---|---|
| train | 36 | 6412 |
| val | 8 | 1436 |
| test | 8 | 1427 |

## 2. Class × posture × split sample counts

| class | posture | train | val | test | total |
|---|---|---|---|---|---|
| C1 | SA | 358 | 80 | 78 | 516 |
| C1 | CA | 360 | 80 | 79 | 519 |
| C1 | HW | 358 | 80 | 80 | 518 |
| C2 | SA | 358 | 80 | 80 | 518 |
| C2 | CA | 358 | 80 | 80 | 518 |
| C2 | HW | 360 | 80 | 79 | 519 |
| C3 | SA | 349 | 80 | 79 | 508 |
| C3 | CA | 356 | 79 | 80 | 515 |
| C3 | HW | 347 | 80 | 79 | 506 |
| C4 | SA | 359 | 80 | 80 | 519 |
| C4 | CA | 349 | 77 | 76 | 502 |
| C4 | HW | 360 | 80 | 79 | 519 |
| C5 | SA | 359 | 80 | 80 | 519 |
| C5 | CA | 356 | 80 | 78 | 514 |
| C5 | HW | 355 | 81 | 80 | 516 |
| C6 | SA | 358 | 80 | 80 | 518 |
| C6 | CA | 353 | 79 | 80 | 512 |
| C6 | HW | 359 | 80 | 80 | 519 |

## 3. n_rows / duration_s basic stats

| field | n | min | p25 | p50 | mean | p75 | max |
|---|---|---|---|---|---|---|---|
| n_rows | 9275 | 107 | 144 | 148 | 147 | 151 | 200 |
| duration_s | 9275 | 2.090 | 2.872 | 2.941 | 2.931 | 3.003 | 3.943 |

## 4. Soft-flag counts

- `boundary_ok=False`: 9
- `length_ok=False`  : 0

These rows are kept in the manifest (soft flags only). Step 2.5
audits operate on the resampled signal, so missing index bounds do
not block bottom-event detection.
