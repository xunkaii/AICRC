# v1seed calibrate | split_seed=123 exp=`b01_5_aug_jitter_scale_strong` seed=`43`

- fitted T = 2.646827

| split | stage | acc | macroF1 | logloss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| train | before | 0.8486 | 0.8469 | 0.3856 | 0.2178 | 0.0533 |
| train | after | 0.8486 | 0.8469 | 0.6564 | 0.3369 | 0.2804 |
| val | before | 0.5268 | 0.5246 | 1.5914 | 0.6826 | 0.2078 |
| val | after | 0.5268 | 0.5246 | 1.2091 | 0.6124 | 0.0445 |
| test | before | 0.5095 | 0.5071 | 1.4178 | 0.6816 | 0.2251 |
| test | after | 0.5095 | 0.5071 | 1.1252 | 0.5974 | 0.0319 |