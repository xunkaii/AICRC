# v1seed calibrate | split_seed=7777 exp=`b01_5_aug_jitter_scale_strong` seed=`42`

- fitted T = 3.479437

| split | stage | acc | macroF1 | logloss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| train | before | 0.9391 | 0.9390 | 0.1810 | 0.0956 | 0.0384 |
| train | after | 0.9391 | 0.9390 | 0.5323 | 0.2523 | 0.3203 |
| val | before | 0.5257 | 0.5160 | 1.9297 | 0.7411 | 0.2897 |
| val | after | 0.5257 | 0.5160 | 1.1577 | 0.6004 | 0.0564 |
| test | before | 0.5067 | 0.4980 | 2.1209 | 0.7699 | 0.3115 |
| test | after | 0.5067 | 0.4980 | 1.2220 | 0.6248 | 0.0458 |