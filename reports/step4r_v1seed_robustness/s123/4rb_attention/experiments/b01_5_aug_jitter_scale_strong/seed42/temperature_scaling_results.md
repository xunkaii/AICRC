# v1seed calibrate | split_seed=123 exp=`b01_5_aug_jitter_scale_strong` seed=`42`

- fitted T = 2.420172

| split | stage | acc | macroF1 | logloss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| train | before | 0.8586 | 0.8573 | 0.3791 | 0.2105 | 0.0607 |
| train | after | 0.8586 | 0.8573 | 0.6131 | 0.3129 | 0.2622 |
| val | before | 0.5204 | 0.5162 | 1.4689 | 0.6793 | 0.2260 |
| val | after | 0.5204 | 0.5162 | 1.1450 | 0.5972 | 0.0253 |
| test | before | 0.5340 | 0.5323 | 1.4148 | 0.6735 | 0.2122 |
| test | after | 0.5340 | 0.5323 | 1.1100 | 0.5993 | 0.0715 |