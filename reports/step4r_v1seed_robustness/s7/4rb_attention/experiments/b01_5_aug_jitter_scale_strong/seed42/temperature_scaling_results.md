# v1seed calibrate | split_seed=7 exp=`b01_5_aug_jitter_scale_strong` seed=`42`

- fitted T = 4.554482

| split | stage | acc | macroF1 | logloss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| train | before | 0.9641 | 0.9640 | 0.1125 | 0.0581 | 0.0345 |
| train | after | 0.9641 | 0.9640 | 0.5550 | 0.2610 | 0.3685 |
| val | before | 0.5127 | 0.5084 | 2.5247 | 0.8282 | 0.3530 |
| val | after | 0.5127 | 0.5084 | 1.2163 | 0.6241 | 0.0420 |
| test | before | 0.4722 | 0.4693 | 2.5776 | 0.8470 | 0.3791 |
| test | after | 0.4722 | 0.4693 | 1.2237 | 0.6277 | 0.0527 |