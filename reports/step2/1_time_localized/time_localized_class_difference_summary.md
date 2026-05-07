# Step 2.5 — Time-localized Class Difference Audit

Goal: identify *where in time* class differences (C1–C6) are
most pronounced. This is descriptive, not a feature decision.

## Method

- Each rep linearly resampled to length 128.
- Channels (timestamp dropped): acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.
- Per (posture, channel, timestep): η² = SSB / SST across classes C1–C6.
- Bottom region: timesteps [38, 89) = middle 40%.
- Full timeline: timesteps [0, 128).

## Per-cell summary

| posture | channel | argmax_t (η²) | η²_max | mean η² (full) | mean η² (bottom) | bottom > full? |
|---|---|---|---|---|---|---|
| SA | acc_x | 50 | 0.2234 | 0.1073 | 0.1489 | **yes** |
| SA | acc_y | 59 | 0.0288 | 0.0136 | 0.0167 | **yes** |
| SA | acc_z | 86 | 0.1173 | 0.0375 | 0.0369 | no |
| SA | gyro_x | 9 | 0.0329 | 0.0082 | 0.0078 | no |
| SA | gyro_y | 73 | 0.1529 | 0.0624 | 0.0733 | **yes** |
| SA | gyro_z | 8 | 0.1928 | 0.0694 | 0.0795 | **yes** |
| CA | acc_x | 64 | 0.0219 | 0.0082 | 0.0108 | **yes** |
| CA | acc_y | 54 | 0.1519 | 0.0676 | 0.0938 | **yes** |
| CA | acc_z | 86 | 0.2060 | 0.0551 | 0.0545 | no |
| CA | gyro_x | 80 | 0.1089 | 0.0382 | 0.0429 | **yes** |
| CA | gyro_y | 18 | 0.0290 | 0.0069 | 0.0053 | no |
| CA | gyro_z | 72 | 0.1842 | 0.0659 | 0.0819 | **yes** |
| HW | acc_x | 54 | 0.1631 | 0.0847 | 0.1460 | **yes** |
| HW | acc_y | 76 | 0.1485 | 0.0587 | 0.1074 | **yes** |
| HW | acc_z | 89 | 0.0874 | 0.0405 | 0.0464 | **yes** |
| HW | gyro_x | 20 | 0.1600 | 0.0548 | 0.0531 | no |
| HW | gyro_y | 101 | 0.1061 | 0.0423 | 0.0284 | no |
| HW | gyro_z | 31 | 0.2681 | 0.1028 | 0.0648 | no |

## Bottom-region concentration

- 11 / 18 (posture × channel) cells have higher mean η² in the bottom region than across the full timeline.
- Argmax timesteps clustered in the bottom region suggest that
  bottom-anchored features are likely to capture class differences.
- This is the descriptive signal Step 2.5 is checking; no
  feature is being selected here.

Per-(posture, channel, timestep) values are in `data/time_localized_class_difference.csv`.
