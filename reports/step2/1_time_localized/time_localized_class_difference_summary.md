# Step 2.5 — 시간 국소화된 클래스 차이 감사

목표: 클래스 차이(C1–C6)가 *시간상 어디에서* 가장 두드러지게 나타나는지
식별한다. 이는 기술적(descriptive) 분석이며, 피처 결정이 아니다.

## 방법

- 각 rep는 길이 128로 선형 리샘플링.
- 채널 (timestamp 제외): acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.
- 각 (posture, channel, timestep)에 대해: 클래스 C1–C6 전반의 η² = SSB / SST.
- Bottom 영역: timesteps [38, 89) = 중간 40%.
- 전체 timeline: timesteps [0, 128).

## 셀별 요약

| posture | channel | argmax_t (η²) | η²_max | mean η² (full) | mean η² (bottom) | bottom > full? |
| ------- | ------- | ------------- | ------ | -------------- | ---------------- | -------------- |
| SA      | acc_x   | 50            | 0.2234 | 0.1073         | 0.1489           | **yes**        |
| SA      | acc_y   | 59            | 0.0288 | 0.0136         | 0.0167           | **yes**        |
| SA      | acc_z   | 86            | 0.1173 | 0.0375         | 0.0369           | no             |
| SA      | gyro_x  | 9             | 0.0329 | 0.0082         | 0.0078           | no             |
| SA      | gyro_y  | 73            | 0.1529 | 0.0624         | 0.0733           | **yes**        |
| SA      | gyro_z  | 8             | 0.1928 | 0.0694         | 0.0795           | **yes**        |
| CA      | acc_x   | 64            | 0.0219 | 0.0082         | 0.0108           | **yes**        |
| CA      | acc_y   | 54            | 0.1519 | 0.0676         | 0.0938           | **yes**        |
| CA      | acc_z   | 86            | 0.2060 | 0.0551         | 0.0545           | no             |
| CA      | gyro_x  | 80            | 0.1089 | 0.0382         | 0.0429           | **yes**        |
| CA      | gyro_y  | 18            | 0.0290 | 0.0069         | 0.0053           | no             |
| CA      | gyro_z  | 72            | 0.1842 | 0.0659         | 0.0819           | **yes**        |
| HW      | acc_x   | 54            | 0.1631 | 0.0847         | 0.1460           | **yes**        |
| HW      | acc_y   | 76            | 0.1485 | 0.0587         | 0.1074           | **yes**        |
| HW      | acc_z   | 89            | 0.0874 | 0.0405         | 0.0464           | **yes**        |
| HW      | gyro_x  | 20            | 0.1600 | 0.0548         | 0.0531           | no             |
| HW      | gyro_y  | 101           | 0.1061 | 0.0423         | 0.0284           | no             |
| HW      | gyro_z  | 31            | 0.2681 | 0.1028         | 0.0648           | no             |

## Bottom 영역 집중도

- 18개 중 11개 (posture × channel) 셀이 bottom 영역에서의 평균 η²가
  전체 timeline 전반보다 높다.
- Argmax timesteps가 bottom 영역에 군집되어 있다는 점은 bottom 앵커
  기반 피처가 클래스 차이를 포착할 가능성이 있음을 시사한다.
- 이는 Step 2.5가 확인하고 있는 기술적(descriptive) 신호이며,
  여기서는 어떤 피처도 선택하지 않는다.

(posture, channel, timestep)별 값은 `data/time_localized_class_difference.csv`에
있다.
