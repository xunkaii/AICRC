# Step 2.5 — Bottom 이벤트 후보 감사

목표: rep 전반에 걸쳐 단일 "bottom" 이벤트가 안정적인 후보 앵커가
될 수 있는지 확인한다. **여기서는 앵커 결정이 확정되지 않는다.**

## 방법

- 원시 신호 상의 bottom 탐색 범위: n_rows의 [25%, 75%).
- `acc_bottom_idx` = 해당 범위에서 acc_z(k=9 평활화)의 argmin.
- `gyro_bottom_idx` = 해당 범위에서 |gyro|(k=9 평활화)의 argmax.
- `ensemble_bottom_idx` = floor((acc + gyro) / 2).
- `bottom_agreement_ratio` = |acc - gyro| / n_rows.
- `bottom_agree_default` = ratio ≤ **0.1** (초기 임계값; 최종 아님).

## 전체

- 감사된 rep 수: **9275**
- `bottom_agree_default` 비율 (임계값 = 0.1): **0.5804** (5383 / 9275)
- `bottom_candidate_warning`이 있는 rep 수: 6252

## (class × posture)별 임계값 = 0.1에서의 일치율

| class | SA | CA | HW |
|---|---|---|---|
| C1 | 0.736 | 0.753 | 0.193 |
| C2 | 0.793 | 0.832 | 0.044 |
| C3 | 0.862 | 0.835 | 0.099 |
| C4 | 0.884 | 0.843 | 0.044 |
| C5 | 0.815 | 0.735 | 0.202 |
| C6 | 0.786 | 0.771 | 0.225 |

- 셀 (class × posture) 총합: 18 / 18
- agree_rate ≥ 70%인 셀: 12
- agree_rate ≥ 50%인 셀: 12
- 최소 셀 agree_rate: 0.044  (C2 × HW)
- 최대 셀 agree_rate: 0.884  (C4 × SA)

## 임계값 민감도

전체 표: `reports/bottom_agreement_threshold_sensitivity.csv`.
임계값 = 0.10은 초기 값이다. Bottom 앵커 결정이 이루어진다면,
Step 2.5 확정 후 이 표를 다시 읽어야 한다.

## Bottom 영역 클래스 차이 신호 (교차 참조)

- 18개 중 11개 (posture × channel) 셀이 bottom 영역에서의 평균 η²가
  전체 timeline보다 높다.
- Bottom 영역 차이 신호: **우세함**.

## 예비 판정

- agree ≥ 70%인 셀: 12 / 18  (GO 다수결 임계값: 10)
- bottom 영역 η²가 전체 범위보다 강한 셀: 11 / 18

- 가장 가까운 분류: **GO**

근거 (휴리스틱이며, 최종 결정 아님):
- GO는 agree ≥ 70%인 셀의 다수 *AND* 우세한 bottom 영역 η² 신호
  둘 다를 요구한다.
- WEAK-GO는 그 중간 경우(대부분 셀에서 50–70% 일치 또는 부분적인
  bottom 영역 η² 신호)를 포괄한다.
- NO-GO는 대부분 셀이 agree < 50%이거나, bottom 영역 η² 신호가
  전체 범위 평균과 같거나 더 작은 경우이다.

이 스크립트는 bottom 앵커를 확정하지 **않는다**; 다음 단계가 결정할 수
있도록 1차 증거만 보고한다.
