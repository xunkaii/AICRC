# Step 3 — 출력 스키마 및 불확실성 정책 설계

이 문서는 **설계 명세서일 뿐이다**. 새로운 피처를 계산하거나, 어떠한
모델도 학습시키거나, 캡션 템플릿을 작성하지 않는다. 이 문서는 Step 4
(모델링)와 이후의 모든 캡션 레이어가 따라야 할 출력 레이어의 형태
및 불확실성 정책을 고정한다.

아래의 제약 사항은 Step 2.5 증거
(`reports/step25_final_synthesis.md`)에서 도출된다; 이 문서는 그것을
재도출하지 않는다.

---

## 1. 목적

Step 3는 다음을 고정한다:

- **시스템이 rep당 방출하도록 허용되는 것의 형태** (클래스 사후,
  클래스 집합 예측, 피처 증거 포인터, 앵커 신뢰도, 불확실성 플래그),
- **그 방출을 캡션 신뢰도 수준에 매핑하는 정책**,
- 증거가 기준을 통과하지 못하는 reps에 대한 **no-call 정책**,
- Step 4가 비교할 수 있는 **정규화 후보**,
- 그리고 Step 4에서 고려 대상이 되지 않는 **명시적 제외 사항**.

다음은 선택하지 않는다:

- 최종 피처 집합,
- 최종 정규화,
- 최종 앵커 규칙,
- 모델 아키텍처,
- 캡션 표현,
- 어떤 성능 기준.

Step 4는 이 문서가 정의하는 설계 공간 내에서 선택한다. Step 4는
별도의 제안 없이 §11의 제외 사항을 다시 열 수 없다.

---

## 2. 출력 레이어가 요구하는 입력

출력 레이어는 rep당 다음을 받아야 한다:

| 입력 | 타입 | 출처 | 필수 여부 |
|---|---|---|---|
| `participant_id` | string | manifest | 예 |
| `rep_id` | string | manifest | 예 |
| `posture` | enum {`SA`, `CA`, `HW`} | manifest | **예 (입력, 공변량 아님)** |
| `class_id_true` | enum {`C1`..`C6`} | manifest | 학습 시에만 |
| `split` | enum {`train`, `val`, `test`} | manifest | 관리용 |
| 피처 증거 벡터 | float[k] | Step 4 모델 | 예 |
| 앵커 정체성 | enum {`ensemble_bottom_idx`, `acc_bottom_idx`} | 자세 조건부 규칙 | 예 |
| `anchor_reliability` | float in [0, 1] | candidate feature bank | 예 |

메모:

- **`posture`는 필수 입력이며**, 선택적 공변량이 아니다. 앵커
  정체성, 피처 해석, 정규화는 모두 이에 따라 분기한다 (§6, §10
  참조). 자세 라벨이 없는 reps는 서비스될 수 없으며 no-call 정책
  (§9)으로 떨어진다.
- 출력 레이어는 `participant_id`-조건부 통계를 절대 보지 않는다
  (§10의 `participant_zscore` 제외 참조).
- `anchor_reliability`는 출력 레이어에 **불확실성 트랙**으로 들어가며,
  피처 트랙으로 들어가지 않는다 (§6 참조).

---

## 3. 클래스 출력 스키마

Rep당 방출은 **반드시** 다음을 포함해야 한다:

```
{
  "class_posterior": {C1: p1, C2: p2, C3: p3, C4: p4, C5: p5, C6: p6},
  "class_set_prediction": [...],          # §4 참조
  "feature_evidence": {...},              # §5 참조
  "anchor_reliability": float in [0, 1],  # §6 참조
  "uncertainty_flags": [...],             # §7 참조
  "caption_confidence_level": enum,       # §8 참조
  "no_call": bool                         # §9 참조
}
```

엄격한 규칙:

- **단일 클래스 argmax 출력은 금지된다.** 소비자는
  `argmax(class_posterior)`를 읽고 이를 답으로 취급하는 것이 허용되지
  않는다. 클래스 집합 예측(§4)이 권위 있는 클래스 형태의 출력이다.
- `class_posterior`는 적절한 분포여야 한다: 6개 클래스가 모두 존재하고,
  비음수이며, 수치적 허용 오차 내에서 1.0으로 합산되어야 한다.
- 스키마는 자세 전반과 신뢰도 수준 전반에 걸쳐 **고정**되어 있다.
  낮은 신뢰도가 필드를 제거하지는 않는다; 이는 `class_set_prediction`
  및 `caption_confidence_level`의 *내용*을 변경하고 `uncertainty_flags`를
  추가한다.
- `class_id_true`는 방출의 일부가 아니다. Step 3 출력은 추론 시점의
  출력이다.

C1..C6는 Step 1에서 잠겨 있다. Step 3는 클래스를 추가, 삭제, 병합,
또는 이름 변경하지 않는다.

---

## 4. 클래스 집합 예측 정책

`class_set_prediction`은 시스템이 이 rep에 대해 공동으로 예측으로
주장할 의지가 있는 **클래스 ID의 리스트**이다. 이는 단일 argmax
라벨을 대체한다.

정책은 **클래스 조건부**이며, Step 2.5 종합의 §7(참조 분리도 동작)에서
도출된다:

| 조건 (class_posterior 기준) | class_set_prediction |
|---|---|
| `C2`가 §8의 confident 임계값을 통과 | `[C2]` |
| top mass가 `{C1, C5, C6}`에 있고 어떤 단일 클래스도 confident 임계값을 통과하지 못함 | `[C1, C5, C6]` (또는 사후 질량이 무시할 수 없는 부분집합; 최소 크기 2) |
| top mass가 `{C3, C4}`에 있고 `C2` 사후가 무시할 수 없음 | `[C3, C4, C2]` (여기서 C2-흡수 헤지가 **필수**) |
| top mass가 `{C3, C4}`에 있고 `C2` 사후가 무시할 만함 | `[C3, C4]` |
| `posture` 입력이 누락된 rep, 또는 위의 어떤 클래스 집합도 §8 기준을 통과하지 못하는 rep, 또는 `anchor_reliability`가 §6의 `anchor_no_call_threshold` 미만이면서 그 클래스 집합이 앵커 유래 증거에 의존하는 rep | `[]` 및 `no_call = true` (§9 참조) |

**앵커 신뢰도가 낮은 경우의 기본 처리는 no-call이 아니다.** Step 2.5의
증거에 따르면 앵커 신뢰도는 캡션 언어를 제한해야 하지만(§6, §8),
클래스 집합 자체를 무효화할 만큼 강한 신호는 아니다. 따라서 rep는
일반적으로 다음과 같이 처리된다:

- `anchor_reliability`가 `anchor_suppression_threshold` 미만이면
  `anchor_unreliable` 플래그를 추가하고 §8에 따라 신뢰도 수준을
  강등(`confident` → `hedged`, `hedged` → `low`)한다. `class_set_prediction`은
  유효하게 유지된다.
- `anchor_reliability`가 더 낮은 `anchor_no_call_threshold` 미만이고
  *그리고* 그 클래스 집합의 캡션이 앵커 유래 증거에 달려 있는 경우에
  한해 §9의 no-call 정책을 따른다.

두 임계값의 정의와 역할은 §6에서, no-call 진입 조건은 §9에서 고정한다.

요구되는 동작:

- **`C2`는 단독으로 설 수 있다.** Step 2.5의 §7은 C2를 참조 모델이
  깨끗하게 처리한 유일한 클래스로 명명했다 (recall 0.66–0.77).
  Step 3는 `[C2]`를 자신감 있는 집합으로 허용한다.
- **`C1`/`C5`/`C6`는 confident 수준에서 단독으로 설 수 없다.**
  Step 2.5의 약 27% 그룹 내 혼동은 이 셋 중 어느 하나에 대한 단정이
  뒷받침되지 않음을 의미한다. 이들은 더 낮은 신뢰도 수준에서만(§8
  참조) 그리고 헤지된 캡션 안에서만 개별적으로 나타날 수 있다.
- **`C3`와 `C4`는 두 축의 모호도를 가진다.** 집합은 (a) C3↔C4
  모호도와 (b) Step 2.5에서 관찰된 C2-흡수 패턴(test에서 C3→C2
  ≈ 50%, C4→C2 ≈ 41%)을 모두 표현해야 한다. C2 사후가 무시할 수
  없을 때 C2 없이 단순한 `[C3, C4]` 집합은 금지된다; "무시할 수
  없는" 정확한 컷오프는 Step 4 calibration의 선택이지만, 스키마는
  `[C3, C4, C2]`를 지원해야 한다.
- 클래스 집합은 **순서가 없다**. 이는 top-K 랭킹이 아니다. 캡션
  레이어는 집합을 공동 헤지("이들 중 어느 것이든 증거와 일관됨")로
  읽어야 하며, 주-부 백업으로 읽으면 안 된다.

Step 3는 reps를 이러한 분기 사이로 이동시키는 수치 임계값을 고정하지
않는다. 이러한 임계값은 Step 4 calibration의 일이다. Step 3는 위의
**분기 구조**만 고정한다.

---

## 5. 피처 증거 스키마

`feature_evidence`는 **어떤 해석 가능한 양이 예측을 이끌었는지**를
설명하는 구조화된 레코드이며, 측정과 휴리스틱을 구별한다.

```
feature_evidence: {
  measurements: [
    {name: str, value: float, normalization: enum, posture_used: enum},
    ...
  ],
  heuristics: [
    {name: str, value: float, kind: "weak_rotational_cue" | ...,
     caveat: str},
    ...
  ]
}
```

규칙:

- 피처 *뱅크* (Step 2.5-3)는 Step 3에서 **확정되지 않았다**. Step 4가
  그 뱅크에서 실제 피처 집합을 선택한다. Step 3는 위의 **형태**와
  뱅크 피처를 `measurements` vs `heuristics`로 **분류**하는 것만
  고정한다.
- **`lateral_proxy_gyro`는 휴리스틱이지 측정이 아니다.** 이는
  `kind = "weak_rotational_cue"`와 IMU가 무릎 각도를 측정하지 않는다는
  caveat 문자열과 함께 `heuristics` 아래에 나타나야 한다. 캡션
  레이어는 이를 knee-valgus 측정으로 렌더링하는 것이 금지된다.
- **`anchor_reliability`는 피처가 아니다.** 이는 `feature_evidence`에
  나타나지 않는다. 이는 자체 필드 아래에 나타난다 (§6).
- 각 measurement 항목은 실제로 적용된 `normalization`(`raw` 또는
  `posture_train_zscore`; §10 참조)과 자세 조건부 분기를 위한
  `posture_used`를 기록해야 한다. 이는 캡션 레이어가 자세별로
  피처를 정직하게 설명할 수 있도록 하기 위함이다.
- 자세 인식 피처 스케일은 피처 값만으로는 해석 불가능함을 의미한다.
  피처를 참조하는 캡션은 단지 값이 아니라 `normalization` 및
  `posture_used` 필드를 가져와야 한다.

Step 3는 다음을 금지한다:

- 휴리스틱을 측정으로 제시하는 것,
- 이 두 버킷 스키마에 존재하지 않는 새로운 피처 클래스(예:
  "diagnostic", "primary")를 추가하는 것,
- `normalization`이 §10 집합 외부에 있는 피처를 방출하는 것.

---

## 6. 앵커 신뢰도 스키마

`anchor_reliability`는 이 rep에 사용된 앵커에 대한 신뢰를 표현하는
[0, 1] 범위의 단일 float이다. 이는 피처 트랙이 아닌 **불확실성
트랙**에 있다.

요구사항:

- **자세 조건부 앵커 정체성 (시작 구조):**
  - `SA`, `CA` → `ensemble_bottom_idx` (acc 및 gyro bottom의 평균).
  - `HW` → `acc_bottom_idx` (acc_z 최솟값만). Step 2.5-2는 HW에서
    gyro |peak|가 스쿼트 bottom이 아닌 상체 회전을 반영함을 보였다;
    HW에서 ensemble을 사용하는 것은 금지된다.
  - rep에 사용된 실제 앵커는 `anchor identity` (§2) 아래에 기록되어
    소비자가 이를 감사할 수 있도록 한다.
- `anchor_reliability`는 자세의 허용된 집합 내에서 후보 앵커 방법
  간의 일치도로부터 계산된다. Step 3는 정확한 공식을 고정하지 않는다
  — Step 2.5-3은 이미 `anchor_reliability` 후보를 산출했다; Step 4는
  이를 정제할 수 있지만 삭제할 수는 없다.
- **`anchor_reliability`는 캡션 생성에서 모션 피처와 결합되지
  않는다.** 이는 어떤 캡션 언어가 허용되는지를 조건화하며 (§8 참조),
  `anchor_unreliable` 불확실성 플래그를 트리거할 수 있지만 (§7 참조),
  피처 증거에는 들어가지 않는다.

Step 3는 컷오프 수치 값을 고정하지 않는다. Step 3는 두 컷오프의
**역할**만 고정하며, 실제 수치 임계값은 Step 4 calibration에서 정한다.

| 임계값 | 역할 |
|---|---|
| `anchor_suppression_threshold` | `anchor_reliability`가 이 값 미만이면, `anchor_unreliable` 플래그를 추가하고(§7) 캡션 레이어는 깊이 언어 및 회복 언어(즉, 특정 bottom 위치에 의존하는 언어)를 **반드시 억제해야 한다**. `class_set_prediction`은 유효하게 유지되며, §8에 따라 신뢰도 수준이 강등될 수 있다. |
| `anchor_no_call_threshold` | `anchor_suppression_threshold`보다 **더 낮은** 임계값. `anchor_reliability`가 이 값 미만이고 *동시에* 해당 rep의 클래스 집합이 앵커 유래 증거에 의존하는 경우에 한해 §9의 no-call 정책이 적용된다. 그 외에는 no-call로 강제하지 않는다. |

다음 두 가지는 변하지 않는다:

- `motion_range_acc_z`가 사용 가능하다면 motion-range-style 언어는
  앵커 신뢰도와 무관하게 방출될 수 있다 — 이 피처는 앵커에
  독립적이기 때문이다.
- 두 임계값의 수치 값과 "어느 클래스 집합이 앵커 유래 증거에
  의존하는가"의 정확한 정의는 Step 4 calibration에서 정해지며,
  Step 3의 범위를 벗어난다. 단, 구조적 기본값으로 `[C2]`와
  `[C3, C4, C2]`가 가장 앵커 의존적이며, `[C1, C5, C6]` 및
  `[C3, C4]`는 덜 의존적이다.

---

## 7. 불확실성 플래그 어휘

`uncertainty_flags`는 **고정된 어휘**에서 가져온 리스트이다. 새로운
플래그는 캡션 시점에 발명될 수 없다.

어휘 (Step 3에 의해 초기 고정):

| 플래그 | 의미 |
|---|---|
| `confident_C2` | `class_set_prediction = [C2]`이고 §8 confident 기준 통과 |
| `within_group_ambiguity_c1_c5_c6` | top mass가 `{C1, C5, C6}`에 있고 어떤 단일 클래스도 confident 기준을 통과하지 못함 |
| `pair_ambiguity_c3_c4` | top mass가 `{C3, C4}`에 있고 C2 질량이 무시할 만함 |
| `pair_plus_c2_absorption` | top mass가 `{C3, C4}`에 있고 C2 질량이 무시할 수 없음 — §4의 필수 헤지 |
| `anchor_unreliable` | `anchor_reliability`가 §6의 `anchor_suppression_threshold` 미만. 이 플래그 자체는 no-call을 의미하지 않는다 — 깊이/회복 언어 억제와 §8 신뢰도 강등을 트리거할 뿐이다. |
| `posture_unknown` | `posture` 입력이 누락되거나 무효; no-call 강제 |
| `low_confidence_no_class_set` | §4의 어떤 분기도 기준을 통과하지 못함; no-call 강제 |

규칙:

- 어휘는 **닫혀 있다**. Step 4는 플래그 추가를 제안할 수 있지만,
  그렇게 하려면 명시적인 Step 3 수정이 필요하며 조용한 확장은
  안 된다.
- 여러 플래그가 동시에 발생할 수 있다 (예: `pair_plus_c2_absorption`
  + `anchor_unreliable`).
- 플래그는 지시적이 아니라 **기술적**이다. 캡션 신뢰도 정책 (§8)이
  플래그를 캡션 동작에 매핑하는 것이다.
- 플래그의 부재는 "높은 신뢰도"가 아니다. 신뢰도는 플래그 리스트가
  아니라 `caption_confidence_level` (§8)에서 읽힌다.

---

## 8. 캡션 신뢰도 수준 정책

`caption_confidence_level`은 enum이며, 다음 중 하나이다:

- `confident`
- `hedged`
- `low`
- `no_call`

`(class_set_prediction, anchor_reliability, uncertainty_flags)`에서
`caption_confidence_level`로의 매핑은 아래 표에 의해 고정된다. Step 3는
**구조**를 고정하고; Step 4가 그 안의 수치 임계값을 calibrate한다.

표에서 "컷오프"는 §6의 `anchor_suppression_threshold`를 의미한다.
`anchor_unreliable` 플래그는 이 임계값 미만에서 부착되며,
**그 자체로 no_call을 의미하지 않는다** — 신뢰도 수준의 강등과
깊이/회복 언어 억제를 트리거할 뿐이다. no_call은 §9에 명시된 조건
(특히 `anchor_reliability`가 더 낮은 `anchor_no_call_threshold` 미만이고
*그리고* 클래스 집합이 앵커 유래 증거에 의존하는 경우)에서만 발생한다.

| class_set_prediction | anchor_reliability | flags가 `anchor_unreliable` 포함 | level |
|---|---|---|---|
| `[C2]` | 컷오프 이상 | 아니오 | `confident` |
| `[C2]` | 컷오프 미만 | 예 | `hedged` (깊이/회복 언어 억제) |
| `[C1, C5, C6]` (또는 2-부분집합) | 컷오프 이상 | 아니오 | `hedged` |
| `[C1, C5, C6]` (또는 2-부분집합) | 컷오프 미만 | 예 | `low` |
| `[C3, C4, C2]` | 컷오프 이상 | 아니오 | `hedged` (C2-흡수 가능성을 언어화 필수) |
| `[C3, C4, C2]` | 컷오프 미만 | 예 | `low` |
| `[C3, C4]` | 컷오프 이상 | 아니오 | `hedged` |
| `[C3, C4]` | 컷오프 미만 | 예 | `low` |
| `[]` | 임의 | 임의 | `no_call` (§9 참조) |

표는 일반적인 강등 경로만 다룬다. `anchor_no_call_threshold` 미만의
앵커 의존적 케이스는 표를 거치지 않고 §4의 마지막 row와 §9에 의해
직접 `class_set_prediction = []` / `no_call`로 라우팅된다.

수준별 요구 동작:

- `confident`: 캡션은 클래스를 명명할 수 있고(`C2`만 이 분기에
  도달함) 앵커가 신뢰할 수 있는 경우 깊이/회복 언어를 사용할 수
  있다.
- `hedged`: 캡션은 `[C2]` 외의 단일 클래스가 아닌 클래스 집합을
  명명해야 한다. `anchor_unreliable`이 존재하면, rep가 어떤 클래스
  집합에 떨어졌는지에 관계없이 깊이 언어와 회복 언어가 억제된다.
- `low`: 캡션은 "신호가 여러 클래스와 일관됨"보다 강한 클래스
  단정을 피해야 하며 증거가 약함을 표면화해야 한다. 휴리스틱
  전용 언어(`lateral_proxy_gyro` 등)는 사용된다면 §5에 따라 휴리스틱으로
  표시되어야 한다.
- `no_call`: 시스템은 구조화된 출력을 방출하지만 캡션 레이어는
  no-call 메시지를 방출한다 (§9 참조). 추측을 **조작하지 않는다**.

Step 3는 이 enum 외의 모든 캡션 신뢰도 수준을 금지한다.

---

## 9. No-call 정책

`no_call`(그리고 `class_set_prediction = []`)은 다음 **세 가지
조건에서만** 발생한다. 다른 어떤 조건도 no-call을 트리거하지 않는다:

1. **`posture_unknown`** — `posture` 입력이 누락되거나 무효이다.
   자세 인식 분기 전체(앵커 정체성, 정규화, 피처 해석)가 이 입력에
   의존하므로, 자세 없이는 어떤 분기도 신뢰성 있게 평가될 수 없다.
2. **`low_confidence_no_class_set`** — §4의 어떤 분기도 §8의
   confident-or-hedged 기준을 통과하지 못한다. 사후가 너무 평평해서
   클래스 집합 헤지조차 뒷받침할 수 없을 때의 catch-all이다.
3. **앵커 의존적 케이스에서 `anchor_reliability`가 §6의
   `anchor_no_call_threshold` 미만**. 즉 다음을 *모두* 만족할 때:
   (a) `anchor_reliability` < `anchor_no_call_threshold` (단순히
   `anchor_suppression_threshold` 미만이 아니라 그보다 더 낮은
   임계값), *그리고* (b) 그 rep의 `class_set_prediction`이 앵커
   유래 증거에 의존한다 (구조적으로 `[C2]`와 `[C3, C4, C2]`가 가장
   앵커 의존적이며, Step 4가 정확한 분류를 calibrate한다).

명시적으로 no-call이 **아닌** 케이스:

- `anchor_reliability`가 `anchor_suppression_threshold`보다 낮지만
  `anchor_no_call_threshold` 이상인 경우는 no-call이 아니다.
  `anchor_unreliable` 플래그가 부착되고 §8에 의해 신뢰도 수준이
  강등될 뿐, 클래스 집합은 유효하다.
- 앵커 신뢰도가 매우 낮아도 그 rep의 클래스 집합이 앵커 유래
  증거에 의존하지 않는 경우(예: 구조적으로 덜 의존적인 `[C1, C5, C6]`
  또는 `[C3, C4]`)는 no-call이 아니다 — 마찬가지로 수준 강등으로만
  처리된다.

세 경우 모두에서:

- 구조화된 출력은 여전히 방출된다 (다운스트림 소비자가 로깅,
  재시도 또는 에스컬레이션할 수 있도록).
- `caption_confidence_level = no_call`.
- 캡션 레이어는 **제한된** no-call 메시지를 생성해야 한다. 클래스를
  조작해서는 안 되며, 휴리스틱을 측정으로 제시해서는 안 되고,
  rep를 조용히 누락해서는 안 된다.
- no-call rep는 실패한 rep와 동일하지 않다. 계산 실패(예: 신호
  로드 오류)는 이 스키마의 범위를 벗어나며 업스트림 파이프라인
  로그에 머문다.

---

## 10. 정규화 후보

Step 4는 정확히 다음 두 가지 정규화 후보를 비교할 수 있다:

1. **`raw + posture_as_input`** — 피처 값이 정규화 없이 통과되며;
   자세는 필수 모델 입력이다 (one-hot 또는 임베디드).
2. **`posture_train_zscore + posture_as_input`** — 피처는 **train
   분할** 자세 조건부 통계만을 사용하여 z-score된다; train 통계가
   val 및 test에 적용된다. 자세는 필수 모델 입력으로 유지된다.

두 후보 모두 Step 2.5-3b가 부과한 "신규 사용자의 첫 rep에서 배포
가능"이라는 제약을 만족한다.

Rep당 방출은 둘 중 어느 것이 사용되었는지 기록해야 한다 (각 measurement
항목의 `normalization` 필드, §5). 단일 배포 파이프라인 내에서 둘을
혼합하는 것은 Step 3의 범위를 벗어난다 — Step 4는 하나에 확정해야
한다.

자세는 두 후보 모두에서 *필수*이다. 자세가 없는 rep는 어느 분기
하에서도 정규화될 수 없으며 no-call로 떨어진다 (§9).

---

## 11. 명시적 제외 사항

다음은 **Step 3의 설계 공간에서 제외되며**, 기저 증거를 재도출하는
별도의 제안 없이 Step 4가 다시 열 수 없다:

- **단일 클래스 argmax 출력.** 스키마 전체에서 금지됨 (§3, §4).
- **`participant_zscore` 정규화, 어떤 형태로든, 메인 파이프라인에서.**
  Step 2.5-3b는 이를 calibration ceiling으로만 유지했다; 이 문서는
  이를 **제외**로 이어간다. 이유:
  - 신규 사용자의 첫 rep에서 배포 불가능 (아직 피험자별 통계가
    없음).
  - 전체 피험자별 calibration으로 적용된 경우에도, 자세 η²를 줄이지
    않는다 (그리고 `depth_proxy`에서는 자세 η²를 0.862에서 0.913으로
    *증가*시킨다).
  `participant_zscore`를 추가하려면 별도의 calibration-flow 제안이
  필요하며, 여기에서 정의된 Step 4의 범위에는 없다.
- **`lateral_proxy_gyro`를 knee-valgus 측정으로 다루는 것.**
  이는 휴리스틱한 약한 회전 단서이다. 이를 무릎 측정으로 표현하는
  캡션은 금지된다 (§5).
- **`anchor_reliability`를 모션 피처로 다루는 것.** 이는 불확실성
  트랙에 있으며 절대 `feature_evidence`에 들어가지 않는다 (§5, §6).
- **자세 전반의 단일 전역 앵커.** Step 2.5-2의 HW 붕괴에 의해
  금지됨 (§6).
- **캡션 생성 시점에 불확실성 플래그 또는 캡션 신뢰도 수준을
  발명하는 것.** 두 어휘 모두 §7과 §8에 의해 닫혀 있다.
- **클래스 재도출.** Step 1의 C1..C6는 잠겨 있다. Step 3는 클래스를
  병합, 삭제, 또는 추가하지 않는다 — C1/C5/C6 모호도 또는 C2 흡수가
  이들을 통합하는 것을 시사할 수 있는 곳에서도. 클래스 구조는
  보존되며; 모호도는 클래스 재정의가 아니라 `class_set_prediction`을
  통해 표현된다.
- **캡션 템플릿 표현.** Step 3의 범위를 완전히 벗어남.
- **성능 기준 / 성공 기준.** Step 3의 범위를 완전히 벗어남.

---

## 12. Step 3 완료 기준

Step 3는 **다음 모두**가 성립할 때 완료된다:

1. 이 문서가 `reports/step3/step3_output_schema_uncertainty_policy.md`에
   존재하며 §1–§11을 다룬다.
2. 출력 스키마(§3)가 Step 4가 새 필드를 발명하지 않고 데이터
   구조로 인스턴스화할 수 있을 만큼 구체적이다.
3. 클래스 집합 예측 분기(§4)가 Step 4가 추가 Step 3 작업 없이
   `class_posterior` 및 §8 임계값의 함수로 구현할 수 있을 만큼
   구체적이다.
4. 피처 증거 스키마(§5)가 측정과 휴리스틱을 구별하며,
   `lateral_proxy_gyro`가 휴리스틱으로 명시적으로 분류되어 있다.
5. `anchor_reliability`가 §6에서 피처가 아닌 불확실성 트랙 필드로
   기술되며, 자세 조건부 앵커 정체성 규칙이 시작 구조로 기록되어
   있다.
6. 불확실성 플래그 어휘(§7)가 닫혀 있고 명명되어 있다.
7. 캡션 신뢰도 수준 enum(§8)이 닫혀 있고 명명되어 있으며, 구조적
   매핑 표가 작성되어 있다.
8. No-call 정책(§9)이 no-call을 트리거하는 모든 조건을 명명한다.
9. 정규화 후보(§10)가 명명된 두 옵션으로 제한되며;
   `participant_zscore`가 §11에서 제외로 명명되어 있다.
10. 명시적 제외 사항(§11)이 별도의 제안 없이 Step 4가 하는 것이
    금지된 모든 것을 명명한다.

Step 3 완료가 **아닌** 것:

- 피처 결정이 아니다.
- 정규화 결정이 아니다 (Step 4가 두 가지 중 하나를 선택함).
- 모델 결정이 아니다.
- 캡션 표현 결정이 아니다.
- 성능 주장이 아니다.

위의 기준이 충족되면, Step 4(모델링)는 이 스키마에 대해 시작될
수 있다.

---

*Step 3 설계 명세서로 생성됨. 이 문서를 만들기 위해 새로운 피처는
계산되지 않았으며, 모델은 학습되지 않았으며, Step 1 또는 Step 2/2.5
파일은 수정되지 않았다.*
