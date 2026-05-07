# Step 7 — Caption Generation Prototype

이 문서는 Step 4 schema output CSV와 Step 4 modeling dataset을 read-only로
읽어, Step 6 template policy를 위반하지 않는 sample별 caption *prototype*
CSV를 생성한 결과를 보고한다. 본 단계는 **prototype**이며, final wording
commit / final raw vs zscore branch commit / final threshold commit /
final UX message commit을 포함하지 않는다.

본 문서는 다음에 종속된다 (충돌 시 종속 문서가 우선):

- `reports/step3/step3_output_schema_uncertainty_policy.md` — 출력 스키마 / 불확실성 정책
- `reports/step4/step4_final_summary.md` — Step 4 최종 상태
- `reports/step5/step5_caption_policy_design.md` — caption policy
- `reports/step6/step6_caption_template_design.md` — caption template family

본 단계는 새 모델 학습 / threshold recalibration / schema output 재생성
/ 새 physical feature 계산 / Step 1~6 산출물 수정을 *수행하지 않는다*.
read 대상: Step 4 schema output CSV 2개 + modeling dataset 1개. 수정 대상
없음. write 대상: Step 7 prototype CSV 2개 + 본 문서.

---

## 1. Step 7 목적

| 항목                | 내용                                                                                                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 목적                | Step 6 template family를 적용하여 sample별 caption prototype CSV를 생성                                                                                                                                |
| 본 단계가 처음 수행하는 것   | 실제 sample별 caption 문장 생성 (per-sample caption CSV)                                                                                                                                             |
| 본 단계가 *수행하지 않는* 것 | final wording commit, final raw/zscore branch commit, final threshold commit, 모델 재학습, threshold 재calibration, 새 feature 계산                                                                    |
| 평가 기준             | Step 6 §12 checklist (14개 항목)를 Step 7 구현용 runtime validation 12개 항목으로 구체화 + Step 7-1 wording refinement에서 *중복/기술 표현 검사* 1개 항목을 추가 (총 13개) — raw / zscore 모든 sample이 13개 항목을 통과해야 prototype 수용 |

---

## 2. 입력 파일과 출력 파일

| 구분   | 경로                                           | 모드        | 비고                                                    |
| ---- | -------------------------------------------- | --------- | ----------------------------------------------------- |
| 입력   | `data/step4/step4_schema_outputs_raw.csv`    | read-only | Step 4 raw branch schema output                       |
| 입력   | `data/step4/step4_schema_outputs_zscore.csv` | read-only | Step 4 zscore branch schema output                    |
| 입력   | `data/step4/step4_modeling_dataset.csv`      | read-only | feature evidence 값 (sample_id 기준 join)                |
| 출력   | `data/step7/step7_captions_raw.csv`          | write     | raw branch caption prototype                          |
| 출력   | `data/step7/step7_captions_zscore.csv`       | write     | zscore branch caption prototype                       |
| 보조   | `data/step7/step7_summary.json`              | write     | 보고서용 요약 (스크립트 자동 생성)                                  |
| 스크립트 | `scripts/generate_step7_captions.py`         | new       | Step 7 generator (read-only join + render + validate) |

실행 명령:

```
python scripts/generate_step7_captions.py
```

### 2.1 CSV 저장 인코딩 (Excel 호환성)

| 항목 | 값 |
|---|---|
| 저장 인코딩 | `utf-8-sig` (UTF-8 with BOM, 0xEF BB BF) |
| 적용 대상 | `data/step7/step7_captions_raw.csv`, `data/step7/step7_captions_zscore.csv` |
| 사유 | Windows Excel이 기본 ANSI / cp949로 추정해 Korean caption_ko를 깨뜨리는 현상 회피 |
| 영향 범위 | 파일 저장 인코딩만 변경. caption 내용 / template policy / 검증 규칙은 변경되지 않음 |
| 호환성 | `pandas.read_csv` / Python `open(..., encoding="utf-8")` 모두 BOM을 투명하게 처리 (별도 디코딩 불필요) |
| 보조 파일 | `data/step7/step7_summary.json`은 BOM 없이 일반 UTF-8로 유지 |

---

## 3. 입력 검증 결과

| 항목 | raw | zscore | modeling |
|---|---:|---:|---:|
| row 수 | 9275 | 9275 | 9275 |
| sample_id unique | OK | OK | OK |
| required columns 존재 | OK | OK | OK |
| split count train/val/test | 6412 / 1436 / 1427 | 6412 / 1436 / 1427 | 6412 / 1436 / 1427 |
| sample_id one-to-one join 후 row 수 | 9275 | 9275 | — |

`schema output CSV에는 feature evidence 값이 없으므로` Step 6 §3.0.1
정책에 따라 `data/step4/step4_modeling_dataset.csv`를 `sample_id` 기준
read-only join하여 feature phrase 선택에만 사용했다.

---

## 4. raw / zscore caption CSV row 수

| branch | 출력 row 수 | 비고                             |
| ------ | -------: | ------------------------------ |
| raw    |     9275 | 입력 row 수와 동일 — drop / dedup 없음 |
| zscore |     9275 | 입력 row 수와 동일 — drop / dedup 없음 |

---

## 5. split별 row 수

| branch | train | val | test |
|---|---:|---:|---:|
| raw | 6412 | 1436 | 1427 |
| zscore | 6412 | 1436 | 1427 |

split 분포는 schema output / modeling dataset과 일관되며, Step 4 §3에서
검증된 split_version `v1_36_8_8`을 그대로 carry한다.

---

## 6. caption_confidence_level 분포

| branch | confident | hedged | low | no_call | 합계 |
|---|---:|---:|---:|---:|---:|
| raw | 723 | 5904 | 230 | 2418 | 9275 |
| zscore | 725 | 5883 | 254 | 2413 | 9275 |

해석:
- 다수가 `hedged` (≈ 63%) — Step 4 §9의 분포와 동일.
- `confident`는 모두 `["C2"]` 단독 (§7 참고).
- `no_call`은 약 26% 수준.

---

## 7. class_set_prediction 분포

| class_set          |  raw | zscore |
| ------------------ | ---: | -----: |
| `["C2"]`           |  746 |    754 |
| `["C1","C5","C6"]` | 2876 |   2842 |
| `["C1","C5"]`      |   10 |     15 |
| `["C1","C6"]`      |  309 |    310 |
| `["C5","C6"]`      |  294 |    325 |
| `["C3","C4"]`      | 1417 |   1386 |
| `["C3","C4","C2"]` | 1205 |   1230 |
| `[]` (no_call)     | 2418 |   2413 |
| 합계                 | 9275 |   9275 |

해석:
- Step 4 §9.2 분포와 일치 (Step 7은 schema output을 그대로 caption 입력으로 사용).
- `["C3","C4","C2"]`은 caption에서 C2를 *반드시* 포함하도록 §11 validation rule 4에서 검증 — 모두 통과 (§11).

---

## 8. template_id 분포

| template_id | raw | zscore | 비고 |
|---|---:|---:|---|
| T-CONF-1 | 723 | 725 | confident + main_feature 사용 — `["C2"]` 단독 |
| T-CONF-2 | 0 | 0 | confident + feature fallback (이번 데이터에서 미발생) |
| T-HEDGE-1 | 5881 | 5854 | hedged + main_feature 사용 (대다수) |
| T-HEDGE-2 | 0 | 0 | hedged + feature fallback (이번 데이터에서 미발생) |
| T-HEDGE-3 | 23 | 29 | `["C2"]` hedged (anchor 강등) — feature 미사용, uncertainty 강조 |
| T-LOW-1 | 0 | 0 | `["C2"]` low 또는 feature fallback (이번 데이터에서 미발생) |
| T-LOW-2 | 230 | 254 | low + main_feature 사용 |
| T-NC-POSTURE | 0 | 0 | `posture_unknown` (이번 데이터에서 미발생) |
| T-NC-ANCHOR | 25 | 22 | anchor-driven no_call |
| T-NC-LOWCONF | 2393 | 2391 | `low_confidence_no_class_set` no_call (catch-all) |

해석:
- 미발생 템플릿(T-CONF-2 / T-HEDGE-2 / T-LOW-1 / T-NC-POSTURE)은 *후보로 정의*되어 있으나, 본 데이터의 입력 분포에서는 발화 조건이 충족되지 않는다. 이는 Step 4 §10에서 `posture_unknown == 0`인 결과와도 일관된다.
- T-NC-ANCHOR row 수(25 / 22)는 Step 4 §10의 anchor-driven no_call 분해(25 / 22)와 정확히 일치한다.

---

## 9. no_call_reason 분포

| no_call_reason                |  raw | zscore |          우선순위 |
| ----------------------------- | ---: | -----: | ------------: |
| `posture_unknown`             |    0 |      0 |       1 (최상위) |
| `anchor_driven`               |   25 |     22 |             2 |
| `low_confidence_no_class_set` | 2393 |   2391 | 3 (catch-all) |
| no_call 합계                    | 2418 |   2413 |             — |

해석:
- 우선순위는 Step 6 §11.4에 따른다 (`posture_unknown > anchor_driven > low_confidence_no_class_set`).
- `posture_unknown`이 0인 사유는 Step 4 §10과 동일 (manifest의 `posture_canonical`이 모두 SA/CA/HW).
- anchor-driven 분해 수치가 Step 4 §10과 정확히 일치 — Step 7은 schema output을 *재해석하지 않고* policy만 적용했다는 의미다.

---

## 10. feature_phrase_fallback_used row 수

| branch | fallback row 수 |   비율 | 비고                                                                                |
| ------ | -------------: | ---: | --------------------------------------------------------------------------------- |
| raw    |              0 | 0.0% | modeling join이 9275/9275 성공 + train 기반 p33/p66 bin이 모든 (feature, posture)에 대해 산출됨 |
| zscore |              0 | 0.0% | 동일                                                                                |

해석:
- Step 6 §3.0.1의 fallback 정책(feature evidence join 실패 시 feature phrase 생략)은 *정의되어 있으나*, 본 데이터에서는 발동되지 않았다.
- 모든 non-no_call row가 main_feature_phrase를 caption에 사용했으며, no_call row는 정의상 main_feature_phrase를 사용하지 않는다.

---

## 11. caption validation 결과

Step 6 §12 checklist는 14개 항목으로 정의되어 있다. Step 7 generator는
이를 **runtime validation 12개 항목**으로 구체화하여 raw / zscore 모든
sample에서 실행한다. Step 7-1 wording refinement(§14)에서 사용자용
중복 / 기술 표현 점검 1개 항목(R13)을 *추가*하여, 현재 runtime은 **총
13개 항목**으로 운영된다. 본 표는 Step 7 runtime 13개 항목과 Step 6
§12의 대응 관계를 보여준다.

| Step 7 runtime 검증 항목 (13개) | 대응 Step 6 §12 항목 |
|---|---|
| 1. no_call이면 caption_ko에 C1~C6 문자열이 없다 | C1 / C12 |
| 2. no_call ↔ class_set == [] 일관성 | C2 / C12 |
| 3. ["C3","C4","C2"]이면 caption_ko에 C2가 포함된다 | C3 |
| 4. ["C3","C4"]이면 caption_ko에 C2가 포함되지 않는다 | C8 (G8 의도 반영) |
| 5. anchor_unreliable이면 depth / recovery / bottom-transition 직접 표현이 없다 | C4 |
| 6. lateral_proxy_gyro / knee valgus / 무릎 각도 / 무릎 외반 표현이 없다 | C5 |
| 7. pred_argmax_debug 문자열이 없다 | C8 |
| 8. threshold 후보값 이름이 없다 | C7 |
| 9. "확실히", "100%", "정답", "모델 실패", "데이터 오류" 표현이 없다 | C13 |
| 10. anchor_reliability를 motion feature처럼 표현하지 않는다 | C6 |
| 11. posterior 확률 수치(`p_C\d=`) 가 없다 | C14 |
| 12. raw vs zscore branch를 final deployed처럼 표현하지 않는다 | C9 |
| 13. *Step 7-1 추가* — caption_ko에 중복 / 기술 표현이 없다 ("가능성 가능성", "모호성 가능성", "패턴에 부합하는 패턴", "단독 패턴에 부합하는 패턴", "모호성 사이의 모호성", "클래스 집합조차", "어떤 클래스도 추정하지 않습니다", "anchor 단서") | (Step 6 §12 위반 아님 — 사용자용 wording 자연화) |

Step 6 §12의 C10 (confident family가 다른 클래스 가능성을 *완전 배제*
하지 않을 것) / C11 (feature 분류 어긋나는 표현 없음) 두 항목은 Step
7의 *render 시점*에서 template / phrase 선택 알고리즘 자체에 의해
보장되며 (Step 6 §10 priority + suppression), 별도 runtime regex
검사가 필요하지 않다.

| branch | pass | fail | fail 비율 |
|---|---:|---:|---:|
| raw | 9275 | 0 | 0.0% |
| zscore | 9275 | 0 | 0.0% |
| 합계 | 18550 | 0 | 0.0% |

콘솔 출력: `Step 7 prototype validation passed`.

이번 prototype은 Step 6 §13 완료 기준의 #6 (template validation
checklist 통과 의무)을 *모든 sample*에서 만족한다.

---

## 12. raw vs zscore 차이 요약

| 항목                       |  raw | zscore | 차이           |
| ------------------------ | ---: | -----: | ------------ |
| confident row 수          |  723 |    725 | +2 (zscore)  |
| hedged row 수             | 5904 |   5883 | −21 (zscore) |
| low row 수                |  230 |    254 | +24 (zscore) |
| no_call row 수            | 2418 |   2413 | −5 (zscore)  |
| anchor-driven no_call    |   25 |     22 | −3 (zscore)  |
| `["C3","C4","C2"]` row 수 | 1205 |   1230 | +25 (zscore) |
| validation pass          | 9275 |   9275 | 동일           |
| feature fallback row     |    0 |      0 | 동일           |

해석:
- 두 branch의 caption 분포는 거의 동일하다. 큰 차이는 hedged/low 사이의 미세한 재분배 (zscore가 hedged를 24만큼 low로 더 강등) 정도다.
- 두 branch 모두 validation을 100% 통과하므로, **이번 prototype 단계에서는 어느 한 branch를 final deployed로 commit하지 않는다** (Step 5 §9 #7 / Step 6 §12 C9 정책 유지).

### 12.1 phrase binning 요약 (caption 문장에 직접 노출되지 않음)

train split 기반 posture별 p33/p66 (참고용 — caption 문장에는 노출하지 않음).

| feature | posture | raw p33 | raw p66 | zscore p33 | zscore p66 | n_train |
|---|---|---:|---:|---:|---:|---:|
| motion_range_acc_z | SA | 4.135 | 5.423 | −0.525 | 0.347 | 2141 |
| motion_range_acc_z | CA | 4.646 | 6.080 | −0.506 | 0.381 | 2132 |
| motion_range_acc_z | HW | 3.603 | 4.986 | −0.571 | 0.352 | 2139 |
| motion_range_gyro_mag | SA | 0.459 | 0.633 | −0.533 | 0.156 | 2141 |
| motion_range_gyro_mag | CA | 0.364 | 0.508 | −0.535 | 0.146 | 2132 |
| motion_range_gyro_mag | HW | 0.912 | 1.184 | −0.481 | 0.283 | 2139 |
| depth_proxy | SA | 7.429 | 8.179 | −0.349 | 0.360 | 2141 |
| depth_proxy | CA | 7.020 | 7.895 | −0.383 | 0.324 | 2132 |
| depth_proxy | HW | −1.498 | 0.399 | −0.438 | 0.432 | 2139 |
| bottom_stability_acc | SA | 0.698 | 0.996 | −0.532 | 0.182 | 2141 |
| bottom_stability_acc | CA | 0.677 | 0.962 | −0.537 | 0.246 | 2132 |
| bottom_stability_acc | HW | 0.333 | 0.554 | −0.590 | 0.127 | 2139 |
| bottom_transition_delta_acc_z | SA | −1.604 | −0.667 | −0.269 | 0.415 | 2141 |
| bottom_transition_delta_acc_z | CA | −1.447 | 0.052 | −0.360 | 0.524 | 2132 |
| bottom_transition_delta_acc_z | HW | −0.089 | 0.048 | −0.233 | 0.211 | 2139 |

해석:
- 위 수치는 **caption phrase 선택을 위한 prototype용 정성 구간**이며, *모델 threshold도 final threshold commit도 아니다* (Step 5 §9 #6 / Step 6 §12 C7 정책 유지).
- `depth_proxy`의 raw 단위가 SA/CA와 HW에서 부호가 다른 것은 자세 조건부 정규화 (Step 4 build_step4_modeling_dataset.py)의 결과이며, caption 문장에서는 자세 간 *수치 비교*를 일절 하지 않는다 (Step 5 §7 정책).
- HW의 `bottom_transition_delta_acc_z` 구간이 매우 좁은(p33 ≈ −0.089, p66 ≈ +0.048) 점은 anchor-dependent feature의 자세별 범위 차이를 반영한다. anchor_unreliable이 발생하면 본 feature는 §10 phrase 선택에서 자동 제거되므로 caption에는 등장하지 않는다.

---

## 13. prototype의 한계 / wording 차원 관찰 사항

본 단계는 prototype이며 wording 품질의 자동 평가는 수행하지 않는다.
다만, 사람 read-through 단계에서 점검할 만한 *문장 차원* 관찰 사항만
표로 정리한다 (정책 위반은 아니며 모두 §11 validation을 통과한다).

| #   | 관찰                                                                                                                                                                                                                                                                                                    | 영향                                                                | Step 7에서 commit하지 않는 이유                                             |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| O1  | T-HEDGE-1 wording 1차 수정에서 `"… 사이의 모호성이 있습니다"` → `"… 가능성이 함께 고려됩니다"`로 변경 후, label 끝이 "가능성"인 `["C3","C4","C2"]` 등에서 `"… 가능성 가능성이 함께 고려됩니다"` 형태의 약한 중복이 남았었다. **Step 7-1 wording refinement(§14)에서 label과 template wording을 통째로 사용자용으로 자연화하여 본 중복 표현은 제거**되었으며, runtime R13 (중복/기술 표현 검사)에서 0건 hit으로 확인됨 | 정책 위반 아님 (Step 6 §12 C1~C14 / G1~G10 모두 통과). wording 차원에서도 R13 통과 | 사용자용 wording 자연화는 *prototype의 일부*이며 final wording commit은 아니다 (§15) |
| O2  | T-HEDGE-3 / T-LOW-1 / T-NC-* 류는 `{main_feature_phrase}` 미사용. main_feature_phrase 컬럼은 빈 문자열로 carry                                                                                                                                                                                                     | 정책 위반 아님 — 추적 컬럼이 의도대로 동작                                         | 문서화 차원에서 본 표에만 명시                                                   |
| O3  | 이번 데이터 분포에서 T-CONF-2 / T-HEDGE-2 / T-LOW-1 / T-NC-POSTURE 4개 template은 발화 조건이 충족되지 않아 미사용                                                                                                                                                                                                             | 정책 위반 아님 — 후보 template은 *조건부 발화*가 정상                              | 데이터 분포 변화 시 자연스럽게 활성화                                               |
| O4  | raw branch와 zscore branch의 wording은 동일 template family를 공유 — 차이는 phrase binning에서만 발생                                                                                                                                                                                                                 | 정책 위반 아님 — 두 branch 모두 prototype                                  | 어느 한 branch도 final commit 아님                                        |

---

## 14. Step 7-1 — Wording refinement

### 14.1 목적

| 항목            | 내용                                                                                                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 목적            | 정책 위반 수정이 *아니라* 사용자용 wording 자연화 (Step 1~6 정책은 변경하지 않음)                                                                                                                          |
| 방법            | `class_set_label` 중립화 + `render()` 10개 template 사용자용 wording + `uncertainty_phrase_from_flags()` 사용자용 phrase + `feature_phrase_for()` 사용자용 phrase + runtime R13 (중복/기술 표현 검사) 추가 |
| 영향 범위         | `scripts/generate_step7_captions.py` 단일 스크립트의 wording 상수 / render / validation 갱신. Step 4 schema output / threshold / model / Step 5~6 정책 *모두 변경 없음*                             |
| validation 결과 | raw 9275 / 0, zscore 9275 / 0 — Step 7 runtime 13개 항목 모두 통과                                                                                                                      |
| commit 상태     | final wording commit *아님* — prototype 단계                                                                                                                                         |

### 14.2 class_set_label 중립화

label에서 "가능성", "모호성", "패턴", "흡수" 같은 *문장 기능* 단어를
제거했다. 가능성 / 모호성 / 판단 보류 같은 문장 기능은 template 쪽에서
처리한다. `["C3","C4","C2"]`에서는 C2가 label에 *반드시* 남도록 한다.

| class_set          | 이전 label            | Step 7-1 label  |
| ------------------ | ------------------- | --------------- |
| `["C2"]`           | C2 단독 패턴            | C2 유형           |
| `["C1","C5","C6"]` | C1·C5·C6 그룹 내 모호성   | C1·C5·C6 유형군    |
| `["C1","C5"]`      | C1·C5 부분집합 가능성      | C1·C5 유형        |
| `["C1","C6"]`      | C1·C6 부분집합 가능성      | C1·C6 유형        |
| `["C5","C6"]`      | C5·C6 부분집합 가능성      | C5·C6 유형        |
| `["C3","C4"]`      | C3·C4 페어 모호성        | C3·C4 유형        |
| `["C3","C4","C2"]` | C3·C4 페어와 C2 흡수 가능성 | C3·C4 유형과 C2 유형 |

### 14.3 template 문장 변경

| template     | 이전                                                                                    | Step 7-1                                                                                          |
| ------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| T-CONF-1     | `{posture_phrase} {class_set_label}에 부합하는 패턴이 관찰됩니다. 보조 단서: {main_feature_phrase}.`   | `{posture_phrase} {class_set_label}에 가까운 동작 패턴이 보입니다. 참고 단서: {main_feature_phrase}.`              |
| T-CONF-2     | `{class_set_label}이 비교적 안정적으로 관찰됩니다 ({posture_basis}).`                               | `{posture_basis}에서 {class_set_label}에 가까운 패턴이 비교적 안정적으로 보입니다.`                                    |
| T-HEDGE-1    | `{posture_phrase} {class_set_label} 가능성이 함께 고려됩니다. 보조 단서: {main_feature_phrase}.`     | `{posture_phrase} {class_set_label} 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: {main_feature_phrase}.`         |
| T-HEDGE-2    | `현재 단서로는 {class_set_label}을 좁히기 어렵습니다. {uncertainty_phrase}.`                         | `현재 신호만으로는 {class_set_label} 안에서 하나로 안정적으로 좁히기 어렵습니다. {uncertainty_phrase}.`                      |
| T-HEDGE-3    | `{posture_phrase} {class_set_label}을 단일 클래스로 단정하기에는 근거가 부족합니다. {uncertainty_phrase}.` | `{posture_phrase} {class_set_label}에 가까운 신호가 있지만, 하나로 말하기에는 신호가 충분하지 않습니다. {uncertainty_phrase}.` |
| T-LOW-1      | `{posture_phrase} 단서가 약하여 {class_set_label}을 더 좁히기 어렵습니다. {uncertainty_phrase}.`      | `{posture_phrase} 신호가 약해서 {class_set_label} 안에서도 하나로 말하기 어렵습니다. {uncertainty_phrase}.`            |
| T-LOW-2      | `여러 패턴이 동시에 가능합니다 ({class_set_label}). 보조 정보: {main_feature_phrase}.`                 | `{class_set_label} 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: {main_feature_phrase}.`                          |
| T-NC-POSTURE | `자세 정보가 확인되지 않아 클래스에 대한 판단을 보류합니다.`                                                   | `자세 정보가 확인되지 않아 동작 유형 판단을 보류합니다.`                                                                 |
| T-NC-ANCHOR  | `anchor 단서를 신뢰하기 어려워, anchor에 의존하는 클래스 후보에 대한 판단을 보류합니다.`                             | `기준 시점 단서를 신뢰하기 어려워, 해당 단서에 의존하는 동작 유형 판단을 보류합니다.`                                                |
| T-NC-LOWCONF | `현재 단서로는 클래스 집합조차 좁힐 수 없을 만큼 근거가 부족합니다. 어떤 클래스도 추정하지 않습니다.`                           | `현재 신호만으로는 동작 유형을 안정적으로 좁히기 어렵습니다. 무리해서 판단하지 않고 보류합니다.`                                           |

원칙:
- "클래스" 대신 "유형" / "동작 유형"을 우선 사용.
- "보조 단서" 대신 "참고 단서" 사용.
- "anchor" 대신 "기준 시점" 사용.
- "클래스 집합", "추정", "근거 부족" 같은 분석용 표현은 사용자용 문장에서 제거.
- no_call 문장에 C1~C6 문자열은 *여전히* 들어가지 않는다 (R1).

### 14.4 uncertainty_phrase 변경

| flag | 이전 | Step 7-1 |
|---|---|---|
| `anchor_unreliable` | anchor 단서가 신뢰하기 어려움 | 기준 시점 단서를 신뢰하기 어려움 |
| `within_group_ambiguity_c1_c5_c6` | C1·C5·C6 그룹 내 모호성이 있음 | C1·C5·C6 유형 안에서 하나로 좁히기 어려움 |
| `pair_ambiguity_c3_c4` | C3·C4 페어 모호성이 있음 | C3·C4 유형 사이에서 하나로 좁히기 어려움 |
| `pair_plus_c2_absorption` | C3·C4 페어와 C2 흡수 가능성이 동시에 고려됨 | C3·C4 유형뿐 아니라 C2 유형과도 비슷하게 보일 수 있음 |
| `low_confidence_no_class_set` | 근거가 부분적임 | 현재 신호가 제한적임 |
| (그 외) | 단서가 부분적임 | 참고할 수 있는 신호가 제한적임 |

`pair_plus_c2_absorption` phrase는 Step 7-1에서도 C2가 *반드시* 남는다
(R3 / Step 6 §4 / §9 #9 정책 유지).

### 14.5 feature phrase 변경

| feature | 이전 | Step 7-1 |
|---|---|---|
| `motion_range_acc_z` (low/mid/high) | 전반 동작 범위가 비교적 좁은 편 / 중간 수준 / 비교적 넓은 편 | 전반적인 움직임 범위가 비교적 작은 편 / 보통 수준 / 비교적 큰 편 |
| `motion_range_gyro_mag` (low/mid/high) | 회전량 단서가 작은 편 / 중간 수준 / 큰 편 | 회전 움직임 단서가 작은 편 / 보통 수준 / 큰 편 |
| `depth_proxy` (low/mid/high) | 정규화된 깊이 단서가 작은 편 / 중간 수준 / 큰 편 | 자세 기준 깊이 관련 단서가 작은 편 / 보통 수준 / 큰 편 |
| `bottom_stability_acc` (low/mid/high) | bottom 부근 안정성 단서가 약한 편 / 중간 수준 / 비교적 안정적 | 기준 시점 주변 안정성 단서가 약한 편 / 보통 수준 / 비교적 안정적인 편 |
| `bottom_transition_delta_acc_z` (low/mid/high) | 전환 단서가 작은 편 / 중간 수준 / 큰 편 | 전환 관련 단서가 작은 편 / 보통 수준 / 큰 편 |

`anchor_unreliable`이 set인 경우 `depth_proxy` / `bottom_stability_acc`
/ `bottom_transition_delta_acc_z` phrase는 *여전히* 자동 제거된다 (Step
6 §10 + R5). knee valgus / 무릎 각도 / 무릎 외반 / raw absolute depth /
자세 간 직접 수치 비교는 *여전히* 금지 (R6 / Step 5 §7).

### 14.6 R13 (중복 / 기술 표현 검사) 결과

| token | raw hit (unique caption 단위) | zscore hit (unique caption 단위) |
|---|---:|---:|
| "가능성 가능성" | 0 | 0 |
| "모호성 가능성" | 0 | 0 |
| "패턴에 부합하는 패턴" | 0 | 0 |
| "단독 패턴에 부합하는 패턴" | 0 | 0 |
| "모호성 사이의 모호성" | 0 | 0 |
| "클래스 집합조차" | 0 | 0 |
| "어떤 클래스도 추정하지 않습니다" | 0 | 0 |
| "anchor 단서" | 0 | 0 |
| **합계** | **0** | **0** |

### 14.7 분포 변화 / unique caption 수

| 항목 | raw | zscore | 변화 (Step 7 → Step 7-1) |
|---|---:|---:|---|
| row 수 | 9275 | 9275 | 동일 |
| caption_confidence_level 분포 | 변화 없음 | 변화 없음 | 동일 |
| class_set_prediction 분포 | 변화 없음 | 변화 없음 | 동일 |
| template_id 분포 | 변화 없음 | 변화 없음 | 동일 (T-HEDGE-1 5881/5854 등 모두 동일) |
| no_call_reason 분포 | 변화 없음 | 변화 없음 | 동일 |
| feature_phrase_fallback_used | 0 | 0 | 동일 |
| validation pass / fail | 9275 / 0 | 9275 / 0 | 동일 (R13 추가에도 fail 0 유지) |
| unique caption count | 65 | 67 | wording 다양성은 유지되며 중복/기술 표현은 0 |

분포가 모두 동일하다는 것은 Step 7-1이 **wording 자연화에만 영향을
주고, schema output 해석 / class_set / template 선택 알고리즘 자체는
변경하지 않았다**는 뜻이다.

### 14.8 갱신된 caption 예시 (raw branch, 각 template 최대 5개 unique)

| template | 예시 |
|---|---|
| T-CONF-1 | `SA 자세에서 C2 유형에 가까운 동작 패턴이 보입니다. 참고 단서: 전반적인 움직임 범위가 비교적 작은 편.` |
| T-CONF-1 | `CA 자세에서 C2 유형에 가까운 동작 패턴이 보입니다. 참고 단서: 전반적인 움직임 범위가 보통 수준.` |
| T-CONF-1 | `HW 자세에서 C2 유형에 가까운 동작 패턴이 보입니다. 참고 단서: 전반적인 움직임 범위가 보통 수준.` |
| T-HEDGE-1 | `SA 자세에서 C3·C4 유형과 C2 유형 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: 전반적인 움직임 범위가 보통 수준.` |
| T-HEDGE-1 | `SA 자세에서 C1·C5·C6 유형군 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: 전반적인 움직임 범위가 비교적 큰 편.` |
| T-HEDGE-1 | `SA 자세에서 C3·C4 유형 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: 전반적인 움직임 범위가 비교적 큰 편.` |
| T-HEDGE-3 | `SA 자세에서 C2 유형에 가까운 신호가 있지만, 하나로 말하기에는 신호가 충분하지 않습니다. 기준 시점 단서를 신뢰하기 어려움.` |
| T-LOW-2 | `C1·C5·C6 유형군 안에서 여러 동작 패턴이 함께 보입니다. 참고 단서: 전반적인 움직임 범위가 비교적 큰 편.` |
| T-NC-ANCHOR | `기준 시점 단서를 신뢰하기 어려워, 해당 단서에 의존하는 동작 유형 판단을 보류합니다.` |
| T-NC-LOWCONF | `현재 신호만으로는 동작 유형을 안정적으로 좁히기 어렵습니다. 무리해서 판단하지 않고 보류합니다.` |

특히:
- `["C3","C4","C2"]` template (T-HEDGE-1)에서 C2가 *여전히* 등장한다 (R3 / Step 6 §9 #9 정책 유지).
- `["C3","C4"]` template에는 C2가 *없다* (R4).
- no_call template (T-NC-ANCHOR / T-NC-LOWCONF)에 C1~C6 문자열은 *없다* (R1).
- "anchor", "클래스 집합조차", "어떤 클래스도 추정하지 않습니다" 같은 기술적 표현은 *제거*되었다 (R13).

---

## 15. 이번 단계에서 commit하지 않는 결정

| 결정 항목 | 상태 | 근거 |
|---|---|---|
| final wording | 미commit | 본 단계는 prototype |
| final raw vs zscore branch | 미commit | Step 5 §9 #7 / Step 6 §12 C9 정책 유지 |
| final threshold | 미commit | Step 5 §9 #6 / Step 6 §12 C7 정책 유지 |
| final UX message (재측정 강제 여부 등) | 미commit | Step 5 §8 / Step 6 §11.3 NC-G4 — 본 정책 범위 밖 |
| 새 모델 학습 | 수행하지 않음 | 본 단계 범위 외 |
| threshold recalibration | 수행하지 않음 | 본 단계 범위 외 |
| schema output 재생성 | 수행하지 않음 | 본 단계 범위 외 |
| 새 physical feature 계산 | 수행하지 않음 | 본 단계 범위 외 |
| Step 1 / 2 / 3 / 4 / 5 / 6 산출물 수정 | 수행하지 않음 | 본 단계 범위 외 |

---

## 16. 다음 단계

| 옵션 | 내용 | 비고 |
|---|---|---|
| Step 7 / Step 7-1 prototype review | 본 prototype CSV의 caption_ko 문장을 사람이 read-through하여 wording 품질을 검토 | 정책 위반은 §11에서 0건이며 R13 (중복/기술 표현)도 0건이지만, 추가 자연화 / 어조 조정 같은 *비정책적* 관찰은 review 단계에서 다룸 |
| 추가 wording refinement (선택) | review 결과 wording을 더 다듬어야 한다면 §14 표 기준 *최소 수정* | Step 6 §13 완료 기준이 깨지지 않도록 §11 13개 항목 재실행 후 본 prototype을 재생성 |
| Step 8 caption evaluation / human review 설계 | 더 큰 평가 / human-in-the-loop / 정량 평가 단계 진입 | 평가 메트릭 / sampling 전략 / IRB / annotator 가이드는 Step 8에서 정의 |

본 단계 prototype이 정책 차원에서 0건 위반을 통과했으며 R13에서도 0건
hit이므로 위 옵션 중 어느 경로로 진행해도 된다. 단, 어떤 경로로
진행하든:

- raw / zscore 중 한 branch를 final deployed로 commit하지 않는다 (Step 5 §9 #7).
- threshold 후보값을 final commit으로 승격하지 않는다 (Step 5 §9 #6).
- §11 검증 13개 항목 (Step 7 12 + Step 7-1 R13)은 후속 단계에서도 모두 유지되어야 한다.
- Step 7-1 wording refinement는 *prototype 단계의 자연화*이며 final wording commit이 아니다 (§15).

---

*본 문서는 Step 7 caption generation prototype 및 Step 7-1 wording
refinement 결과로 작성되었다. 새 모델 학습 / threshold recalibration /
schema output 재생성 / 새 physical feature 계산은 수행하지 않았다.
Step 1 / Step 2 / Step 3 / Step 4 / Step 5 / Step 6 산출물은 수정되지
않았다. data/step4 CSV는 read-only 참고 / read-only join으로만 사용했다.
본 단계의 caption CSV 2개는 prototype이며, final wording / final raw
vs zscore branch는 commit하지 않았다. Step 7-1은 사용자용 wording
자연화이며 정책 변경 / class_set 변경 / no_call 정책 변경 / validation
규칙 완화는 수행하지 않았다.*
