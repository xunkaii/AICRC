# Step 4R — Constraint Knowledge Template Design

본 문서는 wrist IMU 기반 squat 분류 시스템의 caption 생성 단계에서 *신체 제약 지식*을 closed-vocabulary 형태로 LLM에 주입하기 위한 prototype 지식 설계 문서이다. 본 문서는 새 코드를 만들지 않으며, schema·모델·caption 어느 것도 새로 생성하지 않는다.

---

## 메타데이터

| 항목 | 값 |
|---|---|
| 작성 목적 | `scripts/generate_step7_v2_captions.py`의 `build_user_payload()`에 `constraint_template` 필드로 주입하기 위한 지식 설계 문서. LLM이 caption 생성 시 *배경 지식*으로 참조하나, 출력은 schema 사실에 한정한다. |
| 연계 파일 | `reports/step6_v2/step6_v2_final_system_prompt.md` §7 금지 표현 정책 참조 |
| 연계 파일 (구현) | `scripts/generate_step7_v2_captions.py` (`build_user_payload`, 261행) |
| 상태 | **prototype** (10개 entry, 향후 확장 예정) |
| 적용 범위 | Step 4R-C 4rc_contrastive_optional 흐름의 *외부 caption*에는 적용하지 않음. 본 템플릿은 Step 5_v2~7_v2 main pipeline의 caption 표현층에만 주입 후보로 검토된다. |
| 작성일 | 2026-05-15 |

### 주의 사항

- 본 파일의 `caption_safe_expression` 필드는 반드시 `reports/step6_v2/step6_v2_final_system_prompt.md` §7 금지 목록과 충돌하지 않아야 한다. 특히 다음 표현은 본 파일에서도 **절대 사용 금지**.
  - "무릎이 안쪽으로 들어갔다", "knee valgus"
  - "후방 경사", "posterior tilting", "골반이 뒤로 말렸다"
  - "한쪽", "양측", "왼쪽", "오른쪽" (좌우 방향 단정)
  - "관절각"
- `constraint_description`과 `mechanism` 필드는 *내부 지식*이므로 의학·생체역학 용어를 사용해도 좋다. 단 LLM의 caption 출력에 그대로 노출되어서는 안 되므로, system prompt에서 "이 필드는 *내부 배경 지식*이며 caption 본문에 직접 인용하지 말 것"이라는 제약을 함께 주입해야 한다.
- `caption_safe_expression`만 LLM이 caption 본문에 *허용된 어휘로* 풀어쓸 수 있는 표현이다.

---

## 1. Constraint Template Entries

### CT-01 — 발목 dorsiflexion 가동성 제한

| 필드 | 값 |
|---|---|
| constraint_id | CT-01 |
| target_classes | [C2, C3] |
| constraint_name | 발목 가동성 제한 |
| constraint_description | 아킬레스건 단축, 종아리 근육 단축, 발목 dorsiflexion ROM 부족으로 인해 하강 phase에서 발뒤꿈치를 바닥에 유지하면서 깊이 내려가기 어려운 신체 조건. |
| mechanism | 발목이 충분히 접히지 못하면 무게중심을 발 위에 유지하기 위해 체간이 앞으로 더 기울거나, 깊이 자체를 제한하여 발뒤꿈치 들림을 회피하는 보상이 일어난다. 결과적으로 충분한 깊이까지 내려가지 못하는 패턴이 전형적으로 출현한다. |
| caption_safe_expression | "하강 동작이 충분히 진행되지 못한 채 마무리되는 패턴과 관련될 수 있습니다." / "체간이 앞으로 기울며 동작 범위가 제한되는 신호가 동반되는 경우가 많습니다." |
| evidence_basis | 전문가 지식 (스포츠 의학·운동 생리학 일반론) |
| sensor_observability | **false** — 손목 IMU는 발목 ROM 자체를 직접 측정하지 않으며, 체간 기울기와의 결합 효과만 손목 신호의 가속도 z축 성분에 간접 반영됨 |

---

### CT-02 — 고관절 굴곡 가동성 제한

| 필드 | 값 |
|---|---|
| constraint_id | CT-02 |
| target_classes | [C2, C3] |
| constraint_name | 고관절 굴곡 가동성 제한 |
| constraint_description | 햄스트링 단축, 둔근 경직, 또는 고관절 캡슐의 가동 범위 부족으로 hip flexion ROM이 제한된 신체 조건. |
| mechanism | 깊은 squat은 고관절에서 충분한 굴곡 범위를 요구한다. 가동성이 부족하면 일정 깊이에 도달했을 때 골반 정렬이 무너지거나, 추가 하강이 멈춘다. 결과적으로 깊이 부족 계열의 동작 패턴으로 이어진다. |
| caption_safe_expression | "하강 후반부에서 동작 범위가 제한되는 신호가 보이며, 깊이 부족 계열로 해석 가능한 신호입니다." |
| evidence_basis | 전문가 지식 |
| sensor_observability | **false** — 손목 IMU는 hip flexion을 직접 측정할 수 없음. 하강 phase 종결 시점의 가속도 변화로 *간접* 추정만 가능 |

---

### CT-03 — 체간 강성 부족 (코어 약화)

| 필드 | 값 |
|---|---|
| constraint_id | CT-03 |
| target_classes | [C2, C3] |
| constraint_name | 체간 강성 부족 |
| constraint_description | 복부·요추 안정성을 담당하는 깊은 코어 근육군의 약화로 하강 시 체간이 안정적인 정렬을 유지하지 못하는 신체 조건. |
| mechanism | 체간이 안정되지 못하면 무게중심이 흔들리고, 안전을 확보하기 위해 무의식적으로 깊이를 축소한다. 동시에 척추 굴곡과 골반 정렬 변화가 함께 동반될 수 있어 단일 결함이 아닌 복합 패턴으로 나타나는 경우가 많다. |
| caption_safe_expression | "하강 동작이 충분히 진행되지 못하는 패턴이 다른 자세 변화 신호와 함께 나타나는 경향이 있습니다." |
| evidence_basis | 전문가 지식 |
| sensor_observability | **false** — 체간 강성은 손목 IMU로 직접 측정할 수 없음. 손목의 미세한 좌우 흔들림으로 *간접* 시사만 가능 |

---

### CT-04 — 척추 굴곡 동반 패턴

| 필드 | 값 |
|---|---|
| constraint_id | CT-04 |
| target_classes | [C3] |
| constraint_name | 척추 굴곡 동반 패턴 |
| constraint_description | 코어 약화 또는 햄스트링 과긴장으로 하강 후반부에서 요추 굴곡과 골반 회전이 결합되어 발생하는 신체 조건. |
| mechanism | 골반 회전은 척추 굴곡을 동반하고, 무릎 위 부하 분포가 변하며, 무릎 정렬도 동시에 무너지기 쉽다. 단일 결함이 아니라 깊이 부족 + 골반 정렬 변화 + 무릎 정렬 변화가 *동시에* 출현하는 복합 패턴의 기저가 된다. |
| caption_safe_expression | "여러 자세 신호가 동시에 경계 패턴으로 나타나는 복합 오류 후보 패턴과 관련될 수 있습니다." |
| evidence_basis | 전문가 지식 + Step 2.5 empirical evidence (C3는 C2 + 추가 결함의 복합 패턴으로 정의됨) |
| sensor_observability | **false** — 척추 굴곡 자체는 측정 불가. 손목의 자세 변화 패턴으로 *간접* 추정만 가능 |

---

### CT-05 — 다중 가동성-안정성 동시 제약

| 필드 | 값 |
|---|---|
| constraint_id | CT-05 |
| target_classes | [C3] |
| constraint_name | 다중 신체 제약 동시 출현 |
| constraint_description | 발목·고관절·체간 안정성 중 둘 이상이 동시에 부족한 신체 조건. 단일 보상 전략으로 해결되지 않고 여러 보상이 중첩된다. |
| mechanism | 단일 제약은 한 가지 보상으로 해결되지만, 다중 제약은 깊이 부족과 자세 변화가 *동시에* 출현하게 만든다. 이는 복합 결함 클래스의 전형적인 기저 패턴이다. |
| caption_safe_expression | "여러 패턴이 함께 나타나는 복합 오류 후보 신호와 관련될 수 있습니다." |
| evidence_basis | 데이터 기반 (Step 2.5 confusion matrix에서 C3가 C2 및 무릎 관련 후보군과 동시에 혼동되는 양상에서 추론) |
| sensor_observability | **false** — 다중 제약 자체는 측정 불가. 손목 신호의 multi-axis 변동성 증가로 *간접* 시사 |

---

### CT-06 — 고관절 외전근 약화

| 필드 | 값 |
|---|---|
| constraint_id | CT-06 |
| target_classes | [C4, C5, C6] |
| constraint_name | 고관절 외전근 약화 |
| constraint_description | 둔중근(gluteus medius) 및 둔소근의 약화로 frontal plane에서 골반과 대퇴의 정렬을 유지하는 힘이 부족한 신체 조건. |
| mechanism | 외전근이 약하면 하강 시 대퇴가 내측으로 회전하며 무릎 정렬이 무너지는 변화가 발생한다. 좌우 근력 차이에 따라 비대칭하게 출현하기도 하고, 대칭하게 출현하기도 한다. |
| caption_safe_expression | "무릎 관련 오류 후보 패턴과 관련될 수 있는 전형적인 기저 조건입니다. 방향성 단정 없이 후보군 수준으로 해석하는 것이 적절합니다." |
| evidence_basis | 전문가 지식 (Powers 2010, J Orthop Sports Phys Ther 일반론) |
| sensor_observability | **false** — 손목 IMU에서 무릎 정렬은 측정 불가. 하강 phase 중 손목 가속도의 비대칭성으로 *부분* 시사만 가능 |

---

### CT-07 — 족부 회내 / 아치 붕괴

| 필드 | 값 |
|---|---|
| constraint_id | CT-07 |
| target_classes | [C4, C5, C6] |
| constraint_name | 족부 회내 동반 패턴 |
| constraint_description | 후족부 외반, 종골의 내측 기울기, 내측 종아치의 붕괴 등으로 발의 정렬이 내측으로 기우는 신체 조건. |
| mechanism | 족부 정렬이 내측으로 무너지면 경골이 내측으로 회전하고, 무릎 정렬이 연쇄적으로 무너진다. 발-무릎의 운동학적 사슬을 통해 무릎 정렬 변화가 유발되는 패턴이 전형적으로 나타난다. |
| caption_safe_expression | "무릎 관련 오류 후보 패턴이 발 정렬과 연계되어 나타나는 경향이 있습니다." |
| evidence_basis | 전문가 지식 (lower extremity kinematic chain 일반론) |
| sensor_observability | **false** — 손목 IMU에서 발 정렬은 측정 불가 |

---

### CT-08 — 무릎 신전근 약화

| 필드 | 값 |
|---|---|
| constraint_id | CT-08 |
| target_classes | [C2, C4, C5, C6] |
| constraint_name | 무릎 신전근 약화 |
| constraint_description | 대퇴사두근의 근력 부족으로 하강·상승 phase에서 무릎 안정성과 부하 흡수 능력이 저하된 신체 조건. |
| mechanism | 신전근이 약하면 깊은 ROM까지 안전하게 내려가지 못하거나 (깊이 부족 회피), 또는 깊은 자세에서 무릎 정렬 유지에 실패한다 (무릎 관련 후보군). 보상으로 체간이 더 앞으로 기울 수 있다. |
| caption_safe_expression | "하강 동작이 충분히 진행되지 못하거나 무릎 관련 오류 후보 패턴이 함께 나타날 수 있습니다." |
| evidence_basis | 전문가 지식 |
| sensor_observability | **false** — 손목 IMU에서 근력은 측정 불가. 하강 속도 곡선의 변화로 *간접* 시사 |

---

### CT-09 — 균형 잡힌 가동성-안정성 (정상 패턴의 기저)

| 필드 | 값 |
|---|---|
| constraint_id | CT-09 |
| target_classes | [C1] |
| constraint_name | 균형 잡힌 가동성-안정성 |
| constraint_description | 발목·고관절·척추·무릎 신전근이 모두 squat 동작 요구치를 충족하는 신체 조건. 단일 제약 우세 영역이 없으므로 보상 패턴이 출현하지 않는다. |
| mechanism | 모든 관절에서 가동성과 안정성이 균형 잡혀 있으면 보상 전략이 필요하지 않고, 동작이 표준 궤적을 따른다. 손목 IMU에서도 하강·상승 phase의 가속도-자이로 신호가 안정적이고 *반복 가능한* 패턴을 보인다. |
| caption_safe_expression | "동작 신호가 비교적 일관되게 나타나며, 깊이 부족 계열이나 무릎 관련 오류 후보 패턴이 두드러지지 않는 신호입니다." |
| evidence_basis | 전문가 지식 + Step 2.5 empirical evidence (C1은 가장 분리도가 높고 보상 신호가 약함) |
| sensor_observability | **true (부분)** — 손목 IMU는 *보상 부재*를 신호의 일관성으로 간접 관측 가능. 다만 "정상"임을 단정적으로 확정할 수는 없으며, 다른 후보군이 두드러지지 않는다는 형태로만 표현됨 |

---

### CT-10 — 좌우 비대칭 부하 분배

| 필드 | 값 |
|---|---|
| constraint_id | CT-10 |
| target_classes | [C4, C5] |
| constraint_name | 좌우 부하 비대칭 |
| constraint_description | 좌·우 다리 근력 차이, 부상 이력에 따른 회피, 또는 우세 다리 의존 패턴 등으로 하강·상승 phase에서 좌우 부하가 비대칭하게 분포되는 신체 조건. |
| mechanism | 비대칭 부하는 한쪽 다리에서만 정렬 변화가 더 강하게 출현하게 만들어 비대칭 결함으로 분류되는 기저가 된다. 좌우 다리가 같은 신체 조건일 때는 대칭 결함으로, 차이가 있을 때는 비대칭 결함으로 나타난다. |
| caption_safe_expression | "무릎 관련 오류 후보 패턴이 방향성 단정 없이 후보군 수준에서 함께 나타날 수 있습니다." |
| evidence_basis | 전문가 지식 |
| sensor_observability | **false** — wrist IMU 단독으로는 좌우 다리 부하 비대칭을 직접 측정 불가. 양 손목 데이터가 있을 경우에만 부분 시사 가능 |

---

## 2. 클래스별 커버리지 점검

| 클래스 | 정의 | 매핑된 constraint_id | 개수 |
|---|---|---|---:|
| C1 | Normal | CT-09 | 1 |
| C2 | Insufficient depth | CT-01, CT-02, CT-03, CT-08 | 4 |
| C3 | Insufficient depth + posterior tilting + knee valgus | CT-01, CT-02, CT-03, CT-04, CT-05 | 5 |
| C4 | Left-knee valgus | CT-06, CT-07, CT-08, CT-10 | 4 |
| C5 | Right-knee valgus | CT-06, CT-07, CT-08, CT-10 | 4 |
| C6 | Both-knee valgus | CT-06, CT-07, CT-08 | 3 |

요구된 최소 커버리지(C2≥3, C3≥2, C4/C5/C6≥2, C1=1)는 모두 충족됨.

---

## 3. caption_safe_expression 어휘 정합성 검증

본 파일의 모든 `caption_safe_expression`은 `reports/step6_v2/step6_v2_final_system_prompt.md` §7 금지 목록과 다음과 같이 정합한다.

| 금지 표현 | 본 파일의 우회 표현 |
|---|---|
| "무릎이 안쪽으로 들어갔다" / "knee valgus" | "무릎 관련 오류 후보 패턴" |
| "후방 경사" / "posterior tilting" / "골반이 뒤로 말렸다" | "여러 자세 신호가 동시에 경계 패턴으로 나타나는" / "복합 오류 후보 패턴" |
| "한쪽" / "양측" / "왼쪽" / "오른쪽" | "방향성 단정 없이 후보군 수준으로" |
| "관절각" | (사용하지 않음. 모두 "동작 범위 / 동작 신호"로 표현) |
| 단정형 ("확실히", "명확히") | "관련될 수 있습니다" / "동반되는 경우가 많습니다" / "전형적으로 나타나는 패턴입니다" |

본 표는 caption 생성 후 Step 8_v2 자동 검증에서 reverse-check 대상이 된다.

---

## 4. 구현 가이드 (다음 단계에서 사용)

본 문서는 *설계 단계*이며 본 단계에서 코드 수정을 수행하지 않는다. 후속 구현 시 다음 위치를 수정한다.

### 4.1 `scripts/generate_step7_v2_captions.py` 수정 위치 (2군데)

**(A) 파일 상단 — 상수 정의 영역**

본 문서의 10개 entry를 Python `CONSTRAINT_TEMPLATES: list[dict]`로 옮기거나, 본 markdown을 파싱하는 헬퍼를 추가한다. `POSTURE_VOCAB`, `CLASS_VOCAB` 등 기존 상수들과 동일한 영역에 위치시킨다.

**(B) `build_user_payload()` (현재 261행) — return dict 확장**

`build_user_payload()`가 반환하는 dict에 `constraint_template` 필드를 추가한다. 단, schema의 `class_set`과 교집합이 있는 entry만 필터링해 주입한다.

```python
# (예시 — 본 문서는 코드를 수정하지 않으며, 후속 단계에서 적용)
relevant_constraints = [
    t for t in CONSTRAINT_TEMPLATES
    if set(t["target_classes"]) & set(cs)
]
return {
    "task": "schema_to_korean_caption",
    "schema": { ... },
    "posture_vocabulary": POSTURE_VOCAB,
    "class_vocabulary": CLASS_VOCAB,
    "confidence_phrase_pool": CONFIDENCE_PHRASE_POOL,
    "uncertainty_phrase_pool": UNCERTAINTY_PHRASE_POOL,
    "limitation_phrase_pool": LIMITATION_PHRASE_POOL,
    "constraint_template": relevant_constraints,  # 신규 필드
}
```

`no_call=true`인 경우에는 `constraint_template`을 빈 리스트로 주입하거나 키 자체를 생략한다 — schema 단정 금지 정책을 우회하지 않기 위함.

### 4.2 `reports/step6_v2/step6_v2_final_system_prompt.md` 수정 위치

현재 system prompt §1~§8 뒤에 다음과 같은 §9 (또는 [엄격 제약]의 신규 항목)를 *추가 갱신*한다. 본 문서를 *덮어쓰지 말고* 변경 일자와 변경 내역을 함께 추가한다.

> §9 (또는 [엄격 제약] 신규 항목) 예시 — 후속 단계에서 본 문구를 최종 잠근다.
> ```
> 9. user payload에 포함된 `constraint_template`은 *내부 배경 지식*이며, 
>    caption 본문(`caption_ko`)에 의학·생체역학 용어를 그대로 인용하지 마십시오.
>    `constraint_template[i].caption_safe_expression`에 적힌 표현만, 
>    그리고 schema의 `class_set` / `caption_confidence_level` / `no_call`과 
>    *충돌하지 않는 범위에서*만, caption 본문에 풀어 쓸 수 있습니다. 
>    constraint_template에 의존하여 schema 사실을 *확장*하지 마십시오.
> ```

이 변경은 §7의 기존 금지 목록을 *그대로 유지*한 채 추가 제약을 부과하는 형태이며, 충돌을 만들지 않는다.

### 4.3 Golden test case 재검증

본 변경 후 다음 순서로 재검증한다.

```bash
# (1) mock provider로 golden 재실행 — 기존 golden과 schema_faithfulness 비교
python scripts/generate_step7_v2_captions.py --mode golden --provider mock

# (2) 산출물 mismatch 확인
#     data/step7_v2/step7_v2_golden_captions.csv 와
#     reports/step6_v2/step6_v2_golden_test_cases.json 의 정합성 점검

# (3) Step 7_v2 b01.5/seed42 mock rerun (현재 미커밋 변경 영역과 동일)
#     reports/step7_v2/b01_5_seed42/step7_v2_caption_generation_results.md 갱신
```

기대 조건:
- **schema_faithfulness_rate** ≥ 0.99 유지 (현재 0.9998 baseline)
- 신규 어휘 충돌 0건 (Step 8_v2 vocabulary check 통과)
- `constraint_template` 주입 후에도 `class_set narrowing` 위반 0건

위 세 조건이 모두 만족되면 anthropic provider로 본 실행을 진행하며, 위반이 발견되면 본 문서의 `caption_safe_expression`을 entry별로 좁혀 재시도한다.

---

## 5. 본 문서가 변경되는 조건

- Step 8_v2 자동 검증에서 본 문서의 `caption_safe_expression` 중 어떤 entry가 *반복적으로* 위반을 유발할 때 (해당 entry의 표현을 좁힘).
- Step 4R-C clinical corpus와 본 템플릿 사이에 *중복 어휘*가 발견되어 dual-layer design 원칙(internal clinical vs external policy)이 위반될 때.
- 클래스 정의가 변경될 때 (현 시점에서는 변경 없음).
- 새로운 신체 제약 후보 entry가 전문가 검토에서 추가될 때 (확장).

각 변경은 본 문서를 *덮어쓰지 않고*, 변경 일자 및 변경 내역을 추가하여 진행한다.

---

*본 문서는 prototype 지식 설계이며, 코드·schema·caption을 새로 생성하지 않는다. Step 1~3 산출물은 본 문서로 인해 수정되지 않는다.*
