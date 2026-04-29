# AICRC 스쿼트 데이터셋 구조 감사 보고서

- 데이터셋 루트: `C:\Users\user\data\AICRC_DSQ_Data(Combined_segmented)_v2\AICRC_DSQ_Data(Combined_segmented)`
- 참조 논문: *Fine-Grained Motion Recognition in At-Home Fitness*
  (Yun et al., Healthcare 2023, 11, 940)
- 생성 스크립트: `scripts/audit_aicrc_dataset_structure.py`

## 0. Step 1 — 고정된 사용 범위 (LOCKED)

아래 결정사항은 **고정**되며, 이후 모든 단계(manifest 생성, 학습 split,
captioning, 평가)는 이 범위를 따른다.

| 항목 | 값 | 근거 (한 줄 요약) |
|---|---|---|
| 사용 클래스          | **C1, C2, C3, C4, C5, C6** | 논문 Table 1이 정확히 이 여섯 클래스만 정의함. C7은 **제외**. |
| 사용 데이터 타입      | **Segmented 만**           | sensor-to-text는 rep 단위 학습이 필요. Combined는 **제외**. |
| Posture 코드 (canonical) | **SA / CA / HW**       | 논문 Table 2 / Figure 3 약어. 레거시 `AS / AC / AW`는 metadata 별칭으로만 보존. |

이후 모든 코드/문서에서 사용할 canonical posture 매핑:

| 디스크 폴더명          | Canonical (SA/CA/HW) | Legacy (AS/AC/AW) | Figure 3 |
|---|---|---|---|
| `01_Straight_Arms` | **SA** | AS | Figure 3 (a) straight arms |
| `02_Crossed_Arms` | **CA** | AC | Figure 3 (b) crossed arms |
| `03_Hands_on_Waist` | **HW** | AW | Figure 3 (c) hands on waist |

Step 1 scope 규모:

- 6 classes × 3 postures = **18개 조건**
- 18개 조건 × 약 52명 참여자 = **약 935개 subject 서브폴더**
  (한 조건은 51개. 누락된 런 1건 — Section 2 참고)
- 서브폴더당 평균 약 9.9 reps → **약 9,200개 rep-level (signal, caption) 샘플**

이 섹션 아래의 내용은 모두 위 lock을 뒷받침하는 근거 자료이며,
위 범위를 절대 완화하지 않는다.

## 1. 논문 Table 1 / Figure 3 vs 실제 폴더

| 폴더 | 논문 Table 1 대응 | 설명 (논문) | Step 1 scope 포함? |
|---|---|---|---|
| `C1` | C1 | 정상 스쿼트 | **예 (포함)** |
| `C2` | C2 | Insufficient depth (불충분한 하강) | **예 (포함)** |
| `C3` | C3 | Insufficient depth + posterior tilting + knee valgus | **예 (포함)** |
| `C4` | C4 | Left-knee valgus (왼쪽 무릎 안쪽 collapse) | **예 (포함)** |
| `C5` | C5 | Right-knee valgus (오른쪽 무릎 안쪽 collapse) | **예 (포함)** |
| `C6` | C6 | Both-knee valgus (양쪽 무릎 안쪽 collapse) | **예 (포함)** |
| `C7` | 논문에 없음 | 논문 Table 1에 정의되지 않음 (Step 1 제외) | 아니오 (scope 외) |

## 2. C1–C7 인벤토리 (디스크 실측)

디스크상의 모든 `C1..C7` 폴더에는 동일한 3개 posture 서브폴더와
동일한 2개 data-type 서브폴더(`Combined`, `Segmented`)가 존재한다.
Step 1은 C1–C6 × Segmented만 사용하며, 아래 표는 디스크에 무엇이 있는지
기록하기 위한 참고용이다.

| Class | 설명 | 3 postures 모두 존재 | Combined 항목 합 | Segmented 서브폴더 합 | Segmented rep 파일 합 |
|---|---|---|---|---|---|
| C1 | Normal | 예 | 156 | 156 | 1553 |
| C2 | Insufficient depth | 예 | 156 | 156 | 1555 |
| C3 | Insufficient depth with posterior tilting and knee valgus | 예 | 156 | 156 | 1529 |
| C4 | Left-knee valgus | 예 | 155 | 155 | 1540 |
| C5 | Right-knee valgus | 예 | 156 | 156 | 1549 |
| C6 | Both-knee valgus | 예 | 156 | 156 | 1549 |
| C7 | NOT in paper Table 1 (out of scope for Step 1) | 예 | 156 | 155 | 1539 |

주의: `C4 / 02_Crossed_Arms`와 `C7 / 01_Straight_Arms`의 Segmented는
52가 아닌 51 — 원본 데이터 수집 시 1건이 누락된 것으로 보인다.
Step 1 scope에서는 C4 한 조건에만 영향이 있고, 무시 가능한 수준이다.

## 3. Combined vs Segmented — 파일 단위 사실

### Combined  (Step 1에서 **사용하지 않음**)
- (참여자 × class × posture) 당 `.txt` 파일 1개.
- 파일명 패턴: `EP{NN}_{Posture}_{C{n}}_SW_combine.txt`.
- 9개 공백 구분 컬럼, 약 1,500 행. 한 세션 전체(약 10회 스쿼트 + 휴식)가 하나로 이어붙어 있음.
- 컬럼 구성: index, timestamp(ms), ax, ay, az, gx, gy, gz, 그리고 끝에 0으로 채워진 컬럼 2개.
- 제외 이유: 한 파일이 여러 rep과 휴식을 섞어 담고 있어 caption과 정합하려면
  자체 rep-segmenter를 만들어야 함. 본 연구 scope를 벗어남.

### Segmented  (Step 1에서 **사용**)
- (참여자 × class × posture) 당 서브폴더 1개:
  `seg_EP{NN}_{Posture}_{C{n}}_SW_combine/`.
- 서브폴더 내부:
    - `file_YYYY_MM_DD_HH_MM_SS_{rep}_o0.txt` → **파일 1개 = 스쿼트 1회 (1 rep)**
    - 각 rep 파일: 7개 공백 구분 컬럼, 약 120–200 행 (≈ 3초 @ ~50 Hz)
    - `_time.txt`         → rep 경계(초 단위)
    - `_vkeep_st_end.txt` → rep 시작/끝 프레임 index 2행 행렬
- 서브폴더당 평균 약 9.9개의 rep 파일 (대략 10회 반복).
- rep 경계는 논문 저자가 영상을 직접 보면서 수동 분절한 결과(논문 §2.3)이므로
  ground truth 품질이 높다.

## 4. 왜 Segmented만 사용하는가 (lock)

sensor-to-text 학습은 본질적으로 rep 단위 supervision이다. 즉,
스쿼트 1회가 caption 1개에 대응되므로 학습 unit도 rep 단위여야 한다.

- Segmented: 파일 1개 = rep 1개 → (signal, caption) 페어가 직접 나옴.
  C1–C6 합쳐 약 9,200 샘플이며 별도 segmentation 단계가 필요 없음.
- Combined : 파일 1개 = 세션 전체 → segmenter를 처음부터 만들어야 함.
- 논문 Table 4: 수동 분절(manually-segmented) 조건이 자동 windowing 조건보다
  성능이 높음. Segmented를 채택하면 논문 자체가 더 신뢰한 split과 일치한다.

**Step 1 lock**: Segmented만 사용.

## 5. 왜 C7을 제외하는가 (lock)

- 논문 본문: *"the six classes (C1 through C6). The total data collection time was 386.8 min."*
  → C7은 Table 1에 **존재하지 않음**.
- 디스크 실측: `C7/` 폴더는 동일한 posture/data-type 구조와 약 52명의 참여자를
  가지고 있지만, 내부 `_time.txt`가 C1–C6과는 다른 레거시 class 코드
  (예: `C39`)를 참조한다. 이는 C7이 다른 프로토콜이나 미공개 후속 연구를 위해
  수집되었을 가능성을 강하게 시사한다.
- 논문에 정의된 squat-fault 설명이 없으므로 신뢰할 수 있는
  template caption을 작성할 수 없다.

**Step 1 lock**: C7 제외. 재포함하려면 (a) 클래스 semantics 확정,
(b) caption template 정의 두 가지가 명시적으로 문서화되어야 하며,
이는 본 단계 scope를 벗어난다.

## 6. Posture 명칭 — canonical SA / CA / HW (lock)

아티팩트에는 세 가지 명명 규칙이 공존한다. Step 1은 **논문**의 약어
(SA / CA / HW)를 canonical로 표준화한다. 레거시 스크립트 약어
(AS / AC / AW)는 `_time.txt` 내부에서 발견되며, 레거시 전처리와의
join을 위해 metadata 별칭으로만 보존한다.

| Posture (Figure 3)         | 디스크 폴더            | Canonical (paper) | Legacy (script) |
|---|---|---|---|
| (a) straight arms          | `01_Straight_Arms`    | **SA**            | AS              |
| (b) crossed arms           | `02_Crossed_Arms`     | **CA**            | AC              |
| (c) hands on waist         | `03_Hands_on_Waist`   | **HW**            | AW              |

코드/파일명에서 짧은 posture 키가 필요하면 **반드시 SA / CA / HW** 사용.
AS / AC / AW는 레거시 별칭을 명시하는 필드에서만 등장해야 한다.

## 7. 최종 데이터셋 사양 (Step 1)

```
dataset_root : C:\Users\user\data\AICRC_DSQ_Data(Combined_segmented)_v2\AICRC_DSQ_Data(Combined_segmented)
classes      : C1, C2, C3, C4, C5, C6                          # C7 미사용
postures     : SA, CA, HW                                      # canonical
                (folders: 01_Straight_Arms, 02_Crossed_Arms,
                          03_Hands_on_Waist)
data_type    : Segmented                                       # Combined 미사용
unit         : rep 파일 1개 = (signal, caption) 샘플 1개
scale        : 약 935개 subject 서브폴더, 약 9,200개 rep-level 샘플
```

## 8. Manifest 생성 규칙 (Step 1)

manifest의 한 행 = Step 1 scope 내의 rep 파일 1개.

| 컬럼 | 출처 / 규칙 |
|---|---|
| `rep_id`             | `f"{participant_id}_{class_id}_{posture_canonical}_rep{rep_index:02d}"` |
| `participant_id`     | 서브폴더명에서 파싱 (`EP01..EP52`) |
| `class_id`           | 경로에서 추출 (`C1..C6` — C7은 절대 등장 금지) |
| `class_description`  | `CLASS_DESCRIPTIONS[class_id]` |
| `posture_canonical`  | **SA / CA / HW** (posture의 primary key) |
| `posture_legacy`     | AS / AC / AW (별칭 전용) |
| `posture_folder`     | `01_Straight_Arms` / `02_Crossed_Arms` / `03_Hands_on_Waist` |
| `rep_index`          | `file_..._{rep}_o0.txt`의 정수 (1-based) |
| `signal_path`        | rep `.txt` 파일의 절대 경로 |
| `n_rows`             | rep 파일 행 수 (sequence length) |
| `start_idx` / `end_idx` | 부모 서브폴더의 `_vkeep_st_end.txt`에서 |
| `t_start_s` / `t_end_s` | 부모 서브폴더의 `_time.txt`에서 |
| `split`              | participant 단위 split (예: 36 / 8 / 8 train/val/test) |

필터링 규칙:
1. 행 수가 최소 임계값(예: 50 행) 미만인 rep 파일은 truncation으로 간주하여 drop.
2. **`C7` 경로는 manifest 단계에서도 거부**한다 (collector가 이미 `STEP1_CLASSES`로
   제한하지만 방어적으로 한 번 더 차단).
3. **`Combined` 경로는 manifest 단계에서도 거부**한다 (같은 이유).
4. split은 **rep 단위가 아닌 participant 단위**로 자른다 — 한 참여자가
   train과 test에 동시에 들어가면 안 됨.
5. (class × posture)로 stratify하여 모든 fold가 18개 조건을 모두 보도록 한다.

## 9. 조건별 파일 수 표 (Step 1 scope)

전체 표는 `reports/dataset_file_count_by_condition.csv`에도 동일하게 기록됨.

| class | 설명 | posture (canonical) | 폴더 | data_type | file_count | 예시 파일 |
|---|---|---|---|---|---|---|
| C1 | Normal | **SA** | `01_Straight_Arms` | Segmented | 52 | `seg_EP01_Straight_Arms_C1_SW_combine` |
| C1 | Normal | **CA** | `02_Crossed_Arms` | Segmented | 52 | `seg_EP01_Crossed_Arms_C1_SW_combine` |
| C1 | Normal | **HW** | `03_Hands_on_Waist` | Segmented | 52 | `seg_EP01_Hands_on_Waist_C1_SW_combine` |
| C2 | Insufficient depth | **SA** | `01_Straight_Arms` | Segmented | 52 | `seg_EP01_Straight_Arms_C2_SW_combine` |
| C2 | Insufficient depth | **CA** | `02_Crossed_Arms` | Segmented | 52 | `seg_EP01_Crossed_Arms_C2_SW_combine` |
| C2 | Insufficient depth | **HW** | `03_Hands_on_Waist` | Segmented | 52 | `seg_EP01_Hands_on_Waist_C2_SW_combine` |
| C3 | Insufficient depth with posterior tilting and knee valgus | **SA** | `01_Straight_Arms` | Segmented | 52 | `seg_EP01_Straight_Arms_C3_SW_combine` |
| C3 | Insufficient depth with posterior tilting and knee valgus | **CA** | `02_Crossed_Arms` | Segmented | 52 | `seg_EP01_Crossed_Arms_C3_SW_combine` |
| C3 | Insufficient depth with posterior tilting and knee valgus | **HW** | `03_Hands_on_Waist` | Segmented | 52 | `seg_EP01_Hands_on_Waist_C3_SW_combine` |
| C4 | Left-knee valgus | **SA** | `01_Straight_Arms` | Segmented | 52 | `seg_EP01_Straight_Arms_C4_SW_combine` |
| C4 | Left-knee valgus | **CA** | `02_Crossed_Arms` | Segmented | 51 | `seg_EP01_Crossed_Arms_C4_SW_combine` |
| C4 | Left-knee valgus | **HW** | `03_Hands_on_Waist` | Segmented | 52 | `seg_EP01_Hands_on_Waist_C4_SW_combine` |
| C5 | Right-knee valgus | **SA** | `01_Straight_Arms` | Segmented | 52 | `seg_EP01_Straight_Arms_C5_SW_combine` |
| C5 | Right-knee valgus | **CA** | `02_Crossed_Arms` | Segmented | 52 | `seg_EP01_Crossed_Arms_C5_SW_combine` |
| C5 | Right-knee valgus | **HW** | `03_Hands_on_Waist` | Segmented | 52 | `seg_EP01_Hands_on_Waist_C5_SW_combine` |
| C6 | Both-knee valgus | **SA** | `01_Straight_Arms` | Segmented | 52 | `seg_EP01_Straight_Arms_C6_SW_combine` |
| C6 | Both-knee valgus | **CA** | `02_Crossed_Arms` | Segmented | 52 | `seg_EP01_Crossed_Arms_C6_SW_combine` |
| C6 | Both-knee valgus | **HW** | `03_Hands_on_Waist` | Segmented | 52 | `seg_EP01_Hands_on_Waist_C6_SW_combine` |
