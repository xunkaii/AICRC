"""Audit the AICRC squat dataset structure and generate reports.

Step 1 locked scope (as agreed):
  - Classes:    C1-C6 only         (C7 excluded; not in paper Table 1)
  - Data type:  Segmented only     (Combined excluded; rep-level supervision required)
  - Postures:   canonical SA/CA/HW (paper Table 2 / Figure 3 abbreviations)

Outputs:
  reports/dataset_file_count_by_condition.csv   <- Step 1 scope rows only
  reports/dataset_structure_audit.md            <- Step 1 lock + supporting evidence

Run:
  python scripts/audit_aicrc_dataset_structure.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

DATASET_ROOT = Path(
    r"C:\Users\user\data\AICRC_DSQ_Data(Combined_segmented)_v2"
    r"\AICRC_DSQ_Data(Combined_segmented)"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORTS_DIR / "dataset_file_count_by_condition.csv"
MD_PATH = REPORTS_DIR / "dataset_structure_audit.md"

# ---- Step 1 locked scope ----------------------------------------------------
STEP1_CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
STEP1_DATA_TYPE = "Segmented"

# Full filesystem inventory (used only for evidence sections of the report).
ALL_CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
ALL_DATA_TYPES = ["Combined", "Segmented"]
POSTURE_FOLDERS = ["01_Straight_Arms", "02_Crossed_Arms", "03_Hands_on_Waist"]

CLASS_DESCRIPTIONS = {
    "C1": "Normal",
    "C2": "Insufficient depth",
    "C3": "Insufficient depth with posterior tilting and knee valgus",
    "C4": "Left-knee valgus",
    "C5": "Right-knee valgus",
    "C6": "Both-knee valgus",
    "C7": "NOT in paper Table 1 (out of scope for Step 1)",
}

# Canonical posture codes = paper abbreviations (SA / CA / HW).
POSTURE_CANONICAL = {
    "01_Straight_Arms":  "SA",
    "02_Crossed_Arms":   "CA",
    "03_Hands_on_Waist": "HW",
}
POSTURE_LEGACY = {
    "01_Straight_Arms":  "AS",
    "02_Crossed_Arms":   "AC",
    "03_Hands_on_Waist": "AW",
}
POSTURE_FIGURE3 = {
    "01_Straight_Arms":  "Figure 3 (a) straight arms",
    "02_Crossed_Arms":   "Figure 3 (b) crossed arms",
    "03_Hands_on_Waist": "Figure 3 (c) hands on waist",
}


def list_top_level(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(os.listdir(folder))


def count_segmented_reps(seg_root: Path) -> tuple[int, int]:
    """Return (subject_folder_count, total_rep_file_count) under a Segmented dir."""
    subjects = 0
    reps = 0
    if not seg_root.is_dir():
        return 0, 0
    for sub in os.listdir(seg_root):
        sp = seg_root / sub
        if not sp.is_dir():
            continue
        subjects += 1
        for f in os.listdir(sp):
            if f.startswith("file_") and f.endswith(".txt"):
                reps += 1
    return subjects, reps


def collect_step1_rows() -> list[dict]:
    """Step 1 scope rows only: C1-C6 x 3 postures x Segmented."""
    rows: list[dict] = []
    for c in STEP1_CLASSES:
        for ap in POSTURE_FOLDERS:
            folder = DATASET_ROOT / c / ap / STEP1_DATA_TYPE
            entries = list_top_level(folder)
            example = entries[0] if entries else ""
            canonical = POSTURE_CANONICAL[ap]
            legacy = POSTURE_LEGACY[ap]
            label = f"{canonical} (canonical) / {legacy} (legacy script) - {POSTURE_FIGURE3[ap]}"
            rows.append({
                "class": c,
                "class_description": CLASS_DESCRIPTIONS[c],
                "arm_posture": ap,
                "arm_posture_canonical": canonical,
                "arm_posture_label": label,
                "data_type": STEP1_DATA_TYPE,
                "file_count": len(entries),
                "example_file": example,
            })
    return rows


def collect_full_inventory() -> list[dict]:
    """Every class x posture x data_type combination on disk (for evidence only)."""
    rows: list[dict] = []
    for c in ALL_CLASSES:
        for ap in POSTURE_FOLDERS:
            for dt in ALL_DATA_TYPES:
                folder = DATASET_ROOT / c / ap / dt
                entries = list_top_level(folder)
                rows.append({
                    "class": c,
                    "arm_posture": ap,
                    "arm_posture_canonical": POSTURE_CANONICAL[ap],
                    "data_type": dt,
                    "file_count": len(entries),
                    "example_file": entries[0] if entries else "",
                })
    return rows


def write_csv(step1_rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "class", "class_description",
        "arm_posture", "arm_posture_canonical", "arm_posture_label",
        "data_type", "file_count", "example_file",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(step1_rows)


def write_markdown(step1_rows: list[dict], full_rows: list[dict]) -> None:
    seg_stats = {}
    for c in ALL_CLASSES:
        subj_total = 0
        rep_total = 0
        for ap in POSTURE_FOLDERS:
            s, r = count_segmented_reps(DATASET_ROOT / c / ap / "Segmented")
            subj_total += s
            rep_total += r
        seg_stats[c] = (subj_total, rep_total)

    lines: list[str] = []
    add = lines.append

    add("# AICRC 스쿼트 데이터셋 구조 감사 보고서")
    add("")
    add(f"- 데이터셋 루트: `{DATASET_ROOT}`")
    add("- 참조 논문: *Fine-Grained Motion Recognition in At-Home Fitness*")
    add("  (Yun et al., Healthcare 2023, 11, 940)")
    add("- 생성 스크립트: `scripts/audit_aicrc_dataset_structure.py`")
    add("")

    # ---- Section 0: Step 1 locked scope -------------------------------------
    add("## 0. Step 1 — 고정된 사용 범위 (LOCKED)")
    add("")
    add("아래 결정사항은 **고정**되며, 이후 모든 단계(manifest 생성, 학습 split,")
    add("captioning, 평가)는 이 범위를 따른다.")
    add("")
    add("| 항목 | 값 | 근거 (한 줄 요약) |")
    add("|---|---|---|")
    add("| 사용 클래스          | **C1, C2, C3, C4, C5, C6** | 논문 Table 1이 정확히 이 여섯 클래스만 정의함. C7은 **제외**. |")
    add("| 사용 데이터 타입      | **Segmented 만**           | sensor-to-text는 rep 단위 학습이 필요. Combined는 **제외**. |")
    add("| Posture 코드 (canonical) | **SA / CA / HW**       | 논문 Table 2 / Figure 3 약어. 레거시 `AS / AC / AW`는 metadata 별칭으로만 보존. |")
    add("")
    add("이후 모든 코드/문서에서 사용할 canonical posture 매핑:")
    add("")
    add("| 디스크 폴더명          | Canonical (SA/CA/HW) | Legacy (AS/AC/AW) | Figure 3 |")
    add("|---|---|---|---|")
    for ap in POSTURE_FOLDERS:
        fig_kor = {
            "01_Straight_Arms":  "Figure 3 (a) straight arms",
            "02_Crossed_Arms":   "Figure 3 (b) crossed arms",
            "03_Hands_on_Waist": "Figure 3 (c) hands on waist",
        }[ap]
        add(f"| `{ap}` | **{POSTURE_CANONICAL[ap]}** | {POSTURE_LEGACY[ap]} | {fig_kor} |")
    add("")
    add("Step 1 scope 규모:")
    add("")
    add("- 6 classes × 3 postures = **18개 조건**")
    add("- 18개 조건 × 약 52명 참여자 = **약 935개 subject 서브폴더**")
    add("  (한 조건은 51개. 누락된 런 1건 — Section 2 참고)")
    add("- 서브폴더당 평균 약 9.9 reps → **약 9,200개 rep-level (signal, caption) 샘플**")
    add("")
    add("이 섹션 아래의 내용은 모두 위 lock을 뒷받침하는 근거 자료이며,")
    add("위 범위를 절대 완화하지 않는다.")
    add("")

    # ---- Section 1: paper vs disk ------------------------------------------
    add("## 1. 논문 Table 1 / Figure 3 vs 실제 폴더")
    add("")
    add("| 폴더 | 논문 Table 1 대응 | 설명 (논문) | Step 1 scope 포함? |")
    add("|---|---|---|---|")
    for c in ALL_CLASSES:
        if c in STEP1_CLASSES:
            in_scope = "**예 (포함)**"
            paper_id = c
            desc_kor = {
                "C1": "정상 스쿼트",
                "C2": "Insufficient depth (불충분한 하강)",
                "C3": "Insufficient depth + posterior tilting + knee valgus",
                "C4": "Left-knee valgus (왼쪽 무릎 안쪽 collapse)",
                "C5": "Right-knee valgus (오른쪽 무릎 안쪽 collapse)",
                "C6": "Both-knee valgus (양쪽 무릎 안쪽 collapse)",
            }[c]
        else:
            in_scope = "아니오 (scope 외)"
            paper_id = "논문에 없음"
            desc_kor = "논문 Table 1에 정의되지 않음 (Step 1 제외)"
        add(f"| `{c}` | {paper_id} | {desc_kor} | {in_scope} |")
    add("")

    # ---- Section 2: inventory ----------------------------------------------
    add("## 2. C1–C7 인벤토리 (디스크 실측)")
    add("")
    add("디스크상의 모든 `C1..C7` 폴더에는 동일한 3개 posture 서브폴더와")
    add("동일한 2개 data-type 서브폴더(`Combined`, `Segmented`)가 존재한다.")
    add("Step 1은 C1–C6 × Segmented만 사용하며, 아래 표는 디스크에 무엇이 있는지")
    add("기록하기 위한 참고용이다.")
    add("")
    add("| Class | 설명 | 3 postures 모두 존재 | Combined 항목 합 | Segmented 서브폴더 합 | Segmented rep 파일 합 |")
    add("|---|---|---|---|---|---|")
    for c in ALL_CLASSES:
        comb_sum = sum(r["file_count"] for r in full_rows
                       if r["class"] == c and r["data_type"] == "Combined")
        seg_subj, seg_reps = seg_stats[c]
        all_three = all((DATASET_ROOT / c / ap).is_dir() for ap in POSTURE_FOLDERS)
        all_three_kor = "예" if all_three else "**아니오**"
        add(f"| {c} | {CLASS_DESCRIPTIONS[c]} | {all_three_kor} | "
            f"{comb_sum} | {seg_subj} | {seg_reps} |")
    add("")
    add("주의: `C4 / 02_Crossed_Arms`와 `C7 / 01_Straight_Arms`의 Segmented는")
    add("52가 아닌 51 — 원본 데이터 수집 시 1건이 누락된 것으로 보인다.")
    add("Step 1 scope에서는 C4 한 조건에만 영향이 있고, 무시 가능한 수준이다.")
    add("")

    # ---- Section 3: Combined vs Segmented ----------------------------------
    add("## 3. Combined vs Segmented — 파일 단위 사실")
    add("")
    add("### Combined  (Step 1에서 **사용하지 않음**)")
    add("- (참여자 × class × posture) 당 `.txt` 파일 1개.")
    add("- 파일명 패턴: `EP{NN}_{Posture}_{C{n}}_SW_combine.txt`.")
    add("- 9개 공백 구분 컬럼, 약 1,500 행. 한 세션 전체(약 10회 스쿼트 + 휴식)가 하나로 이어붙어 있음.")
    add("- 컬럼 구성: index, timestamp(ms), ax, ay, az, gx, gy, gz, 그리고 끝에 0으로 채워진 컬럼 2개.")
    add("- 제외 이유: 한 파일이 여러 rep과 휴식을 섞어 담고 있어 caption과 정합하려면")
    add("  자체 rep-segmenter를 만들어야 함. 본 연구 scope를 벗어남.")
    add("")
    add("### Segmented  (Step 1에서 **사용**)")
    add("- (참여자 × class × posture) 당 서브폴더 1개:")
    add("  `seg_EP{NN}_{Posture}_{C{n}}_SW_combine/`.")
    add("- 서브폴더 내부:")
    add("    - `file_YYYY_MM_DD_HH_MM_SS_{rep}_o0.txt` → **파일 1개 = 스쿼트 1회 (1 rep)**")
    add("    - 각 rep 파일: 7개 공백 구분 컬럼, 약 120–200 행 (≈ 3초 @ ~50 Hz)")
    add("    - `_time.txt`         → rep 경계(초 단위)")
    add("    - `_vkeep_st_end.txt` → rep 시작/끝 프레임 index 2행 행렬")
    add("- 서브폴더당 평균 약 9.9개의 rep 파일 (대략 10회 반복).")
    add("- rep 경계는 논문 저자가 영상을 직접 보면서 수동 분절한 결과(논문 §2.3)이므로")
    add("  ground truth 품질이 높다.")
    add("")

    # ---- Section 4: why segmented ------------------------------------------
    add("## 4. 왜 Segmented만 사용하는가 (lock)")
    add("")
    add("sensor-to-text 학습은 본질적으로 rep 단위 supervision이다. 즉,")
    add("스쿼트 1회가 caption 1개에 대응되므로 학습 unit도 rep 단위여야 한다.")
    add("")
    add("- Segmented: 파일 1개 = rep 1개 → (signal, caption) 페어가 직접 나옴.")
    add("  C1–C6 합쳐 약 9,200 샘플이며 별도 segmentation 단계가 필요 없음.")
    add("- Combined : 파일 1개 = 세션 전체 → segmenter를 처음부터 만들어야 함.")
    add("- 논문 Table 4: 수동 분절(manually-segmented) 조건이 자동 windowing 조건보다")
    add("  성능이 높음. Segmented를 채택하면 논문 자체가 더 신뢰한 split과 일치한다.")
    add("")
    add("**Step 1 lock**: Segmented만 사용.")
    add("")

    # ---- Section 5: C7 -----------------------------------------------------
    add("## 5. 왜 C7을 제외하는가 (lock)")
    add("")
    add("- 논문 본문: *\"the six classes (C1 through C6). The total data collection time was 386.8 min.\"*")
    add("  → C7은 Table 1에 **존재하지 않음**.")
    add("- 디스크 실측: `C7/` 폴더는 동일한 posture/data-type 구조와 약 52명의 참여자를")
    add("  가지고 있지만, 내부 `_time.txt`가 C1–C6과는 다른 레거시 class 코드")
    add("  (예: `C39`)를 참조한다. 이는 C7이 다른 프로토콜이나 미공개 후속 연구를 위해")
    add("  수집되었을 가능성을 강하게 시사한다.")
    add("- 논문에 정의된 squat-fault 설명이 없으므로 신뢰할 수 있는")
    add("  template caption을 작성할 수 없다.")
    add("")
    add("**Step 1 lock**: C7 제외. 재포함하려면 (a) 클래스 semantics 확정,")
    add("(b) caption template 정의 두 가지가 명시적으로 문서화되어야 하며,")
    add("이는 본 단계 scope를 벗어난다.")
    add("")

    # ---- Section 6: naming -------------------------------------------------
    add("## 6. Posture 명칭 — canonical SA / CA / HW (lock)")
    add("")
    add("아티팩트에는 세 가지 명명 규칙이 공존한다. Step 1은 **논문**의 약어")
    add("(SA / CA / HW)를 canonical로 표준화한다. 레거시 스크립트 약어")
    add("(AS / AC / AW)는 `_time.txt` 내부에서 발견되며, 레거시 전처리와의")
    add("join을 위해 metadata 별칭으로만 보존한다.")
    add("")
    add("| Posture (Figure 3)         | 디스크 폴더            | Canonical (paper) | Legacy (script) |")
    add("|---|---|---|---|")
    add("| (a) straight arms          | `01_Straight_Arms`    | **SA**            | AS              |")
    add("| (b) crossed arms           | `02_Crossed_Arms`     | **CA**            | AC              |")
    add("| (c) hands on waist         | `03_Hands_on_Waist`   | **HW**            | AW              |")
    add("")
    add("코드/파일명에서 짧은 posture 키가 필요하면 **반드시 SA / CA / HW** 사용.")
    add("AS / AC / AW는 레거시 별칭을 명시하는 필드에서만 등장해야 한다.")
    add("")

    # ---- Section 7: final dataset block ------------------------------------
    add("## 7. 최종 데이터셋 사양 (Step 1)")
    add("")
    add("```")
    add("dataset_root : C:\\Users\\user\\data\\AICRC_DSQ_Data(Combined_segmented)_v2"
        "\\AICRC_DSQ_Data(Combined_segmented)")
    add("classes      : C1, C2, C3, C4, C5, C6                          # C7 미사용")
    add("postures     : SA, CA, HW                                      # canonical")
    add("                (folders: 01_Straight_Arms, 02_Crossed_Arms,")
    add("                          03_Hands_on_Waist)")
    add("data_type    : Segmented                                       # Combined 미사용")
    add("unit         : rep 파일 1개 = (signal, caption) 샘플 1개")
    add("scale        : 약 935개 subject 서브폴더, 약 9,200개 rep-level 샘플")
    add("```")
    add("")

    # ---- Section 8: manifest -----------------------------------------------
    add("## 8. Manifest 생성 규칙 (Step 1)")
    add("")
    add("manifest의 한 행 = Step 1 scope 내의 rep 파일 1개.")
    add("")
    add("| 컬럼 | 출처 / 규칙 |")
    add("|---|---|")
    add("| `rep_id`             | `f\"{participant_id}_{class_id}_{posture_canonical}_rep{rep_index:02d}\"` |")
    add("| `participant_id`     | 서브폴더명에서 파싱 (`EP01..EP52`) |")
    add("| `class_id`           | 경로에서 추출 (`C1..C6` — C7은 절대 등장 금지) |")
    add("| `class_description`  | `CLASS_DESCRIPTIONS[class_id]` |")
    add("| `posture_canonical`  | **SA / CA / HW** (posture의 primary key) |")
    add("| `posture_legacy`     | AS / AC / AW (별칭 전용) |")
    add("| `posture_folder`     | `01_Straight_Arms` / `02_Crossed_Arms` / `03_Hands_on_Waist` |")
    add("| `rep_index`          | `file_..._{rep}_o0.txt`의 정수 (1-based) |")
    add("| `signal_path`        | rep `.txt` 파일의 절대 경로 |")
    add("| `n_rows`             | rep 파일 행 수 (sequence length) |")
    add("| `start_idx` / `end_idx` | 부모 서브폴더의 `_vkeep_st_end.txt`에서 |")
    add("| `t_start_s` / `t_end_s` | 부모 서브폴더의 `_time.txt`에서 |")
    add("| `split`              | participant 단위 split (예: 36 / 8 / 8 train/val/test) |")
    add("")
    add("필터링 규칙:")
    add("1. 행 수가 최소 임계값(예: 50 행) 미만인 rep 파일은 truncation으로 간주하여 drop.")
    add("2. **`C7` 경로는 manifest 단계에서도 거부**한다 (collector가 이미 `STEP1_CLASSES`로")
    add("   제한하지만 방어적으로 한 번 더 차단).")
    add("3. **`Combined` 경로는 manifest 단계에서도 거부**한다 (같은 이유).")
    add("4. split은 **rep 단위가 아닌 participant 단위**로 자른다 — 한 참여자가")
    add("   train과 test에 동시에 들어가면 안 됨.")
    add("5. (class × posture)로 stratify하여 모든 fold가 18개 조건을 모두 보도록 한다.")
    add("")

    # ---- Section 9: per-condition table ------------------------------------
    add("## 9. 조건별 파일 수 표 (Step 1 scope)")
    add("")
    add("전체 표는 `reports/dataset_file_count_by_condition.csv`에도 동일하게 기록됨.")
    add("")
    add("| class | 설명 | posture (canonical) | 폴더 | data_type | file_count | 예시 파일 |")
    add("|---|---|---|---|---|---|---|")
    for r in step1_rows:
        add(f"| {r['class']} | {r['class_description']} | "
            f"**{r['arm_posture_canonical']}** | `{r['arm_posture']}` | "
            f"{r['data_type']} | {r['file_count']} | `{r['example_file']}` |")
    add("")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not DATASET_ROOT.is_dir():
        print(f"ERROR: dataset root not found: {DATASET_ROOT}", file=sys.stderr)
        return 1
    step1_rows = collect_step1_rows()
    full_rows = collect_full_inventory()
    write_csv(step1_rows)
    write_markdown(step1_rows, full_rows)
    print(f"wrote {CSV_PATH}  (Step 1 scope: {len(step1_rows)} rows)")
    print(f"wrote {MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
