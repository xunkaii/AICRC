"""Step 4R-B post-processing 3/3 — Attention phase analysis.

Reads (read-only):
    data/step4r/4rb_attention/step4r_sequence_dataset.npz
    checkpoints/step4r/4rb_attention/best_bigru_attention.pt
    models/step4r_attention_rnn.py
    data/step2/bottom_event_audit.csv
    data/manifest_split.csv
    data/step4r/4rb_attention/step4r_bigru_attention_schema_outputs_calibrated.csv

Writes:
    data/step4r/4rb_attention/step4r_bigru_attention_attention_weights.npz
    data/step4r/4rb_attention/step4r_bigru_attention_attention_summary.csv
    reports/step4r/4rb_attention/step4r_attention_phase_by_class.csv
    reports/step4r/4rb_attention/step4r_attention_phase_by_posture.csv
    reports/step4r/4rb_attention/step4r_attention_phase_by_correctness.csv
    reports/step4r/4rb_attention/step4r_attention_phase_analysis.md

Phase definition (timestep 0..127):
    descending          [0, 42]    (43 steps)
    bottom_transition   [43, 85]   (43 steps)
    ascending_recovery  [86, 127]  (42 steps)

Anchor (posture-aware, from bottom_event_audit.csv):
    SA, CA -> ensemble_bottom_idx
    HW     -> acc_bottom_idx
    Rescale to 0..127:  anchor_idx_128 = round(anchor_idx_orig * 128 / n_rows)

Run:
    python scripts/analyze_step4r_attention_phase.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.step4r_attention_rnn import Step4RBiGRUAttention  # noqa: E402


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

INPUT_NPZ = PROJECT_ROOT / "data" / "step4r" / "4rb_attention" / "step4r_sequence_dataset.npz"
INPUT_CKPT = PROJECT_ROOT / "checkpoints" / "step4r" / "4rb_attention" / "best_bigru_attention.pt"
INPUT_BOTTOM_AUDIT = PROJECT_ROOT / "data" / "step2" / "bottom_event_audit.csv"
INPUT_MANIFEST = PROJECT_ROOT / "data" / "manifest_split.csv"
INPUT_SCHEMA_CSV = (
    PROJECT_ROOT / "data" / "step4r" / "4rb_attention"
    / "step4r_bigru_attention_schema_outputs_calibrated.csv"
)

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4rb_attention"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4rb_attention"

OUTPUT_ATTN_NPZ = OUTPUT_DATA_DIR / "step4r_bigru_attention_attention_weights.npz"
OUTPUT_SUMMARY_CSV = OUTPUT_DATA_DIR / "step4r_bigru_attention_attention_summary.csv"
OUTPUT_BY_CLASS_CSV = OUTPUT_REPORT_DIR / "step4r_attention_phase_by_class.csv"
OUTPUT_BY_POSTURE_CSV = OUTPUT_REPORT_DIR / "step4r_attention_phase_by_posture.csv"
OUTPUT_BY_CORRECTNESS_CSV = OUTPUT_REPORT_DIR / "step4r_attention_phase_by_correctness.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step4r_attention_phase_analysis.md"

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
SPLITS = ["train", "val", "test"]
POSTURES = ["SA", "CA", "HW"]
POSTURE_TO_IDX = {p: i for i, p in enumerate(POSTURES)}
SEQ_LEN = 128

PHASES = [
    ("descending", 0, 43),           # [0, 42]
    ("bottom_transition", 43, 86),   # [43, 85]
    ("ascending_recovery", 86, 128), # [86, 127]
]
PHASE_NAMES = [p[0] for p in PHASES]

BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _posture_onehot(posture_str_arr: np.ndarray) -> np.ndarray:
    n = len(posture_str_arr)
    out = np.zeros((n, len(POSTURES)), dtype=np.float32)
    for i, p in enumerate(posture_str_arr):
        out[i, POSTURE_TO_IDX[p]] = 1.0
    return out


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pp = np.clip(p, eps, 1.0)
    return -(pp * np.log(pp)).sum(axis=-1)


def _peak_phase_for_ts(ts: int) -> str:
    for name, lo, hi in PHASES:
        if lo <= ts < hi:
            return name
    return "ascending_recovery"  # fallback for ts == 127 already covered


def _phase_masses(attn: np.ndarray) -> np.ndarray:
    """Return (N, 3) phase mass: descending, bottom_transition, ascending_recovery."""
    out = np.empty((attn.shape[0], len(PHASES)), dtype=np.float64)
    for i, (_, lo, hi) in enumerate(PHASES):
        out[:, i] = attn[:, lo:hi].sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_attention(
    model: nn.Module,
    X: np.ndarray,
    posture_oh: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(posture_oh).float(),
        torch.from_numpy(y).long(),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    attn_list = []
    model.eval()
    with torch.no_grad():
        for X_b, P_b, _ in loader:
            X_b = X_b.to(device, non_blocking=True)
            P_b = P_b.to(device, non_blocking=True)
            _, attn_w = model(X_b, P_b)
            attn_list.append(attn_w.cpu().numpy())
    return np.concatenate(attn_list, axis=0)


# ---------------------------------------------------------------------------
# Anchor handling
# ---------------------------------------------------------------------------

def _build_anchor_table(sample_ids: np.ndarray) -> pd.DataFrame:
    """Return DataFrame indexed by sample_id with columns:
    n_rows_audit, acc_bottom_idx, ensemble_bottom_idx, n_rows_manifest.
    """
    if not INPUT_BOTTOM_AUDIT.exists():
        raise FileNotFoundError(f"Bottom event audit not found: {INPUT_BOTTOM_AUDIT}")
    audit = pd.read_csv(
        INPUT_BOTTOM_AUDIT,
        usecols=["sample_id", "n_rows", "acc_bottom_idx", "ensemble_bottom_idx"], encoding="utf-8-sig"
    )
    audit = audit.rename(columns={"n_rows": "n_rows_audit"})
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {INPUT_MANIFEST}")
    manifest = pd.read_csv(INPUT_MANIFEST, usecols=["sample_id", "n_rows"], encoding="utf-8-sig")
    manifest = manifest.rename(columns={"n_rows": "n_rows_manifest"})
    df = pd.merge(audit, manifest, on="sample_id", how="outer")
    df = df.set_index("sample_id")
    # Reindex on the dataset's sample_ids (preserves order, fills missing with NaN).
    df = df.reindex(sample_ids)
    return df


def _anchor_idx_128(
    posture: np.ndarray,
    anchor_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (anchor_idx_orig, anchor_idx_128). NaN where missing."""
    n = len(posture)
    anchor_orig = np.full(n, np.nan, dtype=np.float64)
    anchor_128 = np.full(n, np.nan, dtype=np.float64)

    acc_idx = anchor_table["acc_bottom_idx"].to_numpy()
    ens_idx = anchor_table["ensemble_bottom_idx"].to_numpy()
    n_rows_aud = anchor_table["n_rows_audit"].to_numpy()
    n_rows_man = anchor_table["n_rows_manifest"].to_numpy()

    for i in range(n):
        p = posture[i]
        if p in ("SA", "CA"):
            a = ens_idx[i]
        elif p == "HW":
            a = acc_idx[i]
        else:
            a = np.nan
        if pd.isna(a):
            continue
        # prefer manifest n_rows, fallback to audit n_rows
        n_rows = n_rows_man[i] if not pd.isna(n_rows_man[i]) else n_rows_aud[i]
        if pd.isna(n_rows) or float(n_rows) <= 0:
            anchor_orig[i] = float(a)
            continue
        anchor_orig[i] = float(a)
        idx128 = int(round(float(a) * SEQ_LEN / float(n_rows)))
        anchor_128[i] = float(np.clip(idx128, 0, SEQ_LEN - 1))
    return anchor_orig, anchor_128


# ---------------------------------------------------------------------------
# Aggregation by group
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate per-sample summary by group_col into wide rows."""
    rows = []
    for g, sub in df.groupby(group_col, dropna=False):
        n = len(sub)
        peak_phase_counts = sub["attention_peak_phase"].value_counts().to_dict()
        most_common = sub["attention_peak_phase"].mode().iloc[0] if len(sub) else "—"
        ad = sub["anchor_distance"].dropna()
        rows.append({
            group_col: g,
            "n": n,
            "attention_entropy_mean": float(sub["attention_entropy"].mean()),
            "attention_entropy_p25": float(np.percentile(sub["attention_entropy"], 25)) if n > 0 else float("nan"),
            "attention_entropy_p50": float(np.percentile(sub["attention_entropy"], 50)) if n > 0 else float("nan"),
            "attention_entropy_p75": float(np.percentile(sub["attention_entropy"], 75)) if n > 0 else float("nan"),
            "descending_mass_mean": float(sub["descending_mass"].mean()),
            "bottom_transition_mass_mean": float(sub["bottom_transition_mass"].mean()),
            "ascending_recovery_mass_mean": float(sub["ascending_recovery_mass"].mean()),
            "peak_phase_descending_count": int(peak_phase_counts.get("descending", 0)),
            "peak_phase_bottom_transition_count": int(peak_phase_counts.get("bottom_transition", 0)),
            "peak_phase_ascending_recovery_count": int(peak_phase_counts.get("ascending_recovery", 0)),
            "most_common_peak_phase": most_common,
            "anchor_distance_n": int(len(ad)),
            "anchor_distance_mean": float(ad.mean()) if len(ad) > 0 else float("nan"),
            "anchor_distance_p50": float(np.percentile(ad, 50)) if len(ad) > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _markdown_report(
    df: pd.DataFrame,
    by_class: pd.DataFrame,
    by_posture: pd.DataFrame,
    by_correct: pd.DataFrame,
    by_no_call: pd.DataFrame,
    by_ambiguity: pd.DataFrame,
    overall: dict,
) -> str:
    L = []
    L.append("# Step 4R-B 후처리 3/3 — Attention Phase Analysis 결과")
    L.append("")
    L.append("- 생성 스크립트: `scripts/analyze_step4r_attention_phase.py`")
    L.append("- attention_weights는 best checkpoint로 전체 sample에 대해 재추출했다.")
    L.append("- phase 정의:")
    L.append("  - `descending`: timestep 0~42")
    L.append("  - `bottom_transition`: timestep 43~85")
    L.append("  - `ascending_recovery`: timestep 86~127")
    L.append(
        "- anchor: posture-aware (`SA/CA → ensemble_bottom_idx`, `HW → acc_bottom_idx`), "
        "원본 timestep을 `n_rows`로 0~127 범위로 rescale."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 0. 주의 문구 — attention 해석의 한계")
    L.append("")
    L.append(
        "- attention은 정답 근거가 아니라 **모델 내부 근거의 보조 분석**이다."
    )
    L.append(
        "- attention entropy를 **센서 관측 가능성의 직접 증거로 과장하지 않는다**."
    )
    L.append(
        "- 본 분석은 posterior entropy, confusion pattern, attention phase를 *함께* "
        "해석할 때만 reframing 문서 §4의 이론 기반(정보이론·관측이론)과 연결된다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 단계의 위치")
    L.append("")
    L.append(
        "calibrated 분류기 위에서, 모델이 시계열의 어느 phase에 *주의를 기울였는지*를 "
        "정량화한다. reframing 문서 §4.3 (Attention 기반 시계열 근거 분석)과 §8.2 "
        "(posterior entropy + attention entropy + confusion pattern)의 입력이 된다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. attention entropy 전체 요약")
    L.append("")
    L.append("`-Σ a_t log a_t` (자연로그). T=128 균등분포 상한 ≈ 4.8520.")
    L.append("")
    L.append("| split | n | entropy mean | entropy p25 | entropy p50 | entropy p75 | peak ts mean |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        sub = df[df["split"] == s]
        n = len(sub)
        if n == 0:
            continue
        L.append(
            f"| {s} | {n} | {sub['attention_entropy'].mean():.4f} | "
            f"{np.percentile(sub['attention_entropy'], 25):.4f} | "
            f"{np.percentile(sub['attention_entropy'], 50):.4f} | "
            f"{np.percentile(sub['attention_entropy'], 75):.4f} | "
            f"{sub['attention_peak_timestep'].mean():.2f} |"
        )
    L.append("")
    L.append("**전체 (all splits):**")
    L.append("")
    L.append(
        f"- entropy mean = {df['attention_entropy'].mean():.4f}, "
        f"p25 = {np.percentile(df['attention_entropy'], 25):.4f}, "
        f"p50 = {np.percentile(df['attention_entropy'], 50):.4f}, "
        f"p75 = {np.percentile(df['attention_entropy'], 75):.4f}"
    )
    L.append("")
    L.append("**전체 phase mass 평균:**")
    L.append("")
    L.append("| phase | mean mass |")
    L.append("|---|---:|")
    L.append(f"| descending | {df['descending_mass'].mean():.4f} |")
    L.append(f"| bottom_transition | {df['bottom_transition_mass'].mean():.4f} |")
    L.append(f"| ascending_recovery | {df['ascending_recovery_mass'].mean():.4f} |")
    L.append("")
    L.append("**전체 peak phase 분포:**")
    L.append("")
    pcounts = df["attention_peak_phase"].value_counts()
    L.append("| peak phase | count | rate |")
    L.append("|---|---:|---:|")
    for p in PHASE_NAMES:
        c = int(pcounts.get(p, 0))
        L.append(f"| {p} | {c} | {c / len(df):.4f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. Class별 attention peak phase / phase mass")
    L.append("")
    L.append(_format_group_table(by_class, "class_id"))
    L.append("")
    L.append(
        "Step 2.5 §3 (시간 국소화된 클래스 차이)에서 SA/CA의 acc_z η²가 bottom **이후** "
        "에 정점이라고 보고됐다. 본 모델의 peak phase 분포가 그 관찰과 일관되는지(예: "
        "ascending_recovery 비중이 substantial한지) 확인한다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. Posture별 attention peak phase / phase mass")
    L.append("")
    L.append(_format_group_table(by_posture, "posture_canonical"))
    L.append("")
    L.append(
        "Step 2.5 §4의 HW 자세에서 acc-bottom과 gyro-peak가 다른 물리 이벤트라는 점, "
        "그리고 §6의 자세 인식 정규화 효과 등을 고려해 자세별 attention 패턴 차이를 본다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. correct vs incorrect 비교")
    L.append("")
    L.append(_format_group_table(by_correct, "correct"))
    L.append("")
    L.append(
        "correct 행에서 attention entropy가 더 낮고 phase 분포가 더 집중되는지(또는 "
        "특정 phase에 mass가 쏠리는지)를 본다. 차이가 작으면 attention이 정답 여부에 "
        "직접적으로 정렬되지 않는다는 신호이다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. no_call vs non_no_call 비교")
    L.append("")
    L.append(_format_group_table(by_no_call, "no_call"))
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. Ambiguity group별 비교")
    L.append("")
    L.append(_format_group_table(by_ambiguity, "ambiguity_group"))
    L.append("")
    L.append(
        "Step 2.5의 ambiguity 패턴을 본 모델의 attention이 어떻게 다루는지 본다 — "
        "특히 within_group_c1_c5_c6와 pair_plus_c2_absorption 그룹에서 entropy/phase "
        "mass가 confident_C2와 *얼마나 다른지*가 핵심."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. anchor-attention distance 요약")
    L.append("")
    ad = df["anchor_distance"].dropna()
    L.append(f"- distance 가용 sample 수: {len(ad)} / {len(df)}")
    if len(ad) > 0:
        L.append(
            f"- distance mean = {ad.mean():.2f}, p25 = {np.percentile(ad, 25):.2f}, "
            f"p50 = {np.percentile(ad, 50):.2f}, p75 = {np.percentile(ad, 75):.2f}, "
            f"p95 = {np.percentile(ad, 95):.2f}"
        )
    L.append("")
    L.append("**posture별 anchor-attention distance:**")
    L.append("")
    L.append("| posture | n with distance | mean | p25 | p50 | p75 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for p in POSTURES:
        sub = df[(df["posture_canonical"] == p) & df["anchor_distance"].notna()]
        n = len(sub)
        if n == 0:
            L.append(f"| {p} | 0 | — | — | — | — |")
            continue
        adp = sub["anchor_distance"]
        L.append(
            f"| {p} | {n} | {adp.mean():.2f} | "
            f"{np.percentile(adp, 25):.2f} | {np.percentile(adp, 50):.2f} | "
            f"{np.percentile(adp, 75):.2f} |"
        )
    L.append("")
    L.append(
        "anchor-attention distance는 *해석 보조 지표*다. 작다고 해서 모델이 옳다는 "
        "의미가 아니며, 크다고 해서 틀렸다는 의미도 아니다 — 자세별 anchor 정의 자체가 "
        "Step 2.5의 한계(특히 HW에서 acc/gyro 일치도 0.044~0.225)를 안고 있다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4rb_attention/step4r_bigru_attention_attention_weights.npz`")
    L.append("  - 키: `attention_weights` (N, 128), `sample_id`, `split`, `class_id`, `posture_canonical`")
    L.append("- `data/step4r/4rb_attention/step4r_bigru_attention_attention_summary.csv` (per-sample)")
    L.append("- `reports/step4r/4rb_attention/step4r_attention_phase_by_class.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_attention_phase_by_posture.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_attention_phase_by_correctness.csv`")
    L.append("- `reports/step4r/4rb_attention/step4r_attention_phase_analysis.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*본 보고서는 자동 생성된다. 기존 Step 1 ~ 4 / 4R-A / 4R-B 산출물은 수정되지 않으며, 기존 attention 관련 컬럼을 가진 다른 CSV는 영향 없음.*")
    L.append("")
    return "\n".join(L)


def _format_group_table(g: pd.DataFrame, group_col: str) -> str:
    if len(g) == 0:
        return "_(group empty)_"
    lines = []
    lines.append(
        f"| {group_col} | n | entropy mean | desc mass | bot mass | rec mass | most common peak | distance mean (n) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for _, r in g.iterrows():
        dm = "—" if pd.isna(r["anchor_distance_mean"]) else f"{r['anchor_distance_mean']:.2f}"
        lines.append(
            f"| {r[group_col]} | {int(r['n'])} | "
            f"{r['attention_entropy_mean']:.4f} | "
            f"{r['descending_mass_mean']:.4f} | "
            f"{r['bottom_transition_mass_mean']:.4f} | "
            f"{r['ascending_recovery_mass_mean']:.4f} | "
            f"{r['most_common_peak_phase']} | "
            f"{dm} ({int(r['anchor_distance_n'])}) |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-B post 3/3 — Attention phase analysis")
    print("=" * 64)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"Sequence dataset not found: {INPUT_NPZ}")
    if not INPUT_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {INPUT_CKPT}")
    if not INPUT_SCHEMA_CSV.exists():
        raise FileNotFoundError(
            f"Schema CSV not found: {INPUT_SCHEMA_CSV}. "
            "Run scripts/generate_step4r_attention_schema_outputs.py first."
        )

    print(f"loading {INPUT_NPZ}")
    z = np.load(INPUT_NPZ, allow_pickle=True)
    X = z["X_norm"].astype(np.float32)
    y = z["y"].astype(np.int64)
    sample_id = z["sample_id"].astype(str)
    class_id = z["class_id"].astype(str)
    posture = z["posture_canonical"].astype(str)
    split = z["split"].astype(str)
    posture_oh = _posture_onehot(posture)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    print(f"loading checkpoint -> {INPUT_CKPT}")
    ckpt = torch.load(INPUT_CKPT, map_location=device)
    model = Step4RBiGRUAttention(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print("re-extracting attention weights ...")
    t0 = time.time()
    attn_w = _infer_attention(model, X, posture_oh, y, device)
    print(f"  done in {time.time() - t0:.1f}s. attention_weights shape={attn_w.shape}")
    if attn_w.shape != (len(sample_id), SEQ_LEN):
        raise ValueError(
            f"attention shape {attn_w.shape} != expected ({len(sample_id)}, {SEQ_LEN})."
        )

    # Phase computations.
    attn_entropy = _entropy(attn_w)                # (N,)
    peak_ts = attn_w.argmax(axis=1).astype(np.int64)  # (N,)
    peak_phase = np.array([_peak_phase_for_ts(int(t)) for t in peak_ts])
    masses = _phase_masses(attn_w)                 # (N, 3)

    # Anchor.
    print("loading anchor / n_rows tables ...")
    anchor_table = _build_anchor_table(sample_id)
    anchor_orig, anchor_128 = _anchor_idx_128(posture, anchor_table)
    anchor_distance = np.where(
        np.isnan(anchor_128),
        np.nan,
        np.abs(peak_ts.astype(np.float64) - anchor_128),
    )
    n_anchor_avail = int((~np.isnan(anchor_128)).sum())
    print(f"  anchor_idx_128 available for {n_anchor_avail} / {len(sample_id)} samples")

    # Schema CSV join (no_call, ambiguity_group, top1_class).
    print(f"joining schema CSV {INPUT_SCHEMA_CSV.name}")
    sc = pd.read_csv(
        INPUT_SCHEMA_CSV,
        usecols=[
            "sample_id", "no_call", "ambiguity_group", "top1_class",
            "predictive_entropy_calibrated",
        ], encoding="utf-8-sig"
    ).set_index("sample_id")
    sc = sc.reindex(sample_id)

    correct = (sc["top1_class"].to_numpy() == class_id).astype(bool)

    # Per-sample summary DataFrame.
    df = pd.DataFrame({
        "sample_id": sample_id,
        "split": split,
        "class_id": class_id,
        "posture_canonical": posture,
        "attention_entropy": attn_entropy,
        "attention_peak_timestep": peak_ts,
        "attention_peak_phase": peak_phase,
        "descending_mass": masses[:, 0],
        "bottom_transition_mass": masses[:, 1],
        "ascending_recovery_mass": masses[:, 2],
        "anchor_idx_orig": anchor_orig,
        "anchor_idx_128": anchor_128,
        "anchor_distance": anchor_distance,
        "predictive_entropy_calibrated": sc["predictive_entropy_calibrated"].to_numpy(),
        "top1_class": sc["top1_class"].to_numpy(),
        "correct": correct,
        "no_call": sc["no_call"].astype(bool).to_numpy(),
        "ambiguity_group": sc["ambiguity_group"].to_numpy(),
    })
    df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"saved per-sample summary -> {OUTPUT_SUMMARY_CSV}")

    # Save attention_weights npz.
    np.savez(
        OUTPUT_ATTN_NPZ,
        attention_weights=attn_w.astype(np.float32),
        sample_id=sample_id,
        split=split,
        class_id=class_id,
        posture_canonical=posture,
    )
    print(f"saved attention weights -> {OUTPUT_ATTN_NPZ}")

    # Aggregations.
    by_class = _aggregate(df, "class_id")
    by_class.to_csv(OUTPUT_BY_CLASS_CSV, index=False, encoding="utf-8-sig")
    print(f"saved by_class -> {OUTPUT_BY_CLASS_CSV}")

    by_posture = _aggregate(df, "posture_canonical")
    by_posture.to_csv(OUTPUT_BY_POSTURE_CSV, index=False, encoding="utf-8-sig")
    print(f"saved by_posture -> {OUTPUT_BY_POSTURE_CSV}")

    by_correct = _aggregate(df, "correct")
    by_correct.to_csv(OUTPUT_BY_CORRECTNESS_CSV, index=False, encoding="utf-8-sig")
    print(f"saved by_correctness -> {OUTPUT_BY_CORRECTNESS_CSV}")

    by_no_call = _aggregate(df, "no_call")
    by_ambiguity = _aggregate(df, "ambiguity_group")

    # Markdown.
    overall = {
        "n_total": len(df),
        "anchor_available": n_anchor_avail,
    }
    md = _markdown_report(df, by_class, by_posture, by_correct, by_no_call, by_ambiguity, overall)
    OUTPUT_REPORT_MD.write_text(md, encoding="utf-8")
    print(f"saved report -> {OUTPUT_REPORT_MD}")

    # Console summary
    print()
    print("attention entropy mean by split:")
    for s in SPLITS:
        sub = df[df["split"] == s]
        print(f"  {s:5s} n={len(sub):4d}  entropy_mean={sub['attention_entropy'].mean():.4f}")
    print()
    print("most common attention_peak_phase by class:")
    for c in CLASSES:
        sub = df[df["class_id"] == c]
        if len(sub) == 0:
            continue
        mc = sub["attention_peak_phase"].mode().iloc[0]
        counts = sub["attention_peak_phase"].value_counts().to_dict()
        print(f"  {c}: {mc}  (counts={counts})")
    print()
    if n_anchor_avail > 0:
        ad = df["anchor_distance"].dropna()
        print(
            f"anchor-attention distance summary (n={len(ad)}): "
            f"mean={ad.mean():.2f}, p50={np.percentile(ad, 50):.2f}, "
            f"p95={np.percentile(ad, 95):.2f}"
        )
    else:
        print("anchor-attention distance: no samples with both peak and anchor")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
