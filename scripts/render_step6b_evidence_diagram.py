"""
Render presentation slide 6-B SVG from Step 2.5 evidence CSVs.

Inputs (read at runtime — no hard-coded numbers):
  reports/step2/4_separability/reference_separability_ambiguous_groups.csv
  reports/step2/4_separability/reference_separability_confusion_by_setting.csv
  reports/step2/2_bottom_event/bottom_agreement_threshold_sensitivity.csv
  data/step2/time_localized_class_difference.csv     (for the 11/18 cells stat)

Output:
  reports/step4r/step6b_evidence_diagram.svg

Run:
  python -X utf8 scripts/render_step6b_evidence_diagram.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMBIG_CSV = ROOT / "reports" / "step2" / "4_separability" / "reference_separability_ambiguous_groups.csv"
BOTTOM_CSV = ROOT / "reports" / "step2" / "2_bottom_event" / "bottom_agreement_threshold_sensitivity.csv"
TIME_CSV = ROOT / "data" / "step2" / "time_localized_class_difference.csv"
OUT_SVG = ROOT / "reports" / "step4r" / "step6b_evidence_diagram.svg"

# Reporting choices ----------------------------------------------------------
AMBIG_SETTING = "raw_core_features"  # canonical pre-normalization reference
AMBIG_SPLIT = "test"
BOTTOM_THRESHOLD = 0.10                # threshold used for the posture bars
BOTTOM_REGION = (38, 89)               # timesteps [38, 89) → middle 40%


def load_ambiguity() -> dict[str, float]:
    with open(AMBIG_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["setting"] == AMBIG_SETTING and row["split"] == AMBIG_SPLIT:
                return {
                    "c1_c5_c6_internal_confusion_rate": float(row["c1_c5_c6_internal_confusion_rate"]),
                    "c3_c4_pair_confusion_rate": float(row["c3_c4_pair_confusion_rate"]),
                    "c2_recall": float(row["c2_recall"]),
                    "n_high_grp": int(row["n_high_grp"]),
                    "n_med_pair": int(row["n_med_pair"]),
                    "n_c2": int(row["n_c2"]),
                }
    raise RuntimeError(f"row not found: {AMBIG_SETTING}/{AMBIG_SPLIT}")


def load_posture_agree(threshold: float) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(BOTTOM_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["scope"] == "posture" and abs(float(row["threshold"]) - threshold) < 1e-6:
                out[row["scope_value"]] = float(row["agree_rate"])
    return out


def compute_bottom_vs_full_cells() -> tuple[int, int]:
    """Count (posture × channel) cells where mean η² is higher in the bottom region
    than across the full timeline. Returns (n_bottom_higher, n_total)."""
    eta_full: dict[tuple[str, str], list[float]] = defaultdict(list)
    eta_bot: dict[tuple[str, str], list[float]] = defaultdict(list)
    lo, hi = BOTTOM_REGION
    with open(TIME_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = int(row["timestep"])
            key = (row["posture"], row["channel"])
            v = float(row["eta_squared"])
            eta_full[key].append(v)
            if lo <= t < hi:
                eta_bot[key].append(v)
    n_higher = 0
    n_total = len(eta_full)
    for key, full_vals in eta_full.items():
        bot_vals = eta_bot[key]
        if not bot_vals:
            continue
        if sum(bot_vals) / len(bot_vals) > sum(full_vals) / len(full_vals):
            n_higher += 1
    return n_higher, n_total


# --------------------------------------------------------- SVG construction
def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_svg(amb: dict, post: dict[str, float], cells_hi: int, cells_n: int) -> str:
    c1_c5_c6 = amb["c1_c5_c6_internal_confusion_rate"]
    c3_c4 = amb["c3_c4_pair_confusion_rate"]
    c2 = amb["c2_recall"]
    sa, ca, hw = post["SA"], post["CA"], post["HW"]

    # Bar chart geometry (right panel)
    chart_x0, chart_y0 = 480, 200
    chart_w, chart_h = 340, 150
    bar_w = 70
    gap = (chart_w - 3 * bar_w) / 4
    # Bars go up to chart_y0 + chart_h baseline; value scales to chart_h.
    def bar(x_idx: int, value: float, fill: str, label: str):
        x = chart_x0 + gap + x_idx * (bar_w + gap)
        h = value * chart_h
        y = chart_y0 + chart_h - h
        return f"""
    <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="{fill}" />
    <text x="{x + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="#111827">{fmt_pct(value)}</text>
    <text x="{x + bar_w/2:.1f}" y="{chart_y0 + chart_h + 18:.1f}" text-anchor="middle" font-size="13" font-weight="600" fill="#374151">{label}</text>"""

    bars_svg = (
        bar(0, sa, "#6B7280", "SA")
        + bar(1, ca, "#6B7280", "CA")
        + bar(2, hw, "#DC2626", "HW")
    )

    # Y-axis gridlines (0, 25, 50, 75, 100)
    grid = ""
    for pct in (0, 25, 50, 75, 100):
        y = chart_y0 + chart_h - chart_h * (pct / 100)
        grid += (
            f'\n    <line x1="{chart_x0}" y1="{y:.1f}" x2="{chart_x0 + chart_w}" y2="{y:.1f}" '
            f'stroke="#E5E7EB" stroke-width="1" stroke-dasharray="3,3" />'
            f'\n    <text x="{chart_x0 - 6:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#9CA3AF">{pct}%</text>'
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 500"
     font-family="'Malgun Gothic','맑은 고딕','Apple SD Gothic Neo','Noto Sans KR',sans-serif">

  <defs>
    <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
      <feOffset dx="0" dy="1" result="b"/>
      <feComponentTransfer><feFuncA type="linear" slope="0.18"/></feComponentTransfer>
      <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <rect width="860" height="500" fill="#FFFFFF"/>

  <!-- ===================== Purple header ===================== -->
  <rect x="0" y="0" width="860" height="46" fill="#5B21B6"/>
  <text x="24" y="30" font-size="18" font-weight="700" fill="#FFFFFF">Step 2.5 · 주요 결과</text>
  <text x="836" y="30" text-anchor="end" font-size="11" fill="#DDD6FE" font-style="italic">
    setting = {AMBIG_SETTING} · split = {AMBIG_SPLIT} · threshold = {BOTTOM_THRESHOLD:.2f}
  </text>

  <!-- ===================== LEFT: Ambiguity groups ===================== -->
  <text x="40" y="74" font-size="14" font-weight="700" fill="#1F2937">Ambiguity group structure</text>
  <text x="40" y="92" font-size="10.5" fill="#6B7280">reference LR · test split — 그룹별 혼동 구조</text>

  <!-- C2 group: separable (green) -->
  <g filter="url(#softShadow)">
    <rect x="30" y="108" width="400" height="78" rx="10" fill="#ECFDF5" stroke="#059669" stroke-width="2"/>
  </g>
  <circle cx="68" cy="147" r="22" fill="#10B981" stroke="#047857" stroke-width="2"/>
  <text x="68" y="152" text-anchor="middle" font-size="14" font-weight="700" fill="#FFFFFF">C2</text>
  <text x="102" y="135" font-size="13" font-weight="700" fill="#065F46">비교적 분리 가능</text>
  <text x="102" y="153" font-size="11" fill="#047857">C2 단독 recall이 가장 높음 — 단언 가능 후보</text>
  <text x="410" y="142" text-anchor="end" font-size="20" font-weight="700" fill="#047857">{fmt_pct(c2)}</text>
  <text x="410" y="160" text-anchor="end" font-size="10" fill="#059669">c2_recall (n={amb["n_c2"]})</text>

  <!-- C1/C5/C6 group: internal ambiguity (orange) -->
  <g filter="url(#softShadow)">
    <rect x="30" y="200" width="400" height="78" rx="10" fill="#FFF7ED" stroke="#EA580C" stroke-width="2"/>
  </g>
  <circle cx="50" cy="239" r="16" fill="#FB923C" stroke="#C2410C" stroke-width="1.5"/>
  <text x="50" y="244" text-anchor="middle" font-size="11" font-weight="700" fill="#FFFFFF">C1</text>
  <circle cx="82" cy="239" r="16" fill="#FB923C" stroke="#C2410C" stroke-width="1.5"/>
  <text x="82" y="244" text-anchor="middle" font-size="11" font-weight="700" fill="#FFFFFF">C5</text>
  <circle cx="114" cy="239" r="16" fill="#FB923C" stroke="#C2410C" stroke-width="1.5"/>
  <text x="114" y="244" text-anchor="middle" font-size="11" font-weight="700" fill="#FFFFFF">C6</text>
  <text x="142" y="227" font-size="13" font-weight="700" fill="#7C2D12">그룹 내부 모호성</text>
  <text x="142" y="245" font-size="11" fill="#9A3412">C1 / C5 / C6 셋이 서로 혼동 — 그룹으로만 출력</text>
  <text x="410" y="234" text-anchor="end" font-size="20" font-weight="700" fill="#C2410C">{fmt_pct(c1_c5_c6)}</text>
  <text x="410" y="252" text-anchor="end" font-size="10" fill="#EA580C">internal confusion (n={amb["n_high_grp"]})</text>

  <!-- C3/C4 pair + C2 absorption (red) -->
  <g filter="url(#softShadow)">
    <rect x="30" y="292" width="400" height="78" rx="10" fill="#FEF2F2" stroke="#DC2626" stroke-width="2"/>
  </g>
  <circle cx="58" cy="331" r="18" fill="#EF4444" stroke="#991B1B" stroke-width="1.5"/>
  <text x="58" y="336" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">C3</text>
  <circle cx="98" cy="331" r="18" fill="#EF4444" stroke="#991B1B" stroke-width="1.5"/>
  <text x="98" y="336" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">C4</text>
  <text x="124" y="335" font-size="14" fill="#7F1D1D">↔</text>
  <circle cx="146" cy="331" r="13" fill="#FCA5A5" stroke="#B91C1C" stroke-width="1.2" stroke-dasharray="3,2"/>
  <text x="146" y="335" text-anchor="middle" font-size="10" font-weight="700" fill="#7F1D1D">C2</text>
  <text x="172" y="319" font-size="13" font-weight="700" fill="#7F1D1D">상호 혼동 + C2 흡수</text>
  <text x="172" y="337" font-size="11" fill="#991B1B">C3↔C4 pair 모호 · C2 쪽으로 흡수되는 경향</text>
  <text x="410" y="326" text-anchor="end" font-size="20" font-weight="700" fill="#B91C1C">{fmt_pct(c3_c4)}</text>
  <text x="410" y="344" text-anchor="end" font-size="10" fill="#DC2626">pair confusion (n={amb["n_med_pair"]})</text>

  <!-- Divider -->
  <line x1="450" y1="70" x2="450" y2="385" stroke="#E5E7EB" stroke-width="1"/>

  <!-- ===================== RIGHT: Bottom-region evidence ===================== -->
  <text x="475" y="74" font-size="14" font-weight="700" fill="#1F2937">Bottom-region evidence</text>
  <text x="475" y="92" font-size="10.5" fill="#6B7280">손목 IMU bottom 영역의 정보성 — posture별로 다름</text>

  <!-- 11/18 callout -->
  <g filter="url(#softShadow)">
    <rect x="475" y="108" width="345" height="68" rx="10" fill="#F5F3FF" stroke="#7C3AED" stroke-width="2"/>
  </g>
  <text x="495" y="135" font-size="32" font-weight="800" fill="#5B21B6">{cells_hi} / {cells_n}</text>
  <text x="568" y="128" font-size="11" font-weight="700" fill="#5B21B6">posture × channel 조합</text>
  <text x="568" y="146" font-size="10.5" fill="#6D28D9">bottom 영역에서 class 변별력이 더 강함</text>
  <text x="568" y="164" font-size="10" font-style="italic" fill="#7C3AED">→ bottom-anchored feature가 의미 있는 후보</text>

  <!-- Bar chart label -->
  <text x="475" y="195" font-size="11.5" font-weight="700" fill="#374151">Bottom event agreement by posture (acc-gyro)</text>

  <!-- Gridlines and bars -->{grid}
  {bars_svg}
  <text x="480" y="382" font-size="10" fill="#9CA3AF">threshold = {BOTTOM_THRESHOLD:.2f} · scope = posture</text>

  <!-- HW callout arrow -->
  <line x1="755" y1="305" x2="755" y2="340" stroke="#DC2626" stroke-width="1.4" stroke-dasharray="3,2"/>
  <text x="755" y="300" text-anchor="middle" font-size="10" font-weight="700" fill="#B91C1C">HW: 손목 신호 ↔ bottom 이벤트 어긋남</text>

  <!-- ===================== Bottom conclusion bar ===================== -->
  <g filter="url(#softShadow)">
    <rect x="20" y="410" width="820" height="68" rx="10" fill="#1E1B4B" stroke="#5B21B6" stroke-width="2"/>
  </g>
  <text x="40" y="438" font-size="11" font-weight="700" fill="#C4B5FD" letter-spacing="0.5">CONCLUSION</text>
  <text x="40" y="463" font-size="14" font-weight="700" fill="#FFFFFF">
    → single label 단정 불가 · class_set + uncertainty_flag + confidence_level 필요
  </text>
</svg>
"""


def main() -> None:
    amb = load_ambiguity()
    posture = load_posture_agree(BOTTOM_THRESHOLD)
    cells_hi, cells_n = compute_bottom_vs_full_cells()

    print(f"ambiguity  | c2={amb['c2_recall']:.3f}  c1c5c6={amb['c1_c5_c6_internal_confusion_rate']:.3f}  c3c4={amb['c3_c4_pair_confusion_rate']:.3f}")
    print(f"posture@{BOTTOM_THRESHOLD:.2f} | SA={posture['SA']:.3f}  CA={posture['CA']:.3f}  HW={posture['HW']:.3f}")
    print(f"bottom>full cells: {cells_hi}/{cells_n}")

    svg = build_svg(amb, posture, cells_hi, cells_n)
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
