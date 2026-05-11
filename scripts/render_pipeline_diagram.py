"""
Pipeline diagram for the AICRC_v2 thesis (uncertainty-aware sensor-to-schema-to-caption).

Renders an overview figure for presentation slides:
  Sensor (Wrist IMU)  ->  Schema (4R-A / 4R-B / 4R-C)  ->  Caption (LLM presentation)
                                                            -> Step 8_v2 auto validation

Output:
  reports/step4r/pipeline_diagram.png
  reports/step4r/pipeline_diagram.pdf
  reports/step4r/pipeline_diagram.svg

Run:
  conda run -n aicrc_step4 python -X utf8 scripts/render_pipeline_diagram.py
"""
from __future__ import annotations

from pathlib import Path

from graphviz import Digraph


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "step4r"
OUT_STEM = OUT_DIR / "pipeline_diagram"

# Windows-safe Korean font. Falls back gracefully if not installed.
KOREAN_FONT = "Malgun Gothic"


def build_graph() -> Digraph:
    g = Digraph("aicrc_v2_pipeline", format="png")

    # Canvas / global attrs — tuned for 16:9 slide insertion.
    g.attr(
        rankdir="LR",
        bgcolor="white",
        nodesep="0.45",
        ranksep="0.85",
        margin="0.25",
        fontname=KOREAN_FONT,
        fontsize="14",
        compound="true",
        splines="spline",
    )
    g.attr(
        "node",
        fontname=KOREAN_FONT,
        fontsize="13",
        shape="box",
        style="rounded,filled",
        color="#333333",
        penwidth="1.4",
    )
    g.attr(
        "edge",
        fontname=KOREAN_FONT,
        fontsize="11",
        color="#444444",
        penwidth="1.4",
        arrowsize="0.9",
    )

    # ------------------------------------------------------------------ Sensor
    with g.subgraph(name="cluster_sensor") as s:
        s.attr(
            label="① Sensor",
            labeljust="l",
            style="rounded,filled",
            fillcolor="#E8F1FB",
            color="#1F6FB2",
            fontsize="16",
            fontcolor="#1F6FB2",
            penwidth="2",
            margin="14",
        )
        s.node(
            "imu",
            "손목 IMU\\n(acc x/y/z · gyro x/y/z)\\nlen=128, 6ch + posture",
            fillcolor="#FFFFFF",
            color="#1F6FB2",
        )
        s.node(
            "feat",
            "Step 4 feature\\n(handcrafted)",
            fillcolor="#FFFFFF",
            color="#1F6FB2",
            shape="note",
        )
        s.node(
            "seq",
            "Step 4R sequence\\n(raw window)",
            fillcolor="#FFFFFF",
            color="#1F6FB2",
            shape="note",
        )
        s.edge("imu", "feat", style="dotted")
        s.edge("imu", "seq", style="dotted")

    # ------------------------------------------------------------------ Schema
    with g.subgraph(name="cluster_schema") as s:
        s.attr(
            label="② Schema  (uncertainty-aware)",
            labeljust="l",
            style="rounded,filled",
            fillcolor="#FDEFE2",
            color="#C76A1A",
            fontsize="16",
            fontcolor="#C76A1A",
            penwidth="2",
            margin="14",
        )

        s.node(
            "m_a",
            "4R-A\\nHistGradientBoosting\\n(feature ceiling)",
            fillcolor="#FFFFFF",
            color="#C76A1A",
        )
        s.node(
            "m_b",
            "4R-B  ★ main\\nBiGRU + Attention\\n(raw sensor → schema)",
            fillcolor="#FFF6E6",
            color="#C76A1A",
            penwidth="2.4",
        )
        s.node(
            "m_c",
            "4R-C (optional)\\nIMU ↔ Text\\ncontrastive alignment",
            fillcolor="#FFFFFF",
            color="#C76A1A",
            style="rounded,filled,dashed",
        )

        s.node(
            "schema_out",
            "{class_set · posterior\\n| caption_confidence_level\\n| uncertainty_flags · no_call\\n| posterior/attention entropy}",
            shape="record",
            fillcolor="#FFFFFF",
            color="#C76A1A",
        )

        s.edge("m_a", "schema_out")
        s.edge("m_b", "schema_out", penwidth="2.2")
        s.edge("m_c", "schema_out", style="dashed")

    # ----------------------------------------------------------------- Caption
    with g.subgraph(name="cluster_caption") as s:
        s.attr(
            label="③ Caption  (Korean, schema-grounded)",
            labeljust="l",
            style="rounded,filled",
            fillcolor="#E9F4EC",
            color="#2E7D4F",
            fontsize="16",
            fontcolor="#2E7D4F",
            penwidth="2",
            margin="14",
        )
        s.node(
            "llm",
            "LLM = presentation layer\\n(분류 X, 판단 X)\\nclosed vocabulary",
            fillcolor="#FFFFFF",
            color="#2E7D4F",
        )
        s.node(
            "caption",
            "한국어 caption\\n(schema-faithful)",
            fillcolor="#FFFFFF",
            color="#2E7D4F",
        )
        s.node(
            "validator",
            "Step 8_v2\\n자동 schema-faithfulness 검증\\n(rule + LLM judge)",
            fillcolor="#FFFFFF",
            color="#2E7D4F",
            shape="component",
        )
        s.edge("llm", "caption")
        s.edge("caption", "validator", label="auto-eval")

    # ------------------------------------------------------ Cross-cluster flow
    g.edge("feat", "m_a", label="tabular")
    g.edge("seq", "m_b", label="window")
    g.edge("seq", "m_c", label="encoder", style="dashed")
    g.edge("schema_out", "llm", label="closed-vocab payload", penwidth="2.0")

    # ----------------------------------------------------------- Side comments
    # Explicit exclusions — pinned at the bottom as a note.
    g.node(
        "exclusions",
        (
            "본 논문 범위에서 제외:\\l"
            "  · sensor-to-text generation (LLM 학습 X)\\l"
            "  · 개인 의미 모형\\l"
            "  · human review (future work)\\l"
        ),
        shape="note",
        fillcolor="#F4F4F4",
        color="#888888",
        fontcolor="#555555",
        fontsize="11",
    )
    g.node(
        "principles",
        (
            "핵심 원칙\\l"
            "  · Text는 학습 target이 아니다\\l"
            "  · LLM은 schema 표현층\\l"
            "  · 관측되지 않은 양은 말하지 않는다\\l"
        ),
        shape="note",
        fillcolor="#F4F4F4",
        color="#888888",
        fontcolor="#555555",
        fontsize="11",
    )

    # Force notes to the bottom rank.
    with g.subgraph() as foot:
        foot.attr(rank="sink")
        foot.node("principles")
        foot.node("exclusions")

    return g


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = build_graph()

    # Render in multiple formats. cleanup=True removes the intermediate .gv source.
    for fmt in ("png", "pdf", "svg"):
        g.format = fmt
        out_path = g.render(filename=str(OUT_STEM), cleanup=True)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
