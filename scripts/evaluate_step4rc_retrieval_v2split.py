"""Step 4R-C (v2split) Day 4+ — Retrieval evaluation + t-SNE (v2split).

Wrapper around evaluate_step4rc_retrieval.py. Monkey-patches input/output
paths to v2split tree.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\dl_env\\python.exe' -X utf8 \
        scripts/evaluate_step4rc_retrieval_v2split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_step4rc_retrieval as base  # noqa: E402


def _patch() -> None:
    v2_data = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    v2_report = (
        PROJECT_ROOT / "reports" / "step4r_v2split" / "4rc_contrastive_optional"
    )
    base.INPUT_JOINT = v2_data / "joint_embeddings.npz"
    base.INPUT_CORPUS = v2_data / "text_corpus.csv"
    base.OUTPUT_DATA_DIR = v2_data
    base.OUTPUT_REPORT_DIR = v2_report
    base.OUTPUT_METRICS_CSV = v2_data / "retrieval_metrics.csv"
    base.OUTPUT_BREAKDOWN_CSV = v2_report / "retrieval_breakdown.csv"
    base.OUTPUT_TSNE_CLASS = v2_report / "tsne_by_class.png"
    base.OUTPUT_TSNE_AMBIG = v2_report / "tsne_by_ambiguity_group.png"
    base.OUTPUT_FINAL_MD = v2_report / "results_final.md"


def main() -> int:
    _patch()
    print("=" * 64)
    print("Step 4R-C (v2split) Day 4+ — Retrieval evaluation")
    print(f"  INPUT_JOINT       -> {base.INPUT_JOINT}")
    print(f"  OUTPUT_FINAL_MD   -> {base.OUTPUT_FINAL_MD}")
    print("=" * 64)
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
