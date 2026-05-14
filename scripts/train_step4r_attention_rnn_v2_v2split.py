"""Step 4R-B (v2split) — Parametrized BiGRU+Attention training under v2 split.

Thin wrapper around scripts/train_step4r_attention_rnn_v2.py. Imports the
base module (read-only, unmodified), monkey-patches the INPUT_NPZ path and
the _output_paths helper so that all v2split outputs land in:

    data/step4r_v2split/4rb_attention/experiments/{exp_id}/seed{N}/
    checkpoints/step4r_v2split/4rb_attention/experiments/{exp_id}/seed{N}/
    reports/step4r_v2split/4rb_attention/experiments/{exp_id}/seed{N}/

then forwards CLI argv unchanged to the base main().

For the b01.5 recipe (confirmed 4R-B baseline), call:
    & 'C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe' -X utf8 \
        scripts/train_step4r_attention_rnn_v2_v2split.py \
        --seed 42 --exp-id b01_5_aug_jitter_scale_strong \
        --aug jitter_scale --jitter-sigma 0.05 --scale-low 0.85 --scale-high 1.15
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_step4r_attention_rnn_v2 as base  # noqa: E402


def _output_paths_v2split(exp_id: str, seed: int) -> dict:
    rel = Path("experiments") / exp_id / f"seed{seed}"
    data_dir = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention" / rel
    )
    report_dir = (
        PROJECT_ROOT / "reports" / "step4r_v2split" / "4rb_attention" / rel
    )
    ckpt_dir = (
        PROJECT_ROOT / "checkpoints" / "step4r_v2split" / "4rb_attention" / rel
    )
    for d in (data_dir, report_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "preds": data_dir / "predictions.csv",
        "metrics": report_dir / "metrics.csv",
        "confusion": report_dir / "confusion.csv",
        "training_log": report_dir / "training_log.csv",
        "report_md": report_dir / "results.md",
        "best_ckpt": ckpt_dir / "best.pt",
        "data_dir": data_dir,
        "report_dir": report_dir,
        "ckpt_dir": ckpt_dir,
    }


def _patch() -> None:
    base.INPUT_NPZ = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention"
        / "step4r_sequence_dataset.npz"
    )
    base._output_paths = _output_paths_v2split


def main(argv: list[str] | None = None) -> int:
    _patch()
    print("=" * 64)
    print("Step 4R-B (v2split) — Attention RNN training")
    print(f"  INPUT_NPZ -> {base.INPUT_NPZ}")
    print("=" * 64)
    return base.main(argv)


if __name__ == "__main__":
    sys.exit(main())
