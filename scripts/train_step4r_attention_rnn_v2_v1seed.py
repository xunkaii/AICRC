"""b01.5 training under parametrized split seed (v1-ratio robustness study).

Wrapper around train_step4r_attention_rnn_v2.py. Monkey-patches INPUT_NPZ
and _output_paths to land under data/step4r_v1seed_robustness/s{SEED}/.

The learning seed and other b01.5 hyperparameters are passed via the base
script's CLI (--seed, --aug, --jitter-sigma, --scale-low, --scale-high).

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe' -X utf8 \
        scripts/train_step4r_attention_rnn_v2_v1seed.py \
        --split-seed 7 \
        --seed 42 --exp-id b01_5_aug_jitter_scale_strong \
        --aug jitter_scale --jitter-sigma 0.05 --scale-low 0.85 --scale-high 1.15
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_step4r_attention_rnn_v2 as base  # noqa: E402


def _peel_split_seed(argv: list[str] | None) -> tuple[int, list[str]]:
    """Extract --split-seed N from argv and return (seed, remaining_argv)."""
    if argv is None:
        argv = sys.argv[1:]
    rest: list[str] = []
    split_seed: int | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--split-seed":
            split_seed = int(argv[i + 1])
            i += 2
            continue
        if a.startswith("--split-seed="):
            split_seed = int(a.split("=", 1)[1])
            i += 1
            continue
        rest.append(a)
        i += 1
    if split_seed is None:
        raise SystemExit("--split-seed is required")
    return split_seed, rest


def main(argv=None) -> int:
    split_seed, rest = _peel_split_seed(argv)
    root_data = (
        PROJECT_ROOT / "data" / "step4r_v1seed_robustness" / f"s{split_seed}"
    )
    root_report = (
        PROJECT_ROOT / "reports" / "step4r_v1seed_robustness" / f"s{split_seed}"
    )
    root_ckpt = (
        PROJECT_ROOT / "checkpoints" / "step4r_v1seed_robustness"
        / f"s{split_seed}"
    )

    base.INPUT_NPZ = root_data / "4rb_attention" / "step4r_sequence_dataset.npz"

    def _output_paths_v1seed(exp_id: str, seed: int) -> dict:
        rel = Path("experiments") / exp_id / f"seed{seed}"
        data_dir = root_data / "4rb_attention" / rel
        report_dir = root_report / "4rb_attention" / rel
        ckpt_dir = root_ckpt / "4rb_attention" / rel
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

    base._output_paths = _output_paths_v1seed
    print("=" * 64)
    print(f"v1seed b01.5 train | split_seed={split_seed}")
    print(f"  INPUT_NPZ -> {base.INPUT_NPZ}")
    print("=" * 64)
    return base.main(rest)


if __name__ == "__main__":
    sys.exit(main())
