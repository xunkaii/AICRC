"""Step 4R-B (v2split) — Temperature scaling under v2 participant split.

Wrapper around calibrate_step4r_attention_temperature_v2.py. The base
module's main() inlines its paths so we cannot monkey-patch them; we
override `base.main` with a new main() that uses v2split paths and reuses
all helper functions and constants from the base module unchanged.

Run:
    & 'C:\\Users\\user\\anaconda3\\envs\\aicrc_env\\python.exe' -X utf8 \
        scripts/calibrate_step4r_attention_temperature_v2_v2split.py \
        --seed 42 --exp-id b01_5_aug_jitter_scale_strong
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import calibrate_step4r_attention_temperature_v2 as base  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--exp-id", type=str, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    print("=" * 64)
    print(f"Step 4R-B (v2split) calibrate | exp_id={args.exp_id} | seed={args.seed}")
    print("=" * 64)

    rel = Path("experiments") / args.exp_id / f"seed{args.seed}"
    data_dir = PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention" / rel
    report_dir = PROJECT_ROOT / "reports" / "step4r_v2split" / "4rb_attention" / rel
    ckpt = (
        PROJECT_ROOT / "checkpoints" / "step4r_v2split" / "4rb_attention" / rel
        / "best.pt"
    )
    input_npz = (
        PROJECT_ROOT / "data" / "step4r_v2split" / "4rb_attention"
        / "step4r_sequence_dataset.npz"
    )
    for d in (data_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    out_npz = data_dir / "logits_calibrated.npz"
    out_preds_csv = data_dir / "predictions_calibrated.csv"
    out_metrics_csv = report_dir / "temperature_scaling_metrics.csv"
    out_report_md = report_dir / "temperature_scaling_results.md"

    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    if not input_npz.exists():
        raise FileNotFoundError(f"input npz missing: {input_npz}")

    z = np.load(input_npz, allow_pickle=True)
    data = {
        "X_norm": z["X_norm"].astype(np.float32),
        "y": z["y"].astype(np.int64),
        "class_id": z["class_id"].astype(str),
        "posture_canonical": z["posture_canonical"].astype(str),
        "sample_id": z["sample_id"].astype(str),
        "participant_id": z["participant_id"].astype(str),
        "split": z["split"].astype(str),
    }
    posture_oh = base._posture_onehot(data["posture_canonical"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading checkpoint -> {ckpt}")
    chk = torch.load(ckpt, map_location=device)
    model_class_name = chk.get("model_class", "Step4RBiGRUAttention")
    model_cls = base.MODEL_CLASS_REGISTRY.get(model_class_name)
    if model_cls is None:
        raise ValueError(
            f"unknown model_class in checkpoint: {model_class_name!r}"
        )
    model = model_cls(**chk["config"]).to(device)
    model.load_state_dict(chk["state_dict"])
    print(
        f"  model_class={model_class_name} best epoch={chk['epoch']} "
        f"val_macroF1={chk['val_macro_f1']:.4f}"
    )

    print("recomputing logits ...")
    t0 = time.time()
    logits_raw, attn_w = base._infer_logits(
        model, data["X_norm"], posture_oh, data["y"], device,
    )
    print(f"  done in {time.time() - t0:.1f}s. logits shape={logits_raw.shape}")

    val_mask = data["split"] == "val"
    T, hist = base._fit_temperature(
        logits_raw[val_mask], data["y"][val_mask], device,
    )
    print(
        f"fitted T = {T:.6f} (LBFGS closures={len(hist['iters'])}, "
        f"final NLL={hist['nll'][-1]:.6f})"
    )

    logits_cal = logits_raw / T
    probs_raw = base._softmax_np(logits_raw)
    probs_cal = base._softmax_np(logits_cal)

    before: dict = {}
    after: dict = {}
    for s in base.SPLITS:
        m = data["split"] == s
        if not m.any():
            continue
        before[s] = base._split_metrics(probs_raw[m], data["y"][m])
        after[s] = base._split_metrics(probs_cal[m], data["y"][m])
        b, a = before[s], after[s]
        print(
            f"  {s:5s} before  acc={b['accuracy']:.4f}  "
            f"macroF1={b['macro_f1']:.4f}  "
            f"logloss={b['log_loss']:.4f}  ECE={b['ece_15bin']:.4f}"
        )
        print(
            f"  {s:5s} after   acc={a['accuracy']:.4f}  "
            f"macroF1={a['macro_f1']:.4f}  "
            f"logloss={a['log_loss']:.4f}  ECE={a['ece_15bin']:.4f}"
        )

    np.savez(
        out_npz,
        logits_raw=logits_raw.astype(np.float32),
        logits_calibrated=logits_cal.astype(np.float32),
        probs_raw=probs_raw.astype(np.float32),
        probs_calibrated=probs_cal.astype(np.float32),
        attention_weights=attn_w.astype(np.float32),
        y=data["y"],
        class_id=data["class_id"],
        sample_id=data["sample_id"],
        participant_id=data["participant_id"],
        posture_canonical=data["posture_canonical"],
        split=data["split"],
        temperature=np.array([T], dtype=np.float64),
        seed=np.array([args.seed], dtype=np.int64),
    )
    print(f"saved npz -> {out_npz}")

    pred_idx = probs_cal.argmax(axis=1)
    sorted_p = np.sort(probs_cal, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    pe = base._entropy(probs_cal)
    pred_df = pd.DataFrame({
        "sample_id": data["sample_id"],
        "participant_id": data["participant_id"],
        "class_id": data["class_id"],
        "posture_canonical": data["posture_canonical"],
        "split": data["split"],
        "pred_argmax_calibrated": [base.CLASSES[i] for i in pred_idx],
        "top1_prob_calibrated": probs_cal.max(axis=1),
        "top1_top2_margin_calibrated": margin,
        "predictive_entropy_calibrated": pe,
        **{f"p_{c}_raw": probs_raw[:, i] for i, c in enumerate(base.CLASSES)},
        **{f"p_{c}_calibrated": probs_cal[:, i] for i, c in enumerate(base.CLASSES)},
        "temperature": [T] * len(data["y"]),
    })
    pred_df.to_csv(out_preds_csv, index=False, encoding="utf-8-sig")
    print(f"saved calibrated predictions -> {out_preds_csv}")

    rows = []
    for s in base.SPLITS:
        for stage, m in [("before", before.get(s, {})), ("after", after.get(s, {}))]:
            for k, v in m.items():
                if k == "n":
                    continue
                rows.append({"split": s, "stage": stage, "metric": k, "value": float(v)})
            if m:
                rows.append({"split": s, "stage": stage, "metric": "n", "value": float(m["n"])})
    rows.append({"split": "—", "stage": "—", "metric": "fitted_temperature", "value": float(T)})
    rows.append({"split": "—", "stage": "—", "metric": "lbfgs_n_closures", "value": float(len(hist["iters"]))})
    rows.append({"split": "—", "stage": "—", "metric": "final_val_nll", "value": float(hist["nll"][-1])})
    pd.DataFrame(rows).to_csv(out_metrics_csv, index=False, encoding="utf-8-sig")
    print(f"saved metrics -> {out_metrics_csv}")

    L = []
    L.append(f"# Step 4R-B (v2split) calibrate | exp=`{args.exp_id}` seed=`{args.seed}`")
    L.append("")
    L.append(f"- fitted T = {T:.6f}")
    L.append("")
    L.append("| split | stage | acc | macroF1 | logloss | Brier | ECE |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for s in base.SPLITS:
        b, a = before[s], after[s]
        L.append(
            f"| {s} | before | {b['accuracy']:.4f} | {b['macro_f1']:.4f} | "
            f"{b['log_loss']:.4f} | {b['brier_multiclass']:.4f} | {b['ece_15bin']:.4f} |"
        )
        L.append(
            f"| {s} | after | {a['accuracy']:.4f} | {a['macro_f1']:.4f} | "
            f"{a['log_loss']:.4f} | {a['brier_multiclass']:.4f} | {a['ece_15bin']:.4f} |"
        )
    out_report_md.write_text("\n".join(L), encoding="utf-8")
    print(f"saved report -> {out_report_md}")

    print()
    print("=" * 64)
    print(f"Done. T = {T:.6f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
