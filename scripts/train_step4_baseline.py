"""Step 4 - Train baseline (multinomial logistic regression).

Reads (read-only):
    data/step4/step4_modeling_dataset.csv

Writes:
    data/step4/step4_predictions_raw.csv
    data/step4/step4_predictions_zscore.csv
    data/step4/step4_model_raw.joblib
    data/step4/step4_model_zscore.joblib

Behavior locked by:
    reports/step4/step4_feature_model_decision_memo.md      (sec. 7)
    reports/step4/step4_modeling_calibration_plan.md        (sec. 4-6)
    reports/step3/step3_output_schema_uncertainty_policy.md (sec. 10)

Trains the Step 4 primary baseline under both normalization candidates
that Step 3 sec. 10 / Step 4 plan sec. 5 keep on the table:

    (a) raw    + posture_as_input
    (b) zscore + posture_as_input    [posture_train_zscore]

Per-rep class posteriors are emitted for train/val/test rows so that
threshold calibration (calibrate_step4_thresholds.py) and schema-output
generation (generate_step4_schema_outputs.py) can be authored in
separate scripts. This script does NOT calibrate thresholds, does NOT
produce schema-shaped outputs, and does NOT write captions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "step4" / "step4_modeling_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "step4"
OUTPUT_PREDICTIONS = {
    "raw": OUTPUT_DIR / "step4_predictions_raw.csv",
    "zscore": OUTPUT_DIR / "step4_predictions_zscore.csv",
}
OUTPUT_MODELS = {
    "raw": OUTPUT_DIR / "step4_model_raw.joblib",
    "zscore": OUTPUT_DIR / "step4_model_zscore.joblib",
}

EXPECTED_ROW_COUNT = 9275

LABEL_COL = "class_id"
ALLOWED_CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
SPLIT_COL = "split"
ALLOWED_SPLITS = ["train", "val", "test"]

POSTURE_ONEHOT = ["posture_SA", "posture_CA", "posture_HW"]

FEATURE_SETS = {
    "raw": [
        "motion_range_acc_z",
        "depth_proxy",
        "motion_range_gyro_mag",
        "bottom_stability_acc",
        "bottom_transition_delta_acc_z",
    ] + POSTURE_ONEHOT,
    "zscore": [
        "motion_range_acc_z_zscore",
        "depth_proxy_zscore",
        "motion_range_gyro_mag_zscore",
        "bottom_stability_acc_zscore",
        "bottom_transition_delta_acc_z_zscore",
    ] + POSTURE_ONEHOT,
}

CARRY_COLS = [
    "sample_id",
    "rep_id",
    "participant_id",
    "class_id",
    "posture",
    "split",
    "anchor_reliability",
    "anchor_type",
]

RANDOM_STATE = 42
MAX_ITER = 5000


def _check_dataset(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Input dataset row count {len(df)} != expected {EXPECTED_ROW_COUNT}."
        )
    required = (
        set(CARRY_COLS)
        | set(FEATURE_SETS["raw"])
        | set(FEATURE_SETS["zscore"])
    )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Input dataset missing required columns: {missing}. "
            f"available columns: {list(df.columns)}"
        )
    bad_classes = sorted(set(df[LABEL_COL].unique()) - set(ALLOWED_CLASSES))
    if bad_classes:
        raise ValueError(
            f"class_id contains values outside {ALLOWED_CLASSES}: {bad_classes}"
        )
    bad_splits = sorted(set(df[SPLIT_COL].unique()) - set(ALLOWED_SPLITS))
    if bad_splits:
        raise ValueError(
            f"split contains values outside {ALLOWED_SPLITS}: {bad_splits}"
        )


def _train_one(
    df: pd.DataFrame,
    feature_cols: list,
    condition_name: str,
) -> tuple[LogisticRegression, pd.DataFrame]:
    train_mask = df[SPLIT_COL] == "train"
    X_train = df.loc[train_mask, feature_cols].to_numpy(dtype="float64")
    y_train = df.loc[train_mask, LABEL_COL].to_numpy()
    print(
        f"[{condition_name}] training on n_train={int(train_mask.sum())}, "
        f"n_features={len(feature_cols)}"
    )

    # Note: sklearn's default multi_class='auto' combined with solver='lbfgs'
    # yields a multinomial logistic regression for >2 classes. This is the
    # primary baseline spec in the Step 4 decision memo (sec. 7). We do not
    # set multi_class explicitly so this script is robust across sklearn
    # versions where that parameter is being deprecated.
    model = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    n_iter = int(model.n_iter_[0]) if hasattr(model, "n_iter_") else -1
    print(
        f"[{condition_name}] fitted. classes_={list(model.classes_)}, "
        f"n_iter={n_iter}"
    )

    classes = model.classes_.tolist()
    if classes != ALLOWED_CLASSES:
        raise ValueError(
            f"[{condition_name}] model.classes_ ordering unexpected: "
            f"got {classes}, expected {ALLOWED_CLASSES}."
        )

    X_all = df[feature_cols].to_numpy(dtype="float64")
    proba = model.predict_proba(X_all)
    proba_df = pd.DataFrame(
        proba,
        index=df.index,
        columns=[f"p_{c}" for c in classes],
    )
    pred_argmax = pd.Series(
        model.classes_[proba.argmax(axis=1)],
        index=df.index,
        name="pred_argmax_debug",
    )

    out = pd.concat(
        [
            df[CARRY_COLS].reset_index(drop=True),
            proba_df.reset_index(drop=True),
            pred_argmax.reset_index(drop=True),
        ],
        axis=1,
    )
    return model, out


def _metrics(out: pd.DataFrame, condition_name: str) -> dict:
    print(f"[{condition_name}] split metrics (pred_argmax_debug vs class_id):")
    summary = {}
    for s in ALLOWED_SPLITS:
        mask = out[SPLIT_COL] == s
        y_true = out.loc[mask, LABEL_COL]
        y_pred = out.loc[mask, "pred_argmax_debug"]
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=ALLOWED_CLASSES,
            zero_division=0,
        )
        f1w = f1_score(
            y_true,
            y_pred,
            average="weighted",
            labels=ALLOWED_CLASSES,
            zero_division=0,
        )
        summary[s] = {
            "n": int(mask.sum()),
            "accuracy": acc,
            "macro_f1": f1m,
            "weighted_f1": f1w,
        }
        print(
            f"  {s:5s} n={int(mask.sum()):4d}  "
            f"acc={acc:.4f}  macro_F1={f1m:.4f}  weighted_F1={f1w:.4f}"
        )
    return summary


def _validate_predictions(
    out: pd.DataFrame,
    input_split_counts: dict,
    condition_name: str,
) -> None:
    if len(out) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"[{condition_name}] row count {len(out)} != {EXPECTED_ROW_COUNT}."
        )
    out_split_counts = out[SPLIT_COL].value_counts().sort_index().to_dict()
    if out_split_counts != input_split_counts:
        raise ValueError(
            f"[{condition_name}] split counts disagree with input. "
            f"input: {input_split_counts}, output: {out_split_counts}"
        )

    p_cols = [f"p_{c}" for c in ALLOWED_CLASSES]
    p = out[p_cols].to_numpy(dtype="float64")
    if np.isnan(p).any():
        n = int(np.isnan(p).sum())
        raise ValueError(f"[{condition_name}] NaN in probability columns: {n} cells.")
    if np.isinf(p).any():
        n = int(np.isinf(p).sum())
        raise ValueError(f"[{condition_name}] +/-inf in probability columns: {n} cells.")
    if (p < 0).any():
        n = int((p < 0).sum())
        raise ValueError(f"[{condition_name}] negative probabilities: {n} cells.")
    sums = p.sum(axis=1)
    bad = ~np.isclose(sums, 1.0, atol=1e-6)
    if bad.any():
        n = int(bad.sum())
        raise ValueError(
            f"[{condition_name}] probability row-sum != 1.0 in {n} rows. "
            f"min sum={float(sums.min())}, max sum={float(sums.max())}."
        )
    print(
        f"[{condition_name}] probability validation: passed "
        f"(no NaN/inf/negatives; all row sums == 1 within 1e-6)"
    )


def main() -> int:
    print("=" * 64)
    print("Step 4 - Train baseline (multinomial LR + L2 + balanced)")
    print("=" * 64)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}. "
            f"Run scripts/build_step4_modeling_dataset.py first."
        )

    df = pd.read_csv(INPUT_FILE)
    print(f"loaded {len(df)} rows from {INPUT_FILE}")
    input_split_counts = df[SPLIT_COL].value_counts().sort_index().to_dict()
    print(f"split counts: {input_split_counts}")
    _check_dataset(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries: dict = {}
    for condition_name, feature_cols in FEATURE_SETS.items():
        print()
        print("-" * 64)
        print(f"condition: {condition_name}")
        print(f"feature columns ({len(feature_cols)}): {feature_cols}")
        print("-" * 64)
        model, out = _train_one(df, feature_cols, condition_name)
        all_summaries[condition_name] = _metrics(out, condition_name)
        _validate_predictions(out, input_split_counts, condition_name)

        pred_path = OUTPUT_PREDICTIONS[condition_name]
        model_path = OUTPUT_MODELS[condition_name]
        out.to_csv(pred_path, index=False)
        joblib.dump(model, model_path)
        print(f"[{condition_name}] saved predictions -> {pred_path}")
        print(f"[{condition_name}] saved model       -> {model_path}")
        print(f"[{condition_name}] training complete.")

    print()
    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"input rows                  : {len(df)}")
    print(f"input split counts          : {input_split_counts}")
    for cond, summary in all_summaries.items():
        for s in ALLOWED_SPLITS:
            m = summary[s]
            print(
                f"{cond:6s} {s:5s} acc={m['accuracy']:.4f}  "
                f"macro_F1={m['macro_f1']:.4f}  weighted_F1={m['weighted_f1']:.4f}"
            )
    print(f"saved predictions raw       : {OUTPUT_PREDICTIONS['raw']}")
    print(f"saved predictions zscore    : {OUTPUT_PREDICTIONS['zscore']}")
    print(f"saved model raw             : {OUTPUT_MODELS['raw']}")
    print(f"saved model zscore          : {OUTPUT_MODELS['zscore']}")
    print(
        "note: pred_argmax_debug is for debugging only and is NOT the "
        "final schema output. class_set_prediction will be produced by "
        "scripts/generate_step4_schema_outputs.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
