"""Step 4R-A — Feature-based ceiling (HistGradientBoosting) reference experiment.

Reads (read-only):
    data/step4/step4_modeling_dataset.csv

Writes:
    data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_raw.csv
    data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_zscore.csv
    data/step4r/4ra_feature_ceiling/step4r_hgb_schema_outputs_raw.csv
    data/step4r/4ra_feature_ceiling/step4r_hgb_schema_outputs_zscore.csv
    reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_metrics.csv
    reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_confusion.csv
    reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_results.md

This is a feature-based ceiling reference experiment, not the project's main
contribution. Existing Step 4 LR baseline files are NOT modified — they remain
as legacy/reference. HGB posterior replaces LR posterior at the schema input;
all schema rules and thresholds follow Step 3 / Step 4 (see
generate_step4_schema_outputs.py).

Reference docs:
    reports/step4_research_reframing.md                      (4R-A definition)
    reports/step3/step3_output_schema_uncertainty_policy.md   (schema policy)
    reports/step4/step4_threshold_calibration.md              (threshold values)
    scripts/generate_step4_schema_outputs.py                  (schema rule reference)

Run:
    python scripts/train_step4r_feature_ceiling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "step4" / "step4_modeling_dataset.csv"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "step4r" / "4ra_feature_ceiling"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "reports" / "step4r" / "4ra_feature_ceiling"

OUTPUT_PRED = {
    "raw": OUTPUT_DATA_DIR / "step4r_hgb_predictions_raw.csv",
    "zscore": OUTPUT_DATA_DIR / "step4r_hgb_predictions_zscore.csv",
}
OUTPUT_SCHEMA = {
    "raw": OUTPUT_DATA_DIR / "step4r_hgb_schema_outputs_raw.csv",
    "zscore": OUTPUT_DATA_DIR / "step4r_hgb_schema_outputs_zscore.csv",
}
OUTPUT_METRICS_CSV = OUTPUT_REPORT_DIR / "step4r_feature_ceiling_metrics.csv"
OUTPUT_CONFUSION_CSV = OUTPUT_REPORT_DIR / "step4r_feature_ceiling_confusion.csv"
OUTPUT_REPORT_MD = OUTPUT_REPORT_DIR / "step4r_feature_ceiling_results.md"

EXPECTED_ROW_COUNT = 9275

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
P_COLS = [f"p_{c}" for c in CLASSES]
SPLITS = ["train", "val", "test"]

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

OUTPUT_COL_ORDER = CARRY_COLS + P_COLS + [
    "pred_argmax_debug",
    "class_set_prediction",
    "uncertainty_flags",
    "caption_confidence_level",
    "no_call",
]

# Thresholds copied verbatim from reports/step4/step4_threshold_calibration.md.
# These are LR-derived val candidates. We intentionally reuse them so that the
# HGB ceiling is evaluated under the *same Step 3 / Step 4 schema policy* as
# the LR baseline. Threshold re-calibration for HGB is out of scope here.
THRESHOLDS = {
    "raw": {
        "confident_C2_threshold": 0.3800,
        "non_trivial_C2_threshold": 0.1123,
        "within_group_threshold": 0.1650,
        "anchor_suppression_threshold": 0.5000,
        "anchor_no_call_threshold": 0.2500,
    },
    "zscore": {
        "confident_C2_threshold": 0.3900,
        "non_trivial_C2_threshold": 0.1059,
        "within_group_threshold": 0.1646,
        "anchor_suppression_threshold": 0.5000,
        "anchor_no_call_threshold": 0.2500,
    },
}

ALLOWED_POSTURES = {"SA", "CA", "HW"}
ALLOWED_LEVELS = {"confident", "hedged", "low", "no_call"}
ALLOWED_FLAGS = {
    "confident_C2",
    "within_group_ambiguity_c1_c5_c6",
    "pair_ambiguity_c3_c4",
    "pair_plus_c2_absorption",
    "anchor_unreliable",
    "posture_unknown",
    "low_confidence_no_class_set",
}
ALLOWED_CLASS_SETS = {
    tuple(s) for s in [
        [],
        ["C2"],
        ["C1", "C5", "C6"],
        ["C1", "C5"],
        ["C1", "C6"],
        ["C5", "C6"],
        ["C3", "C4"],
        ["C3", "C4", "C2"],
    ]
}
ANCHOR_DEPENDENT_SETS = {tuple(s) for s in [["C2"], ["C3", "C4", "C2"]]}

RANDOM_STATE = 42

# HGB hyperparameters — single ceiling point, no grid search in this experiment.
HGB_PARAMS = dict(
    learning_rate=0.1,
    max_iter=200,
    max_depth=None,
    l2_regularization=0.0,
    early_stopping=False,
    random_state=RANDOM_STATE,
)

# LR baseline reference numbers (from reports/step4/step4_final_summary.md, test split).
LR_BASELINE = {
    "raw": {
        "test_accuracy": 0.3210,
        "test_macro_f1": 0.2843,
        "test_weighted_f1": 0.2841,
        "test_log_loss": 1.6173,
        "test_brier": 0.7746,
        "test_ece_15bin": 0.0320,
        "test_c2_recall": 0.8159,
        "test_c1_c5_c6_internal": 0.2462,
        "test_c3_c4_pair": 0.2199,
        "test_c3_to_c2_absorb": 0.3277,
        "test_c4_to_c2_absorb": 0.2723,
    },
    "zscore": {
        "test_accuracy": 0.3203,
        "test_macro_f1": 0.2885,
        "test_weighted_f1": 0.2884,
        "test_log_loss": 1.6141,
        "test_brier": 0.7734,
        "test_ece_15bin": 0.0260,
        "test_c2_recall": 0.8159,
        "test_c1_c5_c6_internal": 0.2238,
        "test_c3_c4_pair": 0.2008,
        "test_c3_to_c2_absorb": 0.3361,
        "test_c4_to_c2_absorb": 0.2681,
    },
}


# ---------------------------------------------------------------------------
# Dataset checks
# ---------------------------------------------------------------------------

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
        raise ValueError(f"Input dataset missing required columns: {missing}.")
    bad_classes = sorted(set(df["class_id"].unique()) - set(CLASSES))
    if bad_classes:
        raise ValueError(f"class_id contains values outside {CLASSES}: {bad_classes}")
    bad_splits = sorted(set(df["split"].unique()) - set(SPLITS))
    if bad_splits:
        raise ValueError(f"split contains values outside {SPLITS}: {bad_splits}")


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------

def _train_and_predict(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    feats = FEATURE_SETS[condition]
    train_mask = df["split"] == "train"
    X_train = df.loc[train_mask, feats].to_numpy(dtype="float64")
    y_train = df.loc[train_mask, "class_id"].to_numpy()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print(
        f"[{condition}] training HGB on n_train={int(train_mask.sum())}, "
        f"n_features={len(feats)}"
    )
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    n_iter = int(getattr(model, "n_iter_", -1) or -1)
    print(
        f"[{condition}] fitted. classes_={list(model.classes_)}, "
        f"n_iter={n_iter}"
    )

    classes = list(model.classes_)
    if classes != CLASSES:
        raise ValueError(
            f"[{condition}] model.classes_ ordering unexpected: "
            f"got {classes}, expected {CLASSES}."
        )

    X_all = df[feats].to_numpy(dtype="float64")
    proba = model.predict_proba(X_all)
    proba_df = pd.DataFrame(proba, index=df.index, columns=P_COLS)
    pred_argmax = pd.Series(
        np.array(CLASSES)[proba.argmax(axis=1)],
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
    _validate_predictions(out, condition)
    return out


def _validate_predictions(out: pd.DataFrame, condition: str) -> None:
    if len(out) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"[{condition}] predictions row count {len(out)} != {EXPECTED_ROW_COUNT}."
        )
    p = out[P_COLS].to_numpy(dtype="float64")
    if np.isnan(p).any():
        raise ValueError(f"[{condition}] NaN in probability columns.")
    if np.isinf(p).any():
        raise ValueError(f"[{condition}] inf in probability columns.")
    if (p < 0).any():
        raise ValueError(f"[{condition}] negative probabilities.")
    sums = p.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError(
            f"[{condition}] probability rows do not sum to 1 within 1e-6 "
            f"(min={sums.min():.6f}, max={sums.max():.6f})."
        )


# ---------------------------------------------------------------------------
# Schema assignment (replicated from generate_step4_schema_outputs.py)
# ---------------------------------------------------------------------------

def _assign_schema(row: dict, thr: dict) -> dict:
    posture = row["posture"]
    if pd.isna(posture) or posture not in ALLOWED_POSTURES:
        return {
            "class_set": [],
            "flags": ["posture_unknown"],
            "level": "no_call",
            "no_call": True,
        }
    p_c2 = float(row["p_C2"])
    argmax = row["pred_argmax_debug"]

    if p_c2 >= thr["confident_C2_threshold"] and argmax == "C2":
        return {
            "class_set": ["C2"],
            "flags": ["confident_C2"],
            "level": "confident",
            "no_call": False,
        }

    if argmax in ("C1", "C5", "C6"):
        group = ["C1", "C5", "C6"]
        wgt = thr["within_group_threshold"]
        included = [c for c in group if float(row[f"p_{c}"]) >= wgt]
        class_set = included if len(included) >= 2 else group
        return {
            "class_set": class_set,
            "flags": ["within_group_ambiguity_c1_c5_c6"],
            "level": "hedged",
            "no_call": False,
        }

    if argmax in ("C3", "C4"):
        if p_c2 >= thr["non_trivial_C2_threshold"]:
            return {
                "class_set": ["C3", "C4", "C2"],
                "flags": ["pair_plus_c2_absorption"],
                "level": "hedged",
                "no_call": False,
            }
        return {
            "class_set": ["C3", "C4"],
            "flags": ["pair_ambiguity_c3_c4"],
            "level": "hedged",
            "no_call": False,
        }

    return {
        "class_set": [],
        "flags": ["low_confidence_no_class_set"],
        "level": "no_call",
        "no_call": True,
    }


def _apply_anchor(result: dict, row: dict, thr: dict) -> dict:
    ar = row["anchor_reliability"]
    if pd.isna(ar):
        raise ValueError(
            f"anchor_reliability is NaN for sample_id={row.get('sample_id')!r}."
        )
    ar = float(ar)
    flags = list(result["flags"])
    is_unreliable = ar < thr["anchor_suppression_threshold"]
    is_below_no_call = ar < thr["anchor_no_call_threshold"]
    is_anchor_dep = tuple(result["class_set"]) in ANCHOR_DEPENDENT_SETS
    if is_unreliable and "anchor_unreliable" not in flags:
        flags.append("anchor_unreliable")
    if result["no_call"]:
        return {
            "class_set": result["class_set"],
            "flags": flags,
            "level": result["level"],
            "no_call": True,
        }
    if is_below_no_call and is_anchor_dep:
        return {
            "class_set": [],
            "flags": flags,
            "level": "no_call",
            "no_call": True,
        }
    if is_unreliable:
        new_level = result["level"]
        if new_level == "confident":
            new_level = "hedged"
        elif new_level == "hedged":
            new_level = "low"
        return {
            "class_set": result["class_set"],
            "flags": flags,
            "level": new_level,
            "no_call": False,
        }
    return {
        "class_set": result["class_set"],
        "flags": flags,
        "level": result["level"],
        "no_call": False,
    }


def _process_row(row: dict, thr: dict) -> dict:
    base = _assign_schema(row, thr)
    final = _apply_anchor(base, row, thr)
    deduped_flags = list(dict.fromkeys(final["flags"]))
    return {
        "class_set_prediction": json.dumps(final["class_set"]),
        "uncertainty_flags": json.dumps(deduped_flags),
        "caption_confidence_level": final["level"],
        "no_call": bool(final["no_call"]),
    }


def _generate_schema_outputs(pred_df: pd.DataFrame, condition: str) -> pd.DataFrame:
    thr = THRESHOLDS[condition]
    if thr["anchor_no_call_threshold"] >= thr["anchor_suppression_threshold"]:
        raise ValueError(
            f"[{condition}] threshold relation violated: "
            f"anchor_no_call_threshold ({thr['anchor_no_call_threshold']}) "
            f"must be < anchor_suppression_threshold "
            f"({thr['anchor_suppression_threshold']})."
        )
    records = pred_df.to_dict("records")
    new_cols = pd.DataFrame([_process_row(r, thr) for r in records])
    out = pd.concat(
        [pred_df.reset_index(drop=True), new_cols.reset_index(drop=True)],
        axis=1,
    )[OUTPUT_COL_ORDER]
    _validate_schema(out, condition)
    return out


def _validate_schema(out: pd.DataFrame, condition: str) -> None:
    if len(out) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"[{condition}] schema row count {len(out)} != {EXPECTED_ROW_COUNT}."
        )
    p = out[P_COLS].to_numpy(dtype="float64")
    if np.isnan(p).any() or np.isinf(p).any() or (p < 0).any():
        raise ValueError(f"[{condition}] schema p_* columns invalid.")
    sums = p.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError(f"[{condition}] schema p_* rows do not sum to 1.")
    bad_cs = []
    for s in out["class_set_prediction"]:
        cs = json.loads(s)
        if tuple(cs) not in ALLOWED_CLASS_SETS:
            bad_cs.append(cs)
    if bad_cs:
        raise ValueError(
            f"[{condition}] class_set_prediction outside whitelist: {bad_cs[:5]}"
        )
    bad_flags = []
    for s in out["uncertainty_flags"]:
        flags = json.loads(s)
        if len(flags) != len(set(flags)):
            raise ValueError(
                f"[{condition}] duplicate uncertainty_flags entry: {flags}"
            )
        for f in flags:
            if f not in ALLOWED_FLAGS:
                bad_flags.append(f)
    if bad_flags:
        raise ValueError(
            f"[{condition}] uncertainty_flags outside vocab: {bad_flags[:5]}"
        )
    bad_levels = sorted(set(out["caption_confidence_level"].unique()) - ALLOWED_LEVELS)
    if bad_levels:
        raise ValueError(
            f"[{condition}] caption_confidence_level outside enum: {bad_levels}"
        )
    coerced = out["no_call"].astype(bool)
    empty_cs = out["class_set_prediction"].apply(lambda s: json.loads(s) == [])
    if (coerced != empty_cs).any():
        raise ValueError(
            f"[{condition}] no_call <-> class_set==[] consistency violated."
        )
    if ((coerced) & (out["caption_confidence_level"] != "no_call")).any():
        raise ValueError(
            f"[{condition}] no_call rows have level != 'no_call'."
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ece(probs: np.ndarray, y_correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error using top-1 confidence."""
    top1 = probs.max(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(top1, edges[1:-1]), 0, n_bins - 1)
    n = len(top1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_conf = float(top1[mask].mean())
        bin_acc = float(y_correct[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _multiclass_brier(probs: np.ndarray, y_idx: np.ndarray, n_classes: int) -> float:
    y_oh = np.zeros((len(y_idx), n_classes), dtype=float)
    y_oh[np.arange(len(y_idx)), y_idx] = 1.0
    return float(((probs - y_oh) ** 2).sum(axis=1).mean())


def _ambiguity(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    g135 = ["C1", "C5", "C6"]
    in_g = np.isin(y_true, g135)
    if in_g.any():
        pig = np.isin(y_pred[in_g], g135)
        out["c1_c5_c6_internal"] = float(
            ((y_pred[in_g] != y_true[in_g]) & pig).sum() / in_g.sum()
        )
    else:
        out["c1_c5_c6_internal"] = float("nan")
    g34 = ["C3", "C4"]
    in_p = np.isin(y_true, g34)
    if in_p.any():
        pip = np.isin(y_pred[in_p], g34)
        out["c3_c4_pair"] = float(
            ((y_pred[in_p] != y_true[in_p]) & pip).sum() / in_p.sum()
        )
    else:
        out["c3_c4_pair"] = float("nan")
    is_c3 = y_true == "C3"
    out["c3_to_c2_absorb"] = (
        float((y_pred[is_c3] == "C2").sum() / is_c3.sum()) if is_c3.any() else float("nan")
    )
    is_c4 = y_true == "C4"
    out["c4_to_c2_absorb"] = (
        float((y_pred[is_c4] == "C2").sum() / is_c4.sum()) if is_c4.any() else float("nan")
    )
    is_c2 = y_true == "C2"
    out["c2_recall"] = (
        float((y_pred[is_c2] == "C2").sum() / is_c2.sum()) if is_c2.any() else float("nan")
    )
    return out


def _evaluate_split(sub: pd.DataFrame) -> dict:
    y_true = sub["class_id"].to_numpy()
    y_pred = sub["pred_argmax_debug"].to_numpy()
    probs = sub[P_COLS].to_numpy(dtype="float64")
    cidx = {c: i for i, c in enumerate(CLASSES)}
    y_idx = np.array([cidx[c] for c in y_true])
    correct = (y_pred == y_true).astype(float)

    top1 = probs.max(axis=1)
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "n": int(len(sub)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=CLASSES, zero_division=0)
        ),
        "log_loss": float(log_loss(y_true, probs, labels=CLASSES)),
        "brier_multiclass": _multiclass_brier(probs, y_idx, len(CLASSES)),
        "ece_15bin": _ece(probs, correct, n_bins=15),
        "top1_prob_mean": float(top1.mean()),
        "top1_prob_p05": float(np.percentile(top1, 5)),
        "top1_prob_p25": float(np.percentile(top1, 25)),
        "top1_prob_p50": float(np.percentile(top1, 50)),
        "top1_prob_p75": float(np.percentile(top1, 75)),
        "top1_prob_p95": float(np.percentile(top1, 95)),
        "top1_top2_margin_mean": float(margin.mean()),
        "top1_top2_margin_p05": float(np.percentile(margin, 5)),
        "top1_top2_margin_p25": float(np.percentile(margin, 25)),
        "top1_top2_margin_p50": float(np.percentile(margin, 50)),
        "top1_top2_margin_p75": float(np.percentile(margin, 75)),
        "top1_top2_margin_p95": float(np.percentile(margin, 95)),
        "per_class": per_class,
        **{f"amb_{k}": v for k, v in _ambiguity(y_true, y_pred).items()},
    }


def _evaluate_all(pred_df: pd.DataFrame) -> dict:
    res = {}
    for s in SPLITS:
        sub = pred_df.loc[pred_df["split"] == s]
        if len(sub) == 0:
            continue
        res[s] = _evaluate_split(sub)
    return res


def _confusion_long(pred_df: pd.DataFrame, condition: str) -> pd.DataFrame:
    rows = []
    for s in SPLITS:
        sub = pred_df.loc[pred_df["split"] == s]
        if len(sub) == 0:
            continue
        cm = confusion_matrix(sub["class_id"], sub["pred_argmax_debug"], labels=CLASSES)
        for i, t in enumerate(CLASSES):
            for j, p in enumerate(CLASSES):
                rows.append(
                    {
                        "condition": condition,
                        "split": s,
                        "true_class": t,
                        "pred_class": p,
                        "count": int(cm[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def _flatten_metrics(metrics_by_cond: dict) -> pd.DataFrame:
    rows = []
    for cond, splits in metrics_by_cond.items():
        for s, m in splits.items():
            base = {"condition": cond, "split": s, "n": m["n"]}
            for k in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "log_loss",
                "brier_multiclass",
                "ece_15bin",
                "top1_prob_mean",
                "top1_prob_p05",
                "top1_prob_p25",
                "top1_prob_p50",
                "top1_prob_p75",
                "top1_prob_p95",
                "top1_top2_margin_mean",
                "top1_top2_margin_p05",
                "top1_top2_margin_p25",
                "top1_top2_margin_p50",
                "top1_top2_margin_p75",
                "top1_top2_margin_p95",
                "amb_c2_recall",
                "amb_c1_c5_c6_internal",
                "amb_c3_c4_pair",
                "amb_c3_to_c2_absorb",
                "amb_c4_to_c2_absorb",
            ]:
                rows.append({**base, "metric": k, "value": m[k]})
            for c in CLASSES:
                pc = m["per_class"][c]
                rows.append({**base, "metric": f"precision_{c}", "value": pc["precision"]})
                rows.append({**base, "metric": f"recall_{c}", "value": pc["recall"]})
                rows.append({**base, "metric": f"f1_{c}", "value": pc["f1"]})
                rows.append({**base, "metric": f"support_{c}", "value": pc["support"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema summary (for markdown report)
# ---------------------------------------------------------------------------

def _schema_summary(schema_df: pd.DataFrame) -> dict:
    out = {}
    levels = schema_df["caption_confidence_level"].value_counts().to_dict()
    for k in ["confident", "hedged", "low", "no_call"]:
        levels.setdefault(k, 0)
    out["level"] = levels
    cs = schema_df["class_set_prediction"].value_counts().sort_index().to_dict()
    out["class_set"] = cs
    flag_counts: dict = {}
    for s in schema_df["uncertainty_flags"]:
        for f in json.loads(s):
            flag_counts[f] = flag_counts.get(f, 0) + 1
    out["flags"] = flag_counts
    n_no_call = int(schema_df["no_call"].astype(bool).sum())
    out["n_no_call"] = n_no_call
    test_mask = schema_df["split"] == "test"
    cm = confusion_matrix(
        schema_df.loc[test_mask, "class_id"],
        schema_df.loc[test_mask, "pred_argmax_debug"],
        labels=CLASSES,
    )
    out["confusion_test"] = cm.tolist()
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _markdown_report(metrics_by_cond: dict, schema_summary: dict) -> str:
    L = []
    L.append("# Step 4R-A — Feature-based Ceiling 결과 보고서")
    L.append("")
    L.append("- 생성 스크립트: `scripts/train_step4r_feature_ceiling.py` (단일 실행으로 모든 산출물 생성)")
    L.append("- 입력: `data/step4/step4_modeling_dataset.csv` (read-only)")
    L.append("- 출력 디렉토리: `data/step4r/4ra_feature_ceiling/`, `reports/step4r/4ra_feature_ceiling/`")
    L.append(
        "- 모델: HistGradientBoostingClassifier "
        "(class_weight=balanced via sample_weight, learning_rate=0.1, "
        "max_iter=200, max_depth=None, l2_regularization=0.0, "
        "early_stopping=False, random_state=42)"
    )
    L.append("- 입력 feature: 기존 Step 4의 main 5 features (raw 또는 zscore branch) + posture one-hot 3개")
    L.append("- split: `v1_36_8_8` (participant-disjoint, 기존 split 컬럼 그대로 사용)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 본 실험의 위치")
    L.append("")
    L.append(
        "1. 본 실험은 **AICRC_v2의 main contribution이 아니다.** "
        "main pipeline은 `reports/step4_research_reframing.md`에서 정의된 "
        "uncertainty-aware sensor-to-schema-to-caption pipeline이며, "
        "본 실험은 그 안의 **Step 4R-A — feature-based ceiling reference**이다."
    )
    L.append(
        "2. 본 실험은 기존 Step 4 LR baseline과 **비교**되며, 그것을 *대체하지 않는다*. "
        "기존 `data/step4/`, `reports/step4/` 산출물은 legacy/reference로 보존된다. "
        "비교 결과는 §3에 정리한다."
    )
    L.append(
        "3. HGB가 LR 대비 분류 또는 calibration 점수를 올려도 그 자체로는 "
        "**sensor-to-text learning을 의미하지 않는다.** 본 실험은 raw IMU sequence를 "
        "직접 학습하지 않으며, caption 생성도 수행하지 않는다."
    )
    L.append(
        "4. 본 실험 결과는 후속 **Step 4R-B — BiGRU + Attention raw IMU "
        "sensor-to-schema** 실험의 **비교 기준**으로 사용된다 "
        "(reframing 문서 §5.5의 모델 채택 기준)."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. 분류·Calibration 지표 요약")
    L.append("")
    for cond_label, cond in [("raw", "raw"), ("zscore", "zscore")]:
        L.append(f"### 2.{1 if cond == 'raw' else 2} {cond_label} branch")
        L.append("")
        L.append("| split | n | accuracy | balanced_acc | macro F1 | weighted F1 | log loss | Brier | ECE (15-bin) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in SPLITS:
            m = metrics_by_cond[cond][s]
            L.append(
                f"| {s} | {m['n']} | {m['accuracy']:.4f} | "
                f"{m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | "
                f"{m['weighted_f1']:.4f} | {m['log_loss']:.4f} | "
                f"{m['brier_multiclass']:.4f} | {m['ece_15bin']:.4f} |"
            )
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. LR baseline 대비 비교 (test split)")
    L.append("")
    L.append("LR baseline 수치는 `reports/step4/step4_final_summary.md` 인용. HGB 수치는 본 실험 산출물.")
    L.append("")
    for ci, cond in enumerate(["raw", "zscore"], start=1):
        L.append(f"### 3.{ci} {cond} branch (test)")
        L.append("")
        L.append("| 지표 | LR baseline | HGB ceiling | 차이 (HGB − LR) |")
        L.append("|---|---:|---:|---:|")
        lr = LR_BASELINE[cond]
        h = metrics_by_cond[cond]["test"]
        rows = [
            ("accuracy", lr["test_accuracy"], h["accuracy"]),
            ("macro F1", lr["test_macro_f1"], h["macro_f1"]),
            ("weighted F1", lr["test_weighted_f1"], h["weighted_f1"]),
            ("log loss", lr["test_log_loss"], h["log_loss"]),
            ("Brier (multi)", lr["test_brier"], h["brier_multiclass"]),
            ("ECE (15-bin)", lr["test_ece_15bin"], h["ece_15bin"]),
            ("C2 recall", lr["test_c2_recall"], h["amb_c2_recall"]),
            ("C1/C5/C6 internal", lr["test_c1_c5_c6_internal"], h["amb_c1_c5_c6_internal"]),
            ("C3/C4 pair", lr["test_c3_c4_pair"], h["amb_c3_c4_pair"]),
            ("C3 → C2 absorb", lr["test_c3_to_c2_absorb"], h["amb_c3_to_c2_absorb"]),
            ("C4 → C2 absorb", lr["test_c4_to_c2_absorb"], h["amb_c4_to_c2_absorb"]),
        ]
        for name, a, b in rows:
            L.append(f"| {name} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. Per-class precision / recall / F1 (test split)")
    L.append("")
    for ci, cond in enumerate(["raw", "zscore"], start=1):
        L.append(f"### 4.{ci} {cond} branch (test)")
        L.append("")
        L.append("| class | precision | recall | F1 | support |")
        L.append("|---|---:|---:|---:|---:|")
        for c in CLASSES:
            pc = metrics_by_cond[cond]["test"]["per_class"][c]
            L.append(
                f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
                f"{pc['f1']:.4f} | {pc['support']} |"
            )
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. Top-1 confidence 및 top1-top2 margin 분포 (test split)")
    L.append("")
    for ci, cond in enumerate(["raw", "zscore"], start=1):
        m = metrics_by_cond[cond]["test"]
        L.append(f"### 5.{ci} {cond} branch")
        L.append("")
        L.append("| 통계 | top1 prob | top1-top2 margin |")
        L.append("|---|---:|---:|")
        L.append(f"| mean | {m['top1_prob_mean']:.4f} | {m['top1_top2_margin_mean']:.4f} |")
        L.append(f"| p05 | {m['top1_prob_p05']:.4f} | {m['top1_top2_margin_p05']:.4f} |")
        L.append(f"| p25 | {m['top1_prob_p25']:.4f} | {m['top1_top2_margin_p25']:.4f} |")
        L.append(f"| p50 | {m['top1_prob_p50']:.4f} | {m['top1_top2_margin_p50']:.4f} |")
        L.append(f"| p75 | {m['top1_prob_p75']:.4f} | {m['top1_top2_margin_p75']:.4f} |")
        L.append(f"| p95 | {m['top1_prob_p95']:.4f} | {m['top1_top2_margin_p95']:.4f} |")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. Schema 출력 분포 (전체 9275행)")
    L.append("")
    for ci, cond in enumerate(["raw", "zscore"], start=1):
        ss = schema_summary[cond]
        L.append(f"### 6.{ci} {cond} branch")
        L.append("")
        L.append("**caption_confidence_level**:")
        L.append("")
        L.append("| confident | hedged | low | no_call |")
        L.append("|---:|---:|---:|---:|")
        lv = ss["level"]
        L.append(f"| {lv['confident']} | {lv['hedged']} | {lv['low']} | {lv['no_call']} |")
        L.append("")
        L.append("**class_set_prediction**:")
        L.append("")
        L.append("| class_set | count |")
        L.append("|---|---:|")
        for k, v in ss["class_set"].items():
            L.append(f"| `{k}` | {v} |")
        L.append("")
        L.append("**uncertainty_flags occurrence**:")
        L.append("")
        L.append("| flag | rows containing it |")
        L.append("|---|---:|")
        for k in sorted(ss["flags"].keys()):
            L.append(f"| `{k}` | {ss['flags'][k]} |")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. Confusion matrix (test split)")
    L.append("")
    L.append(
        "전체 train/val/test confusion 데이터는 "
        "`reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_confusion.csv`에 "
        "long format(condition, split, true_class, pred_class, count)으로 저장된다."
    )
    L.append("")
    for ci, cond in enumerate(["raw", "zscore"], start=1):
        L.append(f"### 7.{ci} {cond} branch (test)")
        L.append("")
        cm = schema_summary[cond]["confusion_test"]
        header = "| true \\ pred | " + " | ".join(CLASSES) + " |"
        sep = "|---|" + "---:|" * len(CLASSES)
        L.append(header)
        L.append(sep)
        for i, c_true in enumerate(CLASSES):
            row = "| " + c_true + " | " + " | ".join(str(cm[i][j]) for j in range(len(CLASSES))) + " |"
            L.append(row)
        L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. 본 실험에서 명시적으로 결정하지 않는 사항")
    L.append("")
    L.append(
        "- **HGB hyperparameter는 grid search하지 않았다.** ceiling reference 한 점만 "
        "산출하는 것이 목적이므로 default 위주(learning_rate=0.1, max_iter=200, "
        "early_stopping=False)로 단일 학습을 수행했다."
    )
    L.append(
        "- **HGB 전용 threshold 재calibration은 수행하지 않았다.** schema 출력 생성에는 "
        "LR-derived val threshold(`reports/step4/step4_threshold_calibration.md`)를 "
        "그대로 사용했다. HGB의 posterior 분포는 LR과 다를 수 있으므로 schema 출력 "
        "분포(§6)는 *동일한 정책 하의 비교*로 읽어야 하며, threshold 재calibration은 "
        "별도 후속작업이다."
    )
    L.append(
        "- **본 실험은 main contribution이 아니다.** main pipeline의 학습 본체는 "
        "Step 4R-B(BiGRU + Attention)이며, 본 실험은 그 비교 기준만 제공한다."
    )
    L.append(
        "- **Step 5 ~ 7 caption layer는 본 실험의 범위 밖이다.** 본 실험은 "
        "schema CSV까지만 산출한다."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 산출물 목록")
    L.append("")
    L.append("- `data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_raw.csv`")
    L.append("- `data/step4r/4ra_feature_ceiling/step4r_hgb_predictions_zscore.csv`")
    L.append("- `data/step4r/4ra_feature_ceiling/step4r_hgb_schema_outputs_raw.csv`")
    L.append("- `data/step4r/4ra_feature_ceiling/step4r_hgb_schema_outputs_zscore.csv`")
    L.append("- `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_metrics.csv` (long format)")
    L.append("- `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_confusion.csv` (long format)")
    L.append("- `reports/step4r/4ra_feature_ceiling/step4r_feature_ceiling_results.md` (본 보고서)")
    L.append("")
    L.append("---")
    L.append("")
    L.append(
        "*본 보고서는 `scripts/train_step4r_feature_ceiling.py` 실행 시 "
        "자동 생성된다. 새로운 split / 데이터 / 캡션은 본 실험으로 인해 생성되지 않으며, "
        "기존 Step 1 ~ 4 산출물은 수정되지 않는다.*"
    )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Step 4R-A — Feature-based ceiling (HistGradientBoosting)")
    print("=" * 64)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}.")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"loaded {len(df)} rows from {INPUT_FILE}")
    _check_dataset(df)

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_by_cond: dict = {}
    schema_summary: dict = {}
    confusion_frames = []

    for condition in ["raw", "zscore"]:
        print()
        print("-" * 64)
        print(f"condition: {condition}")
        print("-" * 64)
        pred_df = _train_and_predict(df, condition)
        pred_df.to_csv(OUTPUT_PRED[condition], index=False, encoding="utf-8-sig")
        print(f"[{condition}] saved predictions -> {OUTPUT_PRED[condition]}")

        schema_df = _generate_schema_outputs(pred_df, condition)
        schema_df.to_csv(OUTPUT_SCHEMA[condition], index=False, encoding="utf-8-sig")
        print(f"[{condition}] saved schema outputs -> {OUTPUT_SCHEMA[condition]}")

        metrics_by_cond[condition] = _evaluate_all(pred_df)
        schema_summary[condition] = _schema_summary(schema_df)
        confusion_frames.append(_confusion_long(pred_df, condition))

        for s in SPLITS:
            m = metrics_by_cond[condition][s]
            print(
                f"[{condition}] {s:5s} n={m['n']:4d}  acc={m['accuracy']:.4f}  "
                f"macro_F1={m['macro_f1']:.4f}  log_loss={m['log_loss']:.4f}  "
                f"ECE={m['ece_15bin']:.4f}"
            )

    metrics_df = _flatten_metrics(metrics_by_cond)
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
    print(f"saved metrics -> {OUTPUT_METRICS_CSV}")

    confusion_df = pd.concat(confusion_frames, axis=0, ignore_index=True)
    confusion_df.to_csv(OUTPUT_CONFUSION_CSV, index=False, encoding="utf-8-sig")
    print(f"saved confusion -> {OUTPUT_CONFUSION_CSV}")

    md = _markdown_report(metrics_by_cond, schema_summary)
    OUTPUT_REPORT_MD.write_text(md, encoding="utf-8")
    print(f"saved report -> {OUTPUT_REPORT_MD}")

    print()
    print("=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
