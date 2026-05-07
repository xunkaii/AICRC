"""Step 2.5-4 — Reference Separability Audit (v2).

This is a *reference* separability audit, not a final model. It fits a
small multinomial logistic regression on three feature settings and asks:
how much of C1..C6 structure is captured by these features, before we
pick anything? Nothing here is a feature-selection or model commitment.

Inputs (read-only):
  data/manifest_split.csv
  data/step2/normalization_feasibility_features.csv

Outputs:
  data/step2/reference_separability_predictions.csv
  reports/reference_separability_audit.md
  reports/reference_separability_metrics.csv
  reports/reference_separability_confusion_by_setting.csv
  reports/reference_separability_ambiguous_groups.csv

Note on sklearn:
  In this environment, `import sklearn` silently aborts the interpreter
  (the local numpy is a MINGW-W64 build that crashes inside sklearn's
  long-double initialization). The script therefore implements
  multinomial LR via `scipy.optimize.minimize` with L-BFGS-B; the
  semantics match sklearn's `LogisticRegression(multi_class='multinomial',
  solver='lbfgs', class_weight='balanced')`. The report makes this
  fallback explicit.

Run:
  python scripts/audit_step25_reference_separability_v2.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MANIFEST_PATH = DATA_DIR / "manifest_split.csv"
FEATURES_PATH = DATA_DIR / "step2" / "normalization_feasibility_features.csv"

CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")
POSTURES = ("SA", "CA", "HW")
SPLITS = ("train", "val", "test")

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
POSTURE_TO_IDX = {p: i for i, p in enumerate(POSTURES)}

# Confusion structure tags.
HIGH_CONFUSION_GROUP = ("C1", "C5", "C6")
MEDIUM_CONFUSION_PAIR = ("C3", "C4")
CLEAN_REFERENCE_CLASS = "C2"

SETTINGS = {
    "raw_core_features": (
        "raw_motion_range_acc_z",
        "raw_depth_proxy",
        "raw_bottom_recovery_slope_acc_z",
    ),
    "posture_train_zscore_core_features": (
        "posture_train_zscore_motion_range_acc_z",
        "posture_train_zscore_depth_proxy",
        "posture_train_zscore_bottom_recovery_slope_acc_z",
    ),
    "posture_train_robust_core_features": (
        "posture_train_robust_motion_range_acc_z",
        "posture_train_robust_depth_proxy",
        "posture_train_robust_bottom_recovery_slope_acc_z",
    ),
}

LR_L2 = 1e-3
LR_MAX_ITER = 1000
LR_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv_dict(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(f"ERROR: required input not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(s):
    if s is None or s == "":
        return None
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def fmt_float(v, ndigits=6):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return f"{float(v):.{ndigits}f}"


# ---------------------------------------------------------------------------
# Manual StandardScaler / Multinomial LR (sklearn substitute)
# ---------------------------------------------------------------------------

class StandardScaler:
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _balanced_sample_weight(y: np.ndarray, n_classes: int) -> np.ndarray:
    n_samples = y.shape[0]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    cw = n_samples / (n_classes * counts)
    return cw[y]


def fit_multinomial_lr(X: np.ndarray, y: np.ndarray,
                       n_classes: int,
                       l2: float = LR_L2,
                       max_iter: int = LR_MAX_ITER,
                       random_state: int = LR_RANDOM_STATE):
    """Multinomial LR with L2 + balanced class weights via L-BFGS-B."""
    n_samples, n_features = X.shape
    sw = _balanced_sample_weight(y, n_classes)
    sw_norm = sw / sw.sum() * n_samples  # mean 1

    y_oh = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_oh[np.arange(n_samples), y] = 1.0

    def loss_grad(theta):
        W = theta[: n_classes * n_features].reshape(n_classes, n_features)
        b = theta[n_classes * n_features:]
        logits = X @ W.T + b  # (N, K)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        z = e.sum(axis=1, keepdims=True)
        log_p = (logits - m) - np.log(z)
        p = e / z

        loss_per = -(y_oh * log_p).sum(axis=1)
        loss = float((sw_norm * loss_per).sum() / n_samples)
        loss += 0.5 * l2 * float((W * W).sum())

        delta = (p - y_oh) * sw_norm[:, None] / n_samples  # (N, K)
        grad_W = delta.T @ X + l2 * W
        grad_b = delta.sum(axis=0)
        grad = np.concatenate([grad_W.ravel(), grad_b])
        return loss, grad

    rng = np.random.RandomState(random_state)
    theta0 = rng.normal(0.0, 0.01, n_classes * n_features + n_classes)
    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "gtol": 1e-7})
    W = res.x[: n_classes * n_features].reshape(n_classes, n_features)
    b = res.x[n_classes * n_features:]
    return W, b, res


def predict_proba(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    logits = X @ W.T + b
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Metrics (sklearn substitute)
# ---------------------------------------------------------------------------

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    p = np.zeros(n_classes)
    r = np.zeros(n_classes)
    f = np.zeros(n_classes)
    s = np.zeros(n_classes, dtype=int)
    for k in range(n_classes):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        s[k] = int((y_true == k).sum())
        p[k] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r[k] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f[k] = 2 * p[k] * r[k] / (p[k] + r[k]) if (p[k] + r[k]) > 0 else 0.0
    return p, r, f, s


def macro_f1(y_true, y_pred, n_classes):
    _, _, f, _ = precision_recall_f1(y_true, y_pred, n_classes)
    return float(f.mean())


def weighted_f1(y_true, y_pred, n_classes):
    _, _, f, s = precision_recall_f1(y_true, y_pred, n_classes)
    total = int(s.sum())
    if total == 0:
        return 0.0
    return float((f * s).sum() / total)


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_xy(rows, feature_cols):
    """Return X (n, 3+3 one-hot), y (n,), and meta (list of dicts)."""
    X_list = []
    y_list = []
    meta = []
    for r in rows:
        vals = [parse_float(r.get(c)) for c in feature_cols]
        if any(v is None for v in vals):
            continue
        if r["class_id"] not in CLASS_TO_IDX:
            continue
        if r["posture_canonical"] not in POSTURE_TO_IDX:
            continue
        oh = [0.0, 0.0, 0.0]
        oh[POSTURE_TO_IDX[r["posture_canonical"]]] = 1.0
        X_list.append(vals + oh)
        y_list.append(CLASS_TO_IDX[r["class_id"]])
        meta.append(r)
    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, meta


# ---------------------------------------------------------------------------
# Per-setting evaluation
# ---------------------------------------------------------------------------

def ambiguous_flag(class_id: str) -> str:
    if class_id in HIGH_CONFUSION_GROUP:
        return "high_confusion_group"
    if class_id in MEDIUM_CONFUSION_PAIR:
        return "medium_confusion_pair"
    if class_id == CLEAN_REFERENCE_CLASS:
        return "clean_reference_class"
    return ""


def ambiguous_metrics(y_true_idx, y_pred_idx):
    high = {CLASS_TO_IDX[c] for c in HIGH_CONFUSION_GROUP}
    med = {CLASS_TO_IDX[c] for c in MEDIUM_CONFUSION_PAIR}
    c2 = CLASS_TO_IDX[CLEAN_REFERENCE_CLASS]

    n_high = int(np.sum(np.isin(y_true_idx, list(high))))
    n_high_internal = int(np.sum(
        np.isin(y_true_idx, list(high))
        & np.isin(y_pred_idx, list(high))
        & (y_true_idx != y_pred_idx)
    ))
    rate_high = (n_high_internal / n_high) if n_high else float("nan")

    n_med = int(np.sum(np.isin(y_true_idx, list(med))))
    n_med_conf = int(np.sum(
        np.isin(y_true_idx, list(med))
        & np.isin(y_pred_idx, list(med))
        & (y_true_idx != y_pred_idx)
    ))
    rate_med = (n_med_conf / n_med) if n_med else float("nan")

    n_c2 = int(np.sum(y_true_idx == c2))
    n_c2_correct = int(np.sum((y_true_idx == c2) & (y_pred_idx == c2)))
    c2_rec = (n_c2_correct / n_c2) if n_c2 else float("nan")

    return {
        "n_high_grp": n_high,
        "c1_c5_c6_internal_confusion_rate": rate_high,
        "n_med_pair": n_med,
        "c3_c4_pair_confusion_rate": rate_med,
        "n_c2": n_c2,
        "c2_recall": c2_rec,
    }


def evaluate_split_predictions(setting_name, split_name, y_true_idx,
                                probs, meta):
    """Build per-row prediction records for the predictions CSV."""
    preds_idx = probs.argmax(axis=1)
    out_rows = []
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    for i, r in enumerate(meta):
        p = probs[i]
        top1 = int(sorted_idx[i, 0])
        top2 = int(sorted_idx[i, 1])
        rec = {
            "sample_id": r["sample_id"],
            "split": split_name,
            "class_id": r["class_id"],
            "posture_canonical": r["posture_canonical"],
            "setting": setting_name,
            "y_true": r["class_id"],
            "y_pred": CLASSES[int(preds_idx[i])],
            "top1_prob": float(p[top1]),
            "top2_class": CLASSES[top2],
            "top2_prob": float(p[top2]),
            "top1_top2_margin": float(p[top1] - p[top2]),
            "ambiguous_group_flag": ambiguous_flag(r["class_id"]),
        }
        for k, c in enumerate(CLASSES):
            rec[f"prob_{c}"] = float(p[k])
        out_rows.append(rec)
    return out_rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

PRED_COLS = [
    "sample_id", "split", "class_id", "posture_canonical", "setting",
    "y_true", "y_pred",
    "prob_C1", "prob_C2", "prob_C3", "prob_C4", "prob_C5", "prob_C6",
    "top1_prob", "top2_class", "top2_prob", "top1_top2_margin",
    "ambiguous_group_flag",
]


def write_predictions_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PRED_COLS)
        for r in rows:
            out = []
            for c in PRED_COLS:
                v = r.get(c, "")
                if isinstance(v, float):
                    out.append(fmt_float(v))
                elif v is None:
                    out.append("")
                else:
                    out.append(str(v))
            w.writerow(out)


def write_metrics_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "split", "scope", "scope_value", "metric", "value"])
        for r in rows:
            w.writerow([
                r["setting"], r["split"], r["scope"], r["scope_value"],
                r["metric"],
                fmt_float(r["value"]) if isinstance(r["value"], float)
                    else str(r["value"]),
            ])


def write_confusion_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "split", "y_true", "y_pred", "count"])
        for r in rows:
            w.writerow([r["setting"], r["split"],
                        r["y_true"], r["y_pred"], int(r["count"])])


def write_ambiguous_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "setting", "split",
            "n_high_grp", "c1_c5_c6_internal_confusion_rate",
            "n_med_pair", "c3_c4_pair_confusion_rate",
            "n_c2", "c2_recall",
        ])
        for r in rows:
            w.writerow([
                r["setting"], r["split"],
                r["n_high_grp"], fmt_float(r["c1_c5_c6_internal_confusion_rate"]),
                r["n_med_pair"], fmt_float(r["c3_c4_pair_confusion_rate"]),
                r["n_c2"], fmt_float(r["c2_recall"]),
            ])


def write_audit_md(metrics_rows, ambiguous_rows, path: Path):
    # Build quick lookup for primary metrics.
    m_idx = defaultdict(dict)
    for r in metrics_rows:
        if r["scope"] == "overall":
            m_idx[(r["setting"], r["split"])][r["metric"]] = r["value"]

    a_idx = {(r["setting"], r["split"]): r for r in ambiguous_rows}

    lines: list[str] = []
    add = lines.append
    add("# Step 2.5-4 — Reference Separability Audit")
    add("")
    add("**Scope.** This is a *reference separability audit*, not a final")
    add("model. The goal is to ask how much of C1..C6 structure is captured")
    add("by a small candidate feature set under three normalization regimes.")
    add("Nothing on this list is adopted, rejected, or tuned for performance.")
    add("")

    add("## 1. Settings compared")
    add("")
    add("| setting | continuous features | normalization | posture input |")
    add("|---|---|---|---|")
    add("| `raw_core_features` | motion_range_acc_z, depth_proxy, "
        "bottom_recovery_slope_acc_z | none | one-hot SA/CA/HW |")
    add("| `posture_train_zscore_core_features` | same three features | "
        "per-posture z-score (train fit) | one-hot SA/CA/HW |")
    add("| `posture_train_robust_core_features` | same three features | "
        "per-posture median/IQR (train fit) | one-hot SA/CA/HW |")
    add("")
    add("**Why participant z-score is excluded.** It needs prior reps from")
    add("the new user (calibration) and cannot run on a first rep, so it is")
    add("not a deployable main-pipeline candidate. Step 2.5-3b kept it as a")
    add("ceiling reference; this audit does not train it.")
    add("")
    add("**Why posture-train normalizations are kept.** Both fit statistics")
    add("on **train rows only** and apply train statistics to val/test, so")
    add("they generalize to a new user's first rep given the user's posture.")
    add("")
    add("**Posture as model input.** All three settings include posture as a")
    add("3-dim one-hot input. Normalization and posture-as-input are not")
    add("mutually exclusive; both can carry signal.")
    add("")

    add("## 2. Reference model")
    add("")
    add("- Multinomial logistic regression, L2-regularized, balanced class")
    add("  weights, random_state=42.")
    add("- StandardScaler fit on **train** features only, applied to all splits.")
    add("- This is a *reference separability* model. It is intentionally")
    add("  simple. Numbers below should not be read as final-model")
    add("  performance estimates.")
    add("")
    add("**sklearn fallback note.** The MINGW-W64 numpy in this environment")
    add("aborts on `import sklearn`. The model is therefore implemented via")
    add("`scipy.optimize.minimize(method='L-BFGS-B')` with an explicit")
    add("multinomial-CE + L2 objective, equivalent to sklearn's")
    add("`LogisticRegression(multi_class='multinomial', solver='lbfgs', "
        "class_weight='balanced')`.")
    add("")

    add("## 3. Headline metrics (val / test)")
    add("")
    add("| setting | split | accuracy | macro F1 | weighted F1 |")
    add("|---|---|---|---|---|")
    for s in SETTINGS:
        for sp in ("val", "test"):
            row = m_idx.get((s, sp), {})
            add(f"| `{s}` | {sp} | "
                f"{fmt_float(row.get('accuracy'), 4)} | "
                f"{fmt_float(row.get('macro_f1'), 4)} | "
                f"{fmt_float(row.get('weighted_f1'), 4)} |")
    add("")

    add("## 4. Ambiguous-group breakdown (test split)")
    add("")
    add("Definitions (reused from prior audits):")
    add("")
    add("- `high_confusion_group` = C1, C5, C6 — share a normal/single/both-knee")
    add("  pattern and are expected to overlap.")
    add("- `medium_confusion_pair` = C3, C4 — distinct but related faults.")
    add("- `clean_reference_class` = C2 — should be most cleanly identified.")
    add("")
    add("| setting | C1/C5/C6 internal confusion | C3/C4 pair confusion | "
        "C2 recall |")
    add("|---|---|---|---|")
    for s in SETTINGS:
        a = a_idx.get((s, "test"))
        if not a:
            continue
        add(f"| `{s}` | "
            f"{fmt_float(a['c1_c5_c6_internal_confusion_rate'], 4)} | "
            f"{fmt_float(a['c3_c4_pair_confusion_rate'], 4)} | "
            f"{fmt_float(a['c2_recall'], 4)} |")
    add("")
    add("Reading guide:")
    add("")
    add("- `C1/C5/C6 internal confusion` is the share of C1∪C5∪C6 reps that")
    add("  the reference model assigns *to a different class within the same")
    add("  group*. High values mean the features cannot pull these three")
    add("  classes apart — which is exactly the expected difficulty.")
    add("- `C3/C4 pair confusion` is the share of C3∪C4 reps misrouted to")
    add("  the *other* class in the pair.")
    add("- `C2 recall` is the share of C2 reps correctly identified.")
    add("")

    add("## 5. Per-class breakdown")
    add("")
    add("Full table: `reports/reference_separability_metrics.csv`. Per-")
    add("posture macro F1 is also in that file (`scope=posture`).")
    add("")

    add("## 6. Confusion matrices")
    add("")
    add("Full table (long format, all settings × splits): "
        "`reports/reference_separability_confusion_by_setting.csv`.")
    add("")

    add("## 7. Does normalization help?")
    add("")
    add("Compare the headline rows for the same split across settings.")
    add("Step 2.5-3b had already shown that:")
    add("")
    add("- `depth_proxy` raw is dominated by posture (η²≈0.86 on posture vs")
    add("  ≈0.004 on class), and posture-aware z-score lifts class η² to")
    add("  ≈0.03 while collapsing posture η² to ≈0.")
    add("")
    add("Whether that translates to better separability under a model is")
    add("what this audit measures. **Direction**, not magnitude, is the")
    add("signal here.")
    add("")

    add("## 8. Implications for caption uncertainty")
    add("")
    add("If `C1/C5/C6 internal confusion` and `C3/C4 pair confusion` remain")
    add("substantial across all three settings, that is *evidence* that")
    add("downstream sensor-to-text outputs should express uncertainty")
    add("between these classes, rather than always committing to a single")
    add("label. Step 2.5-4 only flags this; it does not design the caption.")
    add("")

    add("## 9. What this audit does NOT decide")
    add("")
    add("- Which features survive into modeling.")
    add("- Whether the final model should be logistic regression, a tree")
    add("  ensemble, a small MLP, or something else.")
    add("- Whether any normalization scheme is the official choice.")
    add("- The numerical performance bar to clear; numbers here are a")
    add("  reference, not a target.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading inputs ...")
    print(f"  manifest : {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  features : {FEATURES_PATH.relative_to(PROJECT_ROOT)}")
    feat_rows = read_csv_dict(FEATURES_PATH)
    feat_rows = [r for r in feat_rows
                 if r["class_id"] in CLASS_TO_IDX
                 and r["posture_canonical"] in POSTURE_TO_IDX
                 and r["split"] in SPLITS]
    print(f"  loaded samples: {len(feat_rows)}")

    train_rows = [r for r in feat_rows if r["split"] == "train"]
    val_rows   = [r for r in feat_rows if r["split"] == "val"]
    test_rows  = [r for r in feat_rows if r["split"] == "test"]
    print(f"  train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    n_classes = len(CLASSES)
    all_predictions: list[dict] = []
    all_metrics: list[dict] = []
    all_confusions: list[dict] = []
    all_ambiguous: list[dict] = []

    for setting_name, feat_cols in SETTINGS.items():
        print(f"\nFitting setting: {setting_name}")
        X_train, y_train, meta_train = build_xy(train_rows, feat_cols)
        X_val,   y_val,   meta_val   = build_xy(val_rows,   feat_cols)
        X_test,  y_test,  meta_test  = build_xy(test_rows,  feat_cols)
        print(f"  X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")

        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_va = scaler.transform(X_val)
        X_te = scaler.transform(X_test)

        W, b, opt = fit_multinomial_lr(X_tr, y_train, n_classes,
                                        l2=LR_L2, max_iter=LR_MAX_ITER,
                                        random_state=LR_RANDOM_STATE)
        print(f"  optim: success={opt.success}  iter={opt.nit}  loss={opt.fun:.6f}")

        for split_name, X_s, y_arr, meta in (
            ("train", X_tr, y_train, meta_train),
            ("val",   X_va, y_val,   meta_val),
            ("test",  X_te, y_test,  meta_test),
        ):
            probs = predict_proba(W, b, X_s)
            preds = probs.argmax(axis=1)

            # Predictions: val + test only, all three settings.
            if split_name in ("val", "test"):
                all_predictions.extend(
                    evaluate_split_predictions(setting_name, split_name,
                                               y_arr, probs, meta)
                )

            # Headline metrics.
            acc = accuracy(y_arr, preds)
            mf1 = macro_f1(y_arr, preds, n_classes)
            wf1 = weighted_f1(y_arr, preds, n_classes)
            for metric, value in (("accuracy", acc),
                                   ("macro_f1", mf1),
                                   ("weighted_f1", wf1)):
                all_metrics.append({
                    "setting": setting_name, "split": split_name,
                    "scope": "overall", "scope_value": "all",
                    "metric": metric, "value": value,
                })

            # Per-class.
            p_arr, r_arr, f_arr, s_arr = precision_recall_f1(y_arr, preds, n_classes)
            for ci, c in enumerate(CLASSES):
                for metric, value in (
                    ("precision", float(p_arr[ci])),
                    ("recall",    float(r_arr[ci])),
                    ("f1",        float(f_arr[ci])),
                    ("support",   int(s_arr[ci])),
                ):
                    all_metrics.append({
                        "setting": setting_name, "split": split_name,
                        "scope": "class", "scope_value": c,
                        "metric": metric, "value": value,
                    })

            # Per-posture macro F1.
            posture_arr = np.array([m["posture_canonical"] for m in meta])
            for p in POSTURES:
                mask = posture_arr == p
                if mask.any():
                    p_mf1 = macro_f1(y_arr[mask], preds[mask], n_classes)
                else:
                    p_mf1 = float("nan")
                all_metrics.append({
                    "setting": setting_name, "split": split_name,
                    "scope": "posture", "scope_value": p,
                    "metric": "macro_f1", "value": p_mf1,
                })

            # Confusion.
            cm = confusion_matrix(y_arr, preds, n_classes)
            for i, ci in enumerate(CLASSES):
                for j, cj in enumerate(CLASSES):
                    all_confusions.append({
                        "setting": setting_name, "split": split_name,
                        "y_true": ci, "y_pred": cj, "count": int(cm[i, j]),
                    })

            # Ambiguous group metrics.
            ambig = ambiguous_metrics(y_arr, preds)
            ambig.update({"setting": setting_name, "split": split_name})
            all_ambiguous.append(ambig)

    print("\nWriting outputs ...")
    write_predictions_csv(all_predictions,
                          DATA_DIR / "step2" / "reference_separability_predictions.csv")
    write_metrics_csv(all_metrics,
                      REPORTS_DIR / "reference_separability_metrics.csv")
    write_confusion_csv(all_confusions,
                        REPORTS_DIR / "reference_separability_confusion_by_setting.csv")
    write_ambiguous_csv(all_ambiguous,
                        REPORTS_DIR / "reference_separability_ambiguous_groups.csv")
    write_audit_md(all_metrics, all_ambiguous,
                   REPORTS_DIR / "reference_separability_audit.md")

    # ---- Console summary ----
    print("")
    print("=== Step 2.5-4 reference separability summary ===")
    print(f"loaded samples : {len(feat_rows)}")
    print(f"settings       : {list(SETTINGS.keys())}")
    print("")
    head = {(r["setting"], r["split"], r["metric"]): r["value"]
            for r in all_metrics if r["scope"] == "overall"}
    amb_idx = {(r["setting"], r["split"]): r for r in all_ambiguous}
    for s in SETTINGS:
        print(f"setting: {s}")
        for sp in ("val", "test"):
            acc = head.get((s, sp, "accuracy"), float("nan"))
            mf1 = head.get((s, sp, "macro_f1"), float("nan"))
            a = amb_idx.get((s, sp), {})
            c2 = a.get("c2_recall", float("nan"))
            hi = a.get("c1_c5_c6_internal_confusion_rate", float("nan"))
            md = a.get("c3_c4_pair_confusion_rate", float("nan"))
            print(f"  {sp}: acc={acc:.4f}  macro_f1={mf1:.4f}  "
                  f"C2_rec={c2:.4f}  C1/C5/C6_int_conf={hi:.4f}  "
                  f"C3/C4_conf={md:.4f}")
    print("")
    print("outputs:")
    print("  data/step2/reference_separability_predictions.csv")
    print("  reports/reference_separability_audit.md")
    print("  reports/reference_separability_metrics.csv")
    print("  reports/reference_separability_confusion_by_setting.csv")
    print("  reports/reference_separability_ambiguous_groups.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
