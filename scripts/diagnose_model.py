#!/usr/bin/env python
"""
Subscription-renewal model diagnostic.

Usage (run from project root):
    python scripts/diagnose_model.py

Paths (resolved from params.yaml):
    Model     : models/subscription_renewal_model.joblib
    Test data : data/processed/renewal_test.csv   (13 engineered features + target)

NOTE on SHAP vs permutation importance
---------------------------------------
The saved model is a sklearn Pipeline (LogisticRegression + PolynomialFeatures(degree=2)).
shap.TreeExplainer requires a tree model; the RenewalPredictor explicitly raises an error
for pipeline models and silently falls back to *uniform* importances in the dashboard
(LogisticRegression has no feature_importances_, so np.ones(n) is used -- the dashboard
"Prediction drivers" chart is currently weighted equally for every feature).

Permutation importance is model-agnostic, fast, and operates in the original feature
space, so it gives a meaningful ranking.

NOTE on correlated features
----------------------------
payment_failures appears three times in the feature set:
  payment_failures     -- raw count
  payment_reliability  -- 1 / (payment_failures + 1)
  risk_score           -- composite that includes payment_failures * 1.75

Standard single-feature permutation importance underestimates any one of these because
the others still carry the signal.  Section 2b therefore shuffles all three together
(grouped permutation importance) to measure the combined payment-failure contribution.

For the Section 4 sweep, build_renewal_features recomputes all derived columns each
time payment_failures changes, so the feature row stays internally consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.features.build_renewal_features import build_renewal_features

SEP = "=" * 64


def _roc_auc(model, x: pd.DataFrame, y: pd.Series) -> float:
    return float(roc_auc_score(y, model.predict_proba(x)[:, 1]))


def permutation_importance_grouped(
    model,
    x: pd.DataFrame,
    y: pd.Series,
    groups: list[list[str]],
    n_repeats: int = 30,
    random_state: int = 111,
) -> dict[str, dict]:
    """Return mean/std ROC-AUC drop for each feature group."""
    rng = np.random.default_rng(random_state)
    base = _roc_auc(model, x, y)
    results = {}
    for group in groups:
        drops = []
        for _ in range(n_repeats):
            xp = x.copy()
            idx = rng.permutation(len(xp))
            for col in group:
                xp[col] = xp[col].values[idx]
            drops.append(base - _roc_auc(model, xp, y))
        key = " + ".join(group) if len(group) > 1 else group[0]
        results[key] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return results


# ── 1. Load ───────────────────────────────────────────────────────────────────

config     = load_config()
model_path = resolve_path(config["training"]["artifact_path"])
test_path  = resolve_path(config["processed_data_config"]["test_data_csv"])

bundle    = joblib.load(model_path)
model     = bundle["model"]
metadata  = bundle.get("metadata", {})
threshold = float(
    metadata.get("prediction_threshold",
                 config["training"].get("prediction_threshold", 0.5))
)
target = metadata.get("target", config["raw_data_config"]["target"])

test_df = pd.read_csv(test_path)
test_x  = test_df.drop(columns=[target])
test_y  = test_df[target].astype(int)

inner = getattr(model, "named_steps", {}).get("model")

print(SEP)
print("SECTION 1 -- Artifacts loaded")
print(SEP)
print(f"  Model      : {model_path}")
print(f"  Test data  : {test_path}  ({len(test_df):,} rows, {len(test_x.columns)} features)")
print(f"  Threshold  : {threshold}")
print(f"  Features   : {test_x.columns.tolist()}")
print(f"  Pipeline   : {type(model).__name__} / inner: {type(inner).__name__ if inner else 'N/A'}")
print()

# ── 2a. Per-feature permutation importance ────────────────────────────────────

print(SEP)
print("SECTION 2a -- Per-feature permutation importance (n=30 repeats)")
print("  Each row: mean drop in ROC-AUC when THAT feature alone is shuffled.")
print("  Caveat: correlated siblings (payment_reliability, risk_score) absorb")
print("  signal from payment_failures -- see 2b for grouped result.")
print(SEP)

single_groups = [[col] for col in test_x.columns]
single_results = permutation_importance_grouped(
    model, test_x, test_y, single_groups, n_repeats=30
)

imp_rows = sorted(single_results.items(), key=lambda kv: kv[1]["mean"], reverse=True)

WATCH = {"payment_failures", "support_tickets"}
print(f"\n  {'Rank':<5} {'Feature':<28} {'Mean dAUC':>10}   {'Std':>8}")
print("  " + "-" * 58)
for rank, (feat, vals) in enumerate(imp_rows, 1):
    tag = "  <-- watch" if feat in WATCH else ""
    print(f"  {rank:<5} {feat:<28} {vals['mean']:>+.5f}   +-{vals['std']:.5f}{tag}")

print()

# ── 2b. Grouped importance: all payment-failure signals together ──────────────

print(SEP)
print("SECTION 2b -- Grouped permutation importance")
print("  Shuffles payment_failures + payment_reliability + risk_score together.")
print("  This removes their ability to compensate for each other -- a truer")
print("  measure of how much the model relies on payment-failure information.")
print(SEP)

pf_group  = ["payment_failures", "payment_reliability", "risk_score"]
st_group  = ["support_tickets",  "support_pressure"]
grouped   = permutation_importance_grouped(
    model, test_x, test_y,
    [pf_group, st_group],
    n_repeats=30,
)

for name, vals in grouped.items():
    print(f"\n  Group : {name}")
    print(f"  Mean dAUC : {vals['mean']:+.5f}   +- {vals['std']:.5f}")
    if vals["mean"] < 0.005:
        print("  Verdict   : POSSIBLY UNDER-WEIGHTED -- tiny group effect")
    elif vals["mean"] < 0.05:
        print("  Verdict   : Moderate effect -- detectable but not dominant")
    else:
        print("  Verdict   : Strong effect -- model relies on this group")

print()

# ── 3. Held-out metrics ───────────────────────────────────────────────────────

print(SEP)
print("SECTION 3 -- Held-out metrics (test split, recomputed live)")
print(SEP)

probas = model.predict_proba(test_x)[:, 1]
preds  = (probas >= threshold).astype(int)

acc  = accuracy_score(test_y, preds)
auc  = roc_auc_score(test_y, probas)
prec = precision_score(test_y, preds, zero_division=0)
rec  = recall_score(test_y, preds, zero_division=0)
cm   = confusion_matrix(test_y, preds)

print(f"\n  Accuracy   : {acc:.4f}")
print(f"  ROC-AUC    : {auc:.4f}")
print(f"  Precision  : {prec:.4f}  (predicted renewals that actually renewed)")
print(f"  Recall     : {rec:.4f}  (actual renewals that we caught)")
print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
print(f"  {'':18} Pred churn   Pred renew")
print(f"  Actual churn      {cm[0, 0]:>5}        {cm[0, 1]:>5}")
print(f"  Actual renew      {cm[1, 0]:>5}        {cm[1, 1]:>5}")
fpr = cm[0, 1] / cm[0].sum()
fnr = cm[1, 0] / cm[1].sum()
print(f"\n  False-positive rate : {fpr:.3f}  (churns predicted as renewals)")
print(f"  False-negative rate : {fnr:.3f}  (renewals predicted as churn)")
print()

# ── 4. Controlled probe: sweep payment_failures 0 -> 5 ───────────────────────
#
# Build one baseline account from the 6 raw input signals, then call
# build_renewal_features() at each step so payment_reliability and risk_score
# stay consistent with the new payment_failures value.
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print("SECTION 4 -- Controlled probe: payment_failures swept 0 -> 5")
print("  Derived features (payment_reliability, risk_score) are recomputed")
print("  at each step via build_renewal_features so the row stays consistent.")
print("  Other signals fixed at test-set median; subscription_plan = growth.")
print(SEP)

raw_input_cols = config["raw_data_config"]["input_features"]
numeric_raw    = [c for c in raw_input_cols if c != "subscription_plan"]

baseline_raw: dict[str, object] = {
    col: float(test_x[col].median()) if col in numeric_raw else "growth"
    for col in raw_input_cols
}
baseline_raw["payment_failures"] = 0.0

print("\n  Baseline raw signals:")
for k, v in baseline_raw.items():
    print(f"    {k:<28} = {v}")

print(f"\n  {'pf':<6} {'renewal_proba':>14}  {'delta vs pf=0':>14}  {'label':>8}")
print("  " + "-" * 52)

sweep_results: list[tuple[int, float]] = []
for pf in range(6):
    raw = {**baseline_raw, "payment_failures": float(pf)}
    row_df = build_renewal_features(
        pd.DataFrame([raw]), config=config
    )[test_x.columns]
    p = float(model.predict_proba(row_df)[0, 1])
    sweep_results.append((pf, p))

p0 = sweep_results[0][1]
for pf, p in sweep_results:
    label     = "renew" if p >= threshold else "churn"
    delta_str = "(baseline)" if pf == 0 else f"{p - p0:+.4f}"
    print(f"  {pf:<6} {p:>14.4f}  {delta_str:>14}  {label:>8}")

total_drop = p0 - sweep_results[-1][1]
print(f"\n  Total drop pf=0 -> pf=5 : {total_drop:+.4f}")

if total_drop < 0.05:
    print("  [!] Small drop (<0.05): payment_failures may be under-weighted.")
    print("      Also check the grouped rank in Section 2b.")
elif total_drop < 0.15:
    print("  [i] Moderate effect (0.05-0.15): detectable but not decisive alone.")
else:
    print("  [ok] Strong effect (>0.15): payment_failures meaningfully moves predictions.")
