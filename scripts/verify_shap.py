import sys
sys.path.insert(0, ".")
from src.models.renewal_predictor import RenewalPredictor

pred = RenewalPredictor()

base = {
    "monthly_usage_hours": 22.13,
    "login_frequency": 19.0,
    "last_login_days": 6.1,
    "support_tickets": 3.0,
    "payment_failures": 0.0,
    "subscription_plan": "growth",
}

print("=== pf=0  (proba should be ~0.94) ===")
r0 = pred.predict_one(base)
e0 = pred.explain(base, top_k=8)
print(f"proba : {r0['renewal_probability']:.4f}")
print(f"backend : {e0['explanation_backend']}")
print(f"{'Feature':<28}  {'Value':>8}  {'SHAP':>8}")
print("-" * 50)
for f in e0["top_factors"]:
    v = f"{f['feature_value']:.3f}" if isinstance(f["feature_value"], float) else str(f["feature_value"])
    print(f"  {f['feature']:<26}  {v:>8}  {f['shap_value']:>+8.4f}")

print("\n=== pf=1  (proba should be ~0.71, pf cluster must be NEGATIVE) ===")
base1 = {**base, "payment_failures": 1.0}
r1 = pred.predict_one(base1)
e1 = pred.explain(base1, top_k=8)
print(f"proba : {r1['renewal_probability']:.4f}")
print(f"backend : {e1['explanation_backend']}")
print(f"{'Feature':<28}  {'Value':>8}  {'SHAP':>8}")
print("-" * 50)
for f in e1["top_factors"]:
    v = f"{f['feature_value']:.3f}" if isinstance(f["feature_value"], float) else str(f["feature_value"])
    marker = " <--" if f["feature"] in {"payment_failures", "payment_reliability", "risk_score"} else ""
    print(f"  {f['feature']:<26}  {v:>8}  {f['shap_value']:>+8.4f}{marker}")

print("\n=== Payment-cluster check ===")
pf_cluster = {"payment_failures", "payment_reliability", "risk_score"}
negatives = [f for f in e1["top_factors"] if f["feature"] in pf_cluster and f["shap_value"] < 0]
positives = [f for f in e1["top_factors"] if f["feature"] in pf_cluster and f["shap_value"] >= 0]
if negatives:
    print(f"PASS: {[f['feature'] for f in negatives]} show negative SHAP (push toward churn)")
if positives:
    print(f"WARN: {[f['feature'] for f in positives]} unexpectedly positive")
if not any(f["feature"] in pf_cluster for f in e1["top_factors"]):
    print("WARN: no payment-cluster features in top 8")
