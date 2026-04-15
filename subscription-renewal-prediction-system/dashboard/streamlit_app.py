from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.models.renewal_predictor import RenewalPredictor

PLAN_OPTIONS = ["starter", "growth", "business", "enterprise"]
HIGH_RISK_SCENARIO = {
    "monthly_usage_hours": 8.0,
    "login_frequency": 4.0,
    "last_login_days": 21.0,
    "support_tickets": 4.0,
    "payment_failures": 2.0,
    "subscription_plan": "starter",
}
HIGH_ENGAGEMENT_SCENARIO = {
    "monthly_usage_hours": 58.0,
    "login_frequency": 24.0,
    "last_login_days": 1.0,
    "support_tickets": 0.0,
    "payment_failures": 0.0,
    "subscription_plan": "business",
}


@st.cache_resource
def load_predictor() -> RenewalPredictor:
    return RenewalPredictor()


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    config = load_config()
    train_path = resolve_path(config["processed_data_config"]["train_data_csv"])
    if train_path.exists():
        return pd.read_csv(train_path)
    raw_path = resolve_path(config["raw_data_config"]["raw_data_csv"])
    return pd.read_csv(raw_path)


def default_value(column: str, reference_data: pd.DataFrame) -> float:
    if column in reference_data:
        return float(reference_data[column].median())
    return 0.0


def ensure_session_state(reference_data: pd.DataFrame) -> None:
    defaults = {
        "monthly_usage_hours": default_value("monthly_usage_hours", reference_data),
        "login_frequency": default_value("login_frequency", reference_data),
        "last_login_days": default_value("last_login_days", reference_data),
        "support_tickets": default_value("support_tickets", reference_data),
        "payment_failures": default_value("payment_failures", reference_data),
        "subscription_plan": "growth",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_scenario(scenario: dict[str, float | str]) -> None:
    for key, value in scenario.items():
        st.session_state[key] = value


def get_account_input() -> dict[str, float | str]:
    return {
        "monthly_usage_hours": float(st.session_state["monthly_usage_hours"]),
        "login_frequency": float(st.session_state["login_frequency"]),
        "last_login_days": float(st.session_state["last_login_days"]),
        "support_tickets": float(st.session_state["support_tickets"]),
        "payment_failures": float(st.session_state["payment_failures"]),
        "subscription_plan": str(st.session_state["subscription_plan"]),
    }


def get_risk_profile(renewal_probability: float) -> dict[str, str]:
    if renewal_probability > 0.75:
        return {
            "label": "High Renewal Probability",
            "insight": "User is highly engaged and likely to renew.",
            "icon": "🟢",
            "ui_state": "success",
        }
    if renewal_probability >= 0.4:
        return {
            "label": "Moderate Risk",
            "insight": "User shows moderate engagement. Consider targeted engagement strategies.",
            "icon": "🟠",
            "ui_state": "warning",
        }
    return {
        "label": "High Churn Risk",
        "insight": "User is at high risk of non-renewal. Immediate retention action recommended.",
        "icon": "🔴",
        "ui_state": "error",
    }


def show_risk_banner(risk_profile: dict[str, str]) -> None:
    message = f"{risk_profile['icon']} {risk_profile['label']}"
    if risk_profile["ui_state"] == "success":
        st.success(message)
    elif risk_profile["ui_state"] == "warning":
        st.warning(message)
    else:
        st.error(message)


def render_sidebar(reference_data: pd.DataFrame) -> dict[str, float | str]:
    st.sidebar.header("🧾 Subscription Inputs")
    st.sidebar.caption("Adjust account activity and lifecycle signals for a live forecast.")

    demo_left, demo_right = st.sidebar.columns(2)
    with demo_left:
        if st.button("Test High Risk User", use_container_width=True):
            apply_scenario(HIGH_RISK_SCENARIO)
    with demo_right:
        if st.button("Test High Engagement User", use_container_width=True):
            apply_scenario(HIGH_ENGAGEMENT_SCENARIO)

    st.sidebar.number_input(
        "Monthly usage hours",
        min_value=0.0,
        step=1.0,
        key="monthly_usage_hours",
    )
    st.sidebar.number_input(
        "Login frequency",
        min_value=0.0,
        step=1.0,
        key="login_frequency",
    )
    st.sidebar.number_input(
        "Days since last login",
        min_value=0.0,
        step=1.0,
        key="last_login_days",
    )
    st.sidebar.number_input(
        "Support tickets",
        min_value=0.0,
        step=1.0,
        key="support_tickets",
    )
    st.sidebar.number_input(
        "Payment failures",
        min_value=0.0,
        step=1.0,
        key="payment_failures",
    )
    st.sidebar.selectbox(
        "Subscription plan",
        options=PLAN_OPTIONS,
        key="subscription_plan",
    )

    return get_account_input()


def render_prediction_summary(
    prediction: dict[str, Any],
    explanation: dict[str, Any],
    account: dict[str, float | str],
) -> None:
    renewal_probability = float(prediction["renewal_probability"])
    risk_profile = get_risk_profile(renewal_probability)

    st.subheader("📈 Renewal Decision")
    show_risk_banner(risk_profile)

    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.metric("Business segment", risk_profile["label"])
    with metric_right:
        st.metric("Renewal Probability", f"{renewal_probability:.0%}")

    st.progress(min(max(renewal_probability, 0.0), 1.0))
    st.info(risk_profile["insight"])

    detail_left, detail_right = st.columns([1.05, 0.95])
    with detail_left:
        st.markdown("### 👤 Account Snapshot")
        st.dataframe(pd.DataFrame([account]), hide_index=True, use_container_width=True)
    with detail_right:
        st.markdown("### 🧠 Why this prediction?")
        st.caption(
            "These drivers summarize the strongest feature contributions behind the "
            "renewal forecast so business teams can validate the decision."
        )
        top_factors = pd.DataFrame(explanation["top_factors"]).head(3)
        st.dataframe(top_factors, hide_index=True, use_container_width=True)
        if not top_factors.empty:
            st.bar_chart(
                top_factors.set_index("feature")["shap_value"],
                use_container_width=True,
            )


def render_batch_forecasting(predictor: RenewalPredictor) -> None:
    st.subheader("📦 Batch Forecasting")
    st.write(
        "Upload a CSV with the base subscription lifecycle features to score "
        "multiple accounts at once."
    )
    upload = st.file_uploader("Batch CSV", type="csv")
    if upload is not None:
        batch_df = pd.read_csv(upload)
        predictions = predictor.predict(batch_df)
        result_df = pd.concat(
            [batch_df.reset_index(drop=True), pd.DataFrame(predictions)],
            axis=1,
        )
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            label="Download predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="subscription_renewal_predictions.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="Subscription Renewal Workbench", layout="wide")
    st.title("📊 Subscription Renewal Workbench")
    st.caption(
        "Business-facing renewal intelligence for customer success, lifecycle marketing, "
        "and revenue operations teams."
    )

    predictor = load_predictor()
    reference_data = load_reference_data()
    ensure_session_state(reference_data)
    account = render_sidebar(reference_data)

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("🔍 Single Account Forecast")
        st.write(
            "Generate an on-demand renewal forecast using product usage, billing behavior, "
            "and support engagement signals."
        )
        if st.button("Predict renewal", type="primary", use_container_width=True):
            prediction = predictor.predict_one(account)
            explanation = predictor.explain(account)
            render_prediction_summary(prediction, explanation, account)

    with right:
        render_batch_forecasting(predictor)


if __name__ == "__main__":
    main()
