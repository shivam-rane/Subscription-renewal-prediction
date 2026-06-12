# Redesigned dashboard — layout only, all ML logic preserved from original.
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.json.config.default_engine = "json"
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

# ── Design tokens (Aurora Glass Palette) ──────────────────────────────────────
TOKENS: dict[str, str] = {
    # Background mesh
    "bg_base":             "#13102B",
    "mesh_violet":         "rgba(91,63,214,0.38)",
    "mesh_blue":           "rgba(37,99,235,0.32)",
    "mesh_teal":           "rgba(14,165,164,0.28)",
    # Glass surface
    "glass_bg":            "rgba(255,255,255,0.07)",
    "glass_border":        "rgba(255,255,255,0.16)",
    "glass_blur":          "",
    "glass_blur_hero":     "blur(8px)",
    "glass_radius":        "16px",
    # Derived surfaces & borders
    "sidebar_bg":          "rgba(19,16,43,0.82)",
    "header_bg":           "rgba(19,16,43,0.6)",
    "surface_dim":         "rgba(255,255,255,0.04)",
    "surface_faint":       "rgba(255,255,255,0.05)",
    "surface_stripe":      "rgba(255,255,255,0.03)",
    "border_subtle":       "rgba(255,255,255,0.12)",
    "border_faint":        "rgba(255,255,255,0.1)",
    "border_dimmer":       "rgba(255,255,255,0.08)",
    "border_dashed":       "rgba(255,255,255,0.2)",
    # Text
    "text_primary":        "#F5F6FF",
    "text_secondary":      "rgba(255,255,255,0.72)",
    "text_tertiary":       "rgba(255,255,255,0.45)",
    "text_ghost":          "rgba(255,255,255,0.35)",
    # Renew / success
    "success_fill":        "rgba(110,231,183,0.16)",
    "success_text":        "#A5F3D0",
    "success_accent":      "#6EE7B7",
    "success_border":      "rgba(110,231,183,0.4)",
    "success_pill":        "rgba(110,231,183,0.45)",
    "success_glow":        "rgba(110,231,183,0.5)",
    "success_glow_strong": "rgba(110,231,183,0.8)",
    # Moderate / warning
    "warning_fill":        "rgba(251,191,36,0.16)",
    "warning_text":        "#FCD96B",
    "warning_accent":      "#FBBF24",
    "warning_pill":        "rgba(251,191,36,0.45)",
    # Churn / danger
    "danger_fill":         "rgba(251,113,133,0.16)",
    "danger_text":         "#FDA4AF",
    "danger_accent":       "#FB7185",
    "danger_pill":         "rgba(251,113,133,0.45)",
    # Primary action
    "action_bg":           "rgba(99,102,241,0.22)",
    "action_border":       "rgba(165,180,252,0.4)",
    "action_text":         "#E0E7FF",
    "action_bg_hover":     "rgba(99,102,241,0.4)",
    "action_border_hover": "rgba(165,180,252,0.6)",
    "action_indigo":       "#6366F1",
    # Sidebar brand
    "brand_violet":        "#7F77DD",
    "blob_violet":         "#5B3FD6",
    "brand_glow":          "rgba(127,119,221,0.5)",
    # Table
    "surface_mid":         "rgba(255,255,255,0.08)",
    # Chart
    "text_chart":          "rgba(255,255,255,0.6)",
    "chart_zeroline":      "rgba(255,255,255,0.25)",
}


def glass_card(inner_html: str, pad: str = "16px") -> str:
    T = TOKENS
    return (
        f'<div style="background:{T["glass_bg"]};border:1px solid {T["glass_border"]};'
        f'border-radius:{T["glass_radius"]};padding:{pad};">'
        f"{inner_html}"
        f"</div>"
    )


# Data loading and prediction services
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
            "insight": "Strong renewal signal. Keep the account in expansion and advocacy motions.",
            "ui_state": "success",
        }
    if renewal_probability >= 0.4:
        return {
            "label": "Moderate Risk",
            "insight": "Mixed renewal signal. Trigger targeted engagement before the next billing cycle.",
            "ui_state": "warning",
        }
    return {
        "label": "High Churn Risk",
        "insight": "Low renewal signal. Prioritize immediate retention outreach and executive follow-up.",
        "ui_state": "error",
    }


# Global CSS and styling
def inject_global_css() -> None:
    T = TOKENS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        @import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css');

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        .block-container {{
            padding-top: 0 !important;
            padding-bottom: 24px !important;
            padding-left: 24px !important;
            padding-right: 24px !important;
            max-width: 100% !important;
        }}

        .stApp {{
            background: {T['bg_base']} !important;
            background-image:
                radial-gradient(ellipse at 20% 30%, {T['mesh_violet']} 0%, transparent 58%),
                radial-gradient(ellipse at 82% 18%, {T['mesh_blue']} 0%, transparent 52%),
                radial-gradient(ellipse at 58% 85%, {T['mesh_teal']} 0%, transparent 50%) !important;
            font-family: 'Inter', sans-serif !important;
        }}

        section[data-testid="stSidebar"] {{
            background: rgba(19,16,43,0.45) !important;
            backdrop-filter: blur(8px) !important;
            border-right: 1px solid rgba(255,255,255,0.12) !important;
            min-width: 220px !important;
            max-width: 220px !important;
        }}

        section[data-testid="stSidebar"] > div {{
            padding: 0 !important;
            background: transparent !important;
        }}

        div[data-testid="stDataFrame"] {{
            font-size: 12px !important;
        }}

        /* ── Buttons: gradient underline, no fill ───────────────────────── */
        .stButton > button {{
            background: transparent !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: 2px solid !important;
            border-image: linear-gradient(90deg, #7F77DD, #0EA5A4) 1 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            color: {T['text_primary']} !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            width: 100% !important;
            padding: 6px 4px !important;
            letter-spacing: 0.01em !important;
            white-space: nowrap !important;
            transition: color 0.15s ease, letter-spacing 0.15s ease, transform 0.1s ease !important;
        }}
        .stButton > button:hover {{
            background: transparent !important;
            border-image: linear-gradient(90deg, #7F77DD, #0EA5A4) 1 !important;
            color: #FFFFFF !important;
            letter-spacing: 0.04em !important;
            transform: translateY(-1px) !important;
        }}
        .stButton > button:active {{
            transform: scale(0.98) !important;
        }}
        .stButton > button:focus-visible {{
            outline: 2px solid rgba(127,119,221,0.6) !important;
            outline-offset: 3px !important;
        }}

        /* ── File uploader: glass drop zone ─────────────────────────────── */
        section[data-testid="stFileUploaderDropzone"] {{
            background: rgba(255,255,255,0.06) !important;
            border: 1px dashed rgba(255,255,255,0.22) !important;
            border-radius: 14px !important;
        }}
        section[data-testid="stFileUploaderDropzone"] * {{
            color: {T['text_primary']} !important;
        }}
        section[data-testid="stFileUploaderDropzone"] button {{
            background: rgba(99,102,241,0.2) !important;
            color: #E0E7FF !important;
            border: 1px solid rgba(165,180,252,0.4) !important;
            border-radius: 10px !important;
        }}

        /* ── Number inputs: underline style ─────────────────────────────── */
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {{
            background: transparent !important;
            border: none !important;
            border-bottom: 1px solid rgba(255,255,255,0.18) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            transition: border-bottom-color 0.15s ease !important;
        }}
        div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {{
            border-bottom: 1px solid !important;
            border-image: linear-gradient(90deg, #7F77DD, #0EA5A4) 1 !important;
            box-shadow: none !important;
        }}
        div[data-testid="stNumberInput"] input {{
            color: {T['text_primary']} !important;
            background: transparent !important;
        }}
        div[data-testid="stNumberInput"] input::placeholder {{
            color: {T['text_ghost']} !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: rgba(255,255,255,0.6) !important;
            transition: color 0.12s ease !important;
        }}
        div[data-testid="stNumberInput"] button:hover {{
            background: transparent !important;
            color: {T['text_primary']} !important;
        }}

        /* ── Sidebar number inputs: same underline, higher specificity ───── */
        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"] {{
            background: transparent !important;
            border: none !important;
            border-bottom: 1px solid rgba(255,255,255,0.18) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {{
            border-bottom: 1px solid !important;
            border-image: linear-gradient(90deg, #7F77DD, #0EA5A4) 1 !important;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {{
            background: transparent !important;
            color: {T['text_primary']} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: rgba(255,255,255,0.6) !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:hover {{
            background: transparent !important;
            color: {T['text_primary']} !important;
        }}

        /* ── Sliders: thin aurora signal bar ────────────────────────────── */
        div[data-testid="stSlider"] {{
            padding-top: 2px !important;
            padding-bottom: 2px !important;
            margin-bottom: 6px !important;
        }}
        /* Hide floating thumb value ("153.00" duplicate) */
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {{
            display: none !important;
        }}
        /* Hide min/max end labels ("0.00 … 200.00") */
        div[data-testid="stSlider"] [data-testid="stTickBar"],
        div[data-testid="stSlider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] [data-testid="stTickBarMax"] {{
            display: none !important;
        }}
        /* Track line — empty portion */
        div[data-baseweb="slider"] > div > div {{
            background: rgba(255,255,255,0.12) !important;
            height: 3px !important;
            border-radius: 3px !important;
        }}
        /* Filled portion — default teal, overridden per-signal below */
        div[data-baseweb="slider"] > div > div > div:first-child {{
            background: {T['success_accent']} !important;
            height: 3px !important;
            border-radius: 3px !important;
        }}
        /* Thumb — small, light circle */
        div[data-baseweb="slider"] [role="slider"] {{
            width: 12px !important;
            height: 12px !important;
            border-radius: 50% !important;
            background: {T['text_primary']} !important;
            border: 1.5px solid {T['success_accent']} !important;
            box-shadow: none !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            outline: none !important;
        }}
        div[data-baseweb="slider"] [role="slider"]:focus-visible {{
            box-shadow: 0 0 0 3px rgba(255,255,255,0.15) !important;
        }}
        /* ── Per-signal fill colors ───────────────────────────────────── */
        /* teal — usage, login */
        .st-key-sig_monthly_usage_hours div[data-baseweb="slider"] > div > div > div:first-child,
        .st-key-sig_login_frequency    div[data-baseweb="slider"] > div > div > div:first-child {{
            background: {T['success_accent']} !important;
        }}
        /* coral — last_login, payment */
        .st-key-sig_last_login_days  div[data-baseweb="slider"] > div > div > div:first-child,
        .st-key-sig_payment_failures div[data-baseweb="slider"] > div > div > div:first-child {{
            background: {T['danger_accent']} !important;
        }}
        .st-key-sig_last_login_days  div[data-baseweb="slider"] [role="slider"],
        .st-key-sig_payment_failures div[data-baseweb="slider"] [role="slider"] {{
            border-color: {T['danger_accent']} !important;
        }}
        /* amber — tickets */
        .st-key-sig_support_tickets div[data-baseweb="slider"] > div > div > div:first-child {{
            background: {T['warning_accent']} !important;
        }}
        .st-key-sig_support_tickets div[data-baseweb="slider"] [role="slider"] {{
            border-color: {T['warning_accent']} !important;
        }}
        /* Drag tooltip */
        div[data-baseweb="tooltip"] div {{
            background: {T['bg_base']} !important;
            color: {T['text_primary']} !important;
            border: 1px solid {T['glass_border']} !important;
            border-radius: 6px !important;
            font-size: 11px !important;
        }}

        /* ── Selectbox: underline style ──────────────────────────────────── */
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background: transparent !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: 1px solid rgba(255,255,255,0.18) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            color: {T['text_primary']} !important;
        }}
        div[data-testid="stSelectbox"]:focus-within [data-baseweb="select"] > div {{
            border-bottom: 1px solid !important;
            border-image: linear-gradient(90deg, #7F77DD, #0EA5A4) 1 !important;
            box-shadow: none !important;
        }}
        div[data-testid="stSelectbox"] [data-baseweb="select"] [data-value] {{
            color: {T['text_primary']} !important;
        }}

        /* ── Download button: same underline treatment ───────────────────── */
        .stDownloadButton > button {{
            background: transparent !important;
            border-top: none !important;
            border-left: none !important;
            border-right: none !important;
            border-bottom: 2px solid !important;
            border-image: linear-gradient(90deg, #7F77DD, #0EA5A4) 1 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            color: {T['text_primary']} !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            width: 100% !important;
            padding: 6px 4px !important;
            transition: color 0.15s ease, letter-spacing 0.15s ease, transform 0.1s ease !important;
        }}
        .stDownloadButton > button:hover {{
            background: transparent !important;
            color: #FFFFFF !important;
            letter-spacing: 0.04em !important;
            transform: translateY(-1px) !important;
        }}
        .stDownloadButton > button:active {{
            transform: scale(0.98) !important;
        }}
        .stDownloadButton > button:focus-visible {{
            outline: 2px solid rgba(127,119,221,0.6) !important;
            outline-offset: 3px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Helper functions for UI rendering
def section_label(text: str, icon_class: str | None = None) -> None:
    T = TOKENS
    icon_html = f'<i class="ti {icon_class}" style="font-size:12px;"></i> ' if icon_class else ""
    st.markdown(
        f'<div style="font-size:10px;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:{T["text_tertiary"]};margin-bottom:10px;">{icon_html}{text}</div>',
        unsafe_allow_html=True,
    )


def render_signal_row(label: str, value: float, max_val: float, color: str, unit: str) -> None:
    T = TOKENS
    pct = min(100, int((value / max_val) * 100)) if max_val > 0 else 0
    st.markdown(
        f'<div style="margin-bottom:10px;">'
        f'<div style="font-size:11px;color:{T["text_secondary"]};margin-bottom:4px;">{label}</div>'
        f'<div style="height:3px;background:{T["border_faint"]};border-radius:2px;overflow:hidden;">'
        f'<div style="height:3px;border-radius:2px;background:{color};width:{pct}%;'
        f'box-shadow:0 0 6px {color};"></div>'
        f'</div>'
        f'<div style="font-size:12px;font-weight:500;color:{T["text_primary"]};margin-top:4px;">'
        f'{value:.0f} {unit}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def build_recommendations(account: dict[str, float | str], proba: float) -> list[dict]:
    recs: list[dict] = []
    pf   = float(account["payment_failures"])
    lld  = float(account["last_login_days"])
    lf   = float(account["login_frequency"])
    uh   = float(account["monthly_usage_hours"])
    st_  = float(account["support_tickets"])
    plan = str(account["subscription_plan"])

    if pf >= 1:
        recs.append({"priority": 1, "icon": "ti-credit-card",
            "text": f"Recover billing — {int(pf)} failed payment(s). Trigger a card-update "
                    f"email and have finance confirm the payment method this week."})
    if lld >= 30:
        recs.append({"priority": 2, "icon": "ti-zzz",
            "text": f"Re-engage a dormant account — no login in {int(lld)} days. Send a "
                    f"win-back / what's-new campaign and book a CSM check-in."})
    elif lld >= 14:
        recs.append({"priority": 3, "icon": "ti-clock",
            "text": f"Watch engagement — last login {int(lld)} days ago. Send a value nudge "
                    f"before it goes cold."})
    if lf <= 5 or uh <= 15:
        recs.append({"priority": 3, "icon": "ti-rocket",
            "text": f"Lift adoption — only {int(lf)} logins and {uh:.0f} usage hrs. Invite to "
                    f"onboarding/training and surface unused features."})
    if st_ >= 3:
        recs.append({"priority": 2, "icon": "ti-lifebuoy",
            "text": f"De-risk support — {int(st_)} open tickets signal friction. Have a support "
                    f"lead proactively close the loop and confirm resolution."})

    if proba > 0.75 and not recs:
        next_plan = {"starter": "growth", "growth": "business", "business": "enterprise"}.get(plan, "a higher tier")
        recs.append({"priority": 2, "icon": "ti-trending-up",
            "text": f"Account is healthy — strong renewal signal. Explore an upsell to "
                    f"{next_plan} and ask for a testimonial / referral."})

    if not recs:
        recs.append({"priority": 3, "icon": "ti-check",
            "text": "Stable account — maintain regular touchpoints and monitor for change."})

    recs.sort(key=lambda r: r["priority"])
    return recs[:3]


def render_result(segment: str, probability: float, account: dict[str, float | str]) -> None:
    T = TOKENS
    if segment == "High Churn Risk":
        pill_bg        = T["danger_fill"]
        pill_color     = T["danger_text"]
        pill_border    = T["danger_pill"]
        prob_color     = T["danger_accent"]
        advisory_color = T["danger_accent"]
        advisory_text  = "User is at high risk of non-renewal. Immediate retention action recommended."
    elif segment == "Moderate Risk":
        pill_bg        = T["warning_fill"]
        pill_color     = T["warning_text"]
        pill_border    = T["warning_pill"]
        prob_color     = T["warning_accent"]
        advisory_color = T["warning_accent"]
        advisory_text  = "Mixed renewal signal. Trigger targeted engagement before the next billing cycle."
    else:  # High Renewal Probability
        pill_bg        = T["success_fill"]
        pill_color     = T["success_text"]
        pill_border    = T["success_pill"]
        prob_color     = T["success_accent"]
        advisory_color = T["success_accent"]
        advisory_text  = "Account shows strong renewal intent. Mark for expansion or upsell opportunity."

    prob_pct = int(probability * 100)

    # KPI strip: three equal cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            glass_card(
                f'<div style="font-size:10px;color:{T["text_tertiary"]};margin-bottom:8px;'
                f'letter-spacing:0.05em;text-transform:uppercase;">Renewal probability</div>'
                f'<div style="font-size:40px;line-height:1;font-weight:500;color:{prob_color};'
                f'letter-spacing:-1px;">{prob_pct}%</div>',
                pad="14px 16px",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            glass_card(
                f'<div style="font-size:10px;color:{T["text_tertiary"]};margin-bottom:8px;'
                f'letter-spacing:0.05em;text-transform:uppercase;">Risk segment</div>'
                f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:11px;'
                f'font-weight:500;padding:5px 12px;border-radius:20px;'
                f'background:{pill_bg};color:{pill_color};border:1px solid {pill_border};'
                f'letter-spacing:0.02em;">'
                f'<span style="width:5px;height:5px;border-radius:50%;background:{prob_color};'
                f'box-shadow:0 0 5px {prob_color};display:inline-block;flex-shrink:0;"></span>'
                f'{segment}'
                f'</span>',
                pad="14px 16px",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        recs = build_recommendations(account, probability)
        recs_html = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;">'
            f'<i class="ti {r["icon"]}" style="font-size:13px;color:{T["text_secondary"]};'
            f'flex-shrink:0;margin-top:1px;"></i>'
            f'<span style="font-size:11px;color:{T["text_secondary"]};line-height:1.5;">'
            f'{html.escape(r["text"])}</span>'
            f'</div>'
            for r in recs
        )
        st.markdown(
            glass_card(
                f'<div style="font-size:10px;color:{T["text_tertiary"]};margin-bottom:10px;'
                f'letter-spacing:0.05em;text-transform:uppercase;">Recommended action</div>'
                + recs_html,
                pad="14px 16px",
            ),
            unsafe_allow_html=True,
        )

    # Thin accent stripe with glow
    st.markdown(
        f'<div style="height:2px;background:{T["border_faint"]};border-radius:2px;'
        f'margin:14px 0;overflow:hidden;">'
        f'<div style="height:2px;border-radius:2px;width:{prob_pct}%;'
        f'background:{prob_color};box-shadow:0 0 8px {prob_color};"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Advisory — glass card with segment-coloured left border
    st.markdown(
        f'<div style="background:{T["glass_bg"]};'
        f'border:1px solid {T["glass_border"]};'
        f'border-left:3px solid {advisory_color};'
        f'border-radius:0 {T["glass_radius"]} {T["glass_radius"]} 0;'
        f'padding:10px 14px;line-height:1.6;">'
        f'<div style="font-size:11px;color:{T["text_secondary"]};">{advisory_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def style_dataframe(frame: pd.DataFrame):
    T = TOKENS
    return (
        frame.style.set_properties(
            **{
                "background-color": "transparent",
                "color": T["text_primary"],
                "border-color": T["border_dimmer"],
                "font-size": "12px",
                "padding": "6px 8px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", T["surface_mid"]),
                        ("color", T["text_secondary"]),
                        ("border-color", T["border_dimmer"]),
                        ("font-size", "11px"),
                        ("font-weight", "500"),
                        ("letter-spacing", "0.04em"),
                        ("text-transform", "uppercase"),
                        ("padding", "6px 8px"),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(even) td",
                    "props": [("background-color", T["surface_dim"])],
                },
                {
                    "selector": "tbody tr:nth-child(odd) td",
                    "props": [("background-color", "transparent")],
                },
            ]
        )
        .format(precision=3)
    )


def format_feature_name(feature: str) -> str:
    return feature.replace("_", " ").title()


def make_snapshot_frame(account: dict[str, float | str]) -> pd.DataFrame:
    rows = [
        {"Account Signal": format_feature_name(feature), "Value": value}
        for feature, value in account.items()
    ]
    return pd.DataFrame(rows)


def make_shap_frame(explanation: dict[str, Any]) -> pd.DataFrame:
    factors = pd.DataFrame(explanation["top_factors"])
    if factors.empty:
        return factors
    factors = factors.copy()
    factors["feature"] = factors["feature"].map(format_feature_name)
    return factors.rename(
        columns={
            "feature": "Feature",
            "feature_value": "Feature Value",
            "shap_value": "SHAP Value",
        }
    )


def render_shap_chart(shap_frame: pd.DataFrame) -> None:
    T = TOKENS
    if shap_frame.empty:
        st.markdown(
            f'<div style="font-size:11px;color:{T["text_secondary"]};">'
            f'No prediction drivers are available for this forecast.</div>',
            unsafe_allow_html=True,
        )
        return

    chart_frame = shap_frame.sort_values("SHAP Value", key=lambda s: s.abs())
    colors = [
        T["danger_accent"] if v < 0 else T["success_accent"]
        for v in chart_frame["SHAP Value"].tolist()
    ]

    fig = go.Figure(
        go.Bar(
            x=chart_frame["SHAP Value"],
            y=chart_frame["Feature"],
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=max(180, len(chart_frame) * 34),
        xaxis=dict(
            showgrid=True,
            gridcolor=T["border_subtle"],
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor=T["chart_zeroline"],
            zerolinewidth=1,
            tickfont=dict(color=T["text_chart"], size=10),
            title=None,
        ),
        yaxis=dict(
            tickfont=dict(color=T["text_chart"], size=10),
            title=None,
        ),
        showlegend=False,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# Sidebar rendering
def render_sidebar(reference_data: pd.DataFrame) -> dict[str, float | str]:
    T = TOKENS
    # Brand bar
    st.sidebar.markdown(
        f'<div style="padding:14px 16px;border-bottom:1px solid {T["glass_border"]};'
        f'display:flex;align-items:center;gap:8px;">'
        f'<div style="width:9px;height:9px;border-radius:50%;'
        f'background:linear-gradient(135deg,{T["brand_violet"]},{T["blob_violet"]});'
        f'flex-shrink:0;box-shadow:0 0 8px {T["brand_glow"]};"></div>'
        f'<div>'
        f'<div style="font-size:13px;font-weight:500;color:{T["text_primary"]};'
        f'letter-spacing:-0.2px;">Renewal Workbench</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Account signals section
    st.sidebar.markdown('<div style="padding:14px 16px;">', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div style="font-size:10px;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:{T["text_tertiary"]};margin-bottom:10px;">'
        f'Account signals</div>',
        unsafe_allow_html=True,
    )

    def _signal_label(label: str, val: float, unit: str, color: str) -> None:
        st.sidebar.markdown(
            f'<div style="font-size:11px;color:{T["text_secondary"]};margin-bottom:1px;'
            f'display:flex;justify-content:space-between;">'
            f'<span>{label}</span>'
            f'<span style="color:{color};font-weight:500;">{int(val)} {unit}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _signal_label("Monthly usage hours", float(st.session_state["monthly_usage_hours"]),
                  "hrs", T["success_accent"])
    with st.sidebar:
        with st.container(key="sig_monthly_usage_hours"):
            st.slider("monthly_usage_hours", min_value=0.0, max_value=200.0, step=1.0,
                      key="monthly_usage_hours", label_visibility="collapsed")

    _signal_label("Login frequency", float(st.session_state["login_frequency"]),
                  "sessions", T["success_accent"])
    with st.sidebar:
        with st.container(key="sig_login_frequency"):
            st.slider("login_frequency", min_value=0.0, max_value=50.0, step=1.0,
                      key="login_frequency", label_visibility="collapsed")

    _signal_label("Last login days", float(st.session_state["last_login_days"]),
                  "days", T["danger_accent"])
    with st.sidebar:
        with st.container(key="sig_last_login_days"):
            st.slider("last_login_days", min_value=0.0, max_value=90.0, step=1.0,
                      key="last_login_days", label_visibility="collapsed")

    _signal_label("Support tickets", float(st.session_state["support_tickets"]),
                  "tickets", T["warning_accent"])
    with st.sidebar:
        with st.container(key="sig_support_tickets"):
            st.slider("support_tickets", min_value=0.0, max_value=10.0, step=1.0,
                      key="support_tickets", label_visibility="collapsed")

    _signal_label("Payment failures", float(st.session_state["payment_failures"]),
                  "events", T["danger_accent"])
    with st.sidebar:
        with st.container(key="sig_payment_failures"):
            st.slider("payment_failures", min_value=0.0, max_value=5.0, step=1.0,
                      key="payment_failures", label_visibility="collapsed")

    # Plan selector
    st.sidebar.selectbox(
        "subscription_plan",
        options=PLAN_OPTIONS,
        key="subscription_plan",
        label_visibility="collapsed",
    )
    # Divider
    st.sidebar.markdown(
        f'<div style="height:1px;background:{T["border_faint"]};margin:0 16px;"></div>',
        unsafe_allow_html=True,
    )

    # Quick presets section
    st.sidebar.markdown('<div style="padding:0 16px;">', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div style="font-size:10px;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:{T["text_tertiary"]};margin-bottom:10px;">'
        f'Quick presets</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("High risk", key="high_risk_btn", use_container_width=True,
                  on_click=apply_scenario, args=(HIGH_RISK_SCENARIO,))
    with col2:
        st.button("High engage", key="high_eng_btn", use_container_width=True,
                  on_click=apply_scenario, args=(HIGH_ENGAGEMENT_SCENARIO,))

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    return get_account_input()


# Main content rendering
def render_header() -> None:
    T = TOKENS
    st.markdown(
        f'<div style="margin:-0px -24px 0 -24px;padding:14px 24px;'
        f'border-bottom:1px solid {T["glass_border"]};'
        f'display:flex;align-items:center;justify-content:space-between;'
        f'background:{T["header_bg"]};">'
        f'<div>'
        f'<div style="font-size:15px;font-weight:600;color:{T["text_primary"]};'
        f'letter-spacing:-0.3px;">Subscription Renewal Workbench</div>'
        f'<div style="font-size:11px;color:{T["text_secondary"]};margin-top:2px;'
        f'letter-spacing:0.01em;">'
        f'Customer success · lifecycle marketing · revenue operations'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:5px;'
        f'background:{T["success_fill"]};color:{T["success_text"]};'
        f'font-size:10px;font-weight:500;letter-spacing:0.02em;'
        f'padding:4px 10px;border-radius:20px;border:1px solid {T["success_border"]};">'
        f'<span style="width:6px;height:6px;border-radius:50%;'
        f'background:{T["success_accent"]};display:inline-block;'
        f'box-shadow:0 0 7px {T["success_glow_strong"]};flex-shrink:0;"></span>'
        f'Model live'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_prediction_result(
    prediction: dict[str, Any],
    account: dict[str, float | str],
) -> None:
    renewal_probability = float(prediction["renewal_probability"])
    risk_profile = get_risk_profile(renewal_probability)
    render_result(risk_profile["label"], renewal_probability, account)


def render_single_forecast(
    predictor: RenewalPredictor,
    account: dict[str, float | str],
) -> None:
    T = TOKENS
    # Action bar: label + hint on left, button pinned right
    bar_left, bar_right = st.columns([3, 1])
    with bar_left:
        st.markdown(
            f'<div style="font-size:14px;font-weight:600;color:{T["text_primary"]};'
            f'letter-spacing:-0.2px;">Single account forecast</div>'
            f'<div style="font-size:11px;color:{T["text_secondary"]};margin-top:2px;">'
            f'Adjust signals in the sidebar, then run a prediction.</div>',
            unsafe_allow_html=True,
        )
    with bar_right:
        if st.button("Predict renewal", use_container_width=True):
            st.session_state["latest_account"] = account.copy()
            st.session_state["latest_prediction"] = predictor.predict_one(account)
            st.session_state["latest_explanation"] = predictor.explain(account, top_k=8)

    if "latest_prediction" in st.session_state and "latest_account" in st.session_state:
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        render_prediction_result(
            st.session_state["latest_prediction"],
            st.session_state["latest_account"],
        )
        render_snapshot_and_drivers()
    else:
        st.markdown(
            f'<div style="font-size:11px;color:{T["text_secondary"]};margin-top:28px;'
            f'text-align:center;padding:32px 0;">'
            f'Adjust signals in the sidebar, then run a prediction.</div>',
            unsafe_allow_html=True,
        )


def render_batch_forecasting(predictor: RenewalPredictor) -> None:
    section_label("Batch forecasting")

    upload = st.file_uploader("Drop CSV here", type="csv", label_visibility="collapsed")

    if upload is not None:
        batch_df = pd.read_csv(upload)
        predictions = predictor.predict(batch_df)
        result_df = pd.concat(
            [batch_df.reset_index(drop=True), pd.DataFrame(predictions)],
            axis=1,
        )
        st.dataframe(style_dataframe(result_df), hide_index=True, use_container_width=True)
        render_batch_summary(predictions)
        st.download_button(
            label="Download predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="subscription_renewal_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )



def render_batch_summary(predictions: list[dict[str, float | int | str]]) -> None:
    T = TOKENS
    counts = {
        "High Churn Risk": 0,
        "Moderate Risk": 0,
        "High Renewal Probability": 0,
    }
    for prediction in predictions:
        probability = float(prediction["renewal_probability"])
        counts[get_risk_profile(probability)["label"]] += 1

    st.markdown(
        f'<div style="font-size:11px;color:{T["text_secondary"]};'
        f'margin-top:10px;padding:8px 0;border-top:1px solid {T["border_dimmer"]};">'
        f'{len(predictions)} accounts scored — '
        f'<span style="color:{T["danger_accent"]};font-weight:500;">'
        f'{counts["High Churn Risk"]} High Risk</span>'
        f' · '
        f'<span style="color:{T["warning_accent"]};font-weight:500;">'
        f'{counts["Moderate Risk"]} Moderate</span>'
        f' · '
        f'<span style="color:{T["success_accent"]};font-weight:500;">'
        f'{counts["High Renewal Probability"]} Likely Renewal</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_snapshot_and_drivers() -> None:
    T = TOKENS
    if "latest_account" not in st.session_state or "latest_explanation" not in st.session_state:
        return

    st.markdown(
        f'<div style="height:1px;background:{T["glass_border"]};'
        f'margin:20px 0;"></div>',
        unsafe_allow_html=True,
    )

    snapshot_frame = make_snapshot_frame(st.session_state["latest_account"])
    shap_frame = make_shap_frame(st.session_state["latest_explanation"])

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        section_label("Account snapshot")

        table_html = (
            f'<table style="width:auto;margin:0 auto;border-collapse:collapse;font-size:12px;">'
            f'<tr>'
            f'<th style="text-align:center;white-space:nowrap;color:{T["text_chart"]};font-weight:500;'
            f'padding:4px 0;border-bottom:1px solid {T["border_faint"]};font-size:10px;'
            f'letter-spacing:0.05em;text-transform:uppercase;">Signal</th>'
            f'<th style="text-align:center;white-space:nowrap;color:{T["text_chart"]};font-weight:500;'
            f'padding:4px 0;border-bottom:1px solid {T["border_faint"]};font-size:10px;'
            f'letter-spacing:0.05em;text-transform:uppercase;">Value</th>'
            f'</tr>'
        )

        for _, row in snapshot_frame.iterrows():
            signal_name = row["Account Signal"]
            value = row["Value"]
            value_str = str(value) if isinstance(value, str) else f"{value:.1f}"
            table_html += (
                f'<tr>'
                f'<td style="text-align:center;white-space:nowrap;color:{T["text_secondary"]};'
                f'padding:6px 0;border-bottom:1px solid {T["border_dimmer"]};">'
                f'{html.escape(signal_name)}</td>'
                f'<td style="text-align:center;white-space:nowrap;color:{T["text_secondary"]};'
                f'padding:6px 0;border-bottom:1px solid {T["border_dimmer"]};">'
                f'{html.escape(value_str)}</td>'
                f'</tr>'
            )

        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

    with right_col:
        section_label("Prediction drivers")

        backend = st.session_state.get("latest_explanation", {}).get(
            "explanation_backend", ""
        )
        backend_note = (
            "LinearExplainer on LogisticRegression — exact SHAP in log-odds space"
            if "shap" in backend
            else "feature importance fallback — values are approximate"
        )
        st.markdown(
            f'<div style="font-size:10px;color:{T["text_secondary"]};margin-bottom:10px;">'
            f'  Top SHAP contributions — mint pushes toward renewal, rose toward churn'
            f'  <span style="color:{T["text_tertiary"]};"> · {backend_note}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        render_shap_chart(shap_frame)


def main() -> None:
    st.set_page_config(page_title="Subscription Renewal Workbench", layout="wide")
    inject_global_css()

    predictor = load_predictor()
    reference_data = load_reference_data()
    ensure_session_state(reference_data)
    account = render_sidebar(reference_data)

    render_header()

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["Single forecast", "Batch scoring"])
    with tab_single:
        render_single_forecast(predictor, account)
    with tab_batch:
        render_batch_forecasting(predictor)


if __name__ == "__main__":
    main()
