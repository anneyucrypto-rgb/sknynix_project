"""Single-page H1--H3 research findings dashboard."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.findings import (
    RUNS,
    h2_uses_thirty_minute_artifacts,
    load_run,
    model_conclusions,
    shared_diagnostics,
)


st.set_page_config(page_title="Findings · SKHY Price Discovery", page_icon="↔", layout="wide")

COLORS = {
    "ink": "#153039", "rust": "#b55332", "blue": "#397a8b",
    "gold": "#c69945", "muted": "#68787c", "paper": "#f5f3ee",
    "grid": "#d9d3c6",
}

st.markdown(
    """
    <style>
    .stApp {background:#f5f3ee; color:#172126;}
    [data-testid="stHeader"] {background:rgba(245,243,238,.94);}
    h1,h2,h3 {font-family:Georgia,serif; color:#12323a; letter-spacing:-.025em;}
    h2 {font-size:2.15rem !important; margin-top:3.5rem !important; padding-top:1.25rem; border-top:1px solid #d9d3c6;}
    .findings-hero {padding:2.2rem 0 1.6rem;}
    .findings-hero h1 {font-family:Georgia,serif; font-size:clamp(2.8rem,6vw,5.4rem); line-height:.98; margin:.4rem 0 1rem;}
    .eyebrow {color:#a34b2a; font-size:.76rem; font-weight:750; letter-spacing:.15em; text-transform:uppercase;}
    .deck {font-size:1.08rem; color:#5e6e72; max-width:760px; line-height:1.65;}
    .status-card {height:100%; min-height:154px; background:#fffdf8; border-top:4px solid #123f49; padding:1.15rem 1.2rem; box-shadow:0 1px 0 rgba(20,35,40,.08);}
    .status-card.support {border-color:#397a8b}.status-card.warn {border-color:#b55332}
    .status-label {font-size:.72rem; letter-spacing:.11em; text-transform:uppercase; color:#6e7b7e; font-weight:700;}
    .status-value {font-family:Georgia,serif; font-size:1.55rem; color:#173840; margin:.38rem 0;}
    .status-detail {font-size:.88rem; line-height:1.45; color:#657377;}
    .section-note {font-size:1rem; color:#627176; line-height:1.65; max-width:880px; margin-bottom:1.2rem;}
    .conclusion {background:#123f49; color:#f8f5ec; padding:1.5rem 1.65rem; border-left:6px solid #c69945; margin:1.2rem 0 2rem; line-height:1.65;}
    .conclusion b {color:#f3ca76;}
    div[data-testid="stMetric"] {background:#fffdf8; border:1px solid #e1dccf; padding:.75rem 1rem;}
    .stDataFrame {border:1px solid #ded8ca;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load(label: str):
    return load_run(label)


@st.cache_data(show_spinner="Running shared diagnostics…")
def _diagnostics(label: str) -> pd.DataFrame:
    return shared_diagnostics(load_run(label))


def _card(title: str, status: str, detail: str, css: str = "") -> str:
    return f"""<div class="status-card {css}"><div class="status-label">{title}</div>
    <div class="status-value">{status}</div><div class="status-detail">{detail}</div></div>"""


def _style_axis(axis):
    axis.set_facecolor(COLORS["paper"])
    axis.grid(axis="y", color=COLORS["grid"], linewidth=.7, alpha=.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(COLORS["grid"])
    axis.tick_params(colors=COLORS["muted"], labelsize=9)


def _h1_figure(frame: pd.DataFrame):
    data = frame.dropna(subset=["date_us", "a_d", "ect"]).copy()
    data["date_us"] = pd.to_datetime(data["date_us"])
    data["skhy"] = np.exp(data["a_d"])
    data["equilibrium"] = np.exp(data["a_d"] - data["ect"])
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    figure.patch.set_facecolor(COLORS["paper"])
    axes[0].plot(data["date_us"], data["skhy"], color=COLORS["ink"], linewidth=2, label="SKHY close")
    axes[0].plot(data["date_us"], data["equilibrium"], color=COLORS["rust"], linewidth=1.8, label="Estimated equilibrium")
    axes[0].set_ylabel("Price level (USD)")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].axhline(0, color=COLORS["ink"], linewidth=.8)
    axes[1].fill_between(data["date_us"], 0, data["ect"], where=data["ect"].ge(0), color=COLORS["rust"], alpha=.28)
    axes[1].fill_between(data["date_us"], 0, data["ect"], where=data["ect"].lt(0), color=COLORS["blue"], alpha=.28)
    axes[1].plot(data["date_us"], data["ect"], color=COLORS["ink"], linewidth=1)
    axes[1].set_ylabel("ECT (log gap)")
    for axis in axes:
        _style_axis(axis)
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
    figure.tight_layout()
    return figure


def _price_figure(frame: pd.DataFrame):
    data = frame.dropna(subset=["date_us", "adr_close", "kr_reference_close", "usdkrw_us_close"]).copy()
    data["date_us"] = pd.to_datetime(data["date_us"])
    data["korean_usd"] = .1 * data["kr_reference_close"] / data["usdkrw_us_close"]
    data["skhy_index"] = 100 * data["adr_close"] / data["adr_close"].iloc[0]
    data["korean_index"] = 100 * data["korean_usd"] / data["korean_usd"].iloc[0]
    figure, axis = plt.subplots(figsize=(11, 4.2))
    figure.patch.set_facecolor(COLORS["paper"])
    axis.plot(data["date_us"], data["skhy_index"], color=COLORS["ink"], linewidth=2.1, label="SKHY")
    axis.plot(data["date_us"], data["korean_index"], color=COLORS["gold"], linewidth=2, label="Korean share · FX adjusted")
    axis.set_ylabel("Indexed price (first observation = 100)")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(axis)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    figure.tight_layout()
    return figure


def _h2_response_figure(models: pd.DataFrame):
    horizon = np.arange(30, 391, 30)
    figure, axis = plt.subplots(figsize=(11, 4.5))
    figure.patch.set_facecolor(COLORS["paper"])
    choices = models.loc[models["anchor"].eq("liquid")].drop_duplicates("reference")
    for _, row in choices.iterrows():
        coefficient = float(row["lambda_30"])
        std_error = float(row["lambda_30_std_error"])
        bars = horizon / 30
        response = (1 + coefficient) ** bars
        lower = (1 + coefficient - 1.96 * std_error) ** bars
        upper = (1 + coefficient + 1.96 * std_error) ** bars
        color = COLORS["rust"] if row["reference"] == "static" else COLORS["blue"]
        axis.plot(horizon, response, linewidth=2, color=color, label=f"{str(row['reference']).title()} reference")
        axis.fill_between(horizon, np.minimum(lower, upper), np.maximum(lower, upper), color=color, alpha=.15)
    axis.axhline(1, color=COLORS["ink"], linewidth=.8, linestyle="--")
    axis.set_xlabel("Minutes after 09:30 ET")
    axis.set_ylabel("Implied fraction of initial gap remaining")
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def _direction_figure(h2: pd.DataFrame, h3: pd.DataFrame):
    dynamic = h2.loc[h2["reference"].eq("dynamic") & h2["anchor"].eq("liquid")].iloc[0]
    korean = h3.iloc[0]
    estimates = np.array([float(dynamic["lambda_30"]), float(korean["lambda_k"])])
    errors = np.array([float(dynamic["lambda_30_std_error"]), float(korean["lambda_k_std_error"])]) * 1.96
    labels = ["Korean reference → SKHY\nH2, 30-minute", "SKHY → Korean share\nH3, next session"]
    figure, axis = plt.subplots(figsize=(9, 4.1))
    figure.patch.set_facecolor(COLORS["paper"])
    axis.axhline(0, color=COLORS["ink"], linewidth=.8)
    axis.errorbar([0, 1], estimates, yerr=errors, fmt="o", markersize=9, capsize=6, color=COLORS["rust"], ecolor=COLORS["muted"], linewidth=2)
    axis.set_xticks([0, 1], labels)
    axis.set_ylabel("Adjustment coefficient with 95% CI")
    _style_axis(axis)
    figure.tight_layout()
    return figure


st.markdown(
    """<div class="findings-hero"><div class="eyebrow">Page 2 · Empirical results</div>
    <h1>Research Findings</h1><div class="deck">One equilibrium, two trading venues, three adjustment windows. Select a research run and follow the evidence from long-run parity to the next Korean session.</div></div>""",
    unsafe_allow_html=True,
)

run_label = st.selectbox("Select research run · View model conclusions", list(RUNS), index=0)
data = _load(run_label)
conclusions = model_conclusions(data)

columns = st.columns(3)
columns[0].markdown(_card("H1 · Cointegration", conclusions["h1"]["status"], conclusions["h1"]["detail"], "support"), unsafe_allow_html=True)
columns[1].markdown(_card("H2 · Convergence", conclusions["h2"]["status"], conclusions["h2"]["detail"], "warn"), unsafe_allow_html=True)
columns[2].markdown(_card("H3 · Directionality", conclusions["h3"]["status"], conclusions["h3"]["detail"], "support"), unsafe_allow_html=True)

st.header("H1 — Long-run cointegration")
st.markdown('<div class="section-note">ARDL bounds inference establishes whether SKHY and the FX-adjusted Korean share share a stable daily equilibrium. The lower panel is the estimated equilibrium error: positive values place SKHY above equilibrium.</div>', unsafe_allow_html=True)
h1_model, h1_bounds = data["h1_model"], data["h1_bounds"]
if isinstance(h1_model, pd.DataFrame) and not h1_model.empty:
    model = h1_model.iloc[0]
    bounds = h1_bounds.iloc[0] if isinstance(h1_bounds, pd.DataFrame) and not h1_bounds.empty else pd.Series(dtype=object)
    metrics = st.columns(4)
    metrics[0].metric("Long-run slope θ", f"{float(model['theta']):.3f}")
    metrics[1].metric("Bounds F", f"{float(bounds.get('f_statistic', math.nan)):.2f}")
    metrics[2].metric("Bootstrap p-value", f"{float(bounds.get('bootstrap_p_value', math.nan)):.3f}")
    metrics[3].metric("Residual persistence", f"{float(model['residual_persistence']):.3f}")
    st.pyplot(_h1_figure(data["h1_ect"]), clear_figure=True, width="stretch")
else:
    st.info("H1 has not been estimated for this run.")

st.subheader("SKHY and Korean security price dynamics")
st.markdown('<div class="section-note">Both securities are indexed to 100 for comparison. The Korean ordinary share is converted to the SKHY-equivalent USD basis using contemporaneous USD/KRW.</div>', unsafe_allow_html=True)
if isinstance(data["h3_panel"], pd.DataFrame) and not data["h3_panel"].empty:
    st.pyplot(_price_figure(data["h3_panel"]), clear_figure=True, width="stretch")

st.header("H2 — Error correction and time to convergence")
st.markdown('<div class="section-note">The 30-minute ECM begins at 09:30 ET. Values below one in the response chart imply a shrinking gap; confidence bands incorporate uncertainty in the estimated adjustment coefficient.</div>', unsafe_allow_html=True)
h2_models = data["h2_models"]
h2_ready = h2_uses_thirty_minute_artifacts(data)
if h2_ready:
    preferred = h2_models.loc[h2_models["reference"].eq("dynamic") & h2_models["anchor"].eq("liquid")].iloc[0]
    convergence = data["h2_convergence"]
    event = convergence.loc[convergence["reference"].eq("dynamic") & convergence["anchor"].eq("liquid") & convergence["duration"].eq("duration_clock_minutes")]
    event_probability = 1 - float(event.iloc[0]["no_convergence_rate"]) if not event.empty else math.nan
    metrics = st.columns(4)
    metrics[0].metric("Thirty-minute λ", f"{float(preferred['lambda_30']):.4f}")
    metrics[1].metric("Half-life", "Undefined" if pd.isna(preferred["half_life_minutes"]) else f"{float(preferred['half_life_minutes']):.1f} min")
    metrics[2].metric("90% convergence", "Undefined" if pd.isna(preferred["t90_minutes"]) else f"{float(preferred['t90_minutes']):.1f} min")
    metrics[3].metric("Observed by close", f"{100 * event_probability:.1f}%")
    st.pyplot(_h2_response_figure(h2_models), clear_figure=True, width="stretch")
    st.caption("Implied response paths are model-based, not realized intraday averages. The dynamic-reference coefficient is negative and significant, but its half-life extends well beyond one U.S. regular session.")
else:
    st.warning(
        "Current 30-minute H2 artifacts are unavailable. Legacy five-minute "
        "files are not relabelled as 30-minute estimates. Run "
        "`python -m src.hypotheses.h2` and redeploy the generated H2 artifacts."
    )

st.header("H3 — Directional price discovery")
st.markdown('<div class="section-note">H2 measures Korean-reference-to-SKHY adjustment during the U.S. session. H3 measures the reverse completion step: whether the next Korean session responds to the gap left at the U.S. close.</div>', unsafe_allow_html=True)
h3_model = data["h3_model"]
if isinstance(h3_model, pd.DataFrame) and not h3_model.empty:
    h3 = h3_model.iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Korean λK", f"{float(h3['lambda_k']):.3f}")
    metrics[1].metric("One-sided p-value", f"{float(h3['lambda_k_p_value_positive']):.4f}")
    metrics[2].metric("Target sessions", f"{int(h3['target_sessions'])}")
    metrics[3].metric("Adjusted R²", f"{float(h3['adjusted_r_squared']):.3f}")
    if h2_ready:
        st.pyplot(_direction_figure(h2_models, h3_model), clear_figure=True, width="stretch")
        st.caption("The two coefficients have different horizons and sign conventions: H2 30-minute convergence requires λ < 0; H3 Korean adjustment requires λK > 0.")
    else:
        st.info("The cross-direction coefficient chart will appear after H2 is rebuilt.")
else:
    st.info("H3 has not been estimated for this run.")

st.header("Shared statistical diagnostics")
st.markdown('<div class="section-note">A common battery checks serial correlation, heteroskedasticity, functional form, residual normality, conditional variance, and parameter stability. A p-value below 0.05 flags a possible violation rather than automatically invalidating the model.</div>', unsafe_allow_html=True)
diagnostics = _diagnostics(run_label)
if diagnostics.empty:
    st.info("Diagnostics are not available for this run.")
else:
    display = diagnostics.copy()
    display["Statistic"] = display["Statistic"].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    display["p-value"] = display["p-value"].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("CUSUM uses a Brownian-bridge approximation. CUSUMSQ is shown as a descriptive maximum cumulative-square deviation; no approximate p-value is imposed.")

st.header("Integrated conclusion and Markdown reports")
if h2_ready:
    st.markdown(
        """<div class="conclusion"><b>Integrated finding.</b> H1 identifies a statistically supported daily cross-market equilibrium. H2 detects significant but slow 30-minute correction toward the dynamic equilibrium: the estimated half-life is about 1,056 minutes, and only 9.7% of eligible sessions enter the band before the U.S. close. H3 also finds a positive next-Korean-session response: a 1% larger U.S.-close discrepancy is associated with an approximately 0.502% higher Korean return. Together, the evidence indicates that adjustment begins during U.S. trading but is rarely completed there, with the following Korean session providing an additional convergence channel.</div>""",
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "The integrated conclusion is withheld because this deployment does "
        "not contain the current 30-minute H2 artifacts."
    )

tabs = st.tabs(["H1 report", "H2 report", "H3 report"])
for tab, key in zip(tabs, ("h1_report", "h2_report", "h3_report"), strict=True):
    report = data[key]
    with tab:
        if isinstance(report, Path) and report.is_file():
            content = report.read_text(encoding="utf-8")
            st.download_button("Download Markdown", content, file_name=report.name, mime="text/markdown", key=f"download-{key}")
            with st.expander("Read full report", expanded=False):
                st.markdown(content)
        else:
            st.info("Markdown report is unavailable for this run.")
