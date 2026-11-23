"""Canonical Streamlit home page for SKHY cross-border price discovery."""

import streamlit as st


def render_home() -> None:
    """Render the landing page and its link to the combined findings page."""
    st.set_page_config(
        page_title="SKHY Cross-Border Price Discovery",
        page_icon="↔",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {background: #f5f3ee; color: #172126;}
        [data-testid="stHeader"] {background: rgba(245,243,238,.92);}
        .hero {padding: 5rem 0 3rem; max-width: 920px;}
        .eyebrow {color:#a34b2a; font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;}
        .hero h1 {font-family:Georgia,serif; font-size:clamp(3rem,7vw,6.5rem); line-height:.93; letter-spacing:-.055em; margin:.65rem 0 1.2rem; color:#112b32;}
        .hero p {font-size:1.2rem; line-height:1.7; color:#526267; max-width:720px;}
        .rule {height:1px; background:#d9d3c6; margin:1rem 0 2.25rem;}
        div[data-testid="stPageLink"] a {background:#123f49; color:white !important; padding:1rem 1.35rem; border-radius:4px; text-decoration:none; font-weight:700; width:fit-content;}
        div[data-testid="stPageLink"] a * {color:white !important;}
        div[data-testid="stPageLink"] a:hover {background:#a34b2a; color:white !important;}
        .research-strip {display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:#d9d3c6; margin-top:5rem; border:1px solid #d9d3c6;}
        .research-strip div {background:#f5f3ee; padding:1.4rem;}
        .research-strip b {display:block; color:#123f49; margin-bottom:.4rem;}
        @media(max-width:700px){.research-strip{grid-template-columns:1fr}.hero{padding-top:2rem}}
        </style>
        <div class="hero">
          <div class="eyebrow">SK hynix · United States ↔ Korea</div>
          <h1>Cross-border<br>price discovery.</h1>
          <p>Follow the equilibrium relationship from daily cointegration, through the U.S. trading session, to the next Korean market response.</p>
        </div>
        <div class="rule"></div>
        """,
        unsafe_allow_html=True,
    )

    st.page_link(
        "pages/2_Findings.py",
        label="Open research findings  →",
        icon="📈",
    )

    st.markdown(
        """
        <div class="research-strip">
          <div><b>H1 · Equilibrium</b>Is there a stable long-run relationship?</div>
          <div><b>H2 · Adjustment</b>Does SKHY correct after the U.S. open?</div>
          <div><b>H3 · Direction</b>Does Korea complete the convergence?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_home()
