"""Canonical Streamlit application for cross-border price discovery."""

from pathlib import Path

import streamlit as st


def render_home() -> None:
    """Render the minimal landing page."""
    st.markdown(
        """
        <style>
        .stApp {background: #f5f3ee; color: #172126;}
        [data-testid="stHeader"] {background: rgba(245,243,238,.92);}
        .hero {padding: 5rem 0 3rem; max-width: 920px;}
        .eyebrow {color:#a34b2a; font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;}
        .hero h1 {font-family:Georgia,serif; font-size:clamp(3rem,7vw,6.5rem); line-height:.93; letter-spacing:-.055em; margin:.65rem 0 1.2rem; color:#112b32;}
        .hero p {font-size:1.2rem; line-height:1.7; color:#526267; max-width:720px;}
        @media(max-width:700px){.hero{padding-top:2rem}}
        </style>
        <div class="hero">
          <div class="eyebrow">SK hynix · United States ↔ Korea</div>
          <h1>Cross-border<br>price discovery.</h1>
          <p>Follow the equilibrium relationship from daily cointegration of Assets</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_app() -> None:
    """Configure navigation and run the selected Streamlit page."""
    st.set_page_config(
        page_title="SKHN US - Korean Price Discovery",
        page_icon="↔",
        layout="wide",
    )
    logo = Path(__file__).resolve().parent / "assets" / "cross_border_logo.svg"
    st.logo(str(logo))
    navigation = st.navigation(
        [
            st.Page(render_home, title="Home", icon=":material/home:", default=True),
            st.Page(
                "pages/2_Findings.py",
                title="SKHN US - Korean Price Discovery",
                icon=":material/swap_horiz:",
            ),
        ],
        position="sidebar",
    )
    navigation.run()


if __name__ == "__main__":
    run_app()
