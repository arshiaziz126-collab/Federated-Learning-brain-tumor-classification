"""NeuroFed AI dashboard.  Run with:  streamlit run app.py

The app only *displays* results. Training happens in federated_train.py.
"""
import streamlit as st

import config
from ui.theme import html, inject_css
from views import about, analytics, comparison, diagnose, home, network, privacy_page

st.set_page_config(page_title="NeuroFed AI", page_icon=str(config.ROOT / "assets" / "favicon.png"), layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
st.logo(
    str(config.ROOT / "assets" / "logo.svg"),
    size="large",
    icon_image=str(config.ROOT / "assets" / "logo_icon.svg"),
)

pages = [
    st.Page(home.render, title="Home", url_path="home", default=True),
    st.Page(diagnose.render, title="Diagnose", url_path="diagnose"),
    st.Page(network.render, title="Federated Network", url_path="federated-network"),
    st.Page(analytics.render, title="Federated Analytics", url_path="federated-analytics"),
    st.Page(privacy_page.render, title="Privacy & Security", url_path="privacy-security"),
    st.Page(comparison.render, title="Centralized vs Federated", url_path="centralized-vs-federated"),
    st.Page(about.render, title="About", url_path="about"),
]
nav = st.navigation(pages)

with st.sidebar:
    html("""<div class="nf-side-note"><b>Smarter Networks. Healthier Tomorrow.</b><br>
    Research prototype. Not a clinical diagnostic tool.</div>""")

nav.run()
