"""
app.py - GeoMindAI web application (Streamlit).   Run:  streamlit run app.py

UI only. All analysis is done by the existing modules (agent.py, tools.py, ingest*.py, interp*.py), unchanged.
The previous single-page chat UI is kept as app_classic.py.
"""
import streamlit as st

st.set_page_config(page_title="GeoMindAI", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

from geomind_ui import theme, views  # noqa: E402

theme.inject()
views.init_state()
views.startup()
views.sidebar()
page = st.session_state["page"]
if page == "settings":
    views.settings_page()
elif page == "history":
    views.history()
else:
    views.workspace()
