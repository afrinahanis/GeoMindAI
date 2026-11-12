"""Workspace presentation; analysis remains in the original tool modules."""
import streamlit as st

BLUE, TEAL, INK, MUTED, LINE = "#00877e", "#00877e", "#142332", "#697684", "#e2e7eb"
LOGO = '''<svg width="34" height="30" viewBox="0 0 44 36" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2 9c5-5 9-5 14 0s9 5 14 0 8-4 12-1M2 19c5-5 9-5 14 0s9 5 14 0 8-4 12-1M2 29c5-5 9-5 14 0s9 5 14 0 8-4 12-1" stroke="#00877e" stroke-width="3" stroke-linecap="round"/></svg>'''
CSS = '''<style>
:root { --ink:#142332; --muted:#697684; --line:#e2e7eb; --teal:#00877e; }
#MainMenu, footer { display:none; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding:2.5rem 1.25rem 1rem; max-width:2000px; }
[data-testid="stAppViewContainer"] { background:#fff; }
section[data-testid="stSidebar"] { background:#f8fafb; border-right:1px solid var(--line); width:200px !important; min-width:180px !important; }
section[data-testid="stSidebar"] .stButton button { width:100%; justify-content:flex-start; border:0; background:transparent; text-align:left; }
section[data-testid="stSidebar"] .stButton button[kind="primary"] { background:#def3ef; color:#006e66; }
.gm-brand { display:flex; gap:10px; align-items:center; margin:.3rem 0 1.7rem; }
.gm-brand b { font-size:1.15rem; color:var(--ink); }
.gm-brand span { display:block; font-size:.7rem; color:var(--muted); margin-top:3px; }
.gm-h1 { font-size:1.5rem; line-height:1.25; font-weight:700; color:var(--ink); }
.gm-muted { color:var(--muted); font-size:.82rem; margin-top:4px; }
.gm-recent { color:var(--muted); font-size:.72rem; letter-spacing:.06em; margin:.5rem 0; }
h3 { font-size:1.08rem !important; padding:.2rem 0 .55rem !important; }
h4 { font-size:1rem !important; }
.stButton button, .stDownloadButton button { border-radius:8px; border:1px solid #d8e0e5; background:#fff; color:var(--ink); min-height:2.35rem; }
.stButton button:hover { border-color:var(--teal); color:var(--teal); }
.stButton button[kind="primary"] { background:var(--teal); color:white; border-color:var(--teal); }
.st-key-nav_settings { margin-top:3rem; }
.st-key-project_data, .st-key-main_map, .st-key-chat_panel { background:#fff; border-radius:10px; padding:.85rem; }
.st-key-project_data { border-right:1px solid var(--line); }
.st-key-chat_panel { border-left:1px solid var(--line); }
.st-key-conversation { scrollbar-width:thin; }
.st-key-cp_new button, .st-key-cp_expand button { border:0; background:transparent; padding:.25rem; min-height:2rem; }
.st-key-project_data [data-testid="stTextInput"] input { font-size:.82rem; }
.st-key-chat_panel [data-testid="stSelectbox"] { font-size:.82rem; }
.st-key-nav_ws button, .st-key-nav_hist button { padding:.6rem .8rem; }
.st-key-dslist [data-testid="stVerticalBlock"] { gap:.35rem; }
.st-key-dslist .stButton button { border:0; background:transparent; padding:.2rem .15rem; justify-content:flex-start; text-align:left; min-height:1.9rem; }
.st-key-dslist .stButton button p { font-size:.8rem; text-align:left; overflow-wrap:anywhere; }
.st-key-dslist [data-testid="stExpander"] { border:0; border-radius:6px; }
.st-key-dslist [data-testid="stExpander"] summary { background:#f4f6f8; border-radius:6px; }
.st-key-dslist [data-testid="stExpanderDetails"] { padding:.4rem .1rem; }
[data-testid="stChatMessage"] { background:transparent; padding:.7rem .25rem; gap:.5rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction:row-reverse; margin-left:12%; background:#edf4f3; border-radius:12px; }
[data-testid="stChatMessage"] p { font-size:.88rem; line-height:1.55; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { width:24px; height:24px; }
[data-testid="stChatInput"] { border-radius:10px; border-color:#d8e0e5; }
[data-testid="stExpander"] { border-color:var(--line); border-radius:8px; }
.gm-map-empty { height:560px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; background:#f8fafb; border:1px dashed #d8e0e5; border-radius:8px; color:var(--muted); }
.gm-map-empty b { font-size:1.2rem; color:var(--ink); }
.gm-map-empty p { font-size:.9rem; margin-top:.6rem; }
.gm-note { padding:.8rem; background:#f2f8f6; border-radius:8px; }
@media (max-width:1100px) { section[data-testid="stSidebar"] { width:190px !important; min-width:180px !important; } .block-container { padding-left:.7rem; padding-right:.7rem; } }
@media (max-width:800px) { [data-testid="stHorizontalBlock"] { flex-wrap:wrap; } [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:100%; } .gm-map-empty { height:350px; } }
</style>'''


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.markdown(f'<div class="gm-brand">{LOGO}<div><b>GeoMindAI</b></div></div>', unsafe_allow_html=True)
