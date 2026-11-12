"""Look and feel: colours, logo and CSS. Pure presentation."""
import streamlit as st

BLUE, TEAL, INK, MUTED, LINE = "#2a5db0", "#2f8f83", "#1b2b3a", "#6b7c8d", "#dbe6ef"

LOGO = """<svg width="44" height="36" viewBox="0 0 44 36" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M2 9c5-5 9-5 14 0s9 5 14 0 8-4 12-1" stroke="#2f8f83" stroke-width="3.2" stroke-linecap="round"/>
<path d="M2 19c5-5 9-5 14 0s9 5 14 0 8-4 12-1" stroke="#3b7fb8" stroke-width="3.2" stroke-linecap="round"/>
<path d="M2 29c5-5 9-5 14 0s9 5 14 0 8-4 12-1" stroke="#2a5db0" stroke-width="3.2" stroke-linecap="round"/></svg>"""

CSS = f"""
<style>
:root {{ --blue:{BLUE}; --teal:{TEAL}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:1.2rem; padding-bottom:5rem; max-width:none; }}
section[data-testid="stSidebar"] {{ background:#eaf3fb; width:310px !important; min-width:310px !important; border-right:1px solid var(--line); }}
section[data-testid="stSidebar"] .block-container {{ padding-top:1.6rem; }}
.gm-brand {{ display:flex; gap:12px; align-items:center; margin-bottom:1.4rem; }}
.gm-brand b {{ font-size:1.05rem; color:var(--ink); display:block; line-height:1.1; }}
.gm-brand span {{ font-size:.78rem; color:var(--muted); }}
.gm-h1 {{ font-size:1.9rem; font-weight:800; color:var(--ink); margin:0; }}
.gm-chip {{ display:inline-block; padding:.18rem .7rem; border-radius:999px; background:#dcebf8; color:#2a5db0; font-size:.78rem; font-weight:600; margin-left:.7rem; vertical-align:middle; }}
.gm-badge {{ display:inline-block; padding:.12rem .6rem; border-radius:999px; background:#e3edf7; color:#43607c; font-size:.72rem; font-weight:600; margin-left:.5rem; }}
.gm-card {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; }}
.gm-note {{ background:#e8f2fb; color:#2a4a6d; border-radius:10px; padding:.8rem 1rem; font-size:.92rem; }}
.gm-warn {{ background:#fff8dc; color:#7a5d00; border-radius:10px; padding:.8rem 1rem; font-size:.9rem; margin:.5rem 0; }}
.gm-muted {{ color:var(--muted); font-size:.85rem; }}
.gm-recent {{ font-size:.72rem; letter-spacing:.06em; color:var(--muted); margin:.4rem 0 .4rem .2rem; }}
.gm-hero {{ text-align:center; margin-top:4.5rem; }}
.gm-hero h1 {{ font-size:2.3rem; font-weight:800; color:var(--ink); margin-bottom:.4rem; }}
.gm-hero p {{ color:var(--muted); }}
.gm-sec {{ font-weight:700; color:var(--ink); margin:.2rem 0 .5rem; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{ border-radius:12px; border:1px solid var(--line); background:#fff; color:var(--ink); font-weight:500; }}
.stButton > button:hover {{ border-color:var(--blue); color:var(--blue); }}
.stButton > button[kind="primary"] {{ background:var(--blue); border-color:var(--blue); color:#fff; font-weight:600; }}
.stButton > button[kind="primary"]:hover {{ background:#224f98; color:#fff; }}
section[data-testid="stSidebar"] .stButton > button {{ width:100%; justify-content:center; padding:.7rem 1rem; }}
.st-key-add_data button {{ background:var(--teal) !important; border-color:var(--teal) !important; color:#fff !important; font-weight:600; padding:.7rem 1.4rem; }}
.st-key-nav_settings {{ position:fixed; bottom:18px; left:22px; width:264px; }}
.st-key-nav_settings button {{ background:transparent !important; border:none !important; justify-content:flex-start !important; color:var(--ink) !important; }}

/* dataset list */
.st-key-dslist .stButton > button {{ border:none; background:transparent; padding:.4rem .6rem; font-size:.9rem; }}
.st-key-dslist .stButton > button > div {{ justify-content:flex-start !important; text-align:left; width:100%; }}
.st-key-dslist .stButton > button p {{ text-align:left; word-break:break-all; }}
.st-key-dslist .stButton > button[kind="primary"] {{ background:#dcebf8; color:var(--blue); }}

/* floating chat button */
.st-key-chat_fab {{ position:fixed; right:30px; bottom:26px; z-index:1000; width:auto; }}
.st-key-chat_fab button {{ width:62px; height:62px; border-radius:50% !important; background:var(--teal) !important; border:none !important; color:#fff !important; box-shadow:0 6px 18px rgba(30,70,90,.28); }}
.st-key-chat_fab button p {{ font-size:1.5rem; }}

/* chat panel + cards */
.st-key-chat_panel {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }}
.st-key-hero_cards .stButton > button {{ height:118px; border-radius:16px; font-size:1rem; font-weight:600; }}
.st-key-hero_cards .stButton > button[kind="primary"] {{ background:#dcebf8; color:var(--blue); border-color:var(--blue); }}
[data-testid="stChatMessage"] {{ background:transparent; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:12px; background:#fff; }}
table.gm-tbl {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
table.gm-tbl td {{ padding:.42rem .5rem; border-bottom:1px solid var(--line); vertical-align:top; word-break:break-word; }}
table.gm-tbl td:first-child {{ color:var(--muted); width:42%; }}
div[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:10px; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.markdown(f'<div class="gm-brand">{LOGO}<div><b>GeoMindAI</b><span>Explore. Connect. Explain.</span></div></div>', unsafe_allow_html=True)
