"""Shared GeoMindAI typography, controls and layout styling.

Replace geomind_ui/theme.py. Existing page layout and Python callbacks are kept.
BFA palette: blue #006CB5, green #00A68C, dark green #007C73.
"""
import streamlit as st

BLUE = "#006CB5"
TEAL = "#00A68C"
INK = "#193247"
MUTED = "#647687"
LINE = "#E2E8EE"

# GeoMindAI's existing wave mark; this is not the corporate BFA logo.
LOGO = '''<svg width="34" height="30" viewBox="0 0 44 36" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<path d="M2 9c5-5 9-5 14 0s9 5 14 0 8-4 12-1" stroke="#00A68C" stroke-width="3" stroke-linecap="round"/>
<path d="M2 19c5-5 9-5 14 0s9 5 14 0 8-4 12-1" stroke="#007C73" stroke-width="3" stroke-linecap="round"/>
<path d="M2 29c5-5 9-5 14 0s9 5 14 0 8-4 12-1" stroke="#006CB5" stroke-width="3" stroke-linecap="round"/>
</svg>'''

CSS = '''
<style>
:root {
    --blue: #006CB5;
    --teal: #00A68C;
    --ink: #193247;
    --muted: #647687;
    --line: #E2E8EE;
    --gm-body: .875rem;
    --gm-small: .8125rem;
    --gm-caption: .75rem;
    --gm-section: 1rem;
    --gm-title: 1.5rem;
    --gm-icon: 1.125rem;
    --gm-control: 2.25rem;
    --gm-radius: .5rem;
}
/* Streamlit retains control of the sidebar, header and collapse mechanism. */
#MainMenu, footer { display: none; }
[data-testid="stHeader"] { background: rgba(255,255,255,.96); }
.stApp { color: var(--ink); background: #FFFFFF; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #FFFFFF; }
.block-container { max-width: 1920px; padding: 3.5rem 1.5rem 1.5rem; }
[data-testid="stAppViewContainer"] :is(p, li, label, input, textarea),
[data-testid="stPopoverBody"] :is(p, li, label, input) {
    font-size: var(--gm-body);
    line-height: 1.55;
}
:is(h1, h2, h3, h4, h5, h6) { color: var(--ink); font-weight: 600; letter-spacing: -.015em; }
h1, .gm-h1 { font-size: var(--gm-title) !important; line-height: 1.3; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: var(--gm-section) !important; }
h4, h5, h6 { font-size: .9375rem !important; }
.gm-h1 { margin: 0; font-weight: 650; color: var(--ink); }
.gm-muted { color: var(--muted); font-size: var(--gm-small); line-height: 1.5; margin-top: .25rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    font-size: var(--gm-caption); color: var(--muted); line-height: 1.5;
}
/* Size icons without replacing the Material icon font. */
:is(button, summary) [data-testid="stIconMaterial"] {
    font-size: var(--gm-icon) !important;
    width: var(--gm-icon); height: var(--gm-icon);
    line-height: 1 !important; flex-shrink: 0;
}
:is(.stButton, .stDownloadButton, [data-testid="stPopover"]) button svg,
[data-testid="stExpander"] summary svg {
    width: var(--gm-icon); height: var(--gm-icon); flex-shrink: 0;
}
/* Shared control dimensions. */
:is(.stButton, .stDownloadButton) > button {
    min-height: var(--gm-control); padding: .4rem .75rem;
    border: 1px solid #CFDAE3; border-radius: var(--gm-radius);
    background: #FFFFFF; color: var(--ink); box-shadow: none;
    gap: .5rem; font-size: var(--gm-body); font-weight: 500;
}
:is(.stButton, .stDownloadButton) > button p {
    font-size: var(--gm-body); font-weight: 500; line-height: 1.4;
}
:is(.stButton, .stDownloadButton) > button:hover:not(:disabled) {
    border-color: var(--blue); color: var(--blue); background: #F1F7FC;
}
.stButton > button[kind="primary"] {
    color: white; background: var(--blue); border-color: var(--blue);
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    color: white; background: #005B99; border-color: #005B99;
}
button:disabled { opacity: .55; }
button:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--blue) !important; outline-offset: 2px;
}
[data-testid="stTextInputRootElement"], [data-baseweb="select"] > div {
    border-radius: var(--gm-radius); background: #FFFFFF; min-height: var(--gm-control);
}
[data-testid="stWidgetLabel"] p, [data-baseweb="select"] { font-size: var(--gm-body); }
[data-testid="stExpander"] { border-color: var(--line); border-radius: var(--gm-radius); }
[data-testid="stDataFrame"] { border-radius: var(--gm-radius); }

/* Sidebar: primary navigation and a quieter conversation list. */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
.gm-brand { display: flex; align-items: center; gap: .625rem; margin: .25rem 0 1.25rem; }
.gm-brand > svg { width: 30px; height: 28px; flex-shrink: 0; }
.gm-brand b { display: block; font-size: 1.0625rem; color: var(--ink); line-height: 1.35; }
.gm-brand span { display: block; font-size: var(--gm-caption); color: var(--muted); margin-top: .125rem; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; justify-content: flex-start; text-align: left;
    border: 0; background: transparent; color: #526575;
}
section[data-testid="stSidebar"] .stButton > button p { text-align: left; }
section[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) { background: #F3F6F9; color: var(--ink); }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #EAF3FA; color: var(--blue); }
section[data-testid="stSidebar"] .st-key-nav_new button { background: var(--blue); color: white; justify-content: center; }
section[data-testid="stSidebar"] .st-key-nav_new button:hover { background: #005B99; color: white; }
.st-key-sidebar_nav [data-testid="stVerticalBlock"],
.st-key-sidebar_recent [data-testid="stVerticalBlock"] { gap: .25rem; }
.gm-recent { color: var(--muted); font-size: var(--gm-caption); font-weight: 600; margin: 1rem .75rem .25rem; }
section[data-testid="stSidebar"] .st-key-sidebar_recent .stButton > button p {
    display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    font-size: var(--gm-small); font-weight: 400;
}
section[data-testid="stSidebar"] .st-key-sidebar_recent button[kind="primary"] { background: #F0F5F9; color: var(--ink); }
.st-key-sidebar_footer { margin-top: 1.25rem; padding-top: .75rem; border-top: 1px solid var(--line); }
.st-key-nav_settings { position: static; margin-top: 0; }

/* Panel headers share one height; icon columns do not stretch on expansion. */
:is(.st-key-project_data, .st-key-main_map, .st-key-chat_panel) {
    padding: 1rem; background: white; border: 1px solid var(--line); border-radius: .625rem;
}
:is(.st-key-project_header, .st-key-map_header, .st-key-chat_header) {
    min-height: var(--gm-control);
}
:is(.st-key-project_header, .st-key-map_header, .st-key-chat_header) h3 {
    padding: 0 !important; margin: 0 !important; line-height: var(--gm-control);
}
:is(.st-key-project_header, .st-key-sidebar_recent) [data-testid="stHorizontalBlock"],
.st-key-chat_header [data-testid="stHorizontalBlock"] {
    display: grid; grid-template-columns: minmax(0, 1fr) var(--gm-control);
    align-items: center; gap: .375rem;
}
.st-key-chat_header [data-testid="stHorizontalBlock"] {
    grid-template-columns: minmax(0, 1fr) var(--gm-control) var(--gm-control);
}
:is(.st-key-project_header, .st-key-sidebar_recent, .st-key-chat_header)
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    width: 100% !important; min-width: 0 !important;
}
/* Ellipsis menus: no box or dropdown arrow, with keyboard focus retained. */
:is(.st-key-project_header, .st-key-sidebar_recent) [data-testid="stPopover"] button,
.st-key-chat_header .stButton > button {
    display: flex; align-items: center; justify-content: center;
    width: 100%; min-width: 0; min-height: var(--gm-control);
    padding: .375rem; gap: 0; border: 0 !important;
    background: transparent !important; color: var(--muted); box-shadow: none !important;
}
:is(.st-key-project_header, .st-key-sidebar_recent) [data-testid="stPopover"] button:hover,
.st-key-chat_header .stButton > button:hover { color: var(--blue); }
:is(.st-key-project_header, .st-key-sidebar_recent) [data-testid="stPopover"] button:focus:not(:focus-visible) { outline: none; }
:is(.st-key-project_header, .st-key-sidebar_recent) [data-testid="stPopover"] button svg,
:is(.st-key-project_header, .st-key-sidebar_recent) [data-testid="stPopover"] button > div:has(> svg),
:is(.st-key-project_header, .st-key-sidebar_recent) [data-testid="stPopoverButtonIcon"] { display: none !important; }

/* Dataset list. */
.st-key-dslist [data-testid="stVerticalBlock"] { gap: .375rem; }
.st-key-dslist [data-testid="stExpander"] { border: 0; }
.st-key-dslist [data-testid="stExpander"] summary { background: #F4F7FA; border-radius: var(--gm-radius); }
.st-key-dslist [data-testid="stExpanderDetails"] { padding: .5rem .125rem; }
.st-key-dslist .stButton > button {
    width: 100%; justify-content: flex-start; text-align: left;
    background: transparent; border-color: transparent; padding: .375rem .25rem;
}
.st-key-dslist .stButton > button p { font-size: var(--gm-small); font-weight: 400; text-align: left; overflow-wrap: anywhere; }

/* Chat scope is a compact menu, separate from the conversation. */
.st-key-chat_context [data-testid="stPopover"] { width: fit-content; max-width: 100%; }
.st-key-chat_context [data-testid="stPopover"] button {
    min-height: var(--gm-control); padding: .375rem .625rem;
    border: 0; border-radius: var(--gm-radius); background: #F2F6FA;
    color: #526575; box-shadow: none; gap: .375rem;
}
.st-key-chat_context [data-testid="stPopover"] button p { font-size: var(--gm-small); font-weight: 500; }
.st-key-chat_context [data-testid="stPopover"] button:hover { background: #EAF3FA; color: var(--blue); }
.st-key-conversation { scrollbar-width: thin; border-top: 1px solid var(--line); padding-top: .75rem; }
[data-testid="stChatMessage"] { background: transparent; padding: .75rem .25rem; gap: .625rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse; background: #EAF3FA;
    border-radius: .625rem; margin-left: 8%; padding: .75rem;
}
[data-testid="stChatMessage"] :is(p, li) { font-size: var(--gm-body); line-height: 1.65; }
[data-testid="stChatMessage"] :is(h1, h2, h3, h4) { font-size: var(--gm-section) !important; }
:is([data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]) {
    width: 28px; height: 28px; flex-shrink: 0;
}
[data-testid="stChatInput"] { border: 1px solid #CFDAE3; border-radius: .625rem; background: white; }
[data-testid="stChatInput"] textarea { font-size: var(--gm-body); }
[data-testid="stChatInputSubmitButton"] { color: var(--blue); }
/* Expanded chat: one centred reading column, with the same toolbar. */
.st-key-expanded_chat { max-width: 1040px; width: 100%; margin-inline: auto; }
.st-key-expanded_chat .st-key-chat_panel { border: 0; background: transparent; padding: 0; }
.st-key-expanded_chat .st-key-chat_header h3 { font-size: var(--gm-title) !important; }
.st-key-expanded_chat .st-key-conversation {
    height: clamp(280px, calc(100dvh - 280px), 900px) !important;
}

/* Existing supplementary components use the same scale and spacing. */
.gm-chip, .gm-badge { display: inline-block; padding: .2rem .5rem; border-radius: .375rem; background: #EAF6F2; color: #007C73; font-size: var(--gm-caption); font-weight: 600; margin-left: .5rem; }
.gm-card { padding: 1rem; background: white; border: 1px solid var(--line); border-radius: .625rem; }
.gm-note, .gm-warn { padding: .75rem; border-radius: var(--gm-radius); font-size: var(--gm-body); line-height: 1.55; }
.gm-note { background: #F1F7FB; border-left: 3px solid var(--blue); color: #31536D; }
.gm-warn { background: #FFF8E8; border-left: 3px solid #D8A13A; color: #755314; }
.gm-sec { font-size: var(--gm-body); font-weight: 600; margin: .25rem 0 .5rem; }
.gm-hero { text-align: center; margin: 2rem 0 1.5rem; }
.gm-hero p { color: var(--muted); }
.gm-map-empty { min-height: 350px; height: 560px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1rem; text-align: center; background: #F8FAFC; border: 1px dashed #D6E0E8; border-radius: .625rem; color: var(--muted); }
.gm-map-empty b { color: var(--ink); font-size: var(--gm-section); font-weight: 600; }
.gm-map-empty p { margin-top: .5rem; }
table.gm-tbl { width: 100%; border-collapse: collapse; font-size: var(--gm-body); }
table.gm-tbl td { padding: .5rem; border-bottom: 1px solid var(--line); vertical-align: top; overflow-wrap: anywhere; }
table.gm-tbl td:first-child { width: 40%; color: var(--muted); }
@media (max-width: 1100px) {
    .st-key-workspace_columns > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"],
    .st-key-workspace_columns > [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-workspace_columns > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-workspace_columns > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100%; flex: 1 1 100%; }
}
@media (max-width: 800px) {
    .block-container { padding: 3.5rem .75rem 1rem; }
    :is(.st-key-project_data, .st-key-main_map, .st-key-chat_panel) { padding: .75rem; }
    .gm-map-empty { height: 350px; }
}
</style>
'''


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.markdown(
        f'<div class="gm-brand">{LOGO}<div><b>GeoMindAI</b>'
        '<span>Beicip-Franlab Asia</span></div></div>',
        unsafe_allow_html=True,
    )
