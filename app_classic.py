"""
app.py - Streamlit chat UI.   Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import agent
import ingest
import tools

st.set_page_config(page_title="Exploration Copilot", page_icon="🪨", layout="wide")
st.title("🪨 Exploration Copilot")
st.caption("Reasons over raw seismic/well data and your interpreted faults, polygons and horizons. Drop new files in the folders and press Scan.")

EXAMPLES = [
    "Give me an overview of everything loaded: raw and interpreted data, and caveats.",
    "Based on the interpreted faults, horizon and channels, what is the most likely trap type? Explain step by step.",
    "Rank the prospects and give inline, xline and TWT for each.",
    "How many faults did the interpreter pick, and does automatic detection agree?",
    "Do the interpreted channels agree with the seismic sand probability?",
    "Which wells show the highest sand probability, and do their GR logs agree?",
    "Based on the seismic and well data, is this prospect more likely structural or stratigraphic?",
    "Why do you think that, step by step? What evidence is missing?",
    "Compare wells no-1 and o-1.",
    "What are the key geological risks to investigate next?",
]

# ---- sidebar
import os
import ingest_all
import interp_data

with st.sidebar:
    st.header("Data")
    st.caption(f"Raw: `{ingest.DATA_DIR}`  \nInterpreted: `{interp_data.interp_dir() or 'INTERP_DIR not set'}`")
    have_raw, have_int = ingest.cache_ready(), interp_data.interpreted_ready()
    if not (have_raw or have_int):
        st.warning("Nothing ingested yet.")
        if st.button("Run ingestion (raw + interpreted)", type="primary"):
            with st.spinner("Reading SEG-Y / LAS / shapefiles / horizon ..."):
                ingest_all.run_all()
            st.rerun()
        st.stop()
    c1, c2 = st.columns(2)
    if c1.button("Scan for new data", help="Ingest only new or changed files"):
        with st.spinner("Scanning ..."):
            r = ingest_all.refresh()
        ch = (r["interpreted"] or {}).get("changes", {})
        st.session_state["scan_msg"] = (f"Interpreted: {len(ch.get('new', []))} new, {len(ch.get('changed', []))} changed, "
                                        f"{len(ch.get('removed', []))} removed. Raw: {r['raw'].get('status')}")
        st.rerun()
    if c2.button("Full re-ingest"):
        with st.spinner("Re-ingesting everything ..."):
            ingest_all.run_all(force=True)
        st.rerun()
    if st.session_state.get("scan_msg"):
        st.info(st.session_state["scan_msg"])
    if have_raw:
        cards = ingest.load_cards()
        st.subheader(f"Raw wells ({len(cards['wells'])})")
        st.dataframe(pd.DataFrame(tools.list_wells())[["well", "type", "td_m", "time_depth_confidence"]], hide_index=True)
        st.subheader(f"Raw volumes ({len(cards['volumes'])})")
        for v in cards["volumes"].values():
            st.write(f"• **{v['label']}** ({v['kind']})")
        if cards["warnings"]:
            with st.expander(f"⚠️ {len(cards['warnings'])} raw-data warnings"):
                for w in cards["warnings"]:
                    st.write("-", w)
    else:
        st.info("No raw data loaded: only interpreted-data reasoning is available.")
    if have_int:
        inv = tools.list_datasets()["interpreted"]
        s = inv["summary"]
        st.subheader("Interpreted data")
        f = s.get("faults") or {}
        st.write(f"• **Faults**: {f.get('n_slices', 0)} slices, {f.get('n_features', 0)} traces")
        for k, v in s.get("polygon_classes", {}).items():
            st.write(f"• **{k}**: {v['n_slices']} slices, {v['n_features']} polygons")
        for h in s.get("horizons", []):
            st.write(f"• **Horizon** {h['name']}: {h['n_points']} pts")
        if inv["warnings"]:
            with st.expander(f"⚠️ {len(inv['warnings'])} interpreted-data warnings"):
                for w in inv["warnings"]:
                    st.write("-", w)
    else:
        st.info("No interpreted data loaded (set INTERP_DIR, then Scan).")
    st.subheader("Try asking")
    for q in EXAMPLES:
        if st.button(q, key=q):
            st.session_state["pending"] = q

# ---- chat
if "msgs" not in st.session_state:
    st.session_state.msgs = []


def show(m):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for p in m.get("images", []):
            st.image(p)
        if m.get("trace"):
            with st.expander(f"🔧 Tools used ({len(m['trace'])})"):
                for t in m["trace"]:
                    st.markdown(f"**{t['round']}. `{t['tool']}`** `{t['args']}`")
                    st.json(t["result"], expanded=False)


for m in st.session_state.msgs:
    show(m)

q = st.chat_input("Ask about the wells, seismic or prospect ...") or st.session_state.pop("pending", None)
if q:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[-8:]]
    st.session_state.msgs.append({"role": "user", "content": q})
    show(st.session_state.msgs[-1])
    with st.chat_message("assistant"):
        with st.spinner("Thinking and querying the data ..."):
            try:
                out = agent.run(history, q)
            except Exception as e:
                out = {"answer": f"⚠️ LLM/tool error: `{e}`", "trace": [], "images": []}
    m = {"role": "assistant", "content": out["answer"], "trace": out["trace"], "images": out["images"]}
    st.session_state.msgs.append(m)
    st.rerun()
