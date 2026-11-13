"""Chat execution, conversation persistence and message rendering."""
import time

import pandas as pd
import streamlit as st

from . import store

EXPLORE = ["What data do we have?", "What evidence is missing?", "Give me an overview of everything loaded: raw and interpreted data, and caveats.",
           "How many faults did the interpreter pick, and does automatic detection agree?", "Do the interpreted channels agree with the seismic sand probability?"]
WELL = ["Which wells show the highest sand probability, and do their GR logs agree?", "Compare wells no-1 and o-1.", "Which wells have flagged gas zones and how thick are they?",
        "How reliable is the time-depth tie for each well?"]
SEISMIC = ["Identify whether the trap is stratigraphic, structural or combined in the window 2000-2300 ms. Explain step by step.",
           "Where is the probable gas potential location? Use far minus near stack (high negative), structurally high and high sand probability.",
           "Rank the prospects and give inline, xline and TWT for each.",
           "Define the depositional environment from the sand probability and explain the metrics.",
           "What are the key geological risks to investigate next?"]
HERO = {"Explore data": EXPLORE, "Investigate a well": WELL, "Examine seismic": SEISMIC}
QUICK = ["What data do we have?", "What evidence is missing?"]


def chats():
    if "chats" not in st.session_state:
        st.session_state.chats = store.load_chats()
    return st.session_state.chats


def active():
    cs = chats()
    cid = st.session_state.get("active_chat")
    if cid not in cs:
        cid = store.new_chat(cs)
        st.session_state.active_chat = cid
    return cid, cs[cid]


def scope_labels(selected_labels):
    return ["All project data"] + ([f"Selected data ({len(selected_labels)})"] if selected_labels else [])


def ask_agent(q, selected_labels):
    """Same call as the notebook's ask(): agent.run(history, question). Selected scope only adds a one-line focus hint."""
    import agent
    cid, chat = active()
    history = [{"role": m["role"], "content": m["content"]} for m in chat["msgs"][-8:]]
    text = q
    if st.session_state.get("scope", "All project data").startswith("Selected") and selected_labels:
        text = f"{q}\n\n(Focus on these datasets if relevant: {', '.join(selected_labels)}.)"
    st.session_state.pop("pending", None)
    chat["msgs"].append({"role": "user", "content": q, "t": time.time()})
    if chat["title"] == "New chat":
        chat["title"] = (q[:46] + "...") if len(q) > 48 else q
    store.save_chats(chats())
    st.session_state["_run"] = text


def finish_run(selected_labels):
    """Executes the queued question (shows a spinner inside the conversation)."""
    import agent
    text = st.session_state.pop("_run")
    cid, chat = active()
    history = [{"role": m["role"], "content": m["content"]} for m in chat["msgs"][-9:-1]]
    with st.chat_message("assistant"):
        with st.spinner("Reasoning over the data ..."):
            try:
                out = agent.run(history, text)
            except Exception as e:
                out = {"answer": f"Something went wrong while answering: `{e}`", "trace": [], "images": []}
    chat["msgs"].append({"role": "assistant", "content": out["answer"], "trace": out["trace"], "images": out["images"], "suggestions": out.get("suggestions", []), "t": time.time()})
    store.save_chats(chats())
    st.rerun()


def evidence_table(trace):
    rows = []
    for t in trace:
        r = t["result"] if isinstance(t["result"], dict) else {}
        status = "error" if r.get("error") else "completed"
        rows.append({"Tool": t["tool"], "Status": status, "Arguments": ", ".join(f"{k}={v}" for k, v in (t.get("args") or {}).items()) or "-",
                     "Output": "map/figure" if r.get("image_path") else ("table" if r else "-")})
    return pd.DataFrame(rows)


def render(m, key):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for p in m.get("images", []) or []:
            try:
                st.image(p)
            except Exception:
                pass
        if m.get("trace"):
            n = len(m["trace"])
            st.caption(f"Recorded evidence · {n} tool result(s)")
            with st.expander("Evidence and sources"):
                st.dataframe(evidence_table(m["trace"]), hide_index=True, width="stretch")
                for t in m["trace"]:
                    st.markdown(f"**{t['round']}. `{t['tool']}`** `{t['args']}`")
                    st.json(t["result"], expanded=False)
        render_followups(m, key)


def render_followups(m, key):
    """Offer suggestions only for the latest completed answer in the active chat."""
    if m.get("role") != "assistant" or not m.get("suggestions"):
        return
    _, chat = active()
    if not chat["msgs"] or chat["msgs"][-1] is not m:
        return
    if "_run" in st.session_state or st.session_state.get("pending"):
        return
    with st.container(key=f"followups_{key}"):
        st.caption("Continue exploring")
        for index, question in enumerate(m["suggestions"][:3]):
            if not isinstance(question, str):
                continue
            if st.button(question, key=f"followup_{key}_{index}", width="stretch"):
                st.session_state["pending"] = question
                st.rerun()

def input_box(placeholder, selected_labels):
    q = st.chat_input(placeholder, key=f"ci_{st.session_state.get('page')}")
    q = q or st.session_state.pop("pending", None)
    if q:
        ask_agent(q, selected_labels)
        st.rerun()


