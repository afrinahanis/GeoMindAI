"""Conversation display and map links for outputs returned by the existing tools."""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import catalog as C
from . import store

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


def _pair(value):
    try:
        if len(value) == 2:
            xy = [float(v) for v in value]
            if all(math.isfinite(v) for v in xy):
                return xy
    except (TypeError, ValueError):
        pass
    return None


def _trace_xy(pair):
    """Use the seismic geometry; line numbers are not map coordinates."""
    import ingest as I
    try:
        for name in I.load_cards().get("volumes", {}):
            with np.load(I.CACHE_DIR / f"geom_{name}.npz") as g:
                hits = np.flatnonzero((g["il"] == pair[0]) & (g["xl"] == pair[1]))
                if len(hits):
                    j = hits[0]
                    return _pair([g["x"][j], g["y"][j]])
    except (OSError, KeyError, ValueError):
        pass
    return None


def map_locations(trace):
    """Read locations from tool results only, never from generated prose."""
    points, seen = [], set()
    wells = {i["label"].lower(): i for i in C.build()["items"] if i.get("geom") == "well"}

    def visit(row, label, depth=0):
        if depth > 6 or len(points) >= 30:
            return
        if isinstance(row, list):
            for index, value in enumerate(row[:30]):
                visit(value, f"{label} {index + 1}", depth + 1)
            return
        if not isinstance(row, dict) or row.get("error"):
            return
        xy = _pair(row.get("xy")) or _pair(row.get("crest_xy")) or _pair(row.get("centre_xy"))
        if xy is None:
            xy = _pair([row.get("x"), row.get("y")])
        if xy is None:
            pair = _pair(row.get("inline_xline")) or _pair(row.get("crest_inline_xline")) or _pair(row.get("centre_inline_xline"))
            pair = pair or _pair([row.get("inline"), row.get("xline")])
            if pair:
                xy = _trace_xy(pair)
        well = wells.get(str(row.get("well", "")).lower())
        if xy is None and well and well.get("bbox"):
            xy, label = well["bbox"][:2], well["label"]
        if xy and tuple(xy) not in seen:
            seen.add(tuple(xy))
            if row.get("rank") is not None and "prospect" in label.lower():
                label = f"Prospect {row['rank']}"
            points.append({"label": str(row.get("label") or label), "x": xy[0], "y": xy[1]})
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                visit(value, key.replace("_", " ").title(), depth + 1)

    for entry in trace:
        visit(entry.get("result"), entry.get("tool", "Result").replace("_", " ").title())
    return points


def ask_agent(q, selected_labels):
    _, chat = active()
    text = q
    if selected_labels:
        text += "\n\n(Focus on these datasets if relevant: " + ", ".join(selected_labels) + ".)"
    chat["msgs"].append({"role": "user", "content": q, "t": time.time()})
    if chat["title"] == "New chat":
        chat["title"] = q[:46] + ("…" if len(q) > 46 else "")
    store.save_chats(chats())
    st.session_state["_run"] = text


def finish_run(selected_labels):
    import agent
    text = st.session_state.pop("_run")
    _, chat = active()
    history = [{"role": m["role"], "content": m["content"]} for m in chat["msgs"][-9:-1]]
    with st.spinner("Reviewing project evidence…"):
        try:
            out = agent.run(history, text)
        except Exception as e:
            out = {"answer": f"Unable to answer: {e}", "trace": [], "images": []}
    trace = out.get("trace", [])
    targets = map_locations(trace)
    # These three outputs only repeat existing workspace geometry. Derived maps
    # and sections contain new analysis and remain visible in the conversation.
    context_tools = {"interpreted_map", "interpreted_faults", "interpreted_facies"}
    image_info = {str(t["result"]["image_path"]): t for t in trace
                  if isinstance(t.get("result"), dict) and t["result"].get("image_path")}
    images = list(dict.fromkeys(list(out.get("images", [])) + list(image_info)))
    images = [p for p in images if image_info.get(str(p), {}).get("tool") not in context_tools]
    titles = {p: image_info.get(str(p), {}).get("tool", "Analysis output").replace("_", " ").title() for p in images}
    chat["msgs"].append({"role": "assistant", "content": out.get("answer", ""), "trace": trace,
                         "images": images, "image_titles": titles, "targets": targets, "t": time.time()})
    st.session_state.update(map_targets=targets, map_fit=False, focus=None)
    st.session_state["map_revision"] += 1
    store.save_chats(chats())
    st.rerun()


def restore_map(chat):
    st.session_state.update(map_targets=[], focus=None, map_fit=False)
    for m in reversed(chat["msgs"]):
        if m.get("role") == "assistant":
            st.session_state["map_targets"] = m.get("targets", [])
            break
    st.session_state["map_revision"] += 1


@st.dialog("Analysis output", width="large")
def expand_image(path, title):
    st.subheader(title)
    st.image(path, width="stretch")


def render(m, key):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("targets"):
            if st.button("Locate on map", icon=":material/location_on:", key=f"locate_{key}"):
                st.session_state.update(map_targets=m["targets"], focus=None, map_fit=False, page="workspace")
                st.session_state["map_revision"] += 1
                st.rerun()
        for index, image in enumerate(m.get("images", [])):
            path = Path(image)
            if not path.is_file():
                continue
            title = m.get("image_titles", {}).get(image, "Analysis output")
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.image(str(path), width="stretch")
                left, right = st.columns(2)
                if left.button("Expand", icon=":material/open_in_full:", key=f"expand_{key}_{index}", width="stretch"):
                    expand_image(str(path), title)
                right.download_button("Download", path.read_bytes(), file_name=path.name,
                                      mime="image/png", key=f"download_{key}_{index}", width="stretch")
        if m.get("trace"):
            with st.expander("Evidence & sources"):
                for t in m["trace"]:
                    st.markdown(f"**{t['round']}. {t['tool']}**")
                    st.json(t["result"], expanded=False)


def input_box(placeholder, selected_labels, disabled=False):
    q = st.chat_input(placeholder, key=f"ci_{st.session_state.get('page')}", disabled=disabled)
    q = q or st.session_state.pop("pending", None)
    if q and not disabled:
        ask_agent(q, selected_labels)
        st.rerun()
