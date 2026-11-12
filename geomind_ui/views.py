"""Pages: sidebar, workspace, chat history, settings, add-data dialog. UI only."""
import contextlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from . import catalog as C
from . import chat as CH
from . import mapview, store, theme


# ----------------------------------------------------------------- helpers
def S():
    return st.session_state["settings"]


def init_state():
    ss = st.session_state
    ss.setdefault("page", "workspace")
    ss.setdefault("chat_open", False)
    ss.setdefault("focus", None)
    ss.setdefault("sel_ids", [])
    ss.setdefault("scope", "All project data")
    ss.setdefault("hero", None)
    ss.setdefault("layers", ["survey", "horizon", "polygons", "faults", "wells"])


def goto(page, **kw):
    st.session_state["page"] = page
    for k, v in kw.items():
        st.session_state[k] = v


def count_files(path, cap=50000):
    n = 0
    try:
        for _, _, fs in os.walk(path):
            n += len(fs)
            if n >= cap:
                break
    except Exception:
        pass
    return n


def refresh_catalog():
    C._CACHE.clear()


def do_scan(force=False, raw=None):
    import ingest_all
    try:
        if force:
            ingest_all.run_all(force=True)
            msg = "Full re-ingest finished."
        else:
            r = ingest_all.refresh(reingest_raw=bool(raw))
            ch = (r["interpreted"] or {}).get("changes", {})
            msg = (f"Interpreted: {len(ch.get('new', []))} new, {len(ch.get('changed', []))} changed, {len(ch.get('removed', []))} removed. "
                   f"Raw: {r['raw'].get('status')}.")
    except Exception as e:
        msg = f"Ingestion failed: {type(e).__name__}: {e}"
    refresh_catalog()
    st.session_state["scan_msg"] = msg


def startup():
    """Once per browser session: apply saved folders and (optionally) pick up new interpreted files. Raw data is never re-ingested silently."""
    if st.session_state.get("_started"):
        return
    st.session_state["settings"] = store.load_settings()
    store.apply_settings(S())
    st.session_state["_started"] = True
    if S().get("scan_on_open") and S().get("interp_dir") and Path(S()["interp_dir"]).exists():
        import interp_data as D
        try:
            with st.spinner("Checking for new interpreted files ..."):
                D.scan()
        except Exception as e:
            st.session_state["scan_msg"] = f"Interpreted scan failed: {e}"


def raw_changed():
    try:
        import ingest_all
        ch, _, n = ingest_all._raw_changed()
        return bool(ch and n)
    except Exception:
        return False


# ----------------------------------------------------------------- sidebar
def sidebar():
    page = st.session_state["page"]
    with st.sidebar:
        theme.brand()
        if st.button("Workspace", icon=":material/layers:", type="primary" if page == "workspace" else "secondary", key="nav_ws"):
            goto("workspace")
            st.rerun()
        if st.button("Chat history", icon=":material/chat_bubble_outline:", type="primary" if page == "history" else "secondary", key="nav_hist"):
            goto("history")
            st.rerun()
        with st.expander("Data tools", icon=":material/tune:"):
            import ingest
            import interp_data as D
            st.caption(f"Raw data: {'ingested' if ingest.cache_ready() else 'not ingested'}  \nInterpreted data: {'ingested' if D.interpreted_ready() else 'not ingested'}")
            if st.button("Scan for new data", key="t_scan", help="Ingest only new or changed files"):
                with st.spinner("Scanning ..."):
                    do_scan()
                st.rerun()
            if st.button("Full re-ingest", key="t_full", help="Re-read every raw and interpreted file"):
                with st.spinner("Re-ingesting everything (raw volumes can take several minutes) ..."):
                    do_scan(force=True)
                st.rerun()
            if st.button("Show diagnostics", key="t_diag", help="Compact summary of what each interpreted file contains"):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    try:
                        D.diagnose()
                    except Exception as e:
                        print("diagnose failed:", e)
                st.session_state["diag"] = buf.getvalue()
            if st.session_state.get("diag"):
                st.code(st.session_state["diag"], language="text")
            if st.session_state.get("scan_msg"):
                st.info(st.session_state["scan_msg"])
        st.divider()
        st.markdown('<div class="gm-recent">RECENT</div>', unsafe_allow_html=True)
        cs = CH.chats()
        order = sorted([c for c in cs if cs[c]["msgs"] or c == st.session_state.get("active_chat")], key=lambda c: -cs[c].get("created", 0))
        for cid in order[:12]:
            c1, c2 = st.columns([5, 1])
            if c1.button(cs[cid]["title"], key=f"rc_{cid}", type="primary" if cid == st.session_state.get("active_chat") else "secondary"):
                st.session_state["active_chat"] = cid
                goto("history")
                st.rerun()
            with c2.popover("", icon=":material/more_horiz:"):
                new = st.text_input("Rename", value=cs[cid]["title"], key=f"rn_{cid}")
                if st.button("Save name", key=f"rns_{cid}"):
                    cs[cid]["title"] = new or cs[cid]["title"]
                    store.save_chats(cs)
                    st.rerun()
                if st.button("Delete chat", key=f"del_{cid}"):
                    cs.pop(cid, None)
                    if st.session_state.get("active_chat") == cid:
                        st.session_state.pop("active_chat", None)
                    store.save_chats(cs)
                    st.rerun()
        if st.button("Settings", icon=":material/settings:", key="nav_settings"):
            goto("settings")
            st.rerun()


# ----------------------------------------------------------------- chat panel (workspace)
def chat_panel(sel_labels):
    cid, chat = CH.active()
    with st.container(key="chat_panel"):
        h1, h2, h3 = st.columns([6, 1.2, 1.2])
        h1.markdown("### GeoMindAI")
        h1.caption(chat["title"])
        if h2.button("", icon=":material/open_in_full:", key="cp_expand", help="Open in Chat history"):
            goto("history", chat_open=False)
            st.rerun()
        if h3.button("", icon=":material/close:", key="cp_close", help="Close"):
            st.session_state["chat_open"] = False
            st.rerun()
        opts = CH.scope_labels(sel_labels)
        if st.session_state["scope"] not in opts:
            st.session_state["scope"] = opts[0]
        st.selectbox("Chat scope", opts, key="scope")
        st.caption("Connected workspace data only. Relevant evidence is retrieved per question; not every file is analysed.")
        if st.button("New chat", icon=":material/add_comment:", key="cp_new"):
            if chat["msgs"]:
                st.session_state["active_chat"] = store.new_chat(CH.chats())
            st.rerun()
        box = st.container(height=340, border=False)
        with box:
            if not chat["msgs"]:
                st.caption("Ask about all connected project data.")
                for q in CH.QUICK:
                    if st.button(q, key=f"cpq_{q}"):
                        st.session_state["pending"] = q
                        st.rerun()
            for i, m in enumerate(chat["msgs"]):
                CH.render(m, f"cp{i}")
            if "_run" in st.session_state:
                CH.finish_run(sel_labels)
        CH.input_box("Ask about all project data..." if st.session_state["scope"].startswith("All") else "Ask about the selected data...", sel_labels)


# ----------------------------------------------------------------- workspace
def add_to_selection(i):
    if i not in st.session_state["sel_ids"]:
        st.session_state["sel_ids"] = st.session_state["sel_ids"] + [i]


def remove_from_selection(i):
    st.session_state["sel_ids"] = [x for x in st.session_state["sel_ids"] if x != i]


def set_focus(i):
    st.session_state["focus"] = i


def clear_focus():
    st.session_state["focus"] = None


def overview_panel(cat, focus):
    with st.container(border=True):
        if not focus:
            st.markdown("#### Overview")
            items = cat["items"]
            if not items:
                st.markdown('<div class="gm-note">No data connected yet. Set the data folders in Settings or use Add data.</div>', unsafe_allow_html=True)
                return
            g = pd.Series([i["group"] for i in items]).value_counts()
            st.dataframe(pd.DataFrame({"Group": g.index, "Datasets": g.values}), hide_index=True, width="stretch")
            st.markdown('<div class="gm-note">Select a dataset in the list to zoom the map to it and see its details.</div>', unsafe_allow_html=True)
            for fam in ("raw", "interpreted"):
                w = cat["warnings"][fam]
                if w:
                    with st.expander(f"{len(w)} {fam}-data warning(s)"):
                        for x in w:
                            st.write("-", x)
            return
        st.markdown(f'#### {focus["label"]}<span class="gm-badge">{focus["type"]}</span>', unsafe_allow_html=True)
        t1, t2 = st.tabs(["Overview", "Sources"])
        with t1:
            if not focus.get("bbox"):
                st.markdown('<div class="gm-note">No map location; this data can still be used in chat.</div>', unsafe_allow_html=True)
            rows = C.overview_rows(focus)
            import html
            st.markdown('<table class="gm-tbl">' + "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in rows) + "</table>", unsafe_allow_html=True)
            st.write("")
            if focus["id"] in st.session_state["sel_ids"]:
                st.button("Remove from selected data", key="ov_rm", on_click=remove_from_selection, args=(focus["id"],), width="stretch")
            else:
                st.button("Add to selected data", key="ov_add", on_click=add_to_selection, args=(focus["id"],), width="stretch", type="primary")
            with st.expander("Dataset metadata"):
                st.json(C.metadata(focus), expanded=False)
        with t2:
            st.markdown(f"**File**  \n`{focus.get('path') or 'n/a'}`")
            if focus.get("time_source"):
                st.markdown(f"**Time taken from**  \n{focus['time_source']}")
            if focus.get("coord_mode"):
                st.markdown(f"**Coordinates**  \n{focus['coord_mode']}")
            if focus.get("kind") in ("fault", "polygon", "horizon"):
                w = cat["warnings"]["interpreted"]
                if w:
                    with st.expander(f"{len(w)} interpreted-data warning(s)"):
                        for x in w:
                            st.write("-", x)


def workspace():
    cat = C.build()
    items = cat["items"]
    by_id = {i["id"]: i for i in items}
    focus = by_id.get(st.session_state["focus"])
    sel_ids = [i for i in st.session_state["sel_ids"] if i in by_id]
    st.session_state["sel_ids"] = sel_ids
    sel_labels = [by_id[i]["label"] for i in sel_ids]

    t, b1, b2, b3 = st.columns([5, 1.6, 1.6, 0.6])
    t.markdown(f'<span class="gm-h1">Workspace</span><span class="gm-chip">{"Connected project" if items else "No data connected"}</span>', unsafe_allow_html=True)
    b1.button("Fit to data", icon=":material/center_focus_strong:", key="fit", on_click=clear_focus, width="stretch")
    if b2.button("Add data", icon=":material/add:", key="add_data", width="stretch"):
        add_data_dialog()
    with b3.popover("", icon=":material/tune:"):
        st.caption("Map layers")
        st.multiselect("Layers", ["survey", "horizon", "polygons", "faults", "wells"], key="layers", label_visibility="collapsed")

    if raw_changed():
        st.markdown('<div class="gm-warn">Raw data in the raw folder is new or changed since the last ingestion.</div>', unsafe_allow_html=True)
        if st.button("Ingest raw data now", key="raw_now"):
            with st.spinner("Ingesting raw data (can take several minutes) ..."):
                do_scan(raw=True)
            st.rerun()
    if st.session_state.get("scan_msg"):
        st.caption(st.session_state["scan_msg"])

    chat_open = st.session_state["chat_open"]
    c_list, c_map, c_right = st.columns([1.25, 2.5, 1.6] if chat_open else [1.25, 2.7, 1.5], gap="medium")

    with c_list:
        with st.container(border=True):
            st.markdown("#### Available data")
            q = st.text_input("Search datasets", key="ds_q", placeholder="Search datasets", label_visibility="collapsed").lower().strip()
            with st.container(height=470, border=False, key="dslist"):
                if not items:
                    st.caption("Nothing ingested yet.")
                for grp in ("Seismic volumes", "Wells", "Interpretations"):
                    g = [i for i in items if i["group"] == grp and (not q or q in i["label"].lower() or q in i["type"].lower())]
                    if not g:
                        continue
                    with st.expander(f"{grp} ({len(g)})", expanded=(grp == "Interpretations" or len(g) <= 8)):
                        last = None
                        for i in g:
                            if grp == "Interpretations" and i["sub"] != last:
                                st.caption(i["sub"])
                                last = i["sub"]
                            st.button(i["label"], key=f"ds_{i['id']}", icon=":material/description:", on_click=set_focus, args=(i["id"],),
                                      type="primary" if i["id"] == st.session_state["focus"] else "secondary", width="stretch")

    with c_map:
        with st.container(border=True):
            if not items:
                st.markdown("<div style='height:520px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#e6f1fb;border-radius:10px'>"
                            "<div style='font-size:1.4rem;font-weight:700'>Connect data to build your workspace</div>"
                            "<div class='gm-muted'>Add data or set the data folders in Settings.</div></div>", unsafe_allow_html=True)
            else:
                st.plotly_chart(mapview.make(cat, focus, sel_ids, layers=tuple(st.session_state["layers"])), width="stretch", key="map",
                                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
                st.caption("Map shows every dataset; pick one in the list to zoom to it. Scroll to zoom, drag to pan.")

    with c_right:
        if chat_open:
            chat_panel(sel_labels)
        else:
            overview_panel(cat, focus)

    with st.container(border=True):
        st.markdown("**Selected data**")
        st.multiselect("Selected data", options=list(by_id), format_func=lambda i: by_id[i]["label"], key="sel_ids", placeholder="Choose data from the list or here",
                       label_visibility="collapsed")

    if not chat_open:
        if st.button("", icon=":material/chat:", key="chat_fab", help="Open GeoMindAI chat"):
            st.session_state["chat_open"] = True
            st.rerun()


# ----------------------------------------------------------------- chat history page
def history():
    cat = C.build()
    by_id = {i["id"]: i for i in cat["items"]}
    sel_labels = [by_id[i]["label"] for i in st.session_state["sel_ids"] if i in by_id]
    cid, chat = CH.active()
    left, mid, right = st.columns([1, 3.2, 1])
    with mid:
        opts = CH.scope_labels(sel_labels)
        if st.session_state["scope"] not in opts:
            st.session_state["scope"] = opts[0]
        st.selectbox("Chat scope", opts, key="scope")
        st.caption("Connected workspace data only. Relevant evidence is retrieved per question; not every file is analysed.")
        t, d = st.columns([5, 1.3])
        t.markdown(f"### {chat['title']}")
        if d.button("Data", icon=":material/folder_open:", key="h_data", width="stretch"):
            goto("workspace", chat_open=False)
            st.rerun()
        st.divider()
        if not chat["msgs"]:
            st.markdown('<div class="gm-hero"><h1>What would you like to explore?</h1><p>Connect data. Investigate evidence. Ask the next question.</p></div>', unsafe_allow_html=True)
            with st.container(key="hero_cards"):
                cols = st.columns(3)
                for col, (name, icon) in zip(cols, (("Explore data", ":material/layers:"), ("Investigate a well", ":material/explore:"), ("Examine seismic", ":material/waves:"))):
                    if col.button(name, icon=icon, key=f"hero_{name}", width="stretch", type="primary" if st.session_state["hero"] == name else "secondary"):
                        st.session_state["hero"] = name
                        st.rerun()
            if st.session_state["hero"]:
                st.markdown(f"<div class='gm-sec' style='margin-top:1rem'>{st.session_state['hero']}</div>", unsafe_allow_html=True)
                for q in CH.HERO[st.session_state["hero"]]:
                    if st.button(q, key=f"hq_{q}", width="stretch"):
                        st.session_state["pending"] = q
                        st.rerun()
        for i, m in enumerate(chat["msgs"]):
            CH.render(m, f"h{i}")
        if "_run" in st.session_state:
            CH.finish_run(sel_labels)
    CH.input_box("Ask a question or attach a file...", sel_labels)


# ----------------------------------------------------------------- settings page
CFG_NUM = ["twt_ms", "sample_index", "twt_ms_plus_offset", "unknown"]


def _path_status(p, label):
    if not p:
        st.caption(f"{label}: not set")
    elif Path(p).exists():
        st.markdown(f"<span style='color:#2f8f83'>Folder found</span> <span class='gm-muted'>· {count_files(p):,} files</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#b3261e'>Folder not found on the machine that runs the app</span>", unsafe_allow_html=True)


def settings_page():
    s = S()
    st.markdown('<span class="gm-h1">Settings</span>', unsafe_allow_html=True)
    st.caption("Folders are paths on the machine that runs GeoMindAI. Nothing is ingested or changed until you save.")
    with st.container(border=True):
        st.markdown("#### Data locations")
        raw = st.text_input("Raw data folder", s["raw_dir"], help="SEG-Y stacks and inversions, LAS logs, deviations, checkshots (sub-folders such as stack, inversion, well_data)")
        _path_status(raw, "Raw")
        itp = st.text_input("Interpreted data folder", s["interp_dir"], help="Fault (SHP), Fault (CSV), Polygon (SHP), Horizon (e.g. .../data_export)")
        _path_status(itp, "Interpreted")
        cache = st.text_input("Cache folder", s["cache_dir"] or str(Path("cache").resolve()), help="Where ingested caches are written")
    with st.container(border=True):
        st.markdown("#### How new data is ingested")
        st.markdown('<div class="gm-note"><b>Recommended:</b> interpreted files (shapefiles, fault CSVs, horizon) are small, so they are ingested <b>automatically after upload</b>, '
                    'incrementally (only new or changed files). Raw seismic volumes and logs are large and can take minutes, so they are ingested <b>after upload, when you confirm</b>.</div>', unsafe_allow_html=True)
        a1 = st.toggle("Ingest interpreted data automatically after upload", s["auto_ingest_interpreted"])
        a2 = st.toggle("Ingest raw data automatically after upload", s["auto_ingest_raw"], help="Off: you are asked first (recommended for large SEG-Y)")
        a3 = st.toggle("Look for new interpreted files when the app opens", s["scan_on_open"])
    with st.container(border=True):
        st.markdown("#### Reasoning model")
        u = st.text_input("Model server address (OpenAI-compatible)", s["llm_base_url"] or "http://172.28.34.1:8000/v1")
        m = st.text_input("Model name", s["llm_model"] or "google/gemma-4-26B-A4B-it")
    if st.button("Save settings", type="primary", key="save_set"):
        new = {**s, "raw_dir": raw.strip(), "interp_dir": itp.strip(), "cache_dir": cache.strip(), "auto_ingest_interpreted": a1, "auto_ingest_raw": a2,
               "scan_on_open": a3, "llm_base_url": u.strip(), "llm_model": m.strip()}
        store.save_settings(new)
        st.session_state["settings"] = new
        store.apply_settings(new)
        refresh_catalog()
        st.success("Saved. Use Data tools > Scan for new data (or Full re-ingest after changing a folder) to read the data.")

    with st.container(border=True):
        st.markdown("#### Coordinates and time slices (interpreted data)")
        try:
            import pyproj  # noqa
            st.caption("pyproj is installed: lon/lat shapefiles can be converted with the CRS codes below.")
        except Exception:
            st.caption("pyproj is not installed: without CRS codes the shapefiles are positioned by calibration on the Fault CSV. Install pyproj to use CRS codes.")
        cfgp = Path(s["interp_dir"]) / "data_config.json" if s["interp_dir"] else None
        cfg = {}
        if cfgp and cfgp.exists():
            try:
                cfg = json.loads(cfgp.read_text())
            except Exception:
                cfg = {}
        c1, c2 = st.columns(2)
        shp = c1.text_input("Shapefile CRS", cfg.get("shapefile_crs", ""), placeholder="EPSG:4326")
        seis = c2.text_input("Seismic CRS", cfg.get("seismic_crs", ""), placeholder="EPSG:xxxxx")
        g1, g2, g3 = st.columns(3)
        gm = g1.selectbox("Georeferencing", ["auto", "bbox"], index=["auto", "bbox"].index(cfg.get("georef_method", "auto")) if cfg.get("georef_method", "auto") in ("auto", "bbox") else 0,
                          help="auto: CRS codes or Fault CSV calibration. bbox: stretch to the survey box (approximate)")
        fn = g2.selectbox("Number in fault folder names", CFG_NUM, index=CFG_NUM.index(cfg.get("fault_name_number", "twt_ms")) if cfg.get("fault_name_number", "twt_ms") in CFG_NUM else 0)
        pn = g3.selectbox("Number in polygon folder names", CFG_NUM, index=CFG_NUM.index(cfg.get("polygon_name_number", "twt_ms")) if cfg.get("polygon_name_number", "twt_ms") in CFG_NUM else 0)
        if st.button("Save and re-read interpreted data", key="save_cfg", disabled=not cfgp):
            cfg.update({"georef_method": gm, "fault_name_number": fn, "polygon_name_number": pn})
            if shp.strip():
                cfg["shapefile_crs"] = shp.strip()
            if seis.strip():
                cfg["seismic_crs"] = seis.strip()
            cfgp.write_text(json.dumps(cfg, indent=1))
            import interp_data as D
            with st.spinner("Re-reading interpreted data ..."):
                try:
                    D.scan(force=True)
                    st.success(f"Saved to {cfgp}")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
            refresh_catalog()


# ----------------------------------------------------------------- add data dialog
KINDS = {
    "Interpreted": {"Fault (SHP)": ("Fault (SHP)", True), "Fault (CSV)": ("Fault (CSV)", False), "Polygon (SHP)": ("Polygon (SHP)", True), "Horizon": ("Horizon", False)},
    "Raw": {"Seismic stack (SEG-Y)": ("stack", False), "Inversion (SEG-Y)": ("inversion", False), "Well data (LAS, deviation, checkshot, tops)": ("well_data", False)},
}


@st.dialog("Add data", width="large")
def add_data_dialog():
    s = S()
    fam = st.radio("Data family", list(KINDS), horizontal=True)
    kind = st.selectbox("Type", list(KINDS[fam]))
    sub, per_stem = KINDS[fam][kind]
    root = s["interp_dir"] if fam == "Interpreted" else s["raw_dir"]
    if not root:
        st.markdown(f'<div class="gm-warn">The {fam.lower()} data folder is not set. Set it in Settings first.</div>', unsafe_allow_html=True)
        if st.button("Open Settings"):
            goto("settings")
            st.rerun()
        return
    st.caption(f"Files are saved under `{Path(root) / sub}`")
    if per_stem:
        st.caption("Select the .shp together with its .shx, .dbf and .prj. Each file set goes in its own folder named after the file (the number in the name is read as the TWT time slice, e.g. 1500_Fault_01).")
    if fam == "Raw" and sub != "well_data":
        st.caption("SEG-Y volumes are very large; for those, copy the file into the raw folder on the server and use Scan for new data.")
    files = st.file_uploader("Files", accept_multiple_files=True, key="uploader")
    auto = s["auto_ingest_interpreted"] if fam == "Interpreted" else s["auto_ingest_raw"]
    now = st.checkbox("Ingest right after upload", value=auto)
    if st.button("Upload", type="primary", disabled=not files):
        dest_root = Path(root) / sub
        for f in files:
            d = dest_root / Path(f.name).stem if per_stem else dest_root
            d.mkdir(parents=True, exist_ok=True)
            (d / f.name).write_bytes(f.getbuffer())
        if now:
            with st.spinner("Ingesting ..."):
                do_scan(raw=(fam == "Raw"))
            st.session_state["scan_msg"] = f"Uploaded {len(files)} file(s). " + st.session_state.get("scan_msg", "")
        else:
            st.session_state["scan_msg"] = f"Uploaded {len(files)} file(s). Run Data tools > Scan for new data to read them."
        refresh_catalog()
        st.rerun()
