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
    ss.setdefault("map_revision", 0)
    ss.setdefault("hidden_ids", [])
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

        with st.container(key="sidebar_nav"):
            if st.button(
                "Workspace",
                icon=":material/layers:",
                type="primary" if page == "workspace" else "secondary",
                key="nav_ws",
                width="stretch",
            ):
                goto("workspace")
                st.rerun()

            if st.button(
                "Chat history",
                icon=":material/chat_bubble_outline:",
                type="primary" if page == "history" else "secondary",
                key="nav_hist",
                width="stretch",
            ):
                goto("history")
                st.rerun()

        st.markdown(
            '<div class="gm-recent">Recent conversations</div>',
            unsafe_allow_html=True,
        )

        cs = CH.chats()
        active = st.session_state.get("active_chat")
        # Keep explicitly created empty conversations visible too.
        order = sorted(cs, key=lambda cid: -cs[cid].get("created", 0))

        with st.container(key="sidebar_recent"):
            if not order:
                st.caption("Your conversations will appear here.")

            for cid in order[:12]:
                title, menu = st.columns(
                    [5, 1],
                    gap="small",
                    vertical_alignment="center",
                )

                if title.button(
                    cs[cid]["title"],
                    key=f"rc_{cid}",
                    help=cs[cid]["title"],
                    type="primary" if cid == active else "secondary",
                    width="stretch",
                ):
                    st.session_state["active_chat"] = cid
                    goto("workspace")
                    st.rerun()

                with menu.popover(
                    "",
                    icon=":material/more_horiz:",
                    help="Conversation options",
                ):
                    name = st.text_input(
                        "Conversation name",
                        value=cs[cid]["title"],
                        key=f"rn_{cid}",
                    )

                    if st.button("Save name", key=f"rns_{cid}"):
                        cs[cid]["title"] = name.strip() or cs[cid]["title"]
                        store.save_chats(cs)
                        st.rerun()

                    if st.button("Delete chat", key=f"del_{cid}"):
                        cs.pop(cid, None)
                        if st.session_state.get("active_chat") == cid:
                            st.session_state.pop("active_chat", None)
                        store.save_chats(cs)
                        st.rerun()

        with st.container(key="sidebar_footer"):
            if st.button(
                "Settings",
                icon=":material/settings:",
                key="nav_settings",
                type="primary" if page == "settings" else "secondary",
                width="stretch",
            ):
                goto("settings")
                st.rerun()


# ----------------------------------------------------------------- workspace helpers
def new_conversation():
    """Run before rendering so the sidebar and Chat open the same fresh chat."""
    cs = CH.chats()
    cid = store.new_chat(cs)
    st.session_state["active_chat"] = cid
    store.save_chats(cs)

    # A fresh conversation has no pending question or previous answer highlight.
    st.session_state.pop("pending", None)
    st.session_state.pop("_run", None)
    st.session_state.pop("context_mode", None)
    st.session_state["sel_ids"] = []
    st.session_state["scope"] = "All project data"
    st.session_state["focus"] = None
    st.session_state["map_targets"] = []
    st.session_state["map_fit"] = False
    st.session_state["map_revision"] = st.session_state.get("map_revision", 0) + 1
    st.session_state["_new_chat_created"] = cid


def clear_focus():
    st.session_state["focus"] = None
    st.session_state["map_revision"] += 1


def clear_highlight():
    clear_focus()


def data_tools():
    import interp_data as D
    if st.button("Scan for new data", key="t_scan", width="stretch"):
        with st.spinner("Scanning …"):
            do_scan()
        st.rerun()
    if st.button("Full re-ingest", key="t_full", width="stretch"):
        with st.spinner("Reading project data …"):
            do_scan(force=True)
        st.rerun()
    if st.button("Show diagnostics", key="t_diag", width="stretch"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                D.diagnose()
            except Exception as e:
                print(e)
        st.session_state["diag"] = buf.getvalue()
    if st.session_state.get("diag"):
        st.code(st.session_state["diag"], language="text")


@st.dialog("Dataset details", width="large")
def dataset_details(item):
    st.subheader(item["label"])
    st.dataframe(pd.DataFrame(C.overview_rows(item), columns=["Property", "Value"]).astype(str),
                 hide_index=True, width="stretch")
    if item.get("path"):
        st.caption("Source")
        st.code(item["path"], language="text")
    with st.expander("Metadata"):
        st.json(C.metadata(item))
    if item.get("bbox") and st.button("Locate dataset", icon=":material/location_on:"):
        st.session_state["focus"] = item["id"]
        st.session_state["map_revision"] += 1
        st.rerun()


def chat_panel(sel_labels, full=False):
    cid, chat = CH.active()
    if st.session_state.pop("_new_chat_created", None) == cid:
        st.toast("New chat created")
    cat = C.build()
    by_id = {item["id"]: item for item in cat["items"]}

    # Migrate the former two-step selector without activating an old selection.
    legacy_mode = st.session_state.pop("context_mode", None)
    ids = [i for i in st.session_state.get("sel_ids", []) if i in by_id]
    if legacy_mode is not None and legacy_mode != "Choose datasets…":
        ids = []
    st.session_state["sel_ids"] = ids

    scope_label = (
        "No data connected" if not by_id else
        "All project data" if not ids else
        f"{len(ids)} dataset{'s' if len(ids) != 1 else ''} selected"
    )

    with st.container(border=False, key="chat_panel"):
        with st.container(key="chat_header"):
            heading, new, expand = st.columns(
                [6, 1, 1], gap="small", vertical_alignment="center"
            )
            heading.markdown("### Chat")
            new.button(
                "", icon=":material/add:", key="cp_new", help="New chat",
                on_click=new_conversation,
            )
            if expand.button(
                "",
                icon=":material/arrow_back:" if full else ":material/open_in_full:",
                key="cp_expand",
                help="Back to workspace" if full else "Expand chat",
            ):
                goto("workspace" if full else "history")
                st.rerun()

        # One compact scope menu; an empty selection means all project data.
        # Scope remains independent of the map's visible layers.
        with st.container(key="chat_context"):
            with st.popover(scope_label, icon=":material/database:"):
                st.markdown("**Data for this chat**")
                ids = st.multiselect(
                    "Focus datasets",
                    list(by_id),
                    key="sel_ids",
                    format_func=lambda i: by_id[i]["label"],
                    placeholder="Choose datasets…",
                    disabled=not bool(by_id),
                )
                if by_id:
                    st.caption("Leave empty to use all project data.")
                else:
                    st.caption("Connect data using Add data or Settings.")

        selected = [by_id[i]["label"] for i in ids]
        st.session_state["scope"] = "Selected data" if selected else "All project data"

        with st.container(
            height=505 if not full else 600, border=False, key="conversation"
        ):
            if not chat["msgs"]:
                st.markdown("#### What would you like to explore?")
                st.caption("Ask about your seismic, wells or interpretations.")
                if not cat["items"]:
                    st.caption("Add project data to start a chat.")
                for q in CH.QUICK:
                    if st.button(
                        q, key=f"cpq_{q}", width="stretch",
                        disabled=not cat["items"],
                    ):
                        st.session_state["pending"] = q
                        st.rerun()
            for i, m in enumerate(chat["msgs"]):
                CH.render(m, f"{cid}_{i}")
            if "_run" in st.session_state:
                CH.finish_run(selected)

        question = st.chat_input(
            "Ask a follow-up…" if chat["msgs"] else "Ask about your project…",
            key=f"layout_chat_input_{cid}",
            disabled=not bool(cat["items"]),
        )
        question = question or st.session_state.pop("pending", None)
        if question and cat["items"]:
            CH.ask_agent(question, selected)
            st.rerun()



def workspace():
    cat = C.build()
    items = cat["items"]
    by_id = {i["id"]: i for i in items}
    focus = by_id.get(st.session_state["focus"])
    t, fit, add = st.columns([6, 1.7, 1.5], vertical_alignment="center")
    t.markdown('<div class="gm-h1">Workspace</div><div class="gm-muted">Project data and interpretation</div>', unsafe_allow_html=True)
    fit.button("Fit to data", icon=":material/center_focus_strong:", key="fit", on_click=clear_focus, width="stretch")
    if add.button("Add data", icon=":material/add:", key="add_data", type="primary", width="stretch"):
        add_data_dialog()
    if raw_changed():
        st.info("Raw files have changed. Use Project data → ⋯ → Scan for new data, then ingest raw data.")
        if st.button("Ingest raw data now"):
            with st.spinner("Reading raw data …"):
                do_scan(raw=True)
            st.rerun()
    if st.session_state.get("scan_msg"):
        st.caption(st.session_state["scan_msg"])
    with st.container(key="workspace_columns"):
        c_list, c_map, c_chat = st.columns([1.05, 2.8, 2.0], gap="medium")
        with c_list:
            with st.container(border=False, key="project_data"):
                with st.container(key="project_header"):
                    title, menu = st.columns([5, 1], gap="small", vertical_alignment="center")
                    title.markdown("### Project data")
                    with menu.popover("", icon=":material/more_horiz:", help="Data tools"):
                        data_tools()
                q = st.text_input("Search datasets", key="ds_q", placeholder="Search datasets…", label_visibility="collapsed").lower().strip()
                with st.container(height=540, border=False, key="dslist"):
                    if not items:
                        st.caption("No data connected")
                        if st.button("Set data folders", key="empty_settings", width="stretch"):
                            goto("settings")
                            st.rerun()
                    hidden = set(st.session_state["hidden_ids"])
                    for grp in ("Seismic volumes", "Wells", "Interpretations"):
                        group = [i for i in items if i["group"] == grp and (not q or q in i["label"].lower() or q in i["type"].lower())]
                        if not group:
                            continue
                        name = "Seismic" if grp == "Seismic volumes" else grp
                        with st.expander(f"{name} ({len(group)})", expanded=True):
                            for item in group:
                                check, label = st.columns([1, 6], gap="small", vertical_alignment="center")
                                visible = check.checkbox("Show " + item["label"], value=item["id"] not in hidden,
                                                         key="visible_" + item["id"], label_visibility="collapsed")
                                if visible:
                                    hidden.discard(item["id"])
                                else:
                                    hidden.add(item["id"])
                                if label.button(item["label"], key="ds_" + item["id"], help="Dataset details", width="stretch"):
                                    dataset_details(item)
                    st.session_state["hidden_ids"] = sorted(hidden)
                st.caption("Tick to show · Click a name for details")
                warnings = cat.get("warnings", {})
                if any(warnings.values()):
                    with st.expander("Data warnings"):
                        for family, messages in warnings.items():
                            for message in messages:
                                st.write(f"{family}: {message}")
        with c_map:
            with st.container(border=False, key="main_map"):
                with st.container(key="map_header"):
                    st.markdown("### Map")
                st.caption("Explore your project layers")
                visible_cat = {**cat, "items": [i for i in items if i["id"] not in st.session_state["hidden_ids"]]}
                if items:
                    visible_focus = focus if focus and focus["id"] not in st.session_state["hidden_ids"] else None
                    fig = mapview.make(visible_cat, visible_focus)
                    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                                      font=dict(family="Arial, sans-serif", size=12, color="#193247"),
                                      height=560, uirevision=f"layout:{st.session_state['map_revision']}:{visible_focus['id'] if visible_focus else 'all'}",
                                      margin=dict(l=8, r=8, t=8, b=8))
                    st.plotly_chart(fig, use_container_width=True, key="map", config={"scrollZoom": True, "displaylogo": False})
                    st.caption("Scroll to zoom · Drag to pan")
                else:
                    st.markdown('<div class="gm-map-empty"><b>Your project map</b><p>Connect seismic, wells or interpreted data<br>to explore their locations here.</p></div>', unsafe_allow_html=True)
        with c_chat:
            chat_panel([])


def history():
    with st.container(key="expanded_chat"):
        chat_panel([], full=True)


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
        st.success("Saved. Use Project data → ⋯ → Scan for new data (or Full re-ingest after changing a folder) to read the data.")

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
            st.session_state["scan_msg"] = f"Uploaded {len(files)} file(s). Run Project data → ⋯ → Scan for new data to read them."
        refresh_catalog()
        st.rerun()
