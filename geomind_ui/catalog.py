"""Read-only view of what has been ingested (from the existing caches), shaped for the workspace list, overview panel and map."""
import json
from pathlib import Path

import numpy as np

import ingest as I
import interp_data as D

CLS_COLOR = {"channel": "#d9a23a", "overbank": "#7fae88", "fault": "#c0392b"}
_FALLBACK = ["#6a7fd1", "#b57bbf", "#4aa3a2", "#d4785f"]
_CACHE = {}


def _sig():
    out = []
    for p in (I.CACHE_DIR / "cards.json", I.CACHE_DIR / "interpreted" / "manifest.pkl"):
        try:
            out.append(p.stat().st_mtime_ns)
        except Exception:
            out.append(0)
    return (str(I.CACHE_DIR), str(D.interp_dir()), tuple(out))


def _bbox_of_rings(rings):
    a = np.vstack([np.asarray(r)[:, :2] for r in rings if len(r)])
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def _pad(b, frac=0.0, minimum=0.0):
    w, h = b[2] - b[0], b[3] - b[1]
    px, py = max(w * frac, minimum), max(h * frac, minimum)
    return [b[0] - px, b[1] - py, b[2] + px, b[3] + py]


def build():
    """Returns {'items': [...], 'survey': bbox|None, 'warnings': {...}}. Cached until the ingestion caches change."""
    sig = _sig()
    if _CACHE.get("sig") == sig:
        return _CACHE["val"]
    items, boxes = [], []
    warnings = {"raw": [], "interpreted": []}
    # ---- raw
    try:
        cards = I.load_cards()
        warnings["raw"] = list(cards.get("warnings", []))
        for k, v in cards["volumes"].items():
            b = None
            if v.get("x_range") and v.get("y_range"):
                b = [v["x_range"][0], v["y_range"][0], v["x_range"][1], v["y_range"][1]]
                boxes.append(b)
            items.append({"id": f"vol:{k}", "label": v["label"], "group": "Seismic volumes", "sub": v.get("kind", "volume"), "type": v.get("kind", "volume").title(),
                          "bbox": b, "path": v.get("path"), "meta": v, "geom": "volume"})
        for w, c in cards["wells"].items():
            b = None
            if c.get("x") is not None:
                b = [c["x"], c["y"], c["x"], c["y"]]
                boxes.append(b)
            items.append({"id": f"well:{w}", "label": w, "group": "Wells", "sub": "Wells", "type": f"Well ({c.get('type', '?')})", "bbox": b,
                          "path": (c.get("las") or {}).get("file"), "meta": c, "geom": "well"})
    except Exception:
        pass
    # ---- interpreted
    try:
        st = D._state()
        recs = list(st["data"].values())
        warnings["interpreted"] = []
        p = I.CACHE_DIR / "interpreted_cards.json"
        if p.exists():
            warnings["interpreted"] = json.loads(p.read_text()).get("warnings", [])
        order = {"fault": 0, "polygon": 1, "horizon": 2}
        recs.sort(key=lambda r: (order.get(r.get("kind", "horizon"), 3), str(r.get("cls")), r.get("time_ms") if r.get("time_ms") is not None else 1e12, r["rel"]))
        root = D.interp_dir()
        for r in recs:
            rel = r["rel"]
            label = Path(rel).name
            if r["type"] == "shp":
                rings = [np.asarray(x)[:, :2] for f in r["feats"] for x in f["rings"] if len(x)]
                b = _bbox_of_rings(rings) if rings else None
                boxes.append(b) if b else None
                kind, cls = r["kind"], r["cls"]
                items.append({"id": f"int:{rel}", "label": label, "group": "Interpretations", "sub": "Faults" if kind == "fault" else cls.title(),
                              "type": "Fault" if kind == "fault" else f"Polygon ({cls})", "kind": kind, "cls": cls, "time_ms": r.get("time_ms"), "time_source": r.get("time_source"),
                              "n": len(r["feats"]), "geom": r.get("geom"), "bbox": b, "rings": rings, "coord_mode": r.get("coord_mode"),
                              "path": str(root / rel) if root else rel, "rel": rel, "fields": r.get("fields")})
            else:
                b = [float(v) for v in r["bbox"]]
                boxes.append(b)
                items.append({"id": f"int:{rel}", "label": label, "group": "Interpretations", "sub": "Horizons", "type": "Horizon", "kind": "horizon", "cls": "horizon",
                              "n": r["n"], "t_range": r.get("t_range"), "bbox": b, "coord_mode": r.get("coord_mode"), "notes": r.get("notes"), "name": r["name"],
                              "path": str(root / rel) if root else rel, "rel": rel, "geom": "horizon"})
    except Exception:
        pass
    survey = None
    boxes = [b for b in boxes if b]
    if boxes:
        a = np.array(boxes)
        survey = [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())]
    val = {"items": items, "survey": survey, "warnings": warnings}
    _CACHE["sig"], _CACHE["val"] = sig, val
    return val


def horizon_points(name=None, max_pts=3500):
    h = D.horizon(name)
    if h is None:
        return None
    x, y, t = np.asarray(h["x"]).ravel(), np.asarray(h["y"]).ravel(), np.asarray(h["t"]).ravel()
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x, y, t = x[ok], y[ok], t[ok]
    if len(x) > max_pts:
        i = np.linspace(0, len(x) - 1, max_pts).astype(int)
        x, y, t = x[i], y[i], t[i]
    return x, y, t


def focus_box(item, survey):
    """Map window for a selected dataset: the dataset itself with some margin (points get a margin of ~8% of the survey)."""
    b = item.get("bbox")
    if not b:
        return survey
    span = max((survey[2] - survey[0]), (survey[3] - survey[1])) if survey else 1000.0
    if b[2] - b[0] < 1e-6 and b[3] - b[1] < 1e-6:
        return _pad(b, 0, span * 0.08)
    return _pad(b, 0.25, span * 0.01)


def overview_rows(item):
    """Label/value rows for the Overview tab."""
    rows = [("Type", item["type"]), ("Group", item["group"])]
    m = item.get("meta") or {}
    if item["geom"] == "volume":
        rows += [("Name", m.get("label")), ("Traces", m.get("n_traces")), ("Samples", f"{m.get('n_samples')} @ {m.get('sample_rate_ms')} ms"),
                 ("Time range (ms)", f"{m.get('time_start_ms')} - {m.get('time_end_ms')}"),
                 ("Inline range", f"{(m.get('inline') or {}).get('min')} - {(m.get('inline') or {}).get('max')}"),
                 ("Xline range", f"{(m.get('xline') or {}).get('min')} - {(m.get('xline') or {}).get('max')}"),
                 ("X range", m.get("x_range")), ("Y range", m.get("y_range"))]
    elif item["geom"] == "well":
        rows += [("Well type", m.get("type")), ("Total depth (m)", m.get("td_m")), ("KB/RT (m)", m.get("rt_m")), ("X, Y", f"{m.get('x')}, {m.get('y')}"),
                 ("Deviation", "yes" if m.get("has_deviation") else "no"), ("Checkshot", "yes" if m.get("has_checkshot") else "no"),
                 ("Markers", "yes" if m.get("has_markers") else "no"), ("LAS logs", "yes" if m.get("has_las") else "no"),
                 ("Time-depth confidence", (m.get("time_depth") or {}).get("confidence")), ("Log curves found", ", ".join((m.get("roles") or {}).keys()) or None)]
    elif item["geom"] == "horizon":
        rows += [("Points", item["n"]), ("TWT range (ms)", item.get("t_range")), ("Coordinates", item.get("coord_mode"))]
    else:
        tm = item.get("time_ms")
        rows += [("Time slice (TWT ms)", tm if tm is not None else "unknown"), ("Time taken from", item.get("time_source")),
                 ("Features", item.get("n")), ("Geometry", item.get("geom")), ("Coordinates", item.get("coord_mode"))]
    b = item.get("bbox")
    if b and item["geom"] != "well":
        rows.append(("Extent X", f"{b[0]:,.0f} - {b[2]:,.0f}"))
        rows.append(("Extent Y", f"{b[1]:,.0f} - {b[3]:,.0f}"))
    return [(k, v) for k, v in rows if v not in (None, "", "None - None")]


def metadata(item):
    keep = {k: v for k, v in item.items() if k not in ("rings", "meta", "bbox")}
    if item.get("meta"):
        keep["meta"] = {k: v for k, v in item["meta"].items() if k not in ("amplitude", "las", "roles", "track_bottom") or k == "roles"}
    return json.loads(json.dumps(keep, default=str))
