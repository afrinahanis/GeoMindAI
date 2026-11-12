"""
tools.py - the functions Gemma can call. Each returns a small JSON-serialisable dict/list.
TOOLS maps name -> function, TOOL_SCHEMAS is the OpenAI-style schema list.
"""
import inspect
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import ingest as I
import interpret as N
from interpret import (logs_to_time, find_gas_zones, thickest_gas_zone, compare_zone_to_seismic,
                       extrapolate_zone_to_map, find_prospects, structure_and_closure,
                       depositional_pattern, trap_evidence, horizon_contour_and_faults,
                       horizon_slice, overlay_structure_and_slice, high_variation_slices, count_faults)
from ai_interp import ai_fault_detection, ai_facies_map
from interp_tools import (list_datasets, scan_for_new_data, interpreted_faults, interpreted_facies, horizon_structure,
                          interpreted_map, well_context, validate_interpreted_vs_seismic, assess_traps, rank_prospects, recommend_trap)
from volmath import volume_math, gas_potential

KNOW_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "knowledge"))
IMG_DIR = I.CACHE_DIR / "images"


# ----------------------------------------------------------------- helpers
def _f(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _stats(a, nd=3):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    p = np.percentile(a, [10, 50, 90])
    return {"n": int(a.size), "min": _f(a.min(), nd), "p10": _f(p[0], nd), "p50": _f(p[1], nd),
            "mean": _f(a.mean(), nd), "p90": _f(p[2], nd), "max": _f(a.max(), nd)}


def _mask(md, lo, hi):
    m = np.ones(len(md), bool)
    if lo is not None:
        m &= md >= lo
    if hi is not None:
        m &= md <= hi
    return m


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return None
    return _f(np.corrcoef(a[m], b[m])[0, 1], 2)


def _smooth_at(well, mnemonic, md_points, win_m=20):
    s = I.load_logs(well)[mnemonic].dropna()
    if len(s) < 10:
        return np.full(len(md_points), np.nan)
    grid = np.arange(s.index.min(), s.index.max(), 1.0)
    y = np.interp(grid, s.index.values, s.values)
    ys = pd.Series(y).rolling(int(win_m), center=True, min_periods=1).mean().values
    return np.interp(md_points, grid, ys, left=np.nan, right=np.nan)


def _track_grid(w, step=5.0):
    t = I.load_track(w)
    md = np.arange(t.md.min(), t.md.max(), step)
    return pd.DataFrame({"md": md, **{c: np.interp(md, t.md, t[c]) for c in ("tvdss", "x", "y", "twt_ms")}})


def _cards():
    return I.load_cards()


# ------------------------------------------------------------------- tools
def list_wells():
    """Compact table of all wells."""
    return [{"well": w, "type": c["type"], "td_m": c["td_m"], "rt_m": c["rt_m"], "x": c["x"], "y": c["y"],
             "max_inclination_deg": _f(c.get("max_inclination_deg"), 1),
             "has_checkshot": c["has_checkshot"], "time_depth_confidence": c["time_depth"]["confidence"],
             "has_markers": c["has_markers"], "log_roles_found": list((c.get("roles") or {}).keys())}
            for w, c in _cards()["wells"].items()]


def get_well_summary(well):
    """Everything known about one well: header data, path, time-depth, log inventory, seismic coverage."""
    w = I.resolve_well(well)
    c = dict(_cards()["wells"][w])
    las = dict(c.get("las") or {})
    las["curves"] = {m: {k: v.get(k) for k in ("unit", "top_m", "base_m", "null_pct")}
                     for m, v in (las.get("curves") or {}).items()}
    c["las"] = las
    c["note"] = "Formation tops/markers are not available in the provided data." if not c["has_markers"] else ""
    return c


def get_log_stats(well, curve, md_from=None, md_to=None):
    """Statistics of one log curve (mnemonic or role like GR/RES/RHOB/NPHI/DT) over an optional MD window."""
    w = I.resolve_well(well)
    df = I.load_logs(w)
    roles = _cards()["wells"][w].get("roles", {})
    mn = roles.get(curve.upper(), curve.upper())
    if mn not in df.columns:
        return {"error": f"curve '{curve}' not found", "available": list(df.columns), "roles": roles}
    s = df[mn][_mask(df.index.values, md_from, md_to)]
    return {"well": w, "curve": mn, "unit": _cards()["wells"][w]["las"]["curves"][mn]["unit"],
            "md_window_m": [md_from, md_to], "null_pct_in_window": _f(100 * s.isna().mean(), 1),
            "stats": _stats(s.values)}


def get_volume_info(name=None):
    """Metadata of one seismic/inversion volume, or a compact list of all volumes."""
    vols = _cards()["volumes"]
    if name is None:
        return [{k: c[k] for k in ("name", "label", "kind", "n_traces", "sample_rate_ms", "time_start_ms",
                                   "time_end_ms", "inline", "xline")} for c in vols.values()]
    return vols[I.resolve_volume(name)]


def sample_volumes_at_well(well, md_from=None, md_to=None):
    """P-imp, S-imp, Vp/Vs and sand probability extracted along the well path (stats + 8-bin profile)."""
    w = I.resolve_well(well)
    ext = I.load_extraction().get(w)
    c = _cards()["wells"][w]
    if not ext:
        return {"well": w, "error": "well path does not intersect any inversion volume",
                "coverage": c["volume_coverage"]}
    out = {"well": w, "time_depth_confidence": c["time_depth"]["confidence"], "volumes": {}}
    for vn, e in ext.items():
        m = _mask(e["md"], md_from, md_to)
        if not m.any():
            continue
        md, val, twt = e["md"][m], e["val"][m], e["twt"][m]
        edges = np.linspace(md.min(), md.max(), 9)
        prof = []
        for a, b in zip(edges[:-1], edges[1:]):
            k = (md >= a) & (md <= b)
            if k.any():
                prof.append({"md_from": _f(a, 0), "md_to": _f(b, 0), "twt_ms": _f(twt[k].mean(), 0),
                             "mean": _f(val[k].mean())})
        out["volumes"][vn] = {"label": _cards()["volumes"][vn]["label"], "md_range_m": [_f(md.min(), 0), _f(md.max(), 0)],
                              "stats": _stats(val), "profile": prof}
    return out


def crosscheck_well(well, md_from=None, md_to=None):
    """Compare logs (GR, resistivity, density, neutron, sonic) with inversion values at the same depths."""
    w = I.resolve_well(well)
    c = _cards()["wells"][w]
    ext = I.load_extraction().get(w)
    if not ext:
        return {"well": w, "error": "no inversion coverage along this well"}
    roles = c.get("roles", {})
    out = {"well": w, "time_depth_confidence": c["time_depth"]["confidence"],
           "note": "Logs smoothed over ~20 m, not rigorously upscaled; correlations are indicative only.",
           "volumes": {}}
    for vn, e in ext.items():
        m = _mask(e["md"], md_from, md_to)
        if m.sum() < 10:
            continue
        md, val = e["md"][m], e["val"][m]
        entry = {"n_points": int(m.sum()), "correlation_with_log": {}}
        for role in ("GR", "RES", "RHOB", "NPHI", "DT"):
            if role in roles:
                entry["correlation_with_log"][role] = _corr(_smooth_at(w, roles[role], md), val)
        if "GR" in roles:
            gr = _smooth_at(w, roles["GR"], md)
            if np.isfinite(gr).sum() > 10:
                lo, hi = np.nanpercentile(gr, [25, 75])
                entry["mean_value_where_GR_low(clean)"] = _f(np.nanmean(val[gr <= lo]))
                entry["mean_value_where_GR_high(shaly)"] = _f(np.nanmean(val[gr >= hi]))
                entry["GR_cutoffs_p25_p75"] = [_f(lo, 1), _f(hi, 1)]
        out["volumes"][vn] = entry
    return out


def compare_wells(well_a, well_b):
    """Side-by-side comparison of two wells (header, median logs, inversion means)."""
    out = {}
    for key, name in (("a", well_a), ("b", well_b)):
        w = I.resolve_well(name)
        c = _cards()["wells"][w]
        med = {r: _f(((c.get("las") or {}).get("curves", {}).get(m) or {}).get("p50"))
               for r, m in (c.get("roles") or {}).items() if r in ("GR", "RES", "RHOB", "NPHI", "DT")}
        out[w] = {"type": c["type"], "td_m": c["td_m"], "rt_m": c["rt_m"], "x": c["x"], "y": c["y"],
                  "max_inclination_deg": _f(c.get("max_inclination_deg"), 1),
                  "time_depth_confidence": c["time_depth"]["confidence"], "log_median_whole_well": med,
                  "inversion_mean_along_well": {v: _f(e.get("mean")) for v, e in (c.get("extraction") or {}).items()}}
    (wa, ca), (wb, cb) = list(out.items())[0], list(out.items())[-1]
    A, B = _cards()["wells"][wa], _cards()["wells"][wb]
    out["surface_distance_m"] = _f(math.hypot(A["x"] - B["x"], A["y"] - B["y"]), 0)
    return out


def rank_wells(volume):
    """Rank all wells by the mean value of one inversion volume along their path."""
    v = I.resolve_volume(volume)
    rows = []
    for w, c in _cards()["wells"].items():
        e = (c.get("extraction") or {}).get(v)
        if e and e.get("n_points"):
            rows.append({"well": w, "mean": _f(e["mean"]), "p10": _f(e["p10"]), "p90": _f(e["p90"]),
                         "md_range_m": [_f(x, 0) for x in e["md_range"]]})
    rows.sort(key=lambda r: -r["mean"])
    return {"volume": v, "label": _cards()["volumes"][v]["label"], "ranking_high_to_low": rows,
            "wells_without_coverage": [w for w in _cards()["wells"] if w not in [r["well"] for r in rows]]}


def stack_amplitudes_at_well(well, twt_min_ms=None, twt_max_ms=None):
    """RMS/mean amplitude of near/mid/far/ultra/full stacks along the well in a TWT window (qualitative)."""
    w = I.resolve_well(well)
    g = _track_grid(w)
    hi = twt_max_ms if twt_max_ms is not None else g.twt_ms.max()
    lo = twt_min_ms if twt_min_ms is not None else hi - 500
    g = g[(g.twt_ms >= lo) & (g.twt_ms <= hi)]
    res = {}
    for vn, c in _cards()["volumes"].items():
        if c["kind"] != "stack":
            continue
        vals, d = I.open_volume(vn).sample_along(g.x, g.y, g.twt_ms)
        ok = vals[np.isfinite(vals)]
        if ok.size:
            res[vn] = {"rms": _f(np.sqrt(np.mean(ok ** 2)), 1), "mean": _f(ok.mean(), 1),
                       "mean_abs": _f(np.abs(ok).mean(), 1), "n": int(ok.size)}
    out = {"well": w, "twt_window_ms": [_f(lo, 0), _f(hi, 0)], "stacks": res,
           "time_depth_confidence": _cards()["wells"][w]["time_depth"]["confidence"],
           "note": "Qualitative amplitude summary, not a rigorous AVO analysis."}
    near = next((v for v in res if "near" in v), None)
    far = next((v for v in res if "far" in v and "ultra" not in v), None)
    if near and far and res[near]["rms"]:
        out["far_over_near_rms"] = _f(res[far]["rms"] / res[near]["rms"], 2)
    if not res:
        out["error"] = "well path outside the stack volumes in this window"
    return out


def lateral_variability(volume, twt_min_ms, twt_max_ms, max_traces=6000):
    """Map-view summary of one volume averaged over a TWT window: spread, gradients, coarse block map."""
    v = I.open_volume(volume)
    k = (v.samples >= twt_min_ms) & (v.samples <= twt_max_ms)
    if not k.any():
        return {"error": f"window outside volume time range {v.samples[0]}-{v.samples[-1]} ms"}
    uil, uxl = np.unique(v.il), np.unique(v.xl)
    s = max(1, int(math.ceil(math.sqrt(len(v.il) / max_traces))))
    sil, sxl = uil[::s], uxl[::s]
    sel = np.where(np.isin(v.il, sil) & np.isin(v.xl, sxl))[0]
    vals = np.array([v.trace(i)[k].mean() for i in sel])
    grid = np.full((len(sil), len(sxl)), np.nan)
    grid[np.searchsorted(sil, v.il[sel]), np.searchsorted(sxl, v.xl[sel])] = vals
    gy, gx = np.gradient(grid)
    gm = np.hypot(gx, gy)
    r, c = np.unravel_index(np.nanargmax(gm), gm.shape)
    rb, cb = min(5, grid.shape[0]), min(5, grid.shape[1])
    blocks = [[_f(np.nanmean(b)) for b in np.array_split(rows, cb, axis=1)]
              for rows in np.array_split(grid, rb, axis=0)]
    wells = {}
    for w, wc in _cards()["wells"].items():
        d, i = v.tree.query([wc["x"], wc["y"]])
        if d <= v.max_dist:
            wells[w] = _f(v.trace(i)[k].mean())
    return {"volume": v.name, "twt_window_ms": [twt_min_ms, twt_max_ms],
            "decimation_every_n_lines": s, "map_stats": _stats(vals),
            "gradient": {"mean": _f(np.nanmean(gm), 4), "p90": _f(np.nanpercentile(gm, 90), 4),
                         "max": _f(np.nanmax(gm), 4), "max_at_inline_xline": [int(sil[r]), int(sxl[c])],
                         "mean_gradient_over_std": _f(np.nanmean(gm) / (np.nanstd(grid) or 1), 3)},
            "coarse_block_map_rows_inline_low_to_high_cols_xline_low_to_high": blocks,
            "inline_range": [int(sil[0]), int(sil[-1])], "xline_range": [int(sxl[0]), int(sxl[-1])],
            "value_at_wells": wells}


def render_slice(volume, kind, index, twt_min_ms=None, twt_max_ms=None):
    """Render an inline/xline section or a time slice to PNG (wells nearby are marked). index = line number, or TWT ms for kind='time'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = I.open_volume(volume)
    vc = _cards()["volumes"][v.name]
    kind = kind.lower()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    wells = {}
    for w, c in _cards()["wells"].items():
        d, i = v.tree.query([c["x"], c["y"]])
        if d <= v.max_dist:
            wells[w] = (v.il[i], v.xl[i], c["x"], c["y"])
    fig, ax = plt.subplots(figsize=(9, 5))
    used = None
    if kind in ("inline", "xline"):
        line, other = (v.il, v.xl) if kind == "inline" else (v.xl, v.il)
        used = int(np.unique(line)[np.argmin(np.abs(np.unique(line) - index))])
        idx = np.where(line == used)[0]
        idx = idx[np.argsort(other[idx])]
        data = np.array([v.trace(i) for i in idx]).T
        t0, t1 = (twt_min_ms or v.samples[0]), (twt_max_ms or v.samples[-1])
        ks = (v.samples >= t0) & (v.samples <= t1)
        data = data[ks]
        vmin, vmax = _scale(data, vc["kind"])
        ax.imshow(data, aspect="auto", cmap=_cmap(vc["kind"]), vmin=vmin, vmax=vmax,
                  extent=[other[idx].min(), other[idx].max(), v.samples[ks][-1], v.samples[ks][0]])
        ax.set_xlabel("xline" if kind == "inline" else "inline")
        ax.set_ylabel("TWT (ms)")
        tol = 3 * max(1, np.median(np.diff(np.unique(line))))
        for w, (wi, wx, _, _) in wells.items():
            wl, wo = (wi, wx) if kind == "inline" else (wx, wi)
            if abs(wl - used) <= tol:
                ax.axvline(wo, color="red", ls="--", lw=0.8)
                ax.text(wo, t0 + 0.03 * (t1 - t0), w, color="red", rotation=90, va="top", fontsize=8)
    elif kind == "time":
        k = int(np.argmin(np.abs(v.samples - index)))
        used = float(v.samples[k])
        sel = np.arange(0, len(v.x), max(1, len(v.x) // 20000))
        vals = np.array([v.trace(i)[k] for i in sel])
        vmin, vmax = _scale(vals, vc["kind"])
        sc = ax.scatter(v.x[sel], v.y[sel], c=vals, s=5, marker="s", cmap=_cmap(vc["kind"]), vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax)
        for w, (_, _, x, y) in wells.items():
            ax.plot(x, y, "r^", ms=6)
            ax.text(x, y, " " + w, color="red", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    else:
        plt.close(fig)
        return {"error": "kind must be inline, xline or time"}
    ax.set_title(f"{vc['label']} - {kind} {used}")
    path = IMG_DIR / f"{v.name}_{kind}_{used}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"image_path": str(path), "volume": v.name, "kind": kind, "index_used": used,
            "color_scale": [_f(vmin), _f(vmax)], "wells_in_volume": list(wells)}


def _scale(a, kind):
    a = np.asarray(a, float)
    if kind == "stack":
        m = np.nanpercentile(np.abs(a), 98)
        return -m, m
    return np.nanpercentile(a, 2), np.nanpercentile(a, 98)


def _cmap(kind):
    return "gray" if kind == "stack" else "viridis"


# ---------------------------------------------------------- knowledge search
def _chunks():
    out = []
    for p in KNOW_DIR.glob("**/*"):
        if p.suffix.lower() in (".md", ".txt") and p.is_file():
            buf = ""
            for para in re.split(r"\n\s*\n", p.read_text(errors="ignore")):
                if len(buf) + len(para) > 800 and buf:
                    out.append((p.name, buf.strip()))
                    buf = ""
                buf += para + "\n\n"
            if buf.strip():
                out.append((p.name, buf.strip()))
    return out


def search_knowledge(query, top_k=3):
    """Search the local knowledge/ notes (basin notes, analogues, trap criteria)."""
    ch = _chunks()
    if not ch:
        return {"error": "knowledge/ folder is empty - no analogue or report notes provided"}
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    docs = [Counter(tok(t)) for _, t in ch]
    df = Counter(w for d in docs for w in d)
    q = tok(query)
    scored = sorted(((sum(d[w] * math.log(1 + len(ch) / df[w]) for w in q if w in d), i)
                     for i, d in enumerate(docs)), reverse=True)[:int(top_k)]
    return [{"source": ch[i][0], "score": _f(s, 2), "text": ch[i][1]} for s, i in scored if s > 0]


# ---------------------------------------------------------------- registry
PARAMS = {
    "well": ("string", "Well name, e.g. 'eo-1'"),
    "well_a": ("string", "First well name"), "well_b": ("string", "Second well name"),
    "curve": ("string", "Log mnemonic or role: GR, RES, RHOB, NPHI, DT"),
    "md_from": ("number", "Start measured depth in m (optional)"),
    "md_to": ("number", "End measured depth in m (optional)"),
    "name": ("string", "Volume name or alias (e.g. 'sand probability', 'vpvs', 'far stack'); omit to list all"),
    "volume": ("string", "Volume name or alias, e.g. 'sand probability', 'p impedance', 'vpvs', 'near stack'"),
    "kind": ("string", "inline, xline or time"),
    "index": ("number", "Inline/xline number, or TWT in ms when kind='time'"),
    "twt_min_ms": ("number", "Window start TWT in ms"), "twt_max_ms": ("number", "Window end TWT in ms"),
    "max_traces": ("integer", "Max traces to read (default 6000)"),
    "dt_ms": ("number", "Time sampling in ms (default 4)"),
    "fluid": ("string", "'gas' (needs density-neutron crossover) or 'hc' (any hydrocarbon)"),
    "min_thickness_m": ("number", "Minimum net pay in m (default 2)"),
    "zone_rank": ("integer", "1 = thickest zone of that well (default 1)"),
    "tolerance": ("number", "Extra tolerance around the well signature, fraction (default 0.15)"),
    "top_n": ("integer", "How many candidates to return (default 5)"),
    "win_ms": ("number", "Scan window thickness in ms (default 40)"),
    "threshold": ("number", "Sand-probability cut-off (default: auto)"),
    "min_separation_ms": ("number", "Minimum spacing between ranked slices in ms (default 20)"),
    "n_classes": ("integer", "Number of facies classes (default 5)"),
    "slice_twt_ms": ("number", "Centre TWT (ms) of the horizon slice"),
    "expression": ("string", "Arithmetic on volumes, e.g. 'far - near' or 'sand * (far - near)'. Names: far, near, mid, full, ultrafar, sand, ip, is, vpvs or exact volume names"),
    "reducer": ("string", "How to collapse the window per trace: mean, rms, min, max, mean_abs or median (default mean)"),
    "select": ("string", "'low' = most negative/smallest values, 'high' = largest values"),
    "percent": ("number", "Share of the map (percent) treated as extreme, default 10"),
    "normalise": ("boolean", "Scale each stack to unit variance before the arithmetic (default false for volume_math, true for gas_potential)"),
    "avo_expression": ("string", "AVO attribute expression, default 'far - near'"),
    "avo_percent": ("number", "Lowest this-percent of the AVO map counts as strongly negative (default 15)"),
    "sand_threshold": ("number", "Minimum sand probability 0-1 (default: top 30 percent of the map)"),
    "structural_percent": ("number", "Shallowest this-percent of the interpreted horizon counts as structurally high (default 40)"),
    "class_name": ("string", "Interpreted polygon class, e.g. 'channel' or 'overbank' (omit for all)"),
    "link_tolerance": ("number", "Distance in map units to link fault traces across slices (default: auto)"),
    "cross_check": ("boolean", "Also validate against raw seismic and automatic fault detection (default true)"),
    "reingest_raw": ("boolean", "Also re-read raw data (default false; only changed files are re-read anyway)"),
    "query": ("string", "Free-text search query"), "top_k": ("integer", "Number of passages"),
}
_FUNCS = [list_wells, get_well_summary, get_log_stats, get_volume_info, sample_volumes_at_well,
          crosscheck_well, compare_wells, rank_wells, stack_amplitudes_at_well, lateral_variability,
          render_slice, search_knowledge,
          logs_to_time, find_gas_zones, thickest_gas_zone, compare_zone_to_seismic, extrapolate_zone_to_map,
          find_prospects, structure_and_closure, depositional_pattern, trap_evidence,
          horizon_contour_and_faults, horizon_slice, overlay_structure_and_slice, high_variation_slices, count_faults,
          ai_fault_detection, ai_facies_map,
          list_datasets, scan_for_new_data, interpreted_faults, interpreted_facies, horizon_structure, interpreted_map,
          well_context, validate_interpreted_vs_seismic, assess_traps, rank_prospects, recommend_trap, volume_math, gas_potential]
TOOLS = {f.__name__: f for f in _FUNCS}


def _schema(fn):
    props, req = {}, []
    for n, p in inspect.signature(fn).parameters.items():
        t, d = PARAMS[n]
        props[n] = {"type": t, "description": d}
        if n == "kind":
            props[n]["enum"] = ["inline", "xline", "time"]
        if p.default is inspect.Parameter.empty:
            req.append(n)
    return {"type": "function", "function": {"name": fn.__name__, "description": inspect.getdoc(fn),
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


TOOL_SCHEMAS = [_schema(f) for f in _FUNCS]
