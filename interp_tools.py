"""
interp_tools.py - reasoning tools over INTERPRETED data (fault/polygon shapefiles, horizon) cross-checked with the RAW data.

Each function returns compact JSON for Gemma and (usually) a PNG map. Nothing here is hard-coded to one dataset:
tools read whatever interp_data.scan() has ingested, and they say clearly when something is missing.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

import ingest as I
import interp_data as D

FAULT_PAD_MS = 250.0     # fault slices within this many ms outside the window still count (faults cut many ms)
POLY_PAD_MS = 20.0       # stratigraphic polygons belong to their own time slice
_WIN = [None, None]      # active TWT window (ms) for the interpreted-data tools; set by assess_traps / recommend_trap


def _in_win(t):
    return t is None or ((_WIN[0] is None or t >= _WIN[0]) and (_WIN[1] is None or t <= _WIN[1]))


def _rec_in_win(r):
    """Slices at one time must fall in the window; section sticks (t_range) must overlap it."""
    pad = FAULT_PAD_MS if r.get("kind") == "fault" else POLY_PAD_MS     # faults are vertical features; polygons belong to their slice
    tr = r.get("t_range")
    if tr:
        return (_WIN[0] is None or tr[1] >= _WIN[0] - pad) and (_WIN[1] is None or tr[0] <= _WIN[1] + pad)
    t = r.get("time_ms")
    return t is None or ((_WIN[0] is None or t >= _WIN[0] - pad) and (_WIN[1] is None or t <= _WIN[1] + pad))


def _slices(kind, cls=None):
    """Interpreted slices inside the active window (slices with unknown time are kept)."""
    return [r for r in D.slices(kind, cls) if _rec_in_win(r)]


SAND_WORDS = ("channel", "lobe", "fan", "bar", "sand", "splay", "crevasse", "mound")
SEAL_WORDS = ("overbank", "levee", "shale", "floodplain", "mud")


# ------------------------------------------------------------------ helpers
def _f(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _why_empty():
    """Ingestion warnings explaining why nothing usable was found."""
    p = I.CACHE_DIR / "interpreted_cards.json"
    try:
        return json.loads(p.read_text()).get("warnings", [])[:8]
    except Exception:
        return []


def _need(kind):
    if not D.interpreted_ready():
        return {"error": "no interpreted data ingested yet. Set INTERP_DIR (current: %s) and run ingest_all.run_all(force=True)" % (D.interp_dir(),),
                "ingestion_warnings": _why_empty()}
    return None


def _is_sand(cls):
    return any(w in cls for w in SAND_WORDS)


def _is_seal(cls):
    return any(w in cls for w in SEAL_WORDS)


def _vol_for_xy():
    try:
        c = I.load_cards()
        names = [k for k, v in c["volumes"].items() if v["kind"] == "inversion"] or list(c["volumes"])
        return I.open_volume(names[0]) if names else None
    except Exception:
        return None


def xy_to_ilxl(x, y):
    v = _vol_for_xy()
    if v is None:
        return None
    d, i = v.tree.query([x, y])
    if d > 4 * v.max_dist:
        return None
    return [int(v.il[i]), int(v.xl[i])]


def _study_bbox():
    h = D.horizon()
    if h is not None:
        return [h["gx"][0], h["gy"][0], h["gx"][-1], h["gy"][-1]]
    arrs = [r for s in D._state()["data"].values() if s["type"] == "shp" for f in s["feats"] for r in f["rings"]]
    return D.bbox_of(arrs) if arrs else None


def _poly_rings(rec):
    """Rings of a polygon slice as (ring, signed_area) with the convention outer = clockwise (area <= 0), holes = counter-clockwise.
    Winding differs between GIS exports, so the sign of the largest ring in the slice is taken as 'outer' and rings are flipped to match."""
    rings = [r for f in rec["feats"] for r in f["rings"] if len(r) >= 3]
    if not rings:
        return []
    areas = [D.ring_signed_area(r) for r in rings]
    outer_sign = np.sign(areas[int(np.argmax(np.abs(areas)))]) or -1.0
    out = []
    for r, a in zip(rings, areas):
        is_outer = np.sign(a) == outer_sign
        cw = a <= 0
        if is_outer and not cw:          # outer ring wound CCW -> make it CW
            r, a = r[::-1], -a
        elif (not is_outer) and cw:      # hole wound CW -> make it CCW
            r, a = r[::-1], -a
        out.append((r, a))
    return out


# ---------------------------------------------------------------- faults
def _fault_lines(twt_min=None, twt_max=None):
    lo = _WIN[0] if twt_min is None else twt_min
    hi = _WIN[1] if twt_max is None else twt_max
    lines = []
    for rec in _slices("fault"):
        t = rec.get("time_ms")
        if rec.get("t_range") is None and t is not None and ((lo is not None and t < lo) or (hi is not None and t > hi)):
            continue
        if rec["geom"] == "point":
            pts = np.vstack([r for f in rec["feats"] for r in f["rings"]])
            lines.append({"slice": rec["key"], "time_ms": t, "coords": pts, "attrs": {}})
            continue
        for f in rec["feats"]:
            for r in f["rings"]:
                tv = f.get("t")
                if tv is not None and len(tv) == len(r) and (lo is not None or hi is not None):
                    m = np.ones(len(r), bool)
                    if lo is not None:
                        m &= tv >= lo
                    if hi is not None:
                        m &= tv <= hi
                    r = r[m]
                if len(r) >= 2:
                    lines.append({"slice": rec["key"], "time_ms": t, "coords": r, "attrs": f["attrs"]})
    return lines


def _default_tol():
    bb = _study_bbox()
    if bb is None:
        return 50.0
    diag = math.hypot(bb[2] - bb[0], bb[3] - bb[1])
    return float(np.clip(0.01 * diag, 20.0, 250.0))


def link_faults(lines, tol):
    """Group fault lines from different slices into unique faults (attribute identity if present, else geometric overlap)."""
    n = len(lines)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)
    idkey = None
    for k in (lines[0]["attrs"] if n else {}):
        if str(k).lower() in ("fault_id", "faultid", "fault_name", "faultname", "fault", "name", "fault_no", "id_fault"):
            vals = [str(l["attrs"].get(k)) for l in lines]
            if 1 < len(set(vals)) < n:
                idkey = k
                break
    if idkey:
        first = {}
        for i, l in enumerate(lines):
            v = str(l["attrs"].get(idkey))
            if v in first:
                union(i, first[v])
            else:
                first[v] = i
        method = f"attribute '{idkey}'"
    else:
        samples = [D.resample(l["coords"], tol / 2) for l in lines]
        P = np.vstack(samples)
        ids = np.concatenate([[i] * len(s) for i, s in enumerate(samples)])
        tree = cKDTree(P)
        for i, s in enumerate(samples):
            hits = tree.query_ball_point(s, r=tol)
            cnt = {}
            for h in hits:
                for j in set(ids[h].tolist()):
                    if j != i and lines[j]["slice"] != lines[i]["slice"]:
                        cnt[j] = cnt.get(j, 0) + 1
            for j, c in cnt.items():
                if c / len(s) >= 0.3:
                    union(i, j)
        method = f"geometric overlap within {tol:.0f} map units across slices"
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), method


def interpreted_faults(twt_min_ms=None, twt_max_ms=None, link_tolerance=None):
    """Count and describe the INTERPRETED faults (from fault shapefiles): per-slice counts, unique faults after linking slices, strike, length, location (x,y, inline/xline), time span, and a map."""
    e = _need("fault")
    if e:
        return e
    lines = _fault_lines(twt_min_ms, twt_max_ms)
    if not lines:
        return {"error": "no interpreted fault features found (check Fault (SHP)/Fault (CSV) folders and the time filter)",
                "ingestion_warnings": _why_empty(), "interpreted_dir": str(D.interp_dir()),
                "hint": "run ingest_all.run_all(force=True) and read the warnings; for CSV faults the raw data must be ingested first so inline/xline can be converted to X/Y"}
    tol = float(link_tolerance) if link_tolerance else _default_tol()
    groups, method = link_faults(lines, tol)
    per_slice = {}
    for l in lines:
        per_slice.setdefault(l["slice"], {"time_ms": l["time_ms"], "n": 0, "length": 0.0})
        per_slice[l["slice"]]["n"] += 1
        per_slice[l["slice"]]["length"] += D.line_length(l["coords"])
    faults = []
    for g in groups:
        pts = np.vstack([D.resample(lines[i]["coords"], tol / 2) for i in g])
        az, major, minor, el = D.pca_axes(pts)
        times = [lines[i]["time_ms"] for i in g if lines[i]["time_ms"] is not None]
        c = pts.mean(0)
        faults.append({"n_slices": len({lines[i]["slice"] for i in g}),
                       "max_trace_length": _f(max(D.line_length(lines[i]["coords"]) for i in g), 0),
                       "strike_deg_from_north": _f(az, 0), "centre_xy": [_f(c[0], 0), _f(c[1], 0)],
                       "centre_inline_xline": xy_to_ilxl(c[0], c[1]),
                       "time_span_ms": [_f(min(times), 0), _f(max(times), 0)] if times else None,
                       "attrs": {k: v for k, v in list(lines[g[0]]["attrs"].items())[:3]}})
    faults.sort(key=lambda f: -(f["max_trace_length"] or 0))
    for i, f in enumerate(faults):
        f["id"] = f"F{i + 1}"
    strikes = np.array([f["strike_deg_from_north"] for f in faults], float)
    img = interpreted_map()
    return {"source": "interpreted fault shapefiles", "n_slices": len(per_slice), "n_fault_traces_total": len(lines),
            "n_unique_faults": len(faults), "linking_method": method, "link_tolerance": _f(tol, 0),
            "per_slice": [{"slice": k, **{a: _f(b, 0) if a != "n" else b for a, b in v.items()}} for k, v in per_slice.items()],
            "faults": faults[:15],
            "dominant_strike_deg_from_north": _f(np.degrees(np.angle(np.mean(np.exp(2j * np.radians(strikes))))) / 2 % 180, 0) if len(strikes) else None,
            "image_path": img.get("image_path"), "kind": "map", "index_used": "interpreted faults",
            "caveat": "Unique-fault count depends on the linking rule; if features carry a fault ID column it is used instead of geometry."}


# --------------------------------------------------------------- polygons
def interpreted_facies(class_name=None):
    """Summarise interpreted stratigraphic polygons (channel, overbank, ...): count, area, shape (length, width, sinuosity, orientation) per slice and class, trend through time, and the most likely depositional element."""
    e = _need("polygon")
    if e:
        return e
    classes = [class_name.lower()] if class_name else D.polygon_classes()
    bb = _study_bbox()
    area_ref = (bb[2] - bb[0]) * (bb[3] - bb[1]) if bb else None
    out = {"classes": {}}
    for cls in classes:
        recs = _slices("polygon", cls)
        if not recs:
            continue
        rows = []
        for r in recs:
            rings = _poly_rings(r)
            area = D.polygon_area([x for x, _ in rings]) if rings else 0.0
            row = {"slice": r["key"], "time_ms": _f(r.get("time_ms"), 0), "n_polygons": len(r["feats"]), "area": _f(area, 0),
                   "area_fraction_of_study": _f(area / area_ref, 3) if area_ref else None}
            if rings:
                big = max(rings, key=lambda t: abs(t[1]))[0]
                az, major, minor, el = D.pca_axes(big)
                per = D.line_length(np.vstack([big, big[:1]]))
                row.update({"largest_azimuth_deg": _f(az, 0), "largest_length": _f(major, 0), "largest_width": _f(minor, 0),
                            "elongation": _f(el, 1), "sinuosity": _f((per / 2) / max(major, 1e-9), 2)})
            rows.append(row)
        out["classes"][cls] = {"role": "reservoir-prone" if _is_sand(cls) else "seal/fine-grained" if _is_seal(cls) else "unclassified",
                               "n_slices": len(recs), "slices": rows}
    if not out["classes"]:
        return {"error": f"no polygon class '{class_name}'", "available": D.polygon_classes()}
    chan = [c for c in out["classes"] if _is_sand(c)]
    inf = []
    if chan:
        rws = [r for r in out["classes"][chan[0]]["slices"] if r.get("elongation")]
        if rws:
            el = float(np.mean([r["elongation"] for r in rws]))
            sn = float(np.mean([r["sinuosity"] for r in rws]))
            seal = [c for c in out["classes"] if _is_seal(c)]
            marine = False
            try:
                marine = all("marine" in str(w.get("type", "")).lower() for w in I.load_cards()["wells"].values())
            except Exception:
                pass
            shape = ("highly sinuous (meandering)" if sn > 1.3 else "moderately sinuous" if sn > 1.1 else "straight to low-sinuosity")
            if el >= 3:
                elem = f"{shape} channel belt" + (" with overbank/levee fines" if seal else "")
                env = ("deep-water (slope) channel-levee system" if marine else "fluvial/distributary channel system")
                inf.append(f"{elem}; elongation {el:.1f}, sinuosity {sn:.2f}")
                inf.append(f"likely depositional environment: {env}" + (" (all wells are Marine)" if marine else ""))
            else:
                inf.append(f"sand body is equant (elongation {el:.1f}): lobe/fan or sheet-like rather than a channel")
    out["interpretation"] = inf
    out["caveat"] = "Environment is inferred from polygon geometry only; confirm with cores, logs and regional context."
    img = interpreted_map()
    out.update({"image_path": img.get("image_path"), "kind": "map", "index_used": "interpreted polygons"})
    return out


# ---------------------------------------------------------------- horizon
def _prep_horizon(windowed=True):
    h = D.horizon()
    if h is None:
        return None
    H = h["H"]
    if windowed and (_WIN[0] is not None or _WIN[1] is not None):
        lo = -np.inf if _WIN[0] is None else _WIN[0]
        hi = np.inf if _WIN[1] is None else _WIN[1]
        H = np.where((H >= lo) & (H <= hi), H, np.nan)
        if not np.isfinite(H).any():
            return None
    valid = np.isfinite(H)
    fill = np.nanmean(H)
    Hs = ndi.gaussian_filter(np.where(valid, H, fill), 1.0)
    Hs = np.where(valid, Hs, np.nan)
    gy_, gx_ = np.gradient(np.where(valid, Hs, fill), h["cell"])
    gy_, gx_ = np.where(valid, gy_, np.nan), np.where(valid, gx_, np.nan)
    interp = lambda arr: RegularGridInterpolator((h["gy"], h["gx"]), arr, bounds_error=False, fill_value=np.nan)
    return {**h, "valid": valid, "Hs": Hs, "ddy": gy_, "ddx": gx_, "iH": interp(Hs), "iGx": interp(gx_), "iGy": interp(gy_)}


def find_closures(P, min_height=None, max_cand=12):
    Hs, valid, cell = P["Hs"], P["valid"], P["cell"]
    relief = float(np.nanmax(Hs) - np.nanmin(Hs))
    min_h = min_height if min_height is not None else max(5.0, 0.05 * relief)
    hn = np.where(valid, Hs, np.nanmax(Hs))
    mins = (hn == ndi.minimum_filter(hn, size=7)) & valid
    cands = sorted(zip(*np.where(mins)), key=lambda rc: hn[rc])[:max_cand]
    taken = np.zeros(hn.shape, bool)
    out = []
    for r, c in cands:
        if taken[r, c]:
            continue
        h0, prev, region, open_ = hn[r, c], None, None, False
        for lvl in np.linspace(h0, h0 + relief, 90)[1:]:
            lab, _ = ndi.label(hn <= lvl)
            reg = lab == lab[r, c]
            if (reg[0, :].any() or reg[-1, :].any() or reg[:, 0].any() or reg[:, -1].any()
                    or (ndi.binary_dilation(reg, structure=np.ones((3, 3))) & ~valid).any()):
                open_ = True
                break
            prev, region = lvl, reg
        height = (prev - h0) if prev is not None else 0.0
        if region is None or height < min_h:
            continue
        taken |= region
        ys, xs = np.where(region)
        out.append({"crest_t": float(h0), "crest_xy": [float(P["gx"][c]), float(P["gy"][r])], "spill_t": float(prev), "height_ms": float(height),
                    "area": float(region.sum() * cell ** 2), "region": region, "cell_rc": (int(r), int(c)),
                    "closed": True, "bounded_by_map_edge_above": bool(open_)})
    return out, relief, min_h


def _fault_tree(t_ref=None, window=80.0):
    lines = _fault_lines()
    if t_ref is not None:
        near = [l for l in lines if l["time_ms"] is not None and abs(l["time_ms"] - t_ref) <= window]
        if near:
            lines = near
    if not lines:
        return None, 0
    ds = _default_tol() / 3
    P = np.vstack([D.resample(l["coords"], ds) for l in lines])
    return cKDTree(P), len(lines)


def _fault_bound_fraction(P, region, tol, tree):
    if tree is None:
        return None
    edge = region & ~ndi.binary_erosion(region)
    ys, xs = np.where(edge)
    if not len(xs):
        return 0.0
    pts = np.c_[P["gx"][xs], P["gy"][ys]]
    d, _ = tree.query(pts)
    return float((d <= 2 * tol).mean())


def horizon_structure():
    """Describe the interpreted horizon: time range, relief, structural highs and closures (crest, spill, height ms, area), dip, and how faults bound each closure; draws contours with faults, channels and wells."""
    P = _prep_horizon()
    if P is None:
        return {"error": "no horizon file ingested (expected x y time text file in a 'Horizon' folder or *.xyt)"}
    cl, relief, min_h = find_closures(P)
    tol = _default_tol()
    gmag = np.hypot(P["ddx"], P["ddy"])
    out = {"horizon": P["name"], "n_points": int(len(P["t"])), "time_range_ms": [_f(np.nanmin(P["t"]), 0), _f(np.nanmax(P["t"]), 0)],
           "relief_ms": _f(relief, 0), "mean_dip_ms_per_km": _f(np.nanmean(gmag) * 1000, 1), "max_dip_ms_per_km": _f(np.nanmax(gmag) * 1000, 1),
           "min_closure_height_ms_used": _f(min_h, 1), "notes": P["notes"], "closures": []}
    for i, c in enumerate(cl):
        tree, nf = _fault_tree(c["crest_t"])
        fb = _fault_bound_fraction(P, c["region"], tol, tree)
        out["closures"].append({"id": f"C{i + 1}", "crest_xy": [_f(v, 0) for v in c["crest_xy"]], "crest_inline_xline": xy_to_ilxl(*c["crest_xy"]),
                                "crest_twt_ms": _f(c["crest_t"], 0), "spill_twt_ms": _f(c["spill_t"], 0), "closure_height_ms": _f(c["height_ms"], 1),
                                "area_km2_if_metres": _f(c["area"] / 1e6, 3), "fault_bounded_fraction_of_outline": _f(fb, 2) if fb is not None else None})
    out["n_closed_highs"] = len(cl)
    img = interpreted_map()
    out.update({"image_path": img.get("image_path"), "kind": "map", "index_used": "horizon structure"})
    return out


# ------------------------------------------------------------------ map
def interpreted_map(slice_twt_ms=None):
    """Draw everything interpreted on one map: horizon contours, fault traces, channel/overbank polygons and wells (all slices, or the slice nearest a given TWT)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (I.CACHE_DIR / "images").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    P = _prep_horizon()
    if P is not None:
        im = ax.imshow(P["Hs"], origin="lower", extent=[P["gx"][0], P["gx"][-1], P["gy"][0], P["gy"][-1]], cmap="viridis_r", aspect="equal")
        plt.colorbar(im, ax=ax, label="horizon TWT (ms)", shrink=0.8)
        ax.contour(P["gx"], P["gy"], P["Hs"], levels=14, colors="white", linewidths=0.5)
    cmap = {}
    for i, cls in enumerate(D.polygon_classes()):
        cmap[cls] = "gold" if _is_sand(cls) else "saddlebrown" if _is_seal(cls) else f"C{i}"
    for cls in D.polygon_classes():
        recs = _slices("polygon", cls)
        if slice_twt_ms is not None:
            with_t = [r for r in recs if r.get("time_ms") is not None]
            if with_t:
                recs = [min(with_t, key=lambda r: abs(r["time_ms"] - slice_twt_ms))]
        for r in recs:
            for ring, _ in _poly_rings(r):
                ax.fill(ring[:, 0], ring[:, 1], color=cmap[cls], alpha=0.25 if len(recs) > 1 else 0.5, lw=0.5)
    lines = _fault_lines()
    if slice_twt_ms is not None:
        with_t = [l for l in lines if l["time_ms"] is not None]
        if with_t:
            tn = min(with_t, key=lambda l: abs(l["time_ms"] - slice_twt_ms))["time_ms"]
            lines = [l for l in lines if l["time_ms"] == tn]
    for l in lines:
        ax.plot(l["coords"][:, 0], l["coords"][:, 1], "r-", lw=1.1, alpha=0.7)
    try:
        for w, c in I.load_cards()["wells"].items():
            ax.plot(c["x"], c["y"], "c^", ms=7, mec="k")
            ax.text(c["x"], c["y"], " " + w, color="cyan", fontsize=8)
    except Exception:
        pass
    xs, ys = [], []
    for l in _fault_lines():
        xs += [l["coords"][:, 0].min(), l["coords"][:, 0].max()]
        ys += [l["coords"][:, 1].min(), l["coords"][:, 1].max()]
    for r in _slices("polygon"):
        for ring, _ in _poly_rings(r):
            xs += [ring[:, 0].min(), ring[:, 0].max()]
            ys += [ring[:, 1].min(), ring[:, 1].max()]
    if P is not None:
        xs += [P["gx"][0], P["gx"][-1]]
        ys += [P["gy"][0], P["gy"][-1]]
    if xs and ys:                                   # zoom to the interpreted data, not to stray points
        mx, my = 0.05 * (max(xs) - min(xs) + 1), 0.05 * (max(ys) - min(ys) + 1)
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - my, max(ys) + my)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="r", lw=1.5, label="faults")] + [Line2D([0], [0], marker="s", color="w", markerfacecolor=c, label=k, ms=9) for k, c in cmap.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Interpreted horizon + faults + stratigraphic polygons" + (f" (slice nearest {slice_twt_ms:.0f} ms)" if slice_twt_ms else " (all slices)"))
    path = I.CACHE_DIR / "images" / f"interpreted_map{'' if slice_twt_ms is None else '_' + str(int(slice_twt_ms))}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"image_path": str(path), "kind": "map", "index_used": "interpreted overlay", "volume": "interpreted"}


# --------------------------------------------------------------- wells
def well_context(well):
    """What interpreted features surround one well: distance to nearest fault, which polygon classes (channel/overbank) it sits in per slice, horizon time at the well, and its gas zone (if logs exist)."""
    e = _need("")
    if e:
        return e
    if not I.cache_ready():
        return {"error": "raw well data not ingested (DATA_DIR missing); well context needs well locations"}
    w = I.resolve_well(well)
    c = I.load_cards()["wells"][w]
    x, y = c["x"], c["y"]
    out = {"well": w, "x": x, "y": y, "inline_xline": xy_to_ilxl(x, y)}
    P = _prep_horizon()
    if P is not None:
        out["horizon_twt_ms_at_well"] = _f(P["iH"]([[y, x]])[0], 0)
    lines = _fault_lines()
    if lines:
        best = None
        for l in lines:
            pts = D.resample(l["coords"], _default_tol() / 3)
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y).min()
            if best is None or d < best[0]:
                best = (d, l["slice"])
        out["nearest_fault_distance"] = _f(best[0], 0)
        out["nearest_fault_slice"] = best[1]
    mem = {}
    for r in _slices("polygon"):
        inside = any(MPath(ring).contains_point((x, y)) for ring, sa in _poly_rings(r) if sa <= 0)
        mem.setdefault(r["cls"], []).append({"slice": r["key"], "time_ms": _f(r.get("time_ms"), 0), "inside": bool(inside)})
    out["polygon_membership"] = {k: {"inside_in_n_slices": sum(v["inside"] for v in vs), "of": len(vs), "detail": vs[:6]} for k, vs in mem.items()}
    try:
        import interpret as N
        z, used = N._zones(w, "gas")
        if z:
            out["thickest_gas_zone"] = z[0]
            tm = (z[0]["top_twt_ms"] + z[0]["base_twt_ms"]) / 2
            rs = [r for r in _slices("polygon") if r.get("time_ms") is not None]
            if rs:
                near = {}
                for r in rs:
                    near.setdefault(r["cls"], []).append(r)
                out["facies_at_gas_zone_time"] = {cls: {"slice": (m := min(v, key=lambda r: abs(r["time_ms"] - tm)))["key"],
                                                         "inside": any(MPath(ring).contains_point((x, y)) for ring, sa in _poly_rings(m) if sa <= 0)}
                                                  for cls, v in near.items()}
    except Exception as ex:
        out["gas_zone_note"] = f"not available ({ex})"
    return out


# ------------------------------------------------- validation vs raw data
def _cell_xy(mp):
    v = mp["vol"]
    r = np.searchsorted(mp["il"], v.il[mp["sel"]])
    c = np.searchsorted(mp["xl"], v.xl[mp["sel"]])
    X, Y = np.full(mp["grid"].shape, np.nan), np.full(mp["grid"].shape, np.nan)
    X[r, c], Y[r, c] = v.x[mp["sel"]], v.y[mp["sel"]]
    return X, Y


def _inside_any(xy, rings):
    m = np.zeros(len(xy), bool)
    for ring in rings:
        m |= MPath(ring).contains_points(xy)
    return m


def _detected_fault_mask():
    import interpret as N
    import ai_interp as A
    r = A.ai_fault_detection()          # falls back to coherence internally
    return N.LAST_FAULTS, N.LAST_FAULTS.get("method", "coherence"), r


def validate_interpreted_vs_seismic():
    """Cross-check the interpreted data against the raw seismic: (a) mean sand probability inside channel vs overbank vs outside polygons, (b) agreement between interpreted faults and the automatic (AI/coherence) fault detection."""
    e = _need("")
    if e:
        return e
    if not I.cache_ready():
        return {"error": "raw seismic not ingested (DATA_DIR missing), so there is nothing to validate against; interpreted-only reasoning is still available via assess_traps"}
    import interpret as N
    out = {"polygon_vs_sand_probability": None, "faults_vs_automatic_detection": None}
    sv = N._vol_name(["sand probability"])
    if sv and _slices("polygon"):
        res, assumed = {}, False
        P = D.horizon()
        for cls in D.polygon_classes():
            vals_in = []
            for r in _slices("polygon", cls):
                t = r.get("time_ms")
                if t is None:
                    if P is None:
                        continue
                    t, assumed = float(np.nanmedian(P["t"])), True
                try:
                    mp = N._read_maps(sv, t - 10, t + 10, max_traces=20000)
                except Exception:
                    continue
                X, Y = _cell_xy(mp)
                ok = np.isfinite(mp["grid"]) & np.isfinite(X)
                xy = np.c_[X[ok], Y[ok]]
                inside = _inside_any(xy, [ring for ring, sa in _poly_rings(r) if sa <= 0])
                vals = mp["grid"][ok]
                if inside.sum() >= 3:
                    vals_in.append((vals[inside].mean(), vals[~inside].mean() if (~inside).any() else np.nan, int(inside.sum())))
            if vals_in:
                res[cls] = {"mean_sand_probability_inside": _f(np.mean([v[0] for v in vals_in]), 3),
                            "mean_sand_probability_outside": _f(np.nanmean([v[1] for v in vals_in]), 3), "n_slices_used": len(vals_in)}
        if res:
            out["polygon_vs_sand_probability"] = res
            if assumed:
                out["polygon_vs_sand_probability_note"] = "slice times unknown: horizon median time +-10 ms used as a proxy"
            sc = [c for c in res if _is_sand(c)]
            sl = [c for c in res if _is_seal(c)]
            if sc and sl:
                d = res[sc[0]]["mean_sand_probability_inside"] - res[sl[0]]["mean_sand_probability_inside"]
                out["consistent_with_seismic"] = bool(d > 0.05)
                out["sand_minus_seal_sand_probability"] = _f(d, 3)
    lines = _fault_lines()
    if lines:
        try:
            last, method, r = _detected_fault_mask()
            seg, det = last["seg"], last["mask"]
            v = seg["vol"]
            il_axis, xl_axis = seg["il"], seg["xl"]
            ref = np.zeros(det.shape, bool)
            for l in lines:
                for p in D.resample(l["coords"], seg["spacing"] / 2):
                    d, i = v.tree.query(p)
                    if d <= 3 * v.max_dist:
                        rr = int(np.clip(np.searchsorted(il_axis, v.il[i]), 0, det.shape[0] - 1))
                        cc = int(np.clip(np.searchsorted(xl_axis, v.xl[i]), 0, det.shape[1] - 1))
                        ref[rr, cc] = True
            tolc = 2
            near_ref, near_det = ndi.binary_dilation(ref, iterations=tolc), ndi.binary_dilation(det, iterations=tolc)
            out["faults_vs_automatic_detection"] = {
                "detector": method, "automatic_fault_pixels": int(det.sum()), "interpreted_fault_pixels": int(ref.sum()),
                "share_of_detected_near_interpreted": _f((det & near_ref).sum() / max(det.sum(), 1), 2),
                "share_of_interpreted_near_detected": _f((ref & near_det).sum() / max(ref.sum(), 1), 2),
                "detected_fault_count": r.get("n_faults", r.get("n_fault_candidates", r.get("n_faults_detected")))}
        except Exception as ex:
            out["faults_vs_automatic_detection"] = {"error": f"{type(ex).__name__}: {ex}"}
    return out


# --------------------------------------------------------- trap reasoning
def _strat_analysis(P, region_zone=None):
    """Updip termination of sand-prone polygons against fine-grained polygons and their orientation to dip."""
    ds = P["cell"] * 2
    gm = np.hypot(P["ddx"], P["ddy"])
    gmin = np.nanpercentile(gm, 25)
    seal_rings = [ring for r in _slices("polygon") if _is_seal(r["cls"]) for ring, sa in _poly_rings(r) if sa <= 0]
    tot = upd = seal_adj = 0
    per_slice, axis_dips = [], []
    pts_upd = []
    for r in _slices("polygon"):
        if not _is_sand(r["cls"]):
            continue
        s_tot = s_upd = s_seal = 0
        for ring, sa in _poly_rings(r):
            if sa > 0:
                continue
            pts = D.resample(np.vstack([ring, ring[:1]]), ds)
            if len(pts) < 4:
                continue
            tang = np.gradient(pts, axis=0)
            tn = np.hypot(tang[:, 0], tang[:, 1]) + 1e-12
            outward = np.c_[-tang[:, 1], tang[:, 0]] / tn[:, None]          # clockwise ring: outward = left normal
            qxy = np.c_[pts[:, 1], pts[:, 0]]
            gx_, gy_ = P["iGx"](qxy), P["iGy"](qxy)
            gmag = np.hypot(gx_, gy_)
            ok = np.isfinite(gmag) & (gmag > gmin)
            up = -np.c_[gx_, gy_] / np.where(gmag > 0, gmag, 1)[:, None]
            dot = (outward * up).sum(1)
            isup = ok & (dot > 0.5)
            s_tot += int(ok.sum())
            s_upd += int(isup.sum())
            if isup.any() and seal_rings:
                probe = pts[isup] + outward[isup] * ds * 0.75
                s_seal += int(_inside_any(probe, seal_rings).sum())
            pts_upd.append(pts[isup])
            az, major, minor, el = D.pca_axes(ring)
            cen = ring.mean(0)
            g = np.array([P["iGx"]([[cen[1], cen[0]]])[0], P["iGy"]([[cen[1], cen[0]]])[0]])
            if np.all(np.isfinite(g)) and np.hypot(*g) > 0:
                dipaz = math.degrees(math.atan2(g[0], g[1])) % 180
                d = abs(az - dipaz) % 180
                axis_dips.append(min(d, 180 - d))
        tot, upd, seal_adj = tot + s_tot, upd + s_upd, seal_adj + s_seal
        per_slice.append({"slice": r["key"], "updip_bounded_fraction": _f(s_upd / s_tot, 2) if s_tot else None})
    pu = np.vstack(pts_upd) if pts_upd and sum(len(p) for p in pts_upd) else np.empty((0, 2))
    return {"updip_bounded_fraction": upd / tot if tot else None, "updip_bounded_with_seal_neighbour_fraction": seal_adj / upd if upd else None,
            "mean_angle_channel_axis_vs_dip_deg": float(np.mean(axis_dips)) if axis_dips else None, "per_slice": per_slice, "updip_points": pu}


def assess_traps(twt_min_ms=None, twt_max_ms=None, cross_check=True):
    """MAIN TRAP TOOL. Scores structural (four-way anticline, fault-dependent) and stratigraphic (channel pinch-out / updip facies change) trapping from the interpreted horizon, faults and polygons, cross-checked with raw seismic and wells; lists unconformity/reef as not assessable when data is missing; returns the leading trap, confidence, risks and next steps."""
    e = _need("")
    if e:
        return e
    _WIN[0], _WIN[1] = (None if twt_min_ms is None else float(twt_min_ms)), (None if twt_max_ms is None else float(twt_max_ms))
    try:
        return _assess_traps(cross_check)
    finally:
        _WIN[0] = _WIN[1] = None


def _assess_traps(cross_check):
    window_note = None
    P = _prep_horizon()
    if P is None and (_WIN[0] is not None or _WIN[1] is not None):
        h = D.horizon()
        if h is None:
            return {"error": "a horizon is needed for trap assessment (none ingested)"}
        window_note = (f"The interpreted horizon spans {np.nanmin(h['t']):.0f}-{np.nanmax(h['t']):.0f} ms TWT and has no points inside the requested "
                       f"{_WIN[0]}-{_WIN[1]} ms window; the whole horizon was used instead. Check that the window is in the same time units.")
        lo, hi = _WIN
        _WIN[0] = _WIN[1] = None
        P = _prep_horizon()
        _WIN[0], _WIN[1] = lo, hi
    if P is None:
        return {"error": "a horizon is needed for trap assessment (none ingested)"}
    tol = _default_tol()
    cl, relief, min_h = find_closures(P)
    ev, notes = {}, []
    # --- A/B structural
    if cl:
        best = max(cl, key=lambda c: c["height_ms"] * c["area"])
        tree, nf = _fault_tree(best["crest_t"])
        fb = _fault_bound_fraction(P, best["region"], tol, tree)
        cut = 0
        if tree is not None:
            ys, xs = np.where(best["region"])
            cut = int(np.sum(tree.query(np.c_[P["gx"][xs], P["gy"][ys]])[0] <= tol))
        sA = min(1.0, 0.5 + 0.5 * min(1.0, best["height_ms"] / (0.3 * relief + 1e-9))) * (1 - 0.5 * (fb or 0))
        ev["four_way_anticline"] = {"status": "SUPPORTED" if (fb or 0) < 0.25 else "PARTLY (outline largely fault-bounded)", "score": _f(sA),
                                    "evidence": {"closure_height_ms": _f(best["height_ms"], 1), "crest_twt_ms": _f(best["crest_t"], 0), "spill_twt_ms": _f(best["spill_t"], 0),
                                                 "area_km2_if_metres": _f(best["area"] / 1e6, 3), "crest_inline_xline": xy_to_ilxl(*best["crest_xy"]),
                                                 "fault_bounded_fraction_of_outline": _f(fb, 2) if fb is not None else None}}
        if fb is not None and fb >= 0.25:
            ev["fault_dependent_closure"] = {"status": "SUPPORTED", "score": _f(min(1.0, 0.4 + fb)),
                                             "evidence": {"fault_bounded_fraction_of_outline": _f(fb, 2), "closure_cells_within_fault_tolerance": cut,
                                                          "note": "seal capacity of the fault (throw, juxtaposition, shale gouge) cannot be assessed from this data"}}
        else:
            ev["fault_dependent_closure"] = {"status": "NOT SUPPORTED" if tree is not None else "NOT AVAILABLE (no faults)", "score": 0.0,
                                             "evidence": {"fault_bounded_fraction_of_outline": _f(fb, 2) if fb is not None else None}}
    else:
        ev["four_way_anticline"] = {"status": "NOT SUPPORTED", "score": 0.0, "evidence": {"closures_found": 0, "min_height_ms_used": _f(min_h, 1)}}
        tree, _ = _fault_tree()
        ev["fault_dependent_closure"] = {"status": "NOT AVAILABLE (no closed high to test)", "score": 0.0, "evidence": {}}
    # --- C stratigraphic
    if any(_is_sand(r["cls"]) for r in _slices("polygon")):
        st = _strat_analysis(P)
        uf = st["updip_bounded_fraction"]
        sf = st["updip_bounded_with_seal_neighbour_fraction"]
        sC = 0.0 if uf is None else min(1.0, 0.3 + 1.5 * uf) * (0.6 + 0.4 * (sf or 0))
        ev["stratigraphic_updip_pinchout"] = {"status": "SUPPORTED" if (uf or 0) >= 0.2 else "WEAK" if (uf or 0) > 0.05 else "NOT SUPPORTED", "score": _f(sC if (uf or 0) > 0.05 else 0.1),
                                              "evidence": {"share_of_sand_body_outline_that_terminates_updip": _f(uf, 2),
                                                           "share_of_that_outline_next_to_fine_grained_polygons": _f(sf, 2) if sf is not None else "no seal-class polygons",
                                                           "mean_angle_between_channel_axis_and_dip_deg": _f(st["mean_angle_channel_axis_vs_dip_deg"], 0),
                                                           "reading": "small angle = body runs along dip, so its updip end can pinch out; ~90 = runs along strike"}}
    else:
        ev["stratigraphic_updip_pinchout"] = {"status": "NOT AVAILABLE (no sand-prone polygons)", "score": 0.0, "evidence": {}}
    # --- D/E not assessable
    ev["unconformity"] = {"status": "NOT AVAILABLE", "score": 0.0, "evidence": {"why": "needs two or more horizons showing truncation/onlap; only one horizon ingested"}}
    reef = [c for c in D.polygon_classes() if "reef" in c or "mound" in c]
    ev["reef_buildup"] = {"status": "NOT AVAILABLE" if not reef else "DATA PRESENT (use interpreted_facies)", "score": 0.0, "evidence": {"why": "no reef/mound polygons or carbonate indicators" if not reef else reef}}
    # --- cross-checks with raw data and wells
    cc = {}
    if cross_check and I.cache_ready():
        try:
            v = validate_interpreted_vs_seismic()
            cc = {k: v.get(k) for k in ("polygon_vs_sand_probability", "consistent_with_seismic", "faults_vs_automatic_detection") if v.get(k) is not None}
        except Exception as ex:
            cc = {"error": f"{type(ex).__name__}: {ex}"}
    sc = {k: (x["score"] or 0) for k, x in ev.items()}
    s_struct = max(sc["four_way_anticline"], sc["fault_dependent_closure"])
    s_strat = sc["stratigraphic_updip_pinchout"]
    if s_struct >= 0.5 and s_strat >= 0.5:
        lead = "combination trap (structural closure with stratigraphic sand-body control)"
    elif s_struct >= 0.5 and s_struct >= s_strat:
        lead = "fault-dependent structural trap" if sc["fault_dependent_closure"] >= sc["four_way_anticline"] and sc["fault_dependent_closure"] > 0 else "four-way dip-closed structural trap (anticline)"
    elif s_strat >= 0.5:
        lead = "stratigraphic trap (updip pinch-out of sand-prone body)"
    else:
        lead = "no trap type clearly supported by the interpreted data"
    support = sum(1 for k in ("four_way_anticline", "fault_dependent_closure", "stratigraphic_updip_pinchout") if sc[k] >= 0.5)
    agree = []
    if cc.get("consistent_with_seismic") is True:
        agree.append("interpreted sand polygons coincide with higher raw sand probability")
    fva = cc.get("faults_vs_automatic_detection") or {}
    if isinstance(fva, dict) and (fva.get("share_of_interpreted_near_detected") or 0) > 0.4:
        agree.append("interpreted faults agree with automatic fault detection")
    conf = "medium-high" if (support >= 1 and len(agree) >= 2) else "medium" if (support >= 1 and agree) else "low-medium" if support >= 1 else "low"
    risks = ["Fault seal and juxtaposition cannot be assessed from fault traces alone",
             "Time-depth ties for wells without checkshots are low confidence",
             "Closure and spill points are measured in TWT on one horizon; depth conversion changes closure height"]
    if any(r.get("time_ms") is None for r in _slices("polygon") + _slices("fault")):
        risks.append("Slice times are unknown, so polygons/faults could not be tied to the horizon time (add slice_times.csv)")
    if not any(_is_seal(r["cls"]) for r in _slices("polygon")):
        risks.append("No fine-grained (overbank/levee) polygons, so lateral seal for the stratigraphic case is unproven")
    nxt = ["Check closure outline and sand-body edge on seismic sections at the crest and at the pinch-out",
           "Map the reservoir interval (not only one horizon) and test amplitude/impedance response at the proposed crest",
           "Quantify fault seal (throw vs reservoir thickness) for the bounding fault(s)"]
    img = interpreted_map()
    extra = []
    if _WIN[0] is not None or _WIN[1] is not None:
        for kind, label in (("polygon", "stratigraphic polygon"), ("fault", "fault")):
            allr = D.slices(kind)
            inn = _slices(kind)
            tm = sorted({round(r["time_ms"]) for r in allr if r.get("time_ms") is not None and r not in inn})
            if allr and not inn:
                extra.append(f"All {len(allr)} {label} slices lie outside the window (their times: {tm[:12]} ms), so there is no {label} evidence inside it")
            elif allr and tm:
                extra.append(f"{len(allr) - len(inn)} {label} slice(s) outside the window were ignored (times {tm[:8]} ms)")
        nn = [r for r in _slices("fault") if r.get("time_ms") is not None and not _in_win(r.get("time_ms")) and not r.get("t_range")]
        if nn:
            extra.append(f"{len(nn)} fault slice(s) just outside the window (within {FAULT_PAD_MS:.0f} ms) were used because faults extend vertically")
    return {"horizon": P["name"], "window_ms": [_WIN[0], _WIN[1]], "window_note": window_note, "evidence_notes": extra, "data_used": {"faults": len(_slices("fault")), "polygon_classes": D.polygon_classes(), "horizon_points": int(len(P["t"]))},
            "trap_types": ev, "leading_assessment": lead, "confidence": conf, "independent_agreements": agree, "cross_checks": cc,
            "risks": risks, "next_steps": nxt, "image_path": img["image_path"], "kind": "map", "index_used": "trap assessment",
            "scores_note": "Scores are screening scores from 0 to 1 built from the evidence shown; they are not probabilities of success."}


# --------------------------------------------------------------- prospects
def rank_prospects(top_n=5):
    """Rank candidate prospects from the interpreted data: closed structural highs and updip sand-body terminations, with coordinates (x,y, inline, xline, TWT), size, fault dependency, reservoir support from polygons and raw sand probability, nearest well."""
    e = _need("")
    if e:
        return e
    P = _prep_horizon()
    if P is None:
        return {"error": "no horizon ingested"}
    tol = _default_tol()
    cl, relief, _ = find_closures(P)
    cards = I.load_cards()["wells"] if I.cache_ready() else {}
    sand_rings = [ring for r in _slices("polygon") if _is_sand(r["cls"]) for ring, sa in _poly_rings(r) if sa <= 0]
    rows = []
    amax = max([c["area"] for c in cl] or [1.0])
    for i, c in enumerate(cl):
        tree, _ = _fault_tree(c["crest_t"])
        fb = _fault_bound_fraction(P, c["region"], tol, tree)
        ys, xs = np.where(c["region"])
        res = None
        if sand_rings:
            sel = np.random.default_rng(0).choice(len(xs), min(400, len(xs)), replace=False)
            res = float(_inside_any(np.c_[P["gx"][xs[sel]], P["gy"][ys[sel]]], sand_rings).mean())
        nw = min(cards.items(), key=lambda kv: math.hypot(kv[1]["x"] - c["crest_xy"][0], kv[1]["y"] - c["crest_xy"][1])) if cards else None
        score = 0.35 * min(1, c["height_ms"] / (0.3 * relief + 1e-9)) + 0.25 * c["area"] / amax + 0.25 * (res if res is not None else 0.5) + 0.15 * (1 - (fb or 0))
        rows.append({"type": "structural closure", "xy": [_f(v, 0) for v in c["crest_xy"]], "inline_xline": xy_to_ilxl(*c["crest_xy"]), "twt_ms": _f(c["crest_t"], 0),
                     "closure_height_ms": _f(c["height_ms"], 1), "area_km2_if_metres": _f(c["area"] / 1e6, 3), "fault_bounded_fraction": _f(fb, 2) if fb is not None else None,
                     "sand_polygon_overlap": _f(res, 2) if res is not None else None, "nearest_well": nw[0] if nw else None,
                     "distance_to_nearest_well": _f(math.hypot(nw[1]["x"] - c["crest_xy"][0], nw[1]["y"] - c["crest_xy"][1]), 0) if nw else None, "score": _f(score)})
    if sand_rings:
        st = _strat_analysis(P)
        pu = st["updip_points"]
        if len(pu) >= 5:
            tree = cKDTree(pu)
            used = np.zeros(len(pu), bool)
            for _ in range(3):
                free = np.where(~used)[0]
                if not len(free):
                    break
                cnt = np.array([len(tree.query_ball_point(pu[j], r=tol * 3)) for j in free])
                j = free[int(np.argmax(cnt))]
                members = tree.query_ball_point(pu[j], r=tol * 3)
                used[members] = True
                cen = pu[members].mean(0)
                tt = float(P["iH"]([[cen[1], cen[0]]])[0])
                nw = min(cards.items(), key=lambda kv: math.hypot(kv[1]["x"] - cen[0], kv[1]["y"] - cen[1])) if cards else None
                score = 0.3 + 0.5 * min(1, len(members) / max(len(pu), 1) * 2) + 0.2 * (st["updip_bounded_with_seal_neighbour_fraction"] or 0)
                rows.append({"type": "stratigraphic: updip termination of sand body", "xy": [_f(cen[0], 0), _f(cen[1], 0)], "inline_xline": xy_to_ilxl(*cen),
                             "twt_ms": _f(tt, 0), "outline_points_updip": len(members), "seal_neighbour_fraction": _f(st["updip_bounded_with_seal_neighbour_fraction"], 2),
                             "nearest_well": nw[0] if nw else None,
                             "distance_to_nearest_well": _f(math.hypot(nw[1]["x"] - cen[0], nw[1]["y"] - cen[1]), 0) if nw else None, "score": _f(score)})
    rows.sort(key=lambda r: -(r["score"] or 0))
    for i, r in enumerate(rows[:int(top_n)]):
        r["rank"] = i + 1
    img = interpreted_map()
    return {"prospects": rows[:int(top_n)], "scoring": "screening score 0-1 = 0.35 closure height + 0.25 closure area + 0.25 reservoir (sand polygon overlap) + 0.15 (1 - fault-bounded share); not a probability of success",
            "image_path": img["image_path"], "kind": "map", "index_used": "prospects"}


# ------------------------------------------------------ inventory / rescan
def list_datasets():
    """Inventory of everything ingested (raw wells and volumes, interpreted faults, polygons, horizons), with warnings and which tools apply."""
    out = {"raw": None, "interpreted": None}
    try:
        c = I.load_cards()
        out["raw"] = {"wells": list(c["wells"]), "n_wells": len(c["wells"]), "volumes": {k: v["label"] for k, v in c["volumes"].items()}, "warnings": c["warnings"][:10]}
    except Exception:
        out["raw"] = "not ingested"
    p = I.CACHE_DIR / "interpreted_cards.json"
    if p.exists():
        inv = json.loads(p.read_text())
        out["interpreted"] = {"summary": inv["summary"], "warnings": inv["warnings"], "files": inv["files"][:40]}
    else:
        out["interpreted"] = "not ingested"
    try:
        out["extents_xy"] = D.layer_extents()
    except Exception:
        pass
    out["raw_dir"] = str(I.DATA_DIR)
    out["interpreted_dir"] = str(D.interp_dir()) if D.interp_dir() else None
    return out


def scan_for_new_data(reingest_raw=False):
    """Look for new or changed data in the raw and interpreted folders and ingest only what changed; use after new files are added."""
    import ingest_all
    return ingest_all.refresh(reingest_raw=bool(reingest_raw))


# ------------------------------------------------ suggestive answer + reasoning
def render_trap(r):
    """Readable answer built straight from the tool result, so the final reply always contains the committed suggestion and its reasoning."""
    L = [f"**SUGGESTED ANSWER:** {r['suggested_answer']}  (confidence: {r['confidence']})", ""]
    if r.get("window_ms") and any(v is not None for v in r["window_ms"]):
        L.append(f"Window analysed: {r['window_ms'][0]}-{r['window_ms'][1]} ms TWT")
        L.append("")
    L.append(f"**WHY THIS ANSWER:** {r['why_this_answer']}. Screening scores (0-1): " + ", ".join(f"{k} {v}" for k, v in r["scores_0_to_1"].items()))
    L += ["", "**REASONING FROM THE DATA:**"]
    for i, c in enumerate(r["reasoning_chain"], 1):
        L.append(f"{i}. [{c['verdict'].upper()}] {c['step']}: {c['observation']}. -> {c['implication']}")
    L += ["", "**WHY NOT THE OTHERS:**"]
    for k, v in r["why_not_the_others"].items():
        L.append(f"- {k}: " + (f"score {v['score_0_to_1']}" if "score_0_to_1" in v else f"{v.get('status')} ({v.get('reason')})"))
    if r.get("where_to_look"):
        L += ["", "**WHERE TO LOOK:**"]
        for p in r["where_to_look"]:
            L.append(f"- #{p.get('rank')} {p.get('type')}: inline/xline {p.get('inline_xline')}, TWT {p.get('twt_ms')} ms, nearest well {p.get('nearest_well')}, score {p.get('score')}")
    if r.get("notes"):
        L += ["", "**NOTES:** " + " | ".join(r["notes"])]
    L += ["", "**WHAT WOULD CHANGE THE ANSWER:** " + " | ".join(r["what_would_change_the_answer"]), "", "**NEXT STEPS:** " + " | ".join(r["next_steps"])]
    if r.get("data_gaps"):
        L += ["", "**DATA GAPS:** " + ", ".join(r["data_gaps"])]
    return "\n".join(L)


def recommend_trap(twt_min_ms=None, twt_max_ms=None):
    """USE THIS FOR ANY 'what trap is it / identify trap type / structural vs stratigraphic vs combined / where to drill' QUESTION. Combines interpreted data (horizon, faults, polygons) and raw data (seismic, sand probability, well gas zones) in an optional TWT window and returns a committed SUGGESTED ANSWER (trap type, confidence), the evidence chain behind it, why the other trap types were not chosen, where to look (inline/xline/TWT), what would change the answer and next steps."""
    lo = None if twt_min_ms is None else float(twt_min_ms)
    hi = None if twt_max_ms is None else float(twt_max_ms)
    A = R = None
    notes, gaps = [], []
    if D.interpreted_ready():
        try:
            A = assess_traps(lo, hi)
            if A.get("error"):
                gaps.append(A["error"])
                A = None
        except Exception as ex:
            gaps.append(f"interpreted assessment failed: {type(ex).__name__}: {ex}")
    else:
        gaps.append("no interpreted data ingested (horizon, faults, polygons)")
    if I.cache_ready():
        try:
            import interpret as N
            R = N.trap_evidence(lo, hi)
        except Exception as ex:
            gaps.append(f"raw seismic trap test unavailable for this window: {ex}")
    else:
        gaps.append("no raw data ingested")
    if A is None and R is None:
        return {"suggested_answer": None, "why": "no usable data", "data_gaps": gaps,
                "next_steps": ["Ingest raw and/or interpreted data (scan_for_new_data), then ask again"]}
    chain, scores = [], {"structural (four-way anticline)": 0.0, "structural (fault-dependent)": 0.0, "stratigraphic": 0.0}
    window = [lo, hi]
    if A:
        if A.get("window_note"):
            notes.append(A["window_note"])
        notes += A.get("evidence_notes", [])
        T = A["trap_types"]
        a, f, s = T["four_way_anticline"], T["fault_dependent_closure"], T["stratigraphic_updip_pinchout"]
        scores["structural (four-way anticline)"] = a["score"] or 0
        scores["structural (fault-dependent)"] = f["score"] or 0
        scores["stratigraphic"] = s["score"] or 0
        e = a["evidence"]
        if e.get("closure_height_ms") is not None:
            chain.append({"step": "Structure", "observation": f"The interpreted horizon has a closed high: crest {e['crest_twt_ms']} ms TWT at inline/xline {e['crest_inline_xline']}, spill {e['spill_twt_ms']} ms, closure height {e['closure_height_ms']} ms, area {e['area_km2_if_metres']} km2 (if map units are metres)",
                          "implication": "A dip-closed structure exists, so a structural trap is geometrically possible" if a["status"] != "NOT SUPPORTED" else "no closure"})
        else:
            chain.append({"step": "Structure", "observation": f"No closed high found on the interpreted horizon (minimum height tested {e.get('min_height_ms_used')} ms)",
                          "implication": "A four-way anticline is not supported; any structural trap would need fault or stratigraphic help"})
        fb = f["evidence"].get("fault_bounded_fraction_of_outline")
        nflt = A["data_used"]["faults"]
        if fb is not None:
            chain.append({"step": "Faults", "observation": f"{nflt} interpreted fault slices are present; {fb * 100:.0f}% of the closure outline lies within fault tolerance",
                          "implication": ("The closure depends on faults for its edge, so fault seal becomes a key risk" if fb >= 0.25 else
                                          "The closure outline is mostly independent of faults, which favours a simple dip closure")})
        elif nflt == 0:
            chain.append({"step": "Faults", "observation": "No interpreted faults in this window", "implication": "A fault-dependent trap cannot be tested"})
        se = s["evidence"]
        if se:
            uf, sf = se.get("share_of_sand_body_outline_that_terminates_updip"), se.get("share_of_that_outline_next_to_fine_grained_polygons")
            ang = se.get("mean_angle_between_channel_axis_and_dip_deg")
            chain.append({"step": "Stratigraphy", "observation": f"{uf * 100:.0f}% of the sand-prone polygon outline faces updip and ends" +
                          (f"; {sf * 100:.0f}% of that edge sits next to fine-grained (overbank) polygons" if isinstance(sf, (int, float)) else "") +
                          (f"; the channel axis is {ang:.0f} degrees from the dip direction" if ang is not None else "") if uf is not None else "no sand polygons",
                          "implication": ("Sand bodies terminate updip against fine-grained rock, the pattern of a pinch-out trap" if (uf or 0) >= 0.2 else
                                          "Sand bodies do not clearly terminate updip, so a pinch-out trap is weak")})
        cc = A.get("cross_checks") or {}
        pv = cc.get("polygon_vs_sand_probability")
        if pv:
            sc_ = [k for k in pv if _is_sand(k)]
            if sc_:
                chain.append({"step": "Raw-data check (sand probability)", "observation": f"Mean sand probability inside {sc_[0]} polygons {pv[sc_[0]]['mean_sand_probability_inside']} vs {pv[sc_[0]]['mean_sand_probability_outside']} outside",
                              "implication": "The seismic supports the interpreted sand body" if cc.get("consistent_with_seismic") else "The seismic does not clearly support the interpreted sand body; treat the stratigraphic case with caution"})
        fv = cc.get("faults_vs_automatic_detection")
        if isinstance(fv, dict) and fv.get("share_of_interpreted_near_detected") is not None:
            chain.append({"step": "Raw-data check (faults)", "observation": f"{fv['share_of_interpreted_near_detected'] * 100:.0f}% of interpreted fault pixels lie near automatic ({fv.get('detector')}) detections",
                          "implication": "Faults are visible in the seismic" if fv["share_of_interpreted_near_detected"] > 0.4 else "Automatic detection does not confirm the interpreted faults; check them on sections"})
        gaps += [k for k, x in T.items() if str(x["status"]).startswith("NOT AVAILABLE")]
    if R:
        lean = R.get("leaning", "inconclusive")
        chain.append({"step": "Raw seismic test", "observation": f"Window {R['window_ms'][0]:.0f}-{R['window_ms'][1]:.0f} ms: structural control index {R.get('structural_control_index_0_to_1')} (sand edges vs structure contours), sand variability {R.get('sand_coefficient_of_variation')}, closed highs on the seismic horizon {sum(1 for h in R['structure']['closed_highs'] if h['closed_within_map'])}",
                      "implication": f"Raw data lean: {lean}"})
        bump = 0.15
        if lean == "structural":
            scores["structural (four-way anticline)"] += bump
        elif lean == "stratigraphic":
            scores["stratigraphic"] += bump
        elif lean.startswith("combination"):
            scores["structural (four-way anticline)"] += bump / 2
            scores["stratigraphic"] += bump / 2
    try:
        import interpret as N
        gz = N.thickest_gas_zone()["ranking"]
        if gz:
            g = gz[0]
            inside = (_in_win(g.get("top_twt_ms")) or _in_win(g.get("base_twt_ms")))
            chain.append({"step": "Well control", "observation": f"Thickest flagged gas zone: {g['well']} {g['net_pay_m']} m net at {g['top_twt_ms']}-{g['base_twt_ms']} ms TWT (time-depth confidence {g['time_depth_confidence']})",
                          "implication": "The reservoir interval lies in the analysed window" if inside else "The thickest gas zone is outside the analysed window, so this window is not tied to proven gas"})
    except Exception:
        pass
    st, sa = max(scores["structural (four-way anticline)"], scores["structural (fault-dependent)"]), scores["stratigraphic"]
    if st >= 0.5 and sa >= 0.5:
        ans, why = "COMBINED trap: structural closure with stratigraphic (sand-body) control", "both structural and stratigraphic evidence pass the screening threshold"
    elif st >= 0.5 and st >= sa:
        fd = scores["structural (fault-dependent)"] >= scores["structural (four-way anticline)"] and scores["structural (fault-dependent)"] > 0
        ans = "STRUCTURAL trap: fault-dependent closure" if fd else "STRUCTURAL trap: four-way dip-closed anticline"
        why = "structural evidence is the strongest; stratigraphic evidence is below the screening threshold"
    elif sa >= 0.5:
        ans, why = "STRATIGRAPHIC trap: updip pinch-out of the sand body", "stratigraphic evidence is the strongest; no convincing structural closure"
    else:
        best = max(scores, key=scores.get)
        ans = f"BEST-SUPPORTED but WEAK: {best} (nothing passes the screening threshold)"
        why = "no trap type reaches the threshold; this is the least-unsupported option, not a finding"
    top = max(scores.values())
    SUP = ("dip-closed structure exists", "mostly independent", "Sand bodies terminate", "seismic supports", "Faults are visible",
           "lies in the analysed window", "lean: structural", "lean: stratigraphic", "lean: combination")
    CON = ("not supported", "does not clearly", "is weak", "cannot be tested", "does not confirm", "Automatic detection does not", "outside the analysed window", "Raw data lean: inconclusive")
    for c in chain:
        im = c.get("implication", "")
        c["verdict"] = "contradicts" if any(w in im for w in CON) else "supports" if any(w in im for w in SUP) else "risk" if "risk" in im else "neutral"
    n_sup = sum(c["verdict"] == "supports" for c in chain)
    n_con = sum(c["verdict"] == "contradicts" for c in chain)
    conf = "medium" if (top >= 0.7 and n_sup >= 4 and n_con == 0 and A and R) else "low-medium" if (top >= 0.5 and n_con <= 1) else "low"
    if notes:                       # window mismatch: whole dataset used instead of the requested window
        conf = "low"
    if n_con:
        notes.append(f"{n_con} cross-check(s) contradict or fail to confirm the interpretation, so confidence was reduced. Resolve these before relying on the answer.")
    rejected = {}
    for k, v in scores.items():
        rejected[k] = {"score_0_to_1": _f(v, 2)}
    if A:
        T = A["trap_types"]
        for k, key in (("unconformity", "unconformity"), ("reef", "reef_buildup")):
            rejected[k] = {"status": T[key]["status"], "reason": T[key]["evidence"].get("why")}
    where = None
    try:
        if A:
            _WIN[0], _WIN[1] = lo, hi
            rp = rank_prospects(3)
            where = [{k: p.get(k) for k in ("rank", "type", "inline_xline", "twt_ms", "nearest_well", "score")} for p in rp.get("prospects", [])]
    except Exception:
        where = None
    finally:
        _WIN[0] = _WIN[1] = None
    img = (A or {}).get("image_path") or (R or {}).get("image_path")
    res = {"suggested_answer": ans, "confidence": conf, "why_this_answer": why, "window_ms": window, "scores_0_to_1": {k: _f(v, 2) for k, v in scores.items()},
            "evidence_summary": {"supports": n_sup, "contradicts": n_con}, "reasoning_chain": chain, "why_not_the_others": rejected, "where_to_look": where, "notes": notes,
            "what_would_change_the_answer": ["Closure at the reservoir top is smaller or fault-bounded on the seismic sections (weakens the structural case)",
                                             "Sand body does not end updip or seal polygons are absent (weakens the stratigraphic case)",
                                             "Well results: water-wet sand at the crest or hydrocarbons at a pinch-out edge"],
            "next_steps": ["Check the proposed crest and sand-body edge on seismic sections at the listed inline/xline",
                           "Pick the reservoir top in the analysed window and re-run",
                           "Assess fault seal (throw vs reservoir thickness) before ranking fault-dependent closures"],
            "data_gaps": sorted(set(gaps)), "image_path": img, "kind": "map", "index_used": "trap recommendation",
            "scores_note": "Screening scores from 0 to 1 built from the evidence above; not probabilities of success."}
    res["answer_markdown"] = render_trap(res)
    return res
