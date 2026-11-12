"""
interpret.py - screening-level interpretation tools (the "think like a geoscientist" layer).

Every function does the real computation in Python (log -> time conversion, net-gas flagging,
seismic attribute comparison, map extrapolation, closure detection, facies-pattern metrics) and
returns numbers + a PNG. Gemma then explains them. Nothing here is a formal interpretation:
results are automated screening with simple, stated cut-offs.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import ingest as I

IMG_DIR = I.CACHE_DIR / "images"
TL_DIR = I.CACHE_DIR / "time_logs"
_MAP_CACHE = {}
LAST_FAULTS = {}   # last automatic fault detection: {'seg','mask','method'} (used to cross-check interpreted faults)


# ----------------------------------------------------------------- helpers
def _f(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _cards():
    return I.load_cards()


def _vol_name(candidates):
    for c in candidates:
        try:
            return I.resolve_volume(c)
        except ValueError:
            continue
    return None


def _read_maps(vname, t0, t1, max_traces=8000, how="mean"):
    """Per-trace value inside a TWT window -> 2D grid [inline, xline] (laterally decimated)."""
    key = (vname, round(t0), round(t1), max_traces, how)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    v = I.open_volume(vname)
    k = (v.samples >= t0) & (v.samples <= t1)
    if not k.any():
        raise ValueError(f"window {t0}-{t1} ms outside volume range {v.samples[0]}-{v.samples[-1]} ms")
    uil, uxl = np.unique(v.il), np.unique(v.xl)
    if len(uil) < 3 or len(uxl) < 3:
        raise ValueError(f"{vname}: survey grid is degenerate ({len(uil)} inlines x {len(uxl)} xlines). "
                         "Re-run ingestion with force=True (ingest_all.run_all(force=True)) so inline/xline are rebuilt from trace coordinates.")
    s = max(1, int(math.ceil(math.sqrt(len(v.il) / max_traces))))
    sil, sxl = uil[::s], uxl[::s]
    sel = np.where(np.isin(v.il, sil) & np.isin(v.xl, sxl))[0]
    ts = v.samples[k]
    vals = np.empty(len(sel))
    for n, i in enumerate(sel):
        seg = v.trace(i)[k]
        vals[n] = seg.mean() if how == "mean" else ts[np.argmax(np.abs(seg))]
    grid = np.full((len(sil), len(sxl)), np.nan)
    grid[np.searchsorted(sil, v.il[sel]), np.searchsorted(sxl, v.xl[sel])] = vals
    out = {"grid": grid, "il": sil, "xl": sxl, "sel": sel, "vol": v, "step": s,
           "spacing": (v.max_dist / 3.0) * s}
    _MAP_CACHE[key] = out
    return out


def _scan_cube(vname, win_ms, step_ms, max_traces=6000):
    """Sliding-window means for every (decimated) trace: array [n_traces, n_windows]."""
    key = ("scan", vname, win_ms, step_ms, max_traces)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    v = I.open_volume(vname)
    dt = float(np.median(np.diff(v.samples)))
    wn, sn = max(2, int(round(win_ms / dt))), max(1, int(round(step_ms / dt)))
    starts = np.arange(0, len(v.samples) - wn + 1, sn)
    uil, uxl = np.unique(v.il), np.unique(v.xl)
    s = max(1, int(math.ceil(math.sqrt(len(v.il) / max_traces))))
    sel = np.where(np.isin(v.il, uil[::s]) & np.isin(v.xl, uxl[::s]))[0]
    arr = np.empty((len(sel), len(starts)))
    for n, i in enumerate(sel):
        c = np.cumsum(np.r_[0, v.trace(i)])
        arr[n] = (c[starts + wn] - c[starts]) / wn
    centers = v.samples[starts] + (wn - 1) / 2 * dt
    out = {"arr": arr, "centers": centers, "sel": sel, "vol": v, "step": s, "win": win_ms}
    _MAP_CACHE[key] = out
    return out


def _well_ilxl(v):
    out = {}
    for w, c in _cards()["wells"].items():
        d, i = v.tree.query([c["x"], c["y"]])
        if d <= v.max_dist:
            out[w] = (int(v.il[i]), int(v.xl[i]))
    return out


def _png(name, m, grid, title, cmap="viridis", contour=None, mark=None, clabel=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    il, xl = m["il"], m["xl"]
    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto",
                   extent=[xl[0], xl[-1], il[0], il[-1]])
    plt.colorbar(im, ax=ax, label=clabel)
    if contour is not None:
        ax.contour(np.linspace(xl[0], xl[-1], grid.shape[1]), np.linspace(il[0], il[-1], grid.shape[0]),
                   contour, levels=[0.5], colors="white", linewidths=1)
    for w, (i, x) in _well_ilxl(m["vol"]).items():
        ax.plot(x, i, "r^", ms=6)
        ax.text(x, i, " " + w, color="red", fontsize=8)
    for lab, (i, x) in (mark or {}).items():
        ax.plot(x, i, "w*", ms=13, mec="k")
        ax.text(x, i, " " + lab, color="white", fontsize=9, weight="bold")
    ax.set_xlabel("xline")
    ax.set_ylabel("inline")
    ax.set_title(title, fontsize=10)
    path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _default_window(twt_min_ms, twt_max_ms):
    v = I.open_volume("sand probability" if _vol_name(["sand probability"]) else list(_cards()["volumes"])[0])
    t0 = v.samples[0] if twt_min_ms is None else twt_min_ms
    t1 = v.samples[-1] if twt_max_ms is None else twt_max_ms
    return float(t0), float(t1)


# --------------------------------------------------------- 1. logs -> time
def logs_to_time(well, dt_ms=4.0):
    """Convert a well's logs from depth (MD) to TWT using its time-depth relation; saves a CSV."""
    w = I.resolve_well(well)
    c = _cards()["wells"][w]
    tr = I.load_track(w)
    if not np.isfinite(tr.twt_ms).any():
        return {"well": w, "error": "no time-depth relation available"}
    df = I.load_logs(w)
    md = df.index.values
    twt = np.interp(md, tr.md, tr.twt_ms, left=np.nan, right=np.nan)
    ok = np.isfinite(twt)
    t = pd.DataFrame(df.values[ok], index=twt[ok], columns=df.columns)
    t = t[~t.index.duplicated()].sort_index()
    grid = np.arange(math.ceil(t.index.min()), t.index.max(), dt_ms)
    out = pd.DataFrame({cn: np.interp(grid, t.index.values[t[cn].notna()], t[cn].dropna().values,
                                      left=np.nan, right=np.nan) if t[cn].notna().sum() > 1 else np.nan
                        for cn in t.columns}, index=grid)
    out.index.name = "TWT_ms"
    TL_DIR.mkdir(parents=True, exist_ok=True)
    path = TL_DIR / f"{w}_time.csv"
    out.to_csv(path)
    return {"well": w, "csv": str(path), "dt_ms": dt_ms, "twt_range_ms": [_f(grid[0], 0), _f(grid[-1], 0)],
            "n_samples": len(out), "curves": list(out.columns),
            "time_depth": c["time_depth"],
            "warning": "LOW confidence: regional-average time-depth, ties may be off by tens of ms"
            if c["time_depth"]["confidence"] != "high" else ""}


# ------------------------------------------------------ 2. gas / HC zones
def _zones(w, fluid="gas", min_thickness_m=2.0):
    c = _cards()["wells"][w]
    roles = c.get("roles", {})
    df = I.load_logs(w)
    need = [r for r in ("GR", "RES") if r not in roles]
    if need:
        raise ValueError(f"{w}: missing {need} curves; cannot flag hydrocarbon zones")
    gr, rt = df[roles["GR"]], df[roles["RES"]]
    ok = gr.notna() & rt.notna()
    lo, hi = np.nanpercentile(gr[ok], [5, 95])
    vsh = ((gr - lo) / max(hi - lo, 1e-6)).clip(0, 1)
    base = np.nanmedian(rt[ok & (vsh > 0.6)]) if (ok & (vsh > 0.6)).sum() > 20 else np.nanpercentile(rt[ok], 50)
    flag = ok & (vsh < 0.35) & (rt > 3 * base)
    used = ["Vsh<0.35 (from GR)", f"RES>3x shale baseline ({base:.2f})"]
    phi = None
    if "RHOB" in roles and "NPHI" in roles:
        phid = ((2.65 - df[roles["RHOB"]]) / 1.65).clip(-0.1, 0.6)
        nphi = df[roles["NPHI"]]
        if nphi.max() > 1.5:
            nphi = nphi / 100.0
        xo = phid - nphi
        phi = phid
        if fluid == "gas" and xo.notna().sum() > 50:
            flag &= xo.fillna(-1) > 0.02
            used.append("density-neutron crossover > 0.02 (gas)")
    elif fluid == "gas":
        used.append("no density/neutron pair -> 'gas' cannot be separated from oil: treat as HC")
    idx = df.index.values
    dz = float(np.median(np.diff(idx))) if len(idx) > 1 else 0.15
    f = flag.reindex(df.index).fillna(False).values
    tr = I.load_track(w)
    zones, i, n = [], 0, len(f)
    gap = max(2.0, 3 * dz)
    while i < n:
        if not f[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (f[j + 1] or (idx[min(n - 1, j + 1 + int(gap / dz))] - idx[j] <= gap and f[j + 1:j + 2 + int(gap / dz)].any())):
            j += 1
        top, bot = idx[i], idx[j]
        net = float(f[i:j + 1].sum() * dz)
        if net >= min_thickness_m:
            tv = np.interp([top, bot], tr.md, tr.tvd)
            tw = np.interp([top, bot], tr.md, tr.twt_ms)
            sl = slice(i, j + 1)
            zones.append({"top_md": _f(top, 1), "base_md": _f(bot, 1), "gross_md_m": _f(bot - top, 1),
                          "net_pay_m": _f(net, 1), "gross_tvd_m": _f(tv[1] - tv[0], 1),
                          "top_twt_ms": _f(tw[0], 1), "base_twt_ms": _f(tw[1], 1),
                          "mean_GR": _f(np.nanmean(gr.values[sl]), 1), "mean_RES": _f(np.nanmean(rt.values[sl]), 2),
                          "mean_porosity_density": _f(np.nanmean(phi.values[sl]), 3) if phi is not None else None})
        i = j + 1
    zones.sort(key=lambda z: -(z["net_pay_m"] or 0))
    return zones, used


def find_gas_zones(well, fluid="gas", min_thickness_m=2.0):
    """Flag net gas/hydrocarbon zones in one well from GR, resistivity and density-neutron crossover; list zones thickest first (depth and time)."""
    w = I.resolve_well(well)
    try:
        z, used = _zones(w, fluid, min_thickness_m)
    except ValueError as e:
        return {"well": w, "error": str(e)}
    return {"well": w, "fluid": fluid, "criteria": used, "n_zones": len(z), "zones": z[:6],
            "time_depth_confidence": _cards()["wells"][w]["time_depth"]["confidence"],
            "note": "Screening cut-offs, no petrophysical calibration (no tops, no water saturation)."}


def thickest_gas_zone(fluid="gas", min_thickness_m=2.0):
    """Scan all wells and return the thickest flagged gas/HC zone per well, ranked."""
    rows = []
    for w in _cards()["wells"]:
        try:
            z, _ = _zones(w, fluid, min_thickness_m)
        except Exception:
            continue
        if z:
            rows.append({"well": w, **z[0], "time_depth_confidence": _cards()["wells"][w]["time_depth"]["confidence"]})
    rows.sort(key=lambda r: -(r["net_pay_m"] or 0))
    return {"fluid": fluid, "ranking": rows[:8], "n_wells_with_zones": len(rows)}


# ------------------------------------------- 3. well zone vs seismic (time)
def _zone(well, zone_rank, fluid):
    w = I.resolve_well(well)
    z, used = _zones(w, fluid)
    if not z or zone_rank > len(z):
        raise ValueError(f"{w}: no {fluid} zone with rank {zone_rank}")
    return w, z[int(zone_rank) - 1]


def compare_zone_to_seismic(well, zone_rank=1, fluid="gas"):
    """Compare a flagged gas zone (in TWT) with inversion volumes and near/far stacks at the well: values inside vs outside the zone and rank among all traces."""
    try:
        w, z = _zone(well, zone_rank, fluid)
    except ValueError as e:
        return {"error": str(e)}
    t0, t1 = z["top_twt_ms"], z["base_twt_ms"]
    ext = I.load_extraction().get(w, {})
    out = {"well": w, "zone": z, "twt_window_ms": [t0, t1], "inversion": {}, "stacks": {},
           "time_depth_confidence": _cards()["wells"][w]["time_depth"]["confidence"]}
    for vn, e in ext.items():
        inz = (e["twt"] >= t0) & (e["twt"] <= t1)
        if inz.sum() < 2:
            out["inversion"][vn] = {"note": "zone thinner than the volume sampling or outside volume"}
            continue
        mp = _read_maps(vn, t0, t1)
        g = mp["grid"][np.isfinite(mp["grid"])]
        zin = float(np.mean(e["val"][inz]))
        out["inversion"][vn] = {"in_zone_mean": _f(zin), "outside_zone_mean_same_well": _f(np.mean(e["val"][~inz])),
                                "percentile_vs_all_traces_same_window": _f(100 * (g < zin).mean(), 0),
                                "map_p10_p90": [_f(np.percentile(g, 10)), _f(np.percentile(g, 90))]}
    tr = I.load_track(w)
    for vn, vc in _cards()["volumes"].items():
        if vc["kind"] != "stack":
            continue
        pad = 60
        tt = np.arange(t0 - pad, t1 + pad, 4.0)
        x = np.interp(tt, tr.twt_ms, tr.x)
        y = np.interp(tt, tr.twt_ms, tr.y)
        vals, _ = I.open_volume(vn).sample_along(x, y, tt)
        inz = (tt >= t0) & (tt <= t1)
        if np.isfinite(vals[inz]).sum() > 1:
            out["stacks"][vn] = {"rms_in_zone": _f(np.sqrt(np.nanmean(vals[inz] ** 2)), 2),
                                 "rms_background_pm60ms": _f(np.sqrt(np.nanmean(vals[~inz] ** 2)), 2)}
    n = next((k for k in out["stacks"] if "near" in k), None)
    f = next((k for k in out["stacks"] if "far" in k and "ultra" not in k), None)
    if n and f and out["stacks"][n]["rms_in_zone"]:
        out["far_over_near_rms_in_zone"] = _f(out["stacks"][f]["rms_in_zone"] / out["stacks"][n]["rms_in_zone"], 2)
    return out


def extrapolate_zone_to_map(well, zone_rank=1, fluid="gas", tolerance=0.15):
    """Use the gas zone's seismic signature at the well (sand probability, Vp/Vs, P-impedance P10-P90 in the zone window) to map where similar response occurs away from the well; returns body size, location (inline/xline) and a map PNG."""
    try:
        w, z = _zone(well, zone_rank, fluid)
    except ValueError as e:
        return {"error": str(e)}
    t0, t1 = z["top_twt_ms"], z["base_twt_ms"]
    ext = I.load_extraction().get(w, {})
    crit, grids, m0 = {}, [], None
    for vn, e in ext.items():
        inz = (e["twt"] >= t0) & (e["twt"] <= t1)
        if inz.sum() < 2:
            continue
        mp = _read_maps(vn, t0 - 4, t1 + 4)
        lo, hi = np.percentile(e["val"][inz], [10, 90])
        pad = tolerance * max(hi - lo, abs(np.mean([lo, hi])) * 0.05)
        crit[vn] = [_f(lo - pad), _f(hi + pad)]
        grids.append((mp["grid"] >= lo - pad) & (mp["grid"] <= hi + pad) & np.isfinite(mp["grid"]))
        m0 = m0 or mp
    if not grids:
        return {"error": "well zone is outside the inversion volumes", "well": w}
    match = np.logical_and.reduce(grids)
    lab, n = ndi.label(match)
    wi = _well_ilxl(m0["vol"]).get(w)
    r = int(np.searchsorted(m0["il"], wi[0])) if wi else None
    c = int(np.searchsorted(m0["xl"], wi[1])) if wi else None
    bodies = []
    for b in range(1, n + 1):
        rr, cc = np.where(lab == b)
        if len(rr) < 4:
            continue
        bodies.append({"size_traces": int(len(rr)), "area_map_units2": _f(len(rr) * m0["spacing"] ** 2, 0),
                       "centre_inline_xline": [int(m0["il"][int(rr.mean())]), int(m0["xl"][int(cc.mean())])],
                       "inline_range": [int(m0["il"][rr.min()]), int(m0["il"][rr.max()])],
                       "xline_range": [int(m0["xl"][cc.min()]), int(m0["xl"][cc.max()])],
                       "contains_well": bool(r is not None and r < lab.shape[0] and c < lab.shape[1] and lab[r, c] == b)})
    bodies.sort(key=lambda b: (-b["contains_well"], -b["size_traces"]))
    sv = _vol_name(["sand probability"])
    bg = _read_maps(sv, t0 - 4, t1 + 4)["grid"] if sv else match.astype(float)
    img = _png(f"extrap_{w}_z{zone_rank}", m0, bg, f"{w} zone signature match (white = match), TWT {t0:.0f}-{t1:.0f} ms",
              contour=match.astype(float), clabel="sand probability" if sv else "match")
    return {"well": w, "zone_twt_ms": [t0, t1], "signature_ranges": crit, "match_fraction_of_map": _f(match.mean(), 3),
            "n_bodies": len(bodies), "bodies_top5": bodies[:5], "image_path": img, "volume": "match_map",
            "kind": "map", "index_used": f"{t0:.0f}-{t1:.0f}ms",
            "note": "Match = all volumes inside the well's P10-P90 (+tolerance). Similar response is not proof of gas."}


# --------------------------------------------------------- 4. prospects
def find_prospects(top_n=5, twt_min_ms=None, twt_max_ms=None, win_ms=40.0):
    """Scan the volumes with a sliding TWT window for the strongest combined sand-probability high + low Vp/Vs + low P-impedance anomalies; returns inline, xline and TWT for the top candidates and a map."""
    sv, vv, pv = _vol_name(["sand probability"]), _vol_name(["vpvs"]), _vol_name(["p impedance"])
    if not sv:
        return {"error": "sand probability volume not found"}
    t0, t1 = _default_window(twt_min_ms, twt_max_ms)
    cubes = {k: _scan_cube(n, win_ms, win_ms / 2) for k, n in (("sand", sv), ("vpvs", vv), ("pimp", pv)) if n}
    base = cubes["sand"]
    ok = (base["centers"] >= t0 + win_ms / 2) & (base["centers"] <= t1 - win_ms / 2)
    if not ok.any():
        return {"error": "window too small for the scan"}
    def z(a):
        return (a - a.mean(0)) / (a.std(0) + 1e-9)
    score = z(base["arr"])
    parts = ["+sand probability"]
    if "vpvs" in cubes:
        score = score - z(cubes["vpvs"]["arr"]); parts.append("-Vp/Vs")
    if "pimp" in cubes:
        score = score - z(cubes["pimp"]["arr"]); parts.append("-P-impedance")
    score = score / len(parts)
    score[:, ~ok] = -np.inf
    v = base["vol"]
    il, xl, ctr = v.il[base["sel"]], v.xl[base["sel"]], base["centers"]
    il_rng, xl_rng = np.ptp(il) + 1, np.ptp(xl) + 1
    order = np.argsort(score, axis=None)[::-1][:20000]
    picks = []
    wells = _well_ilxl(v)
    for flat in order:
        ti, wi_ = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[ti, wi_]):
            break
        cand = (int(il[ti]), int(xl[ti]), float(ctr[wi_]))
        if all(abs(cand[0] - p["inline"]) > 0.15 * il_rng or abs(cand[1] - p["xline"]) > 0.15 * xl_rng
               or abs(cand[2] - p["twt_ms"]) > win_ms for p in picks):
            nw = min(wells.items(), key=lambda kv: math.hypot(kv[1][0] - cand[0], kv[1][1] - cand[1])) if wells else None
            picks.append({"rank": len(picks) + 1, "inline": cand[0], "xline": cand[1], "twt_ms": _f(cand[2], 0),
                          "score_z": _f(score[ti, wi_], 2),
                          "sand_prob_mean": _f(base["arr"][ti, wi_], 2),
                          "vpvs_mean": _f(cubes["vpvs"]["arr"][ti, wi_], 2) if "vpvs" in cubes else None,
                          "p_impedance_mean": _f(cubes["pimp"]["arr"][ti, wi_], 0) if "pimp" in cubes else None,
                          "nearest_well": nw[0] if nw else None,
                          "dist_to_nearest_well_lines": _f(math.hypot(nw[1][0] - cand[0], nw[1][1] - cand[1]), 0) if nw else None})
        if len(picks) >= int(top_n):
            break
    best = picks[0]
    mp = _read_maps(sv, best["twt_ms"] - win_ms / 2, best["twt_ms"] + win_ms / 2)
    img = _png(f"prospect1_{best['twt_ms']:.0f}", mp, mp["grid"], f"Sand probability, TWT {best['twt_ms']:.0f} ms (star = rank 1)",
              mark={"P1": (best["inline"], best["xline"])}, clabel="sand probability")
    return {"scoring": " ".join(parts) + " (z-scores per window)", "window_ms": win_ms, "search_range_ms": [t0, t1],
            "prospects": picks, "image_path": img, "volume": "sand_probability", "kind": "time",
            "index_used": best["twt_ms"],
            "note": "Anomaly ranking, not a trap: closure/seal must be checked with structure_and_closure."}


# ------------------------------------------------ 5. structure & closure
def structure_and_closure(twt_min_ms, twt_max_ms, volume="full stack"):
    """Pick the strongest reflector per trace inside a TWT window (full stack) to make a time-structure map; reports relief, dip and closed highs (crest, spill, closure height ms)."""
    vn = _vol_name([volume, "full stack"])
    mp = _read_maps(vn, twt_min_ms, twt_max_ms, how="argpeak")
    h = mp["grid"].copy()
    fill = np.nanmax(h)
    hn = ndi.median_filter(np.where(np.isfinite(h), h, fill), size=3)
    rngh = np.ptp(hn) or 1.0
    mins = (hn == ndi.minimum_filter(hn, size=5))
    cands = sorted(zip(*np.where(mins)), key=lambda rc: hn[rc])[:10]
    seen = np.zeros(hn.shape, bool)
    highs = []
    for r, c in cands:
        if seen[r, c]:
            continue
        prev, spill, region = None, None, None
        for lvl in np.linspace(hn[r, c], hn[r, c] + rngh, 60)[1:]:
            lab, _ = ndi.label(hn <= lvl)
            reg = lab == lab[r, c]
            if reg[0, :].any() or reg[-1, :].any() or reg[:, 0].any() or reg[:, -1].any():
                spill = lvl
                break
            prev, region = lvl, reg
        closed = spill is not None and region is not None
        height = (prev - hn[r, c]) if closed else 0.0
        if closed:
            seen |= region
        highs.append({"crest_inline_xline": [int(mp["il"][r]), int(mp["xl"][c])], "crest_twt_ms": _f(hn[r, c], 1),
                      "closed_within_map": bool(closed), "closure_height_ms": _f(height, 1),
                      "area_map_units2": _f(region.sum() * mp["spacing"] ** 2, 0) if closed else None})
    highs = [x for x in highs if x["closure_height_ms"] and x["closure_height_ms"] >= 2][:4]
    gy, gx = np.gradient(hn)
    dip = np.hypot(gy, gx) / mp["spacing"]
    img = _png("structure_map", mp, hn, f"Picked horizon TWT (ms), window {twt_min_ms:.0f}-{twt_max_ms:.0f}",
              cmap="viridis_r", clabel="TWT (ms)",
              mark={f"H{i+1}": tuple(x["crest_inline_xline"]) for i, x in enumerate(highs)})
    return {"volume": vn, "window_ms": [twt_min_ms, twt_max_ms], "relief_ms": _f(rngh, 1),
            "mean_dip_ms_per_map_unit": _f(np.mean(dip), 5), "closed_highs": highs, "n_closed_highs": len(highs),
            "image_path": img, "kind": "map", "index_used": f"{twt_min_ms:.0f}-{twt_max_ms:.0f}ms",
            "caveat": "Automatic strongest-amplitude pick; may jump between events. Window should bracket ONE reflector (e.g. the well-tied reservoir top)."}


# ------------------------------------------ 6. depositional pattern
def depositional_pattern(twt_min_ms=None, twt_max_ms=None, threshold=None):
    """Analyse the sand-probability map in a TWT window: sand fraction, body count, elongation, orientation, connectivity, anisotropy; proposes the most likely depositional pattern (channel, lobe/fan, sheet, bar/patchy) with confidence."""
    sv = _vol_name(["sand probability"])
    auto = twt_min_ms is None or twt_max_ms is None
    if auto:
        twt_min_ms, twt_max_ms = _auto_window()
    mp = _read_maps(sv, twt_min_ms, twt_max_ms)
    g = mp["grid"]
    fin = np.isfinite(g)
    thr = float(threshold) if threshold is not None else (0.5 if np.nanmin(g) >= 0 and np.nanmax(g) <= 1 and np.nanpercentile(g, 70) > 0.5 else float(np.nanpercentile(g, 70)))
    b = (g >= thr) & fin
    lab, n = ndi.label(b, structure=np.ones((3, 3)))
    sizes = ndi.sum(b, lab, range(1, n + 1)) if n else np.array([])
    frac = float(b.sum() / max(fin.sum(), 1))
    bodies = []
    for k in np.argsort(sizes)[::-1][:5]:
        rr, cc = np.where(lab == k + 1)
        if len(rr) < 6:
            continue
        cov = np.cov(np.c_[rr, cc].T)
        ev, evec = np.linalg.eigh(cov)
        el = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
        ang = (math.degrees(math.atan2(evec[0, 1], evec[1, 1])) % 180)
        bodies.append({"size_traces": int(len(rr)), "share_of_sand": _f(len(rr) / b.sum(), 2), "elongation": _f(el, 1),
                       "long_axis_deg_from_inline_direction": _f(ang, 0), "length_map_units": _f(4 * math.sqrt(ev[1]) * mp["spacing"], 0),
                       "width_map_units": _f(4 * math.sqrt(ev[0]) * mp["spacing"], 0)})
    gn = np.where(fin, g, np.nanmean(g)) - np.nanmean(g)
    ac_il = float(np.mean(gn[1:, :] * gn[:-1, :]) / (np.mean(gn ** 2) + 1e-12))
    ac_xl = float(np.mean(gn[:, 1:] * gn[:, :-1]) / (np.mean(gn ** 2) + 1e-12))
    big = bodies[0] if bodies else None
    if not big:
        env, why = "no coherent sand body at this cut-off", "no connected body above threshold"
    elif frac > 0.65 and big["share_of_sand"] > 0.8:
        env, why = "sheet / blanket sand (e.g. shoreface, sheet delta, widespread turbidite)", "sand covers most of the map as one connected body"
    elif big["elongation"] >= 3.0 and big["share_of_sand"] > 0.3:
        env, why = "channel / linear sand belt (fluvial, distributary or slope channel)", f"dominant body elongation {big['elongation']}"
    elif 1.5 <= big["elongation"] < 3.0 and big["size_traces"] > 0.08 * fin.sum():
        env, why = "lobe / fan (delta lobe, mouth bar or submarine-fan lobe)", f"moderately elongated large body (elongation {big['elongation']})"
    elif frac < 0.3 and len(bodies) >= 3:
        env, why = "isolated bars / patchy sands (mouth bars, sand patches, reworked shoals)", "several small disconnected bodies"
    else:
        env, why = "mixed / indeterminate", "metrics do not match one clean pattern"
    conf = "medium" if (big and why.startswith(("sand covers", "dominant"))) else "low"
    img = _png(f"depo_{twt_min_ms:.0f}_{twt_max_ms:.0f}", mp, g, f"Sand probability, {twt_min_ms:.0f}-{twt_max_ms:.0f} ms (white = >= {thr:.2f})",
              contour=b.astype(float), clabel="sand probability")
    return {"window_ms": [twt_min_ms, twt_max_ms], "window_chosen_automatically_as_highest_variation_slice": auto,
            "threshold": _f(thr, 2), "sand_fraction": _f(frac, 2),
            "n_bodies": int(n), "largest_bodies": bodies, "lag1_autocorr_along_inline": _f(ac_il, 2),
            "lag1_autocorr_along_xline": _f(ac_xl, 2), "proposed_environment": env, "because": why, "confidence": conf,
            "image_path": img, "volume": "sand_probability", "kind": "map", "index_used": f"{twt_min_ms:.0f}-{twt_max_ms:.0f}ms",
            "caveat": "Pattern from one seismic-derived probability map; needs core/regional context to confirm."}


# --------------------------------------------- 7. trap type evidence
def trap_evidence(twt_min_ms=None, twt_max_ms=None):
    """Structural vs stratigraphic trap test in a TWT window: closed highs on the picked horizon, and whether sand-probability edges follow or cut across time-structure contours."""
    if twt_min_ms is None or twt_max_ms is None:
        twt_min_ms, twt_max_ms = _auto_window(30.0)
    st = structure_and_closure(twt_min_ms, twt_max_ms)
    sv = _vol_name(["sand probability"])
    sand = _read_maps(sv, twt_min_ms, twt_max_ms)["grid"]
    hor = _read_maps(_vol_name(["full stack"]), twt_min_ms, twt_max_ms, how="argpeak")["grid"]
    sg = np.gradient(ndi.gaussian_filter(np.nan_to_num(sand, nan=np.nanmean(sand)), 1))
    hg = np.gradient(ndi.gaussian_filter(np.nan_to_num(hor, nan=np.nanmean(hor)), 1))
    sm, hm = np.hypot(*sg), np.hypot(*hg)
    use = (sm > np.percentile(sm, 70)) & (hm > np.percentile(hm, 30))
    cosang = np.abs((sg[0] * hg[0] + sg[1] * hg[1]) / (sm * hm + 1e-12))
    ctrl = float(cosang[use].mean()) if use.any() else None
    sand_cv = float(np.nanstd(sand) / (abs(np.nanmean(sand)) + 1e-9))
    closed = [h for h in st["closed_highs"] if h["closed_within_map"]]
    items = {
        "closed_structural_high": ("SUPPORTED (structural)" if closed else "CONTRADICTED / not seen on picked horizon"),
        "sand_edges_follow_structure_contours": ("NOT AVAILABLE" if ctrl is None else
                                                 "SUPPORTED (structural)" if ctrl > 0.65 else
                                                 "CONTRADICTED -> edges cut across contours (stratigraphic)" if ctrl < 0.4 else "INCONCLUSIVE"),
        "strong_lateral_sand_variability": "SUPPORTED (stratigraphic)" if sand_cv > 0.35 else "weak lateral variability",
    }
    s_struct = (1 if closed else 0) + (1 if ctrl is not None and ctrl > 0.65 else 0)
    s_strat = (1 if ctrl is not None and ctrl < 0.4 else 0) + (1 if sand_cv > 0.35 else 0)
    lean = ("combination (structural closure + stratigraphic sand control)" if closed and s_strat >= 1 and s_struct >= 1 and ctrl is not None and ctrl < 0.65
            else "structural" if s_struct > s_strat else "stratigraphic" if s_strat > s_struct else "inconclusive")
    return {"window_ms": [twt_min_ms, twt_max_ms], "structure": {k: st[k] for k in ("relief_ms", "closed_highs", "caveat")},
            "structural_control_index_0_to_1": _f(ctrl, 2), "sand_coefficient_of_variation": _f(sand_cv, 2),
            "checklist": items, "leaning": lean, "confidence": "low-medium",
            "image_path": st["image_path"], "volume": "structure_map", "kind": "map", "index_used": st["index_used"],
            "what_would_change_my_mind": ["a better horizon pick at the well-tied reservoir top", "formation tops / fault interpretation",
                                          "a seal analysis", "well results from a pinch-out location"]}


# ============ FOUNDATION MAPS: horizon + contours + faults, 10 ms slices, overlay ============
def prep_cube_for_faults(cube, band=None):
    """Make a cube safe for fault detection on an irregular survey footprint.
    Outside-footprint traces (NaN or all zero) create an artificial step at the survey edge that detectors report as a fault.
    Returns (filled_cube, valid_footprint, edge_band): empty cells are filled with the nearest real trace and a band along the edge is flagged
    so detections there can be discarded."""
    c = np.nan_to_num(cube, nan=0.0)
    valid = (c != 0).any(2) & np.isfinite(cube).any(2)
    if valid.all() or not valid.any():
        return np.where(np.isfinite(cube), cube, 0.0), valid, np.zeros(valid.shape, bool)
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = cube[idx[0], idx[1], :]
    filled = np.where(np.isfinite(filled), filled, 0.0)
    b = band if band is not None else max(3, int(math.ceil(0.04 * min(valid.shape))))
    inner = ndi.binary_erosion(valid, iterations=b, border_value=0)
    return filled, valid, ~inner


def _segments(vname, t0, t1, max_traces=8000):
    """Window samples for every decimated trace: array [n_il, n_xl, n_samples]."""
    key = ("seg", vname, round(t0), round(t1), max_traces)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    v = I.open_volume(vname)
    k = (v.samples >= t0) & (v.samples <= t1)
    if k.sum() < 2:
        raise ValueError(f"window {t0}-{t1} ms too thin or outside {v.samples[0]}-{v.samples[-1]} ms")
    uil, uxl = np.unique(v.il), np.unique(v.xl)
    s = max(1, int(math.ceil(math.sqrt(len(v.il) / max_traces))))
    sil, sxl = uil[::s], uxl[::s]
    sel = np.where(np.isin(v.il, sil) & np.isin(v.xl, sxl))[0]
    cube = np.full((len(sil), len(sxl), int(k.sum())), np.nan)
    ri, ci = np.searchsorted(sil, v.il[sel]), np.searchsorted(sxl, v.xl[sel])
    for n, i in enumerate(sel):
        cube[ri[n], ci[n]] = v.trace(i)[k]
    out = {"cube": cube, "il": sil, "xl": sxl, "vol": v, "step": s, "spacing": (v.max_dist / 3.0) * s}
    _MAP_CACHE[key] = out
    return out


def _coherence(cube):
    """Zero-lag normalised similarity with the 4 neighbours (1 = continuous, low = discontinuity)."""
    c = np.nan_to_num(cube - np.nanmean(cube, axis=2, keepdims=True))
    nrm = np.sqrt((c ** 2).sum(2)) + 1e-9
    acc, cnt = np.zeros(c.shape[:2]), np.zeros(c.shape[:2])
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = c[max(dr, 0):c.shape[0] + min(dr, 0), max(dc, 0):c.shape[1] + min(dc, 0)]
        b = c[max(-dr, 0):c.shape[0] + min(-dr, 0), max(-dc, 0):c.shape[1] + min(-dc, 0)]
        na = nrm[max(dr, 0):c.shape[0] + min(dr, 0), max(dc, 0):c.shape[1] + min(dc, 0)]
        nb = nrm[max(-dr, 0):c.shape[0] + min(-dr, 0), max(-dc, 0):c.shape[1] + min(-dc, 0)]
        sim = (a * b).sum(2) / (na * nb)
        acc[max(dr, 0):c.shape[0] + min(dr, 0), max(dc, 0):c.shape[1] + min(dc, 0)] += sim
        cnt[max(dr, 0):c.shape[0] + min(dr, 0), max(dc, 0):c.shape[1] + min(dc, 0)] += 1
    return acc / np.maximum(cnt, 1)


def _horizon_and_faults(t0, t1, vname):
    mp = _read_maps(vname, t0, t1, how="argpeak")
    h = mp["grid"]
    hn = ndi.median_filter(np.where(np.isfinite(h), h, np.nanmax(h)), size=3)
    seg = _segments(vname, t0, t1)
    coh = _coherence(seg["cube"])
    gy, gx = np.gradient(hn)
    grad = np.hypot(gy, gx)
    low = coh < np.nanpercentile(coh, 10)
    steep = grad > np.nanpercentile(grad, 90)
    cand = low & steep if (low & steep).sum() >= 6 else low
    cand = ndi.binary_opening(cand, structure=np.ones((1, 2))) | ndi.binary_opening(cand, structure=np.ones((2, 1))) | cand
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    faults = []
    for b in range(1, n + 1):
        rr, cc = np.where(lab == b)
        if len(rr) < 5:
            continue
        ev, evec = np.linalg.eigh(np.cov(np.c_[rr, cc].T))
        el = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
        if el < 2.0:
            continue
        faults.append({"length_map_units": _f(4 * math.sqrt(ev[1]) * mp["spacing"], 0), "elongation": _f(el, 1),
                       "strike_deg_from_inline_direction": _f(math.degrees(math.atan2(evec[0, 1], evec[1, 1])) % 180, 0),
                       "centre_inline_xline": [int(mp["il"][int(rr.mean())]), int(mp["xl"][int(cc.mean())])],
                       "mean_coherence": _f(coh[rr, cc].mean(), 2), "mean_horizon_step_ms": _f(grad[rr, cc].mean(), 1)})
    faults.sort(key=lambda f: -f["length_map_units"])
    fmask = np.zeros_like(cand)
    for b in range(1, n + 1):
        rr, cc = np.where(lab == b)
        if len(rr) >= 5:
            fmask[rr, cc] = True
    return mp, hn, coh, faults, fmask


def _plot(name, m, layers, title, wells=True):
    """layers: list of dicts(kind=image|contour|mask, data=..., ...)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    il, xl = m["il"], m["xl"]
    ext = [xl[0], xl[-1], il[0], il[-1]]
    X, Y = np.linspace(xl[0], xl[-1], len(xl)), np.linspace(il[0], il[-1], len(il))
    for L in layers:
        if L["kind"] == "image":
            im = ax.imshow(L["data"], origin="lower", aspect="auto", extent=ext, cmap=L.get("cmap", "gray"),
                           vmin=L.get("vmin"), vmax=L.get("vmax"))
            plt.colorbar(im, ax=ax, label=L.get("label", ""))
        elif L["kind"] == "contour":
            cs = ax.contour(X, Y, L["data"], levels=L.get("levels", 12), colors=L.get("color", "yellow"), linewidths=0.8)
            ax.clabel(cs, fontsize=6, fmt="%.0f")
        elif L["kind"] == "mask":
            mm = np.ma.masked_where(~L["data"], np.ones(L["data"].shape))
            ax.imshow(mm, origin="lower", aspect="auto", extent=ext, cmap="autumn", alpha=0.9)
    if wells:
        for w, (i, x) in _well_ilxl(m["vol"]).items():
            ax.plot(x, i, "c^", ms=6, mec="k")
            ax.text(x, i, " " + w, color="cyan", fontsize=8)
    ax.set_xlabel("xline")
    ax.set_ylabel("inline")
    ax.set_title(title, fontsize=10)
    path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def horizon_contour_and_faults(twt_min_ms, twt_max_ms, volume="full stack"):
    """Step 1: simple time-structure map (picked horizon + contours) and a fault-candidate map (low coherence + steep horizon step). Window should bracket ONE reflector."""
    vn = _vol_name([volume, "full stack"])
    mp, hn, coh, faults, fmask = _horizon_and_faults(twt_min_ms, twt_max_ms, vn)
    img = _plot(f"foundation_structure_{twt_min_ms:.0f}_{twt_max_ms:.0f}", mp,
                [{"kind": "image", "data": hn, "cmap": "viridis_r", "label": "TWT (ms)"},
                 {"kind": "contour", "data": hn, "levels": 12, "color": "white"},
                 {"kind": "mask", "data": fmask}],
                f"Structure contours (TWT ms) + fault candidates (red), {twt_min_ms:.0f}-{twt_max_ms:.0f} ms")
    return {"window_ms": [twt_min_ms, twt_max_ms], "relief_ms": _f(np.ptp(hn), 1),
            "contour_interval_ms_approx": _f(np.ptp(hn) / 12, 1), "n_fault_candidates": len(faults),
            "fault_candidates": faults[:6], "mean_coherence": _f(coh.mean(), 2),
            "image_path": img, "volume": "structure_faults", "kind": "map", "index_used": f"{twt_min_ms:.0f}-{twt_max_ms:.0f}ms",
            "caveat": "Fault candidates are discontinuity proxies on a decimated map (every "
                      f"{_segments(vn, twt_min_ms, twt_max_ms)['step']} lines), not interpreted faults."}


def horizon_slice(slice_twt_ms, win_ms=10.0, volume="sand probability"):
    """Step 2: horizon/time-slice pattern averaged in a thin window (default 10 ms) of any volume (RMS for stacks, mean for inversions)."""
    vn = _vol_name([volume])
    if vn is None:
        return {"error": f"unknown volume {volume}"}
    t0, t1 = slice_twt_ms - win_ms / 2, slice_twt_ms + win_ms / 2
    seg = _segments(vn, t0, t1)
    isstack = _cards()["volumes"][vn]["kind"] == "stack"
    g = np.sqrt(np.nanmean(seg["cube"] ** 2, axis=2)) if isstack else np.nanmean(seg["cube"], axis=2)
    lo, hi = np.nanpercentile(g, [2, 98])
    img = _plot(f"hslice_{vn}_{slice_twt_ms:.0f}_{win_ms:.0f}", seg,
                [{"kind": "image", "data": g, "cmap": "gray" if isstack else "viridis", "vmin": lo, "vmax": hi,
                  "label": ("RMS amplitude" if isstack else "mean value")}],
                f"{_cards()['volumes'][vn]['label']} {t0:.0f}-{t1:.0f} ms")
    gg = g[np.isfinite(g)]
    return {"volume": vn, "window_ms": [t0, t1], "statistic": "RMS" if isstack else "mean",
            "stats": {"min": _f(gg.min()), "p10": _f(np.percentile(gg, 10)), "p50": _f(np.median(gg)),
                      "p90": _f(np.percentile(gg, 90)), "max": _f(gg.max())},
            "image_path": img, "kind": "map", "index_used": f"{t0:.0f}-{t1:.0f}ms"}


def overlay_structure_and_slice(twt_min_ms, twt_max_ms, slice_twt_ms, win_ms=10.0, volume="sand probability"):
    """Step 3: overlay the structure contours and fault candidates (step 1) on the 10 ms horizon slice (step 2) with wells, and quantify whether the slice anomaly follows structure."""
    vn = _vol_name([volume])
    mp, hn, coh, faults, fmask = _horizon_and_faults(twt_min_ms, twt_max_ms, _vol_name(["full stack"]))
    seg = _segments(vn, slice_twt_ms - win_ms / 2, slice_twt_ms + win_ms / 2)
    isstack = _cards()["volumes"][vn]["kind"] == "stack"
    g = np.sqrt(np.nanmean(seg["cube"] ** 2, axis=2)) if isstack else np.nanmean(seg["cube"], axis=2)
    if g.shape != hn.shape:
        return {"error": "volumes have different decimated grids; cannot overlay"}
    lo, hi = np.nanpercentile(g, [2, 98])
    img = _plot(f"overlay_{vn}_{slice_twt_ms:.0f}", mp,
                [{"kind": "image", "data": g, "cmap": "gray" if isstack else "viridis", "vmin": lo, "vmax": hi,
                  "label": f"{vn} ({'RMS' if isstack else 'mean'})"},
                 {"kind": "contour", "data": hn, "levels": 12, "color": "yellow"},
                 {"kind": "mask", "data": fmask}],
                f"{vn} {slice_twt_ms:.0f}±{win_ms/2:.0f} ms + structure (yellow) + faults (red)")
    ok = np.isfinite(g) & np.isfinite(hn)
    corr = float(np.corrcoef(g[ok], hn[ok])[0, 1]) if ok.sum() > 10 and np.std(g[ok]) > 0 else None
    hi_mask = g >= np.nanpercentile(g, 80)
    near_fault = ndi.binary_dilation(fmask, iterations=2)
    on_fault = float((hi_mask & near_fault).sum() / max(hi_mask.sum(), 1))
    return {"slice": {"volume": vn, "twt_ms": slice_twt_ms, "win_ms": win_ms},
            "structure_window_ms": [twt_min_ms, twt_max_ms],
            "corr_slice_vs_structure_twt": _f(corr, 2),
            "reading": ("slice anomaly tracks structural highs/lows" if corr is not None and abs(corr) > 0.5 else
                        "slice anomaly largely independent of structure (stratigraphic/lateral facies control possible)"),
            "share_of_top20pct_anomaly_near_fault_candidates": _f(on_fault, 2),
            "n_fault_candidates": len(faults), "fault_candidates": faults[:4],
            "image_path": img, "volume": "overlay", "kind": "map", "index_used": f"{slice_twt_ms:.0f}ms",
            "caveat": "Horizon pick and fault proxy are automatic; confirm on sections before relying on them."}


# ===================== whole-volume scans (no window needed) =====================
def high_variation_slices(top_n=5, volume="sand probability", min_separation_ms=20.0):
    """Scan EVERY time slice of the sand-probability volume and rank them by lateral variation (std and P5-P95 range across the map); returns the best times and draws the map of the top slice plus std-vs-time."""
    vn = _vol_name([volume])
    if vn is None:
        return {"error": f"unknown volume {volume}"}
    v = I.open_volume(vn)
    dt = float(np.median(np.diff(v.samples)))
    cube = _scan_cube(vn, dt, dt)
    arr, ctr = cube["arr"], cube["centers"]
    std = np.nanstd(arr, axis=0)
    p5, p95 = np.nanpercentile(arr, 5, axis=0), np.nanpercentile(arr, 95, axis=0)
    mn, mx = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
    rank, picked = np.argsort(std)[::-1], []
    for k in rank:
        if all(abs(ctr[k] - ctr[j]) >= min_separation_ms for j in picked):
            picked.append(k)
        if len(picked) >= int(top_n):
            break
    rows = [{"rank": i + 1, "twt_ms": _f(ctr[k], 0), "std": _f(std[k], 3), "p5": _f(p5[k], 2), "p95": _f(p95[k], 2),
             "min": _f(mn[k], 2), "max": _f(mx[k], 2), "range_p95_p5": _f(p95[k] - p5[k], 2)} for i, k in enumerate(picked)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    best = picked[0]
    mp = _read_maps(vn, ctr[best] - dt / 2, ctr[best] + dt / 2)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    im = a.imshow(mp["grid"], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                  extent=[mp["xl"][0], mp["xl"][-1], mp["il"][0], mp["il"][-1]])
    plt.colorbar(im, ax=a, label="sand probability (0-1)")
    for w, (i, x) in _well_ilxl(v).items():
        a.plot(x, i, "r^", ms=6)
        a.text(x, i, " " + w, color="red", fontsize=8)
    a.set_xlabel("xline"); a.set_ylabel("inline"); a.set_title(f"Time slice {ctr[best]:.0f} ms (highest variation, std={std[best]:.2f})")
    b.plot(std, ctr, "k-", lw=1)
    for k in picked:
        b.plot(std[k], ctr[k], "ro")
        b.text(std[k], ctr[k], f" {ctr[k]:.0f}", fontsize=8)
    b.invert_yaxis(); b.set_xlabel("lateral std of sand probability"); b.set_ylabel("TWT (ms)"); b.set_title("Variation vs time")
    path = IMG_DIR / f"high_variation_{ctr[best]:.0f}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"volume": vn, "n_slices_scanned": int(len(ctr)), "sampling_ms": dt, "value_range_whole_volume": [_f(np.nanmin(arr), 2), _f(np.nanmax(arr), 2)],
            "top_slices": rows, "image_path": str(path), "kind": "time", "index_used": _f(ctr[best], 0),
            "definition": "variation = standard deviation of sand probability across the map at each time slice (decimated map)."}


def _auto_window(half=20.0):
    r = high_variation_slices(1)
    t = r["top_slices"][0]["twt_ms"]
    return t - half, t + half


def _order_path(pix):
    """Order skeleton pixels of one branch by walking from an end point."""
    pset, nbrs = set(pix), {}
    for p in pset:
        nbrs[p] = [(p[0] + i, p[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i or j) and (p[0] + i, p[1] + j) in pset]
    ends = [p for p in pset if len(nbrs[p]) == 1]
    cur = min(ends) if ends else min(pset)
    path, seen = [cur], {cur}
    while True:
        nxt = [q for q in nbrs[cur] if q not in seen]
        if not nxt:
            break
        cur = min(nxt, key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        path.append(cur)
        seen.add(cur)
    return path


def _split_straight(path, tol=2.5):
    """Douglas-Peucker style split of an ordered path into near-straight segments."""
    if len(path) < 3:
        return [path]
    p0, p1 = np.array(path[0], float), np.array(path[-1], float)
    d = p1 - p0
    nrm = math.hypot(*d)
    pts = np.array(path, float)
    rel = pts - p0
    dist = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / nrm if nrm > 0 else np.hypot(rel[:, 0], rel[:, 1])
    k = int(np.argmax(dist))
    if dist[k] <= tol or k in (0, len(path) - 1):
        return [path]
    return _split_straight(path[:k + 1], tol) + _split_straight(path[k:], tol)


def _merge_collinear(segs, spacing, pers, low, starts, gap_px=6.0, ang_tol=15.0, off_tol=3.0):
    """Join segments that continue each other across a junction (same strike, small gap and offset)."""
    n = len(segs)
    parent = list(range(n))
    find = lambda a: a if parent[a] == a else find(parent[a])
    ends = [(np.array([f["rr"][0], f["cc"][0]], float), np.array([f["rr"][-1], f["cc"][-1]], float)) for f in segs]
    for i in range(n):
        for j in range(i + 1, n):
            da = abs(segs[i]["strike"] - segs[j]["strike"])
            if min(da, 180 - da) > ang_tol:
                continue
            for a in ends[i]:
                for b in ends[j]:
                    if np.hypot(*(a - b)) <= gap_px:
                        u = ends[i][1] - ends[i][0]
                        u = u / (np.hypot(*u) or 1)
                        off = abs(u[0] * (b - a)[1] - u[1] * (b - a)[0])
                        if off <= off_tol:
                            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        rr = np.concatenate([segs[i]["rr"] for i in g]); cc = np.concatenate([segs[i]["cc"] for i in g])
        dr, dc = np.ptp(rr), np.ptp(cc)
        pts = np.c_[rr, cc].astype(float)
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        seen = low[:, rr, cc].mean(1) > 0.4
        out.append({"rr": rr, "cc": cc, "length": (math.hypot(dr, dc) + 1) * spacing, "elong": math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)),
                    "strike": math.degrees(math.atan2(evec[0, 1], evec[1, 1])) % 180,
                    "persist": float(pers[rr, cc].mean()), "win_idx": np.where(seen)[0], "starts": starts})
    return out


def _fault_lineaments(cube, wn, spacing, exclude=None):
    """Persistent low-coherence lineaments through time -> fault candidates (steep faults only)."""
    nil, nxl, ns = cube.shape
    starts = list(range(0, max(ns - wn, 1), wn)) or [0]
    low = []
    for a in starts:
        c = _coherence(cube[:, :, a:a + wn])
        c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = np.nanmedian(c)
        ref = c[~exclude] if (exclude is not None and (~exclude).any()) else c
        c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = np.nanmedian(ref)
        low.append((c < np.nanpercentile(ref, 10)) & (~exclude if exclude is not None else True))
    low = np.array(low)
    pers = low.mean(0)
    if exclude is not None and exclude.any():
        pers = np.where(exclude, 0.0, pers)
    cand = pers >= max(0.3, float(np.nanpercentile(pers[~exclude] if exclude is not None and (~exclude).any() else pers, 85)))
    cand = ndi.binary_closing(cand, structure=np.ones((3, 3))) | cand
    if exclude is not None:
        cand &= ~exclude
    from skimage.morphology import skeletonize
    skel = skeletonize(cand)
    nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junction = skel & (nb >= 3)
    branches = skel & ~ndi.binary_dilation(junction, structure=np.ones((3, 3)))
    lab, n = ndi.label(branches, structure=np.ones((3, 3)))
    out = []
    for b in range(1, n + 1):
        rr, cc = np.where(lab == b)
        if len(rr) < 6:
            continue
        path = _order_path(list(zip(rr.tolist(), cc.tolist())))
        for seg in _split_straight(path, tol=2.5):
            if len(seg) < 6:
                continue
            sr, sc = np.array([p[0] for p in seg]), np.array([p[1] for p in seg])
            dr, dc = sr[-1] - sr[0], sc[-1] - sc[0]
            length = (math.hypot(dr, dc) + 1) * spacing
            if math.hypot(dr, dc) < 6:
                continue
            seen = low[:, sr, sc].mean(1) > 0.4
            out.append({"rr": sr, "cc": sc, "length": length, "elong": math.hypot(dr, dc) / 2.0,
                        "strike": math.degrees(math.atan2(dc, dr)) % 180,
                        "persist": float(pers[sr, sc].mean()), "win_idx": np.where(seen)[0], "starts": starts})
    out = _merge_collinear(out, spacing, pers, low, starts)
    out.sort(key=lambda f: -f["length"])
    return out, pers, cand


def count_faults(volume="full stack", win_ms=40.0):
    """Count fault candidates over the WHOLE time range: low-coherence lineaments in the full stack that persist across many time windows; returns the number, strike, length, location and time extent of each, plus a map."""
    vn = _vol_name([volume, "full stack"])
    v = I.open_volume(vn)
    dt = float(np.median(np.diff(v.samples)))
    seg = _segments(vn, v.samples[0], v.samples[-1], max_traces=6000)
    wn = max(4, int(round(win_ms / dt)))
    filled, valid, edge = prep_cube_for_faults(seg["cube"])
    fl, pers, cand = _fault_lineaments(filled, wn, seg["spacing"], exclude=edge)
    faults = []
    for i, f in enumerate(fl[:12]):
        t_a = v.samples[0] + dt * f["starts"][f["win_idx"].min()] if len(f["win_idx"]) else None
        t_b = v.samples[0] + dt * (f["starts"][f["win_idx"].max()] + wn) if len(f["win_idx"]) else None
        faults.append({"id": f"F{i+1}", "length_map_units": _f(f["length"], 0), "elongation": _f(f["elong"], 1),
                       "strike_deg_from_inline_direction": _f(f["strike"], 0), "persistence_0_1": _f(f["persist"], 2),
                       "centre_inline_xline": [int(seg["il"][int(f["rr"].mean())]), int(seg["xl"][int(f["cc"].mean())])],
                       "seen_between_twt_ms": [_f(t_a, 0), _f(t_b, 0)]})
    mask = np.zeros(pers.shape, bool)
    for f in fl:
        mask[f["rr"], f["cc"]] = True
    LAST_FAULTS.clear()
    LAST_FAULTS.update({"seg": seg, "mask": mask, "method": "coherence"})
    img = _plot("fault_overview", seg, [{"kind": "image", "data": pers, "cmap": "gray_r", "label": "fraction of windows with discontinuity"},
                                         {"kind": "mask", "data": mask}],
                f"Persistent discontinuity (dark) and fault candidates (red): {len(fl)}")
    return {"volume": vn, "windows_used": int(len(range(0, max(seg['cube'].shape[2] - wn, 1), wn))), "window_ms": win_ms,
            "n_fault_candidates": len(fl), "faults": faults, "image_path": img, "kind": "map",
            "index_used": "all times", "lateral_decimation_every_n_lines": seg["step"],
            "survey_edge_excluded": True, "caveat": "Survey-edge artefacts are masked out. Counts persistent steep discontinuities only; low-angle faults, noise and acquisition footprint can add/remove candidates. Not a fault interpretation."}
