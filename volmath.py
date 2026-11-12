"""
volmath.py - arithmetic on seismic volumes inside a TWT window, plus a composite gas-potential screen.

volume_math(expression, ...)  evaluates e.g. "far_stack - near_stack" or "sand_probability * (far - near)" trace-by-trace inside the window,
                              reduces to a map, and reports where the values are extreme (clusters with inline/xline/x/y and nearest well).
gas_potential(...)            combines three criteria the geoscientist asked for: far-minus-near strongly negative (AVO brightening),
                              structurally high, and high sand probability; ranks zones and checks them against well gas zones.
"""
import ast
import json
import math
import re

import numpy as np
from scipy import ndimage as ndi

import ingest as I
import interpret as N

ALIASES = {"far": "far_stack", "near": "near_stack", "mid": "mid_stack", "full": "full_stack", "ultrafar": "ultrafar_stack",
           "ultra_far": "ultrafar_stack", "sand": "sand_probability", "sandprob": "sand_probability", "ip": "p_impedance",
           "pimp": "p_impedance", "is": "s_impedance", "simp": "s_impedance", "vpvs": "vpvs"}
FUNCS = {"abs": np.abs, "sqrt": lambda a: np.sqrt(np.clip(a, 0, None)), "log": lambda a: np.log(np.clip(a, 1e-12, None)),
         "log10": lambda a: np.log10(np.clip(a, 1e-12, None)), "exp": np.exp, "minimum": np.minimum, "maximum": np.maximum, "sign": np.sign}
KEYSEP = 100003


def _f(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _names(expr):
    tree = ast.parse(expr.strip(), mode="eval")
    return [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in FUNCS]


def _resolve(name):
    cards = I.load_cards()["volumes"]
    low = name.lower()
    if low in cards:
        return low
    if low in ALIASES and ALIASES[low] in cards:
        return ALIASES[low]
    for k, v in cards.items():
        if low in k or low in v["label"].lower().replace(" ", "_").replace("-", "_"):
            return k
    raise ValueError(f"unknown volume '{name}'. Available: {list(cards)} (aliases: far, near, mid, full, ultrafar, sand, ip, is, vpvs)")


def _safe_eval(expr, env):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = ev(n.left), ev(n.right)
            return {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: lambda x, y: np.divide(x, np.where(np.abs(y) < 1e-12, np.nan, y)),
                    ast.Pow: np.power}[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS:
            return FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("only + - * / ** , numbers, volume names and abs/sqrt/log/log10/exp/minimum/maximum/sign are allowed")
    return ev(ast.parse(expr.strip(), mode="eval"))


def _axes(ref, max_traces):
    v = I.open_volume(ref)
    uil, uxl = np.unique(v.il), np.unique(v.xl)
    if len(uil) < 3 or len(uxl) < 3:
        raise ValueError("survey grid is degenerate; run ingest_all.run_all(force=True)")
    s = max(1, int(math.ceil(math.sqrt(len(uil) * len(uxl) / max_traces))))
    return uil[::s], uxl[::s], s


def _block(vname, t0, t1, sil, sxl, tgrid):
    """Samples of one volume inside the window on a common time axis -> (ntraces, nt) array (NaN where the volume has no trace)."""
    v = I.open_volume(vname)
    keys = v.il.astype(np.int64) * KEYSEP + v.xl
    order = np.argsort(keys)
    sk = keys[order]
    tgt = (sil[:, None].astype(np.int64) * KEYSEP + sxl[None, :]).ravel()
    pos = np.clip(np.searchsorted(sk, tgt), 0, len(sk) - 1)
    ok = sk[pos] == tgt
    k = (v.samples >= tgt_lo(tgrid)) & (v.samples <= tgt_hi(tgrid))
    same = k.sum() == len(tgrid) and np.allclose(v.samples[k], tgrid, atol=1e-3)
    out = np.full((len(tgt), len(tgrid)), np.nan)
    for n in np.where(ok)[0]:
        tr = v.trace(order[pos[n]])
        out[n] = tr[k] if same else np.interp(tgrid, v.samples, tr)
    return out, v, order[pos], ok


def tgt_lo(t):
    return t[0] - 1e-3


def tgt_hi(t):
    return t[-1] + 1e-3


REDUCERS = {"mean": lambda a: np.nanmean(a, 1), "rms": lambda a: np.sqrt(np.nanmean(a ** 2, 1)), "min": lambda a: np.nanmin(a, 1),
            "max": lambda a: np.nanmax(a, 1), "mean_abs": lambda a: np.nanmean(np.abs(a), 1), "median": lambda a: np.nanmedian(a, 1)}


def compute_map(expression, t0, t1, reducer="mean", normalise=False, max_traces=15000):
    """-> dict(grid, il, xl, vol, idx(index of each grid cell's trace), ok, names, step)."""
    names = {n: _resolve(n) for n in _names(expression)}
    if not names:
        raise ValueError("expression must contain at least one volume name")
    ref = next(iter(names.values()))
    sil, sxl, s = _axes(ref, max_traces)
    vol0 = I.open_volume(ref)
    dt = float(np.median(np.diff(vol0.samples)))
    tgrid = np.arange(math.ceil(t0 / dt) * dt, t1 + 1e-9, dt)
    if len(tgrid) < 1:
        raise ValueError("window too narrow for the sample rate")
    env, vols, ok_all, idx_ref = {}, {}, None, None
    for alias, vn in names.items():
        if vn not in vols:
            vols[vn] = _block(vn, t0, t1, sil, sxl, tgrid)
        blk, v, idx, ok = vols[vn]
        b = blk.copy()
        if normalise and I.load_cards()["volumes"][vn]["kind"] == "stack":
            sd = np.nanstd(b)
            b = b / sd if sd > 0 else b
        env[alias] = b
        ok_all = ok if ok_all is None else (ok_all & ok)
        idx_ref = idx if idx_ref is None else idx_ref
    res = _safe_eval(expression, env)
    res = np.asarray(res, float)
    if res.ndim == 0:
        res = np.full((len(sil) * len(sxl), len(tgrid)), float(res))
    with np.errstate(all="ignore"):
        vals = REDUCERS[reducer](res)
    vals[~ok_all] = np.nan
    return {"grid": vals.reshape(len(sil), len(sxl)), "il": sil, "xl": sxl, "vol": vol0, "idx": idx_ref.reshape(len(sil), len(sxl)),
            "ok": ok_all.reshape(len(sil), len(sxl)), "names": names, "step": s, "n_samples": len(tgrid), "dt": dt}


def _cell_xy(m):
    v = m["vol"]
    X, Y = np.full(m["grid"].shape, np.nan), np.full(m["grid"].shape, np.nan)
    ok = m["ok"]
    X[ok], Y[ok] = v.x[m["idx"][ok]], v.y[m["idx"][ok]]
    return X, Y


def _cell_area(m):
    X, Y = _cell_xy(m)
    dx = np.nanmedian(np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)))
    dy = np.nanmedian(np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)))
    return float(dx * dy)


def _clusters(mask, m, grid, top=5, min_cells=2):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    X, Y = _cell_xy(m)
    ca = _cell_area(m)
    wells = {}
    try:
        wells = I.load_cards()["wells"]
    except Exception:
        pass
    rows = []
    for k in range(1, n + 1):
        r, c = np.where(lab == k)
        if len(r) < min_cells:
            continue
        cx, cy = float(np.nanmean(X[r, c])), float(np.nanmean(Y[r, c]))
        nw = min(wells.items(), key=lambda kv: math.hypot(kv[1]["x"] - cx, kv[1]["y"] - cy)) if wells else None
        rows.append({"cells": int(len(r)), "area_km2_if_metres": _f(len(r) * ca / 1e6, 3), "inline": int(round(float(np.mean(m["il"][r])))), "xline": int(round(float(np.mean(m["xl"][c])))),
                     "x": _f(cx, 0), "y": _f(cy, 0), "mean_value": _f(np.nanmean(grid[r, c])), "extreme_value": _f(np.nanmin(grid[r, c]) if np.nanmean(grid[r, c]) < np.nanmedian(grid) else np.nanmax(grid[r, c])),
                     "nearest_well": nw[0] if nw else None, "distance_to_nearest_well": _f(math.hypot(nw[1]["x"] - cx, nw[1]["y"] - cy), 0) if nw else None})
    rows.sort(key=lambda r: -r["cells"])
    return rows[:top], lab


def _plot(name, panels, m, wells_il, suptitle, zones=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (I.CACHE_DIR / "images").mkdir(parents=True, exist_ok=True)
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.2 * rows), squeeze=False)
    ext = [m["xl"][0], m["xl"][-1], m["il"][0], m["il"][-1]]
    for ax, (title, g, cmap, vlim) in zip(axs.ravel(), panels):
        kw = {} if vlim is None else {"vmin": vlim[0], "vmax": vlim[1]}
        im = ax.imshow(g, origin="lower", extent=ext, aspect="auto", cmap=cmap, **kw)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for w, (i, x) in wells_il.items():
            ax.plot(x, i, "k^", ms=5, mec="w")
            ax.text(x, i, " " + w, fontsize=7)
        if zones is not None:
            ax.contour(np.linspace(ext[0], ext[1], zones.shape[1]), np.linspace(ext[2], ext[3], zones.shape[0]), zones.astype(float), levels=[0.5], colors="lime", linewidths=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("xline")
        ax.set_ylabel("inline")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    path = I.CACHE_DIR / "images" / f"{name}.png"
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _window_or_default(t0, t1):
    """If the user gave no window, use the interpreted horizon time range (median +-75 ms) when available."""
    if t0 is not None and t1 is not None:
        return float(t0), float(t1), None
    try:
        import interp_data as D
        h = D.horizon()
        if h is not None:
            mid = float(np.nanmedian(h["t"]))
            return mid - 75.0, mid + 75.0, f"no window given: used {mid - 75:.0f}-{mid + 75:.0f} ms around the interpreted horizon"
    except Exception:
        pass
    return None, None, None


def volume_math(expression, twt_min_ms=None, twt_max_ms=None, reducer="mean", select="low", percent=10.0, normalise=False):
    """Do arithmetic on seismic volumes inside a TWT window and find where the result is extreme. expression uses volume names or aliases (far, near, mid, full, ultrafar, sand, ip, is, vpvs), e.g. 'far - near' or 'sand * (far - near)'. reducer = mean|rms|min|max|mean_abs|median over the window; select = low (most negative / smallest) or high; percent = share of the map treated as extreme."""
    twt_min_ms, twt_max_ms, wnote = _window_or_default(twt_min_ms, twt_max_ms)
    if twt_min_ms is None:
        return {"error": "give twt_min_ms and twt_max_ms (the window)"}
    if reducer not in REDUCERS:
        return {"error": f"reducer must be one of {list(REDUCERS)}"}
    try:
        m = compute_map(expression, float(twt_min_ms), float(twt_max_ms), reducer, bool(normalise))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    g = m["grid"]
    vals = g[np.isfinite(g)]
    if not len(vals):
        return {"error": "no valid samples inside the window (check the window and volume time range)"}
    thr = float(np.percentile(vals, percent if select == "low" else 100 - percent))
    mask = (g <= thr) if select == "low" else (g >= thr)
    mask &= np.isfinite(g)
    zones, _ = _clusters(mask, m, g)
    img = _plot("volume_math", [(f"{expression}  [{reducer}, {twt_min_ms:.0f}-{twt_max_ms:.0f} ms]", g, "RdBu", None),
                                (f"extreme {select} {percent:g}% (threshold {thr:.3g})", mask.astype(float), "gray_r", (0, 1))], m,
                N._well_ilxl(m["vol"]), "Volume math", zones=mask)
    out = {"window_note": wnote, "expression": expression, "window_ms": [twt_min_ms, twt_max_ms], "reducer": reducer, "volumes_used": m["names"], "normalised_stacks": bool(normalise),
           "stats": {"min": _f(vals.min()), "p10": _f(np.percentile(vals, 10)), "median": _f(np.median(vals)), "p90": _f(np.percentile(vals, 90)), "max": _f(vals.max())},
           "selection": f"{select} {percent:g}% of the map (threshold {thr:.4g})", "zones": zones, "n_cells": int(np.isfinite(g).sum()),
           "map_decimation_every_n_lines": m["step"], "image_path": img, "kind": "map", "index_used": f"{twt_min_ms:.0f}-{twt_max_ms:.0f}ms", "volume": expression}
    return out


def _rank01(a):
    """Rank-normalise to 0..1 (1 = largest) ignoring NaN."""
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return out
    r = a[ok].argsort().argsort() / (ok.sum() - 1.0)
    out[ok] = r
    return out


def gas_potential(twt_min_ms=None, twt_max_ms=None, avo_expression="far - near", avo_percent=15.0, sand_threshold=None, structural_percent=40.0, normalise=True):
    """Where is the probable gas? Screens three criteria inside a TWT window: (1) far minus near stack strongly NEGATIVE (AVO brightening; the lowest avo_percent of the map), (2) structurally HIGH on the interpreted horizon (shallowest structural_percent), (3) HIGH sand probability (>= sand_threshold, default top 30 percent). Ranks zones where all criteria coincide, checks them against wells with flagged gas, and returns a committed answer."""
    twt_min_ms, twt_max_ms, wnote = _window_or_default(twt_min_ms, twt_max_ms)
    if twt_min_ms is None:
        return {"error": "give twt_min_ms and twt_max_ms (the window)"}
    t0, t1 = float(twt_min_ms), float(twt_max_ms)
    try:
        A = compute_map(avo_expression, t0, t1, "mean", bool(normalise))
    except Exception as e:
        return {"error": f"AVO map failed: {type(e).__name__}: {e}"}
    g = A["grid"]
    notes, crit = ([wnote] if wnote else []), {}
    avo_thr = float(np.nanpercentile(g, avo_percent))
    crit["avo_negative"] = (g <= avo_thr) & np.isfinite(g)
    S = None
    try:
        S = compute_map("sand_probability", t0, t1, "mean", False)
        S = _regrid_like(S, A)
        thr = float(sand_threshold) if sand_threshold is not None else float(np.nanpercentile(S, 70))
        crit["high_sand"] = (S >= thr) & np.isfinite(S)
    except Exception as e:
        notes.append(f"sand probability criterion skipped: {e}")
        thr = None
    H = None
    try:
        import interp_data as D
        h = D.horizon()
        if h is not None:
            from scipy.interpolate import RegularGridInterpolator
            X, Y = _cell_xy(A)
            Hi = RegularGridInterpolator((h["gy"], h["gx"]), h["H"], bounds_error=False, fill_value=np.nan)
            H = Hi(np.c_[Y.ravel(), X.ravel()]).reshape(X.shape)
            if np.isfinite(H).sum() > 10:
                hmed = float(np.nanmedian(H))
                if hmed < t0 - 100 or hmed > t1 + 100:
                    notes.append(f"the interpreted horizon (median {hmed:.0f} ms) lies outside the analysed window {t0:.0f}-{t1:.0f} ms, so the structural criterion is plan-view structural position only")
                hthr = float(np.nanpercentile(H, structural_percent))
                crit["structural_high"] = (H <= hthr) & np.isfinite(H)
            else:
                notes.append("interpreted horizon does not cover this area; structural criterion skipped")
                H = None
        else:
            notes.append("no interpreted horizon ingested; structural criterion skipped")
    except Exception as e:
        notes.append(f"structural criterion skipped: {e}")
    count = sum(c.astype(int) for c in crit.values())
    nmax = len(crit)
    best = count == nmax
    zones, lab = _clusters(best, A, g)
    level = nmax
    if not zones and nmax > 1:
        best = count >= nmax - 1
        zones, lab = _clusters(best, A, g)
        level = nmax - 1
        notes.append(f"no place meets all {nmax} criteria; zones below meet {level} of {nmax}")
    soft = np.nanmean(np.stack([1 - _rank01(g)] + ([_rank01(S)] if S is not None else []) + ([1 - _rank01(H)] if H is not None else [])), 0)
    for z in zones:
        r, c = np.argmin(abs(A["il"] - z["inline"])), np.argmin(abs(A["xl"] - z["xline"]))
        z["criteria_met_at_centre"] = int(count[r, c])
        z["mean_composite_score_0_1"] = _f(np.nanmean(soft[lab == lab[r, c]]) if lab[r, c] > 0 else soft[r, c], 2)
        z["avo_value_at_centre"] = _f(g[r, c])
        if S is not None:
            z["sand_probability_at_centre"] = _f(S[r, c], 2)
        if H is not None:
            z["horizon_twt_at_centre_ms"] = _f(H[r, c], 0)
    zones.sort(key=lambda z: -(z.get("mean_composite_score_0_1") or 0))
    wellchk = []
    try:
        gz = N.thickest_gas_zone()["ranking"]
        wil = N._well_ilxl(A["vol"])
        for w in gz[:8]:
            ov = w["top_twt_ms"] <= t1 and w["base_twt_ms"] >= t0
            if w["well"] in wil:
                i, x = wil[w["well"]]
                r, c = np.argmin(abs(A["il"] - i)), np.argmin(abs(A["xl"] - x))
                wellchk.append({"well": w["well"], "gas_top_twt_ms": w["top_twt_ms"], "gas_base_twt_ms": w["base_twt_ms"], "net_pay_m": w.get("net_pay_m"),
                                "gas_interval_inside_window": bool(ov), "criteria_met_at_well": int(count[r, c]), "of": nmax})
    except Exception:
        pass
    inside = [w for w in wellchk if w["gas_interval_inside_window"]]
    agree = [w for w in inside if w["criteria_met_at_well"] >= max(1, nmax - 1)]
    panels = [(f"{avo_expression} (mean, {t0:.0f}-{t1:.0f} ms){' [stacks normalised]' if normalise else ''}", g, "RdBu", None)]
    if S is not None:
        panels.append(("sand probability (mean)", S, "viridis", (0, 1)))
    if H is not None:
        panels.append(("interpreted horizon TWT (ms)", H, "viridis_r", None))
    panels.append((f"criteria met (0-{nmax}); green = zones", count.astype(float), "magma", (0, nmax)))
    img = _plot("gas_potential", panels, A, N._well_ilxl(A["vol"]), "Probable gas screen", zones=best)
    if zones:
        z = zones[0]
        ans = (f"Most probable gas location: inline {z['inline']}, xline {z['xline']} (x {z['x']}, y {z['y']}), window {t0:.0f}-{t1:.0f} ms"
               f"{', horizon at ' + str(z['horizon_twt_at_centre_ms']) + ' ms' if z.get('horizon_twt_at_centre_ms') else ''}; "
               f"{level} of {nmax} criteria met; nearest well {z['nearest_well']} ({z['distance_to_nearest_well']} map units away).")
    else:
        ans = "No zone meets the criteria together in this window; relax avo_percent / structural_percent or try another window."
    conf = "medium" if (zones and level == nmax and agree) else "low-medium" if zones and level == nmax else "low"
    if inside and not agree:
        notes.append("wells with flagged gas inside the window do NOT sit where the criteria are met; treat the screen with caution")
    chain = [f"AVO: {avo_expression} (mean over the window{', stacks scaled to unit variance' if normalise else ''}); 'high negative' = lowest {avo_percent:g}% of the map, threshold {avo_thr:.3g}"]
    if S is not None:
        chain.append(f"Sand: mean sand probability >= {thr:.2f}" + ("" if sand_threshold is not None else " (top 30% of the map)"))
    if H is not None:
        chain.append(f"Structure: interpreted horizon in the shallowest {structural_percent:g}% of its time range")
    chain.append(f"Combined: {int((count == nmax).sum())} cells meet all {nmax} criteria; zones ranked by a composite rank score")
    res = {"suggested_answer": ans, "confidence": conf, "window_ms": [t0, t1], "criteria": {"avo_expression": avo_expression, "avo_percent": avo_percent, "sand_threshold": thr,
           "structural_percent": structural_percent, "n_criteria_available": nmax}, "reasoning_chain": chain, "zones": zones[:5], "well_check": wellchk,
           "wells_agreeing": [w["well"] for w in agree], "notes": notes,
           "caveats": ["Far-minus-near is a screening attribute: negative values can also come from tuning, hard streaks or noise, so confirm on gathers/AVO crossplots",
                       "Maps are decimated (every N lines) for speed",
                       "Structural height uses the interpreted horizon, which may not be the reservoir top"],
           "next_steps": ["Check the zone on inline/crossline sections and on far/near stacks", "Crossplot intercept-gradient or impedance at the zone against the well gas zones", "Compare with wells outside the zone (water/shale response)"],
           "image_path": img, "kind": "map", "index_used": f"{t0:.0f}-{t1:.0f}ms", "volume": "gas screen"}
    res["answer_markdown"] = render_gas(res)
    return res


def _regrid_like(S, A):
    """Sand map computed on its own grid -> same cells as A (same il/xl axes by construction, so just return the grid)."""
    return S["grid"] if S["grid"].shape == A["grid"].shape and np.array_equal(S["il"], A["il"]) and np.array_equal(S["xl"], A["xl"]) else _resample(S, A)


def _resample(S, A):
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((S["il"], S["xl"]), np.nan_to_num(S["grid"], nan=np.nanmean(S["grid"])), bounds_error=False, fill_value=np.nan)
    II, XX = np.meshgrid(A["il"], A["xl"], indexing="ij")
    return f(np.c_[II.ravel(), XX.ravel()]).reshape(II.shape)


def render_gas(r):
    L = [f"**SUGGESTED ANSWER:** {r['suggested_answer']}  (confidence: {r['confidence']})", "", "**WHY (reasoning from the data):**"]
    L += [f"{i}. {c}" for i, c in enumerate(r["reasoning_chain"], 1)]
    if r["zones"]:
        L += ["", "**WHERE TO LOOK:**"]
        for i, z in enumerate(r["zones"][:3], 1):
            L.append(f"{i}. inline {z['inline']}, xline {z['xline']}, about {z['area_km2_if_metres']} km2 (if metres), criteria met {z.get('criteria_met_at_centre')}, "
                     f"composite score {z.get('mean_composite_score_0_1')}, nearest well {z['nearest_well']}")
    if r["well_check"]:
        L += ["", "**WELL CHECK:** " + "; ".join(f"{w['well']}: gas {w['gas_top_twt_ms']:.0f}-{w['gas_base_twt_ms']:.0f} ms ({'in' if w['gas_interval_inside_window'] else 'outside'} window), criteria met {w['criteria_met_at_well']}/{w['of']}" for w in r["well_check"][:6])]
    if r["notes"]:
        L += ["", "**NOTES:** " + " | ".join(r["notes"])]
    L += ["", "**CAVEATS:** " + " | ".join(r["caveats"]), "", "**NEXT STEPS:** " + " | ".join(r["next_steps"])]
    return "\n".join(L)
