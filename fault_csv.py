"""
fault_csv.py - reads interpreted faults delivered as CSV tables:

    Fault (CSV)/IL_timeslice_Fault_01.csv   inline numbers of the fault points of slice 01
    Fault (CSV)/CL_timeslice_Fault_01.csv   crossline numbers of the same points (paired row by row)
    Fault (CSV)/TWT_timeslice_Fault.csv     two-way time of each slice

The layout is detected, not assumed. Supported shapes (header row optional):
  * IL and CL each one column, same length                      -> points of one slice
  * IL and CL each several columns of the same shape            -> one fault trace per column (blank cells ignored)
  * TWT: one value per slice | 2 columns (slice number, time) | one column per slice | one value per point (all slices stacked)
Inline/xline are converted to map X/Y with the raw seismic geometry, then points are grouped into fault traces.
Run `python fault_csv.py [folder]` to print how a folder was understood.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = re.compile(r"^(?P<ax>il|inline|cl|xl|crossline|xline|twt|time)[_\-\s]", re.I)
TRAIL = re.compile(r"(\d+)$")


def classify(p):
    """-> ('il'|'cl'|'twt', slice_number_or_None) or None if the file is not part of a fault CSV set."""
    m = PREFIX.match(p.stem)
    if not m:
        return None
    ax = m.group("ax").lower()
    ax = "il" if ax in ("il", "inline") else "twt" if ax in ("twt", "time") else "cl"
    t = TRAIL.search(p.stem)
    return ax, (int(t.group(1)) if t else None)


def is_fault_csv(p, rel):
    return p.suffix.lower() == ".csv" and "fault" in rel.lower() and (classify(p) is not None or NAME3.match(p.stem) is not None)


def read_table(path):
    """Numeric table from a CSV with or without header -> (DataFrame of floats, header list or None)."""
    head = Path(path).read_text(errors="ignore")[:4000]
    sep = next((s for s in (",", ";", "\t") if s in head), None)
    if sep is None:
        sep = r"\s+" if re.search(r"\S\s+\S", head) else ","
    raw = pd.read_csv(path, header=None, sep=sep, engine="python" if sep == r"\s+" else "c", dtype=str, skip_blank_lines=True)
    raw = raw.apply(lambda c: c.str.strip() if c.dtype == object else c)
    num = raw.apply(pd.to_numeric, errors="coerce")
    header = None
    if len(raw) and num.iloc[0].isna().any() and raw.iloc[0].notna().any():
        header = [str(v) for v in raw.iloc[0].tolist()]
        num = num.iloc[1:]
    num = num.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    return num.astype(float), header


def _drop_index_col(df):
    """Remove a leading 0..n-1 / 1..n counter column."""
    if df.shape[1] >= 2:
        c = df.iloc[:, 0].dropna().to_numpy()
        if len(c) > 2 and np.all(np.diff(c) == 1) and c[0] in (0, 1):
            return df.iloc[:, 1:]
    return df


def pair(il_df, cl_df, warns, name):
    """-> list of (n,2) arrays of (inline, xline); each array is one explicitly given fault trace/group."""
    a, b = _drop_index_col(il_df), _drop_index_col(cl_df)
    if a.shape[1] == b.shape[1] and a.shape[1] > 1:
        groups = []
        for j in range(a.shape[1]):
            p = np.c_[a.iloc[:, j].to_numpy(), b.iloc[:, j].to_numpy()]
            p = p[np.isfinite(p).all(1)]
            if len(p):
                groups.append(p)
        return groups
    x, y = a.to_numpy().ravel(order="F"), b.to_numpy().ravel(order="F")
    if len(x) != len(y):
        n = min(len(x), len(y))
        warns.append(f"{name}: IL has {len(x)} values but CL has {len(y)}; paired the first {n}")
        x, y = x[:n], y[:n]
    p = np.c_[x, y]
    return [p[np.isfinite(p).all(1)]]


_TREE = {}


def to_xy(il, xl, geom_name):
    """Inline/xline -> map X/Y using the trace geometry of a raw volume (linear interpolation, nearest trace outside the hull)."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree
    import ingest as I
    if geom_name not in _TREE:
        g = np.load(I.CACHE_DIR / f"geom_{geom_name}.npz")
        sel = np.linspace(0, len(g["x"]) - 1, min(80000, len(g["x"]))).astype(int)
        P = np.c_[g["il"][sel], g["xl"][sel]].astype(float)
        _TREE[geom_name] = (LinearNDInterpolator(P, g["x"][sel]), LinearNDInterpolator(P, g["y"][sel]), cKDTree(P), g["x"][sel], g["y"][sel], P)
    fx, fy, tree, gx, gy, P = _TREE[geom_name]
    q = np.c_[il, xl].astype(float)
    x, y = fx(q), fy(q)
    bad = ~np.isfinite(x) | ~np.isfinite(y)
    if bad.any():
        d, i = tree.query(q[bad])
        x[bad], y[bad] = gx[i], gy[i]
    inside = float(np.mean(tree.query(q)[0] <= 3.0))
    return np.c_[x, y], inside


def parse_twt(df, slice_ids, npts, warns):
    """-> {slice_id: time_ms} from the TWT table, or {} if the layout is not understood."""
    if df is None or df.empty:
        return {}
    ids = sorted(slice_ids)
    n = len(ids)
    out = None
    if df.shape[1] == 1 and len(df) == n:
        out = dict(zip(ids, df.iloc[:, 0].tolist()))
    elif df.shape[1] >= 2 and set(ids) <= set(int(v) for v in df.iloc[:, 0].dropna().tolist() if float(v).is_integer()):
        out = {int(k): float(v) for k, v in zip(df.iloc[:, 0], df.iloc[:, 1]) if np.isfinite(k) and float(k).is_integer()}
    elif df.shape[1] == n and n > 1:
        out = {sid: float(np.nanmedian(df.iloc[:, j])) for j, sid in enumerate(ids)}
    elif df.shape[1] == 1 and len(df) == sum(npts.get(i, 0) for i in ids) and len(df) > 0:
        v, o, k = df.iloc[:, 0].to_numpy(), {}, 0
        for sid in ids:
            o[sid] = float(np.nanmedian(v[k:k + npts[sid]]))
            k += npts[sid]
        out = o
    elif df.shape[1] == 1 and len(df) == 1:
        out = {sid: float(df.iloc[0, 0]) for sid in ids}
        warns.append("TWT file has a single value: used for every slice")
    if out is None:
        warns.append(f"TWT table shape {df.shape} not understood for {n} slices; add slice_times.csv or data_config.json")
        return {}
    vals = np.array([v for v in out.values() if np.isfinite(v)])
    if len(vals) and np.nanmedian(np.abs(vals)) < 20:
        out = {k: v * 1000.0 for k, v in out.items()}       # seconds -> ms
        warns.append("TWT values look like seconds: converted to ms")
    return {k: abs(float(v)) for k, v in out.items() if np.isfinite(v)}


def traces_from_points(xy, group_given=False):
    """Group map points into connected fault traces and order each along its main axis. -> (list of (n,2) arrays, n_dropped_points)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    xy = np.unique(np.round(xy, 3), axis=0)
    if len(xy) < 3:
        return [], len(xy)
    tree = cKDTree(xy)
    nn = tree.query(xy, k=2)[0][:, 1]
    r = 2.6 * float(np.median(nn[nn > 0])) if (nn > 0).any() else 1.0
    pairs = np.array(list(tree.query_pairs(r)))
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
        n, lab = connected_components(g, directed=False)
    else:
        n, lab = len(xy), np.arange(len(xy))
    out, dropped = [], 0
    for k in range(n):
        p = xy[lab == k]
        if len(p) < 3:
            dropped += len(p)
            continue
        c = p - p.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        out.append(p[np.argsort(c @ vt[0])])
    return out, dropped


NAME3 = re.compile(r"^(?P<ax>TWT|IL|CL|XL)[_\-\s](?P<line>\d+(?:\.\d+)?)[_\-\s]*Fault[_\-\s]*(?P<k>\d+)?", re.I)


def _cols(df, header):
    """Map columns to roles using the header (or the common 5-column FaultID, StickID, IL, CL, TWT layout)."""
    roles = {}
    if header and len(header) == df.shape[1]:
        for j, h in enumerate(header):
            h = re.sub(r"[^a-z]", "", h.lower())
            if h in ("il", "inline", "iline") and "il" not in roles:
                roles["il"] = j
            elif h in ("cl", "xl", "crossline", "xline") and "cl" not in roles:
                roles["cl"] = j
            elif h.startswith(("twt", "time")) and "t" not in roles:
                roles["t"] = j
            elif h in ("faultid", "fault", "faultno") and "fault" not in roles:
                roles["fault"] = j
            elif h in ("stickid", "stick") and "stick" not in roles:
                roles["stick"] = j
    elif df.shape[1] == 5:
        roles = {"fault": 0, "stick": 1, "il": 2, "cl": 3, "t": 4}
    elif df.shape[1] == 3:
        roles = {"il": 0, "cl": 1, "t": 2}
    return roles if {"il", "cl"} <= set(roles) else None


def table_mode(files):
    """True when the CSVs are self-contained tables (IL, CL and TWT in each file), e.g. TWT_1500_Fault_01.csv, IL_1800_Fault_01.csv."""
    hits = 0
    for p in files[:6]:
        if NAME3.match(p.stem):
            df, h = read_table(p)
            hits += _cols(df, h) is not None
    return hits > 0


def ingest_tables(root, files, geom_name, warns):
    """Self-contained fault tables -> slice records. TWT_<t>_* = map-view trace at time t; IL_<n>_* / CL_<n>_* = stick(s) on a section (vertices carry TWT)."""
    tslices, recs = {}, []
    for p in sorted(files):
        m = NAME3.match(p.stem)
        if not m:
            continue
        df, h = read_table(p)
        c = _cols(df, h)
        if c is None or df.empty:
            warns.append(f"{p.name}: columns not understood (need IL, CL, TWT); header={h}")
            continue
        ax, line, k = m.group("ax").lower(), float(m.group("line")), int(m.group("k") or 1)
        il, xl = df.iloc[:, c["il"]].to_numpy(), df.iloc[:, c["cl"]].to_numpy()
        tt = df.iloc[:, c["t"]].to_numpy() if "t" in c else np.full(len(df), line if ax == "twt" else np.nan)
        ok = np.isfinite(il) & np.isfinite(xl)
        il, xl, tt = il[ok], xl[ok], tt[ok]
        if len(il) < 2:
            continue
        sid = df.iloc[:, c["stick"]].to_numpy()[ok] if "stick" in c else np.zeros(len(il))
        fid = df.iloc[:, c["fault"]].to_numpy()[ok] if "fault" in c else np.zeros(len(il))
        xy, inside = to_xy(il, xl, geom_name) if geom_name else (np.c_[il, xl], 0.0)
        if geom_name and inside < 0.5:
            warns.append(f"{p.name}: only {inside * 100:.0f}% of IL/CL points fall inside the seismic grid; check IL/CL values")
        feats = []
        for f_, s_ in sorted({(a, b) for a, b in zip(fid, sid)}):
            sel = (fid == f_) & (sid == s_)
            ring = xy[sel]
            tv = tt[sel]
            if len(ring) < 2:
                continue
            if ax != "twt":
                o = np.argsort(tv)
                ring, tv = ring[o], tv[o]
            feats.append({"rings": [ring], "geom": "line", "z": None, "t": tv,
                          "attrs": {"FaultID": int(f_), "StickID": int(s_), "file": p.name, "fault_no": k}})
        if not feats:
            continue
        if ax == "twt":
            tslices.setdefault(float(np.nanmedian(tt)) if np.isfinite(np.nanmedian(tt)) else line, []).extend(feats)
        else:
            allt = np.concatenate([f["t"] for f in feats])
            recs.append({"type": "shp", "rel": f"Fault (CSV)/{p.name}", "root": str(root), "kind": "fault", "cls": "fault", "idx": int(line),
                         "key": f"{ax.upper()}{int(line)}_fault_{k:02d}", "geom": "line", "feats": feats, "crs": "", "fields": ["FaultID", "StickID"],
                         "source": "fault_csv", "section": f"{ax.upper()} {int(line)}",
                         "coord_mode": "inline/xline converted to X/Y" if geom_name else "inline/xline (not converted)",
                         "time_ms": float(np.nanmedian(allt)), "t_range": [float(np.nanmin(allt)), float(np.nanmax(allt))],
                         "time_source": "TWT column of the file"})
    for tm, feats in sorted(tslices.items()):
        recs.append({"type": "shp", "rel": f"Fault (CSV)/TWT_{int(tm)}", "root": str(root), "kind": "fault", "cls": "fault", "idx": int(tm),
                     "key": f"timeslice_fault_{int(tm)}", "geom": "line", "feats": feats, "crs": "", "fields": ["FaultID", "StickID"],
                     "source": "fault_csv", "section": None,
                     "coord_mode": "inline/xline converted to X/Y" if geom_name else "inline/xline (not converted)",
                     "time_ms": float(tm), "time_source": "TWT column / file name"})
    if not geom_name:
        warns.append("Fault CSV: raw seismic geometry not available, so inline/xline could not be converted to X/Y. Ingest the raw data first.")
    return recs


def ingest_group(root, files, geom_name, cfg_times, warns):
    """files: list of Paths. -> list of slice records compatible with interp_data (kind 'fault')."""
    if table_mode(sorted(files)):
        return ingest_tables(root, files, geom_name, warns)
    kinds = {"il": {}, "cl": {}, "twt": None}
    for p in files:
        c = classify(p)
        if not c:
            continue
        ax, idx = c
        if ax == "twt":
            kinds["twt"] = p
        elif idx is not None:
            kinds[ax][idx] = p
    ids = sorted(set(kinds["il"]) & set(kinds["cl"]))
    for miss in sorted(set(kinds["il"]) ^ set(kinds["cl"])):
        warns.append(f"Fault CSV slice {miss}: only one of IL / CL file found; skipped")
    if not ids:
        raise ValueError("no matching IL_/CL_ file pairs found")
    pts, npts = {}, {}
    for i in ids:
        a, _ = read_table(kinds["il"][i])
        b, _ = read_table(kinds["cl"][i])
        groups = pair(a, b, warns, f"slice {i}")
        pts[i] = groups
        npts[i] = int(sum(len(g) for g in groups))
    times = {}
    if kinds["twt"] is not None:
        tdf, _ = read_table(kinds["twt"])
        times = parse_twt(tdf, ids, npts, warns)
    recs = []
    for i in ids:
        all_ilxl = np.vstack(pts[i])
        xy_all, inside = to_xy(all_ilxl[:, 0], all_ilxl[:, 1], geom_name) if geom_name else (all_ilxl, 0.0)
        if geom_name and inside < 0.5:
            warns.append(f"Fault CSV slice {i}: only {inside * 100:.0f}% of points fall inside the seismic grid; check that IL/CL are inline/crossline numbers")
        traces, dropped = [], 0
        off = 0
        for g in pts[i]:
            xyg = xy_all[off:off + len(g)]
            off += len(g)
            tr, d = traces_from_points(xyg)
            traces += tr
            dropped += d
        if not traces:
            warns.append(f"Fault CSV slice {i}: no connected fault traces (points too sparse)")
            continue
        recs.append({"type": "shp", "rel": f"Fault (CSV)/slice_{i:02d}", "root": str(root), "kind": "fault", "cls": "fault", "idx": i,
                     "key": f"timeslice_fault_{i:02d}", "geom": "line",
                     "feats": [{"rings": [t], "geom": "line", "z": None, "attrs": {"fault": k + 1, "source": "csv"}} for k, t in enumerate(traces)],
                     "crs": "", "fields": ["fault"], "source": "fault_csv",
                     "coord_mode": "inline/xline converted to X/Y" if geom_name else "inline/xline (raw geometry missing: not converted)",
                     "time_ms": times.get(i), "time_source": kinds["twt"].name if times.get(i) is not None else "unknown",
                     "n_points": int(len(all_ilxl)), "n_points_dropped": int(dropped)})
    if not geom_name:
        warns.append("Fault CSV: raw seismic geometry not available, so inline/xline could not be converted to X/Y. Ingest the raw data first.")
    return recs


def describe(folder=None):
    """Print how a Fault (CSV) folder is read (first rows of each file and the detected layout)."""
    import os
    folder = Path(folder or os.environ.get("INTERP_DIR", "."))
    files = sorted(p for p in folder.rglob("*.csv") if is_fault_csv(p, str(p)))
    if not files:
        print("no IL_/CL_/TWT_ fault CSV files under", folder)
        return
    for p in files:
        df, h = read_table(p)
        print(f"{p.name}: {df.shape[0]} rows x {df.shape[1]} cols; header={h}")
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    describe(sys.argv[1] if len(sys.argv) > 1 else None)
