"""
interp_data.py - ingestion of INTERPRETED data: fault / polygon shapefiles and horizon x-y-time files.

Design goals
  * DYNAMIC: nothing is hard-coded to your folder names. Every folder in INTERP_DIR, EXTRA_DIRS and DATA_DIR is scanned
    recursively and each file is classified from its path + content. New or changed files are picked up by
    scan() (incremental, keyed by size+mtime) - no code change needed.
  * TOLERANT: slice time can come from a config file, attributes, shapefile Z values, the file name, or start+step.
    If it cannot be found the data is still used (just without time-dependent checks) and a warning says so.

Environment variables
  INTERP_DIR   folder with interpreted data (e.g. .../data_export)
  EXTRA_DIRS   more folders, separated by ':' (Linux) / ';' (Windows)
  GEO_CONFIG   optional path to data_config.json (see README)
"""
import json
import math
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import fault_csv
import ingest as I


def interp_dir():
    p = os.environ.get("INTERP_DIR", "").strip()
    return Path(p) if p else None


def extra_dirs():
    return [Path(p) for p in os.environ.get("EXTRA_DIRS", "").split(os.pathsep) if p.strip()]


def icache():
    d = I.CACHE_DIR / "interpreted"
    d.mkdir(parents=True, exist_ok=True)
    return d


FAULT_WORDS = ("fault", "fracture", "lineament")
POLY_WORDS = ("channel", "overbank", "levee", "stratigraph", "polygon", "facies", "lobe", "fan", "bar", "sand", "shale",
              "crevasse", "splay", "floodplain", "mound", "reef", "unconformity")
HORIZON_EXT = {".xyt", ".xyz", ".dat", ".txt", ".csv", ".asc"}
SLICE_RE = re.compile(r"(?:time)?slice[_\-\s]*(?P<cls>[a-zA-Z]+)[_\-\s]*(?P<idx>\d+)$", re.I)


# ---------------------------------------------------------------- config
def load_config():
    cands = []
    if os.environ.get("GEO_CONFIG"):
        cands.append(Path(os.environ["GEO_CONFIG"]))
    if interp_dir():
        cands.append(interp_dir() / "data_config.json")
    cands += [Path("data_config.json")]
    cfg = {}
    for p in cands:
        if p.is_file():
            try:
                cfg = json.loads(p.read_text())
                cfg["_path"] = str(p)
                break
            except Exception as e:
                cfg = {"_error": f"{p}: {e}"}
    st = {}
    for p in ([interp_dir() / "slice_times.csv"] if interp_dir() else []) + [Path("slice_times.csv")]:
        if p.is_file():
            for line in p.read_text().splitlines():
                t = [x.strip() for x in re.split(r"[,;\t]", line)]
                if len(t) >= 2:
                    try:
                        st[t[0].lower()] = float(t[1])
                    except ValueError:
                        pass
            break
    cfg.setdefault("slice_times_ms", {})
    cfg["slice_times_ms"] = {**{k.lower(): v for k, v in cfg["slice_times_ms"].items()}, **st}
    return cfg


# ---------------------------------------------------------------- geometry helpers
def ring_signed_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1]))


def polygon_area(rings):
    """Shapefile convention: outer rings clockwise (negative signed area), holes counter-clockwise."""
    tot = 0.0
    for r in rings:
        if len(r) >= 3:
            sa = ring_signed_area(r)
            tot += -sa if sa < 0 else -sa  # outer adds |a|, holes subtract
    return abs(tot) if tot != 0 else sum(abs(ring_signed_area(r)) for r in rings if len(r) >= 3)


def line_length(coords):
    return float(np.sum(np.hypot(*np.diff(coords, axis=0).T))) if len(coords) > 1 else 0.0


def resample(coords, ds):
    """Points along a polyline every ds map units (keeps vertices)."""
    if len(coords) < 2:
        return coords.copy()
    seg = np.hypot(*np.diff(coords, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    if cum[-1] == 0:
        return coords[:1].copy()
    s = np.arange(0, cum[-1], max(ds, 1e-9))
    s = np.r_[s, cum[-1]]
    return np.c_[np.interp(s, cum, coords[:, 0]), np.interp(s, cum, coords[:, 1])]


def pca_axes(pts):
    """-> (azimuth_deg_from_north_clockwise 0-180, extent_major, extent_minor, elongation)."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0, 1.0
    c = pts - pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov(c.T))
    major = evec[:, 1]
    az = math.degrees(math.atan2(major[0], major[1])) % 180
    pm, pn = c @ major, c @ evec[:, 0]
    ext_major, ext_minor = float(np.ptp(pm)), float(np.ptp(pn))
    return az, ext_major, ext_minor, ext_major / max(ext_minor, 1e-9)


def bbox_of(arrs):
    a = np.vstack([x for x in arrs if len(x)])
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


# ---------------------------------------------------------------- reading files
def read_shp(path):
    import shapefile
    sf = shapefile.Reader(str(path), encoding="utf-8", encodingErrors="replace")
    fields = [f[0] for f in sf.fields[1:]]
    stype = sf.shapeType
    base = {1: "point", 11: "point", 21: "point", 8: "point", 18: "point", 28: "point",
            3: "line", 13: "line", 23: "line", 5: "polygon", 15: "polygon", 25: "polygon", 31: "polygon"}.get(stype, "other")
    out = []
    for sr in sf.iterShapeRecords():
        shp, rec = sr.shape, sr.record
        if shp.shapeType == 0 or not len(shp.points):
            continue
        pts = np.array(shp.points, float)[:, :2]
        parts = list(shp.parts) + [len(pts)] if base != "point" else [0, len(pts)]
        rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1) if parts[i + 1] - parts[i] >= 1]
        z = getattr(shp, "z", None)
        zm = float(np.nanmean(z)) if z is not None and len(z) and np.isfinite(np.nanmean(z)) else None
        try:
            attrs = {f: (rec[i] if i < len(rec) else None) for i, f in enumerate(fields)}
        except Exception:
            attrs = {}
        out.append({"rings": rings, "geom": base, "z": zm, "attrs": attrs})
    prj = Path(path).with_suffix(".prj")
    crs = prj.read_text(errors="ignore")[:120] if prj.exists() else ""
    return out, crs, fields


def read_xyt(path):
    """Tolerant x-y-time reader. Returns arrays x, y, t and a note list."""
    import pandas as pd
    notes = []
    head = [ln for ln in Path(path).read_text(errors="ignore").splitlines()[:40] if ln.strip()]
    sep = "," if sum(ln.count(",") for ln in head) > sum(ln.count(" ") + ln.count("\t") for ln in head) / 4 else r"\s+"
    df = pd.read_csv(path, sep=sep, header=None, comment="#", engine="python" if sep != r"\s+" else "c",
                     on_bad_lines="skip", dtype=str)
    colnames = None
    first = [str(v).strip().lower() for v in df.iloc[0].tolist()] if len(df) else []
    if first and any(re.search(r"[a-z]", v) for v in first):
        colnames = first
        df = df.iloc[1:]
    num = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    num = num.dropna(axis=1, how="all")
    if num.shape[1] < 3:
        raise ValueError("fewer than 3 numeric columns (need x, y, time)")
    ix, iy, it = 0, 1, 2
    if colnames and len(colnames) == df.shape[1]:
        def find(keys):
            for i, c in enumerate(colnames):
                if any(c == k or c.startswith(k) for k in keys):
                    return i
        fx, fy, ft = find(("x", "east", "utmx")), find(("y", "north", "utmy")), find(("t", "time", "twt", "z", "depth"))
        if None not in (fx, fy, ft):
            ix, iy, it = fx, fy, ft
            notes.append(f"columns by header: x={colnames[ix]}, y={colnames[iy]}, t={colnames[it]}")
    elif num.shape[1] >= 5:
        c0, c1 = num.iloc[:, 0], num.iloc[:, 1]
        if (c0 % 1 == 0).all() and (c1 % 1 == 0).all() and c0.abs().max() < 1e5 and c1.abs().max() < 1e5:
            ix, iy, it = 2, 3, 4
            notes.append("5+ columns with integer inline/xline first: using columns 3,4,5 as x,y,time")
    arr = num.iloc[:, [ix, iy, it]].dropna().to_numpy(float)
    x, y, t = arr[:, 0], arr[:, 1], arr[:, 2]
    return x, y, t, notes


NULLS = (-999.0, -999.25, -9999.0, -99999.0, 9999.0, 99999.0, 999999.0, 1e30, 1e20, -1e30)


def clean_xyt(x, y, t):
    """Drop null markers (-999, 9999, 1e30, zeros) and far-out points that would blow up the map extent. -> x, y, t, note list."""
    n0 = len(x)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    for nv in NULLS:
        ok &= (np.abs(t - nv) > 1e-6) & (np.abs(x - nv) > 1e-6) & (np.abs(y - nv) > 1e-6)
    ok &= ~((np.abs(x) < 1e-9) & (np.abs(y) < 1e-9)) & (np.abs(t) > 1e-9)
    x, y, t = x[ok], y[ok], t[ok]
    if len(x) > 20:
        for a in (x, y):
            pass
        keep = np.ones(len(x), bool)
        for a in (x, y):
            lo, hi = np.percentile(a, [1, 99])
            span = max(hi - lo, 1e-9)
            keep &= (a >= lo - 1.0 * span) & (a <= hi + 1.0 * span)
        lo, hi = np.percentile(t, [1, 99])
        span = max(hi - lo, 1e-9)
        keep &= (t >= lo - 2.0 * span) & (t <= hi + 2.0 * span)
        x, y, t = x[keep], y[keep], t[keep]
    notes = [f"{n0 - len(x)} null/outlier points removed from the horizon file"] if n0 - len(x) else []
    return x, y, t, notes


def grid_horizon(x, y, t, n=320):
    """Regular grid of the horizon (masked where no data). Returns dict gx, gy, H, cell."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    if len(x) > 300000:
        sel = np.random.default_rng(0).choice(len(x), 300000, replace=False)
        xs, ys, ts = x[sel], y[sel], t[sel]
    else:
        xs, ys, ts = x, y, t
    sub = np.random.default_rng(1).choice(len(xs), min(4000, len(xs)), replace=False)
    tree = cKDTree(np.c_[xs, ys])
    dd, _ = tree.query(np.c_[xs[sub], ys[sub]], k=2)
    spacing = float(np.median(dd[:, 1])) or 1.0
    xr, yr = np.ptp(xs), np.ptp(ys)
    cell = max(spacing, max(xr, yr) / n)
    gx = np.arange(xs.min(), xs.max() + cell, cell)
    gy = np.arange(ys.min(), ys.max() + cell, cell)
    GX, GY = np.meshgrid(gx, gy)
    H = griddata((xs, ys), ts, (GX, GY), method="linear")
    d, _ = tree.query(np.c_[GX.ravel(), GY.ravel()])
    H[(d > 3 * max(spacing, cell)).reshape(H.shape)] = np.nan
    return {"gx": gx, "gy": gy, "H": H, "cell": float(cell), "point_spacing": spacing}


# ------------------------------------------------------------ coordinate check
def _seismic_extent():
    try:
        c = I.load_cards()
    except Exception:
        return None
    v = [x for x in c.get("volumes", {}).values() if x.get("x_range")]
    if not v:
        return None
    return {"x": [min(a["x_range"][0] for a in v), max(a["x_range"][1] for a in v)],
            "y": [min(a["y_range"][0] for a in v), max(a["y_range"][1] for a in v)],
            "il": [min(a["inline"]["min"] for a in v), max(a["inline"]["max"] for a in v)],
            "xl": [min(a["xline"]["min"] for a in v), max(a["xline"]["max"] for a in v)],
            "name": next(iter(c["volumes"]))}


_IL2XY = {}


def _ilxl_to_xy(il, xl):
    from scipy.interpolate import LinearNDInterpolator
    ext = _seismic_extent()
    if "f" not in _IL2XY:
        g = np.load(I.CACHE_DIR / f"geom_{ext['name']}.npz")
        sel = np.linspace(0, len(g["x"]) - 1, min(60000, len(g["x"]))).astype(int)
        p = np.c_[g["il"][sel], g["xl"][sel]].astype(float)
        _IL2XY["f"] = (LinearNDInterpolator(p, g["x"][sel]), LinearNDInterpolator(p, g["y"][sel]))
    fx, fy = _IL2XY["f"]
    return fx(il, xl), fy(il, xl)


def check_coords(arr_xy, cfg):
    """-> (converted_xy_or_same, mode_string, warning_or_None). arr_xy: Nx2."""
    ext = _seismic_extent()
    mode = cfg.get("coordinate_mode", "auto")
    xo, yo = cfg.get("x_offset", 0.0), cfg.get("y_offset", 0.0)
    a = arr_xy + np.array([xo, yo])
    if ext is None or mode == "xy":
        return a, "xy (not checked against seismic)", None
    if (np.abs(arr_xy[:, 0]).max() <= 180 and np.abs(arr_xy[:, 1]).max() <= 90 and max(abs(ext["x"][0]), abs(ext["y"][0])) > 1000
            and cfg.get("coordinate_mode", "auto") != "xy"):
        return arr_xy, "lonlat", None                  # geographic degrees: converted later by _georeference
    dx, dy = 0.2 * np.ptp(ext["x"]), 0.2 * np.ptp(ext["y"])
    inside = ((a[:, 0] >= ext["x"][0] - dx) & (a[:, 0] <= ext["x"][1] + dx) &
              (a[:, 1] >= ext["y"][0] - dy) & (a[:, 1] <= ext["y"][1] + dy)).mean()
    if inside > 0.5 and mode != "ilxl":
        return a, "xy", None
    for order, (ci, cx) in {"x=xline,y=inline": (1, 0), "x=inline,y=xline": (0, 1)}.items():
        v_il, v_xl = a[:, ci], a[:, cx]
        if ((v_il >= ext["il"][0] - 5) & (v_il <= ext["il"][1] + 5) & (v_xl >= ext["xl"][0] - 5) & (v_xl <= ext["xl"][1] + 5)).mean() > 0.8:
            x, y = _ilxl_to_xy(v_il, v_xl)
            good = np.isfinite(x) & np.isfinite(y)
            out = a.copy()
            out[good] = np.c_[x[good], y[good]]
            return out, f"inline/xline converted to x,y ({order})", "coordinates looked like inline/xline numbers and were converted; set coordinate_mode in data_config.json if wrong"
    return a, "xy (outside seismic extent)", ("coordinates do not overlap the seismic volumes; check CRS / units "
                                                "(set x_offset, y_offset or coordinate_mode in data_config.json)")


# ------------------------------------------------------------ classification
NUM_RE = re.compile(r"^(?P<num>\d{2,5})[_\-\s]+(?P<cls>[A-Za-z]+)(?:[_\-\s]+(?P<k>\d+))?$")


def classify_shp(rel, geom):
    parts = [p for p in Path(rel).parts]
    low = rel.lower()
    cls, idx, key = None, None, None
    for p in list(reversed(parts[:-1])) + [Path(rel).stem]:
        m = SLICE_RE.search(p)
        if m:
            cls, idx, key = m.group("cls").lower(), int(m.group("idx")), p
            break
        m = NUM_RE.match(p)                              # e.g. 1500_Fault_01, 0080_Channel_02
        if m:
            cls, idx, key = m.group("cls").lower(), int(m.group("num")), p
            break
    if cls is None:
        key = parts[-2] if len(parts) > 1 else Path(rel).stem
    if cls and any(w in cls for w in FAULT_WORDS) or (cls is None and any(w in low for w in FAULT_WORDS)):
        kind = "fault"
        cls = cls or "fault"
    elif cls or any(w in low for w in POLY_WORDS):
        kind = "polygon"
        cls = cls or next(w for w in POLY_WORDS if w in low)
    else:
        kind = "fault" if geom == "line" else "polygon"
        cls = cls or Path(rel).stem.lower()
    return kind, cls, idx, key


def find_time(feats, rel, key, idx, idx_all, cfg, seis_t):
    st = cfg.get("slice_times_ms", {})
    for k in (key, Path(rel).stem, str(Path(rel).parent.name)):
        if k and k.lower() in st:
            return float(st[k.lower()]), "config/slice_times"
    for f in feats:
        for name, v in f["attrs"].items():
            if re.fullmatch(r"(time|twt|t_?ms|time_?ms|twt_?ms|slice_?time|t)", str(name).lower()) and isinstance(v, (int, float)) and v == v:
                return float(v), f"attribute {name}"
    zs = [f["z"] for f in feats if f["z"] is not None]
    if zs and np.ptp(zs) < 50 and abs(np.mean(zs)) > 1:
        return abs(float(np.mean(zs))), "shapefile Z values"
    m0 = NUM_RE.match(str(key or ""))
    if m0:
        num = float(m0.group("num"))
        how = cfg.get("fault_name_number", "twt_ms") if any(w in m0.group("cls").lower() for w in FAULT_WORDS) else cfg.get("polygon_name_number", "twt_ms")
        if how == "twt_ms" and 10 <= num <= 8000:
            return num, "number in folder name (TWT ms)"
        if how == "sample_index" and cfg.get("sample_rate_ms"):
            return num * float(cfg["sample_rate_ms"]), "folder number x sample rate"
        if how == "twt_ms_plus_offset" and cfg.get("name_number_offset_ms") is not None:
            return num + float(cfg["name_number_offset_ms"]), "folder number + offset"
    m = re.search(r"(\d{3,5})\s*ms", rel, re.I) or re.search(r"(?:twt|time|t)[_\-]?(\d{3,5})(?!\d)", rel, re.I)
    if m:
        return float(m.group(1)), "file name"
    if idx is not None and cfg.get("slice_time_start_ms") is not None:
        step = cfg.get("slice_time_step_ms", 1.0)
        return float(cfg["slice_time_start_ms"] + step * (idx - min(idx_all)) ), "config start+step*index"
    return None, "unknown"


# ------------------------------------------------------------------- scanning
def _roots():
    seen, roots = set(), []
    for r in ([interp_dir()] if interp_dir() else []) + extra_dirs() + [I.DATA_DIR]:
        if r and Path(r).exists() and str(Path(r).resolve()) not in seen:
            seen.add(str(Path(r).resolve()))
            roots.append(Path(r))
    return roots


def _candidate_files():
    shp, hor, fcsv, other, seen = [], [], [], 0, set()
    for root in _roots():
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.startswith(("~$", ".")):
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            rel = str(p.relative_to(root))
            low = rel.lower()
            ext = p.suffix.lower()
            if fault_csv.is_fault_csv(p, rel):
                fcsv.append((root, p, rel))
            elif ext == ".shp":
                shp.append((root, p, rel))
            elif ext == ".xyt" or (ext in HORIZON_EXT and "horizon" in low and not any(k in low for k in ("dev", "check"))):
                hor.append((root, p, rel))
            else:
                other += 1
    return shp, hor, fcsv, other


def _sig(p):
    s = p.stat()
    side = sum(q.stat().st_mtime_ns for q in p.parent.glob(p.stem + ".*") if q.suffix.lower() in (".dbf", ".shx", ".prj")) if p.suffix.lower() == ".shp" else 0
    return f"{s.st_size}:{s.st_mtime_ns}:{side}"


def scan(force=False):
    """Incrementally (re)ingest all interpreted files. Returns a change report."""
    cfg = load_config()
    mp = icache() / "manifest.pkl"
    old = pickle.loads(mp.read_bytes()) if mp.exists() and not force else {"files": {}, "data": {}}
    new = {"files": {}, "data": {}}
    rep = {"new": [], "changed": [], "removed": [], "unchanged": 0, "warnings": [], "config": cfg.get("_path")}
    if cfg.get("_error"):
        rep["warnings"].append(f"config not read: {cfg['_error']}")
    shp, hor, fcsv, other = _candidate_files()
    for root, p, rel in shp + hor:
        sig = _sig(p)
        key = f"{root.name}/{rel}"
        if old["files"].get(key) == sig and key in old["data"]:
            new["files"][key], new["data"][key] = sig, old["data"][key]
            rep["unchanged"] += 1
            continue
        try:
            rec = _ingest_one(root, p, rel, cfg, rep["warnings"])
        except Exception as e:
            rep["warnings"].append(f"{rel}: could not be read ({type(e).__name__}: {e})")
            continue
        new["files"][key], new["data"][key] = sig, rec
        (rep["changed"] if key in old["files"] else rep["new"]).append(rel)
    _scan_fault_csv(fcsv, old, new, rep, cfg)
    rep["removed"] = [k for k in old["files"] if k not in new["files"]]
    _assign_slice_times(new, cfg, rep["warnings"])
    try:
        _georeference(new, cfg, rep["warnings"])
    except Exception as ex:
        rep["warnings"].append(f"GEOREF failed: {type(ex).__name__}: {ex}")
    try:
        _extent_warnings(new, rep["warnings"])
    except Exception as ex:
        rep["warnings"].append(f"extent check skipped: {ex}")
    mp.write_bytes(pickle.dumps(new))
    inv = build_inventory(new, rep, cfg, other)
    (I.CACHE_DIR / "interpreted_cards.json").write_text(json.dumps(I.clean(inv), indent=1))
    clear_caches()
    return {"changes": {k: rep[k] for k in ("new", "changed", "removed", "unchanged")}, "warnings": rep["warnings"],
            "inventory": inv["summary"]}


def _scan_fault_csv(fcsv, old, new, rep, cfg):
    """Fault CSV sets are read as one group per folder (IL_/CL_ pairs + TWT table)."""
    groups = {}
    for root, p, rel in fcsv:
        groups.setdefault((root, str(Path(rel).parent)), []).append(p)
    for (root, parent), files in groups.items():
        gkey = f"{root.name}/__faultcsv__/{parent}"
        sig = str([(f.name, _sig(f)) for f in sorted(files)])
        reuse = [k for k in old["data"] if k.startswith(gkey + "#")]
        if old["files"].get(gkey) == sig and reuse:
            new["files"][gkey] = sig
            for k in reuse:
                new["data"][k] = old["data"][k]
            rep["unchanged"] += len(files)
            continue
        geom = None
        try:
            ext = _seismic_extent()
            geom = ext["name"] if ext else None
        except Exception:
            geom = None
        try:
            recs = fault_csv.ingest_group(root, files, geom, cfg, rep["warnings"])
        except Exception as e:
            rep["warnings"].append(f"Fault CSV in {parent}: could not be read ({type(e).__name__}: {e})")
            continue
        new["files"][gkey] = sig
        for r in recs:
            new["data"][f"{gkey}#{r['idx']:03d}"] = r
        (rep["changed"] if gkey in old["files"] else rep["new"]).append(f"{parent}/ ({len(files)} CSV files, {len(recs)} slices)")


def _ingest_one(root, p, rel, cfg, warns):
    if p.suffix.lower() == ".shp":
        feats, crs, fields = read_shp(p)
        if not feats:
            raise ValueError("no geometry")
        geom = feats[0]["geom"]
        kind, cls, idx, key = classify_shp(rel, geom)
        allpts = np.vstack([r for f in feats for r in f["rings"]])
        conv, mode, w = check_coords(allpts, cfg)
        if w:
            warns.append(f"{rel}: {w}")
        if mode.startswith("inline") or cfg.get("x_offset") or cfg.get("y_offset"):
            off = 0
            for f in feats:
                for i, r in enumerate(f["rings"]):
                    f["rings"][i] = conv[off:off + len(r)]
                    off += len(r)
        if "GEOGCS" in crs and "PROJCS" not in crs and mode != "lonlat":
            warns.append(f"{rel}: shapefile CRS is geographic (lat/lon); reproject to the seismic CRS or coordinates will not match")
        if geom == "polygon" and kind == "fault":
            geom_note = "fault given as polygons: boundaries used as lines"
            warns.append(f"{rel}: {geom_note}")
        return {"type": "shp", "rel": rel, "root": str(root), "kind": kind, "cls": cls, "idx": idx, "key": key,
                "geom": geom, "feats": feats, "crs": crs, "fields": fields, "coord_mode": mode}
    x, y, t, notes = read_xyt(p)
    x, y, t, cnote0 = clean_xyt(x, y, t)
    unit = cfg.get("horizon_time_unit", "auto")
    note = list(notes) + cnote0
    if unit == "s" or (unit == "auto" and np.nanmedian(np.abs(t)) < 20):
        t = t * 1000.0
        note.append("time converted s -> ms")
    if np.nanmedian(t) < 0:
        t = -t
        note.append("negative times: sign flipped (positive TWT assumed)")
    conv, mode, w = check_coords(np.c_[x, y], cfg)
    if w:
        warns.append(f"{rel}: {w}")
    x, y = conv[:, 0], conv[:, 1]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x, y, t = x[ok], y[ok], t[ok]
    grid = grid_horizon(x, y, t)
    npz = icache() / (re.sub(r"[^A-Za-z0-9_]+", "_", Path(rel).stem) + "_horizon.npz")
    np.savez_compressed(npz, x=x, y=y, t=t, gx=grid["gx"], gy=grid["gy"], H=grid["H"], cell=grid["cell"], spacing=grid["point_spacing"])
    return {"type": "horizon", "rel": rel, "root": str(root), "name": Path(rel).stem, "npz": str(npz), "n": int(len(x)),
            "t_range": [float(np.nanmin(t)), float(np.nanmax(t))], "bbox": [float(x.min()), float(y.min()), float(x.max()), float(y.max())],
            "notes": note, "coord_mode": mode}


def _local_m(ll, lon0, lat0):
    """lon/lat degrees -> local metres (equirectangular, good to ~0.1% over a few tens of km)."""
    return np.c_[(ll[:, 0] - lon0) * 111320.0 * np.cos(np.radians(lat0)), (ll[:, 1] - lat0) * 110574.0]


def _icp_fit(P, Q, groups=None, iters=40):
    """Rigid (rotation + translation, optional mirror) fit of local-metre points P onto seismic X/Y points Q by trimmed ICP.
    Starting rotations every 15 degrees; unit factor 1 (metres) or 3.28084 (feet). Scale is fixed by geography, so thin, nearly
    straight fault traces cannot collapse the solution. Returns (apply_fn, median_residual, description)."""
    from scipy.spatial import cKDTree
    if groups is None:
        trees = {None: (cKDTree(Q), np.arange(len(Q)))}
    else:
        gP, gQ = groups
        trees = {g: (cKDTree(Q[gQ == g]), np.where(gQ == g)[0]) for g in np.unique(gP) if (gQ == g).any()}

    def nn(Pt):
        if groups is None:
            d, j = trees[None][0].query(Pt)
            return d, j
        d = np.full(len(Pt), np.inf)
        j = np.zeros(len(Pt), int)
        for g, (tr, idx) in trees.items():
            m = gP == g
            if m.any():
                dd, jj = tr.query(Pt[m])
                d[m], j[m] = dd, idx[jj]
        return d, j

    def kabsch(A, B):
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        return R, cb - R @ ca
    best = None
    for unit in (1.0, 3.28084):
        for mirror in (1, -1):
            Pm = P * unit * np.array([1, mirror])
            for ang in range(0, 360, 15):
                a = np.radians(ang)
                R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                tr_ = Q.mean(0) - (Pm @ R.T).mean(0)
                for it in range(iters):
                    Pt = Pm @ R.T + tr_
                    d, j = nn(Pt)
                    ok = np.isfinite(d)
                    if ok.sum() < 4:
                        break
                    keep = ok & (d <= np.percentile(d[ok], 60 if it > 8 else 90))
                    R, tr_ = kabsch(Pm[keep], Q[j[keep]])
                d, j = nn(Pm @ R.T + tr_)
                ok = np.isfinite(d)
                score = float(np.median(d[ok])) if ok.any() else np.inf
                if best is None or score < best[0]:
                    best = (score, unit, mirror, R.copy(), tr_.copy())
    score, unit, mirror, R, tr_ = best
    fn = lambda L: (L * unit * np.array([1, mirror])) @ R.T + tr_
    return fn, score, f"rigid fit, unit x{unit:g}{', mirrored' if mirror < 0 else ''}, rotation {np.degrees(np.arctan2(R[1, 0], R[0, 0])):.0f} deg"


def _georeference(state, cfg, warns):
    """Shapefiles in lon/lat -> seismic X/Y. Order: explicit CRS in data_config.json (pyproj) -> calibration on the Fault CSV sticks
    (same faults exist in both) -> optional bounding-box fit (georef_method: 'bbox')."""
    pend = [r for r in state["data"].values() if r["type"] == "shp" and r.get("coord_mode") == "lonlat"]
    if not pend:
        return
    method, info = None, ""
    tf = None
    if cfg.get("shapefile_crs") and cfg.get("seismic_crs"):
        try:
            from pyproj import Transformer
            T = Transformer.from_crs(cfg["shapefile_crs"], cfg["seismic_crs"], always_xy=True)
            tf = lambda a: np.c_[T.transform(a[:, 0], a[:, 1])]
            method = f"pyproj {cfg['shapefile_crs']} -> {cfg['seismic_crs']}"
        except Exception as e:
            warns.append(f"CRS conversion with pyproj failed ({type(e).__name__}: {e}); pip install pyproj, or remove the CRS keys to use auto-calibration")
    if tf is None:
        anchors = [r for r in state["data"].values() if r["type"] == "shp" and r["kind"] == "fault" and r.get("source") == "fault_csv" and r.get("section") is None]
        fpend = [r for r in pend if r["kind"] == "fault" and r.get("time_ms") is not None]
        if anchors and pend and cfg.get("georef_method", "auto") != "bbox":
            def pts(rs):
                return np.vstack([ring for r in rs for f in r["feats"] for ring in f["rings"]])
            allll = np.vstack([ring for r in pend for f in r["feats"] for ring in f["rings"]])
            lon0, lat0 = float(np.median(allll[:, 0])), float(np.median(allll[:, 1]))
            loc = lambda a: _local_m(np.asarray(a, float), lon0, lat0)
            times_a = {round(r["time_ms"]) for r in anchors}
            common = [r for r in fpend if round(r["time_ms"]) in times_a]
            use_groups = len(common) >= 1
            npts_ = lambda r: sum(len(ring) for f in r["feats"] for ring in f["rings"])
            if use_groups:
                Pp = loc(pts(common))
                gP = np.concatenate([[round(r["time_ms"])] * npts_(r) for r in common]).astype(int)
                Qr = [r for r in anchors if round(r["time_ms"]) in {round(c["time_ms"]) for c in common}]
                Qp = pts(Qr)
                gQ = np.concatenate([[round(r["time_ms"])] * npts_(r) for r in Qr]).astype(int)
                fn, med, desc = _icp_fit(Pp, Qp, groups=(gP, gQ))
            else:
                fn, med, desc = _icp_fit(loc(pts([r for r in pend if r["kind"] == "fault"] or pend)), pts(anchors))
            diag = float(np.hypot(*np.ptp(pts(anchors), axis=0)))
            tf = lambda a: fn(loc(a))
            method = f"calibrated on Fault CSV ({'matching time slices' if use_groups else 'all faults'}; {desc}); median residual {med:.0f} map units"
            if med > 0.08 * diag:
                warns.append(f"GEOREF: calibration residual {med:.0f} is large (>8% of the fault extent {diag:.0f}); lon/lat -> X/Y may be wrong. "
                             "Provide shapefile_crs and seismic_crs in data_config.json")
        elif cfg.get("georef_method") == "bbox":
            ref = None
            hs = [r for r in state["data"].values() if r["type"] == "horizon"]
            if hs:
                ref = hs[0]["bbox"]
            if ref is None:
                s = _seismic_extent()
                ref = [s["x"][0], s["y"][0], s["x"][1], s["y"][1]] if s else None
            allp = np.vstack([ring for r in pend for f in r["feats"] for ring in f["rings"]])
            if ref is not None:
                lo, hi = allp.min(0), allp.max(0)
                sx, sy = (ref[2] - ref[0]) / max(hi[0] - lo[0], 1e-12), (ref[3] - ref[1]) / max(hi[1] - lo[1], 1e-12)
                tf = lambda a: np.c_[ref[0] + (a[:, 0] - lo[0]) * sx, ref[1] + (a[:, 1] - lo[1]) * sy]
                method = "bounding-box fit to the horizon/seismic extent (approximate)"
                warns.append("GEOREF: lon/lat layers were stretched to the horizon/seismic bounding box (georef_method='bbox'); positions are approximate")
    if tf is None:
        warns.append("GEOREF: shapefiles are in lon/lat but cannot be tied to the seismic X/Y (no CRS in data_config.json and no Fault CSV to calibrate on). "
                     "Add shapefile_crs / seismic_crs (EPSG codes or PROJ strings) to data_config.json")
        return
    for r in pend:
        for f in r["feats"]:
            f["rings"] = [tf(np.asarray(ring, float)) for ring in f["rings"]]
        r["coord_mode"] = f"lon/lat -> X/Y ({method})"
    state["georef"] = method
    warns.append(f"GEOREF: {len(pend)} lon/lat shapefile layers converted to X/Y: {method}")


def _assign_slice_times(state, cfg, warns):
    by_cls = {}
    for rec in state["data"].values():
        if rec["type"] == "shp":
            by_cls.setdefault((rec["kind"], rec["cls"]), []).append(rec)
    for (kind, cls), recs in by_cls.items():
        idxs = [r["idx"] for r in recs if r["idx"] is not None] or [0]
        for r in recs:
            if r.get("source") == "fault_csv":
                continue
            r["time_ms"], r["time_source"] = find_time(r["feats"], r["rel"], r["key"], r["idx"], idxs, cfg, None)
        if all(r["time_ms"] is None for r in recs):
            warns.append(f"{kind}/{cls}: slice times unknown for {len(recs)} slices (tools still work; time-dependent checks are skipped). "
                         "Add slice_times.csv (name,time_ms) or data_config.json to enable them")


def layer_extents(state=None):
    """Bounding box (x0, y0, x1, y1) of every interpreted layer and of the raw seismic/wells, for consistency checks."""
    st = state or _state()
    ex = {}
    pts = {"faults": [], "polygons": []}
    for r in st["data"].values():
        if r["type"] == "shp":
            pts["faults" if r["kind"] == "fault" else "polygons"] += [ring for f in r["feats"] for ring in f["rings"]]
        else:
            ex[f"horizon:{r['name']}"] = [round(v, 1) for v in r["bbox"]]
    for k, v in pts.items():
        if v:
            ex[k] = [round(float(z), 1) for z in bbox_of(v)]
    s = _seismic_extent()
    if s:
        ex["seismic"] = [round(s["x"][0], 1), round(s["y"][0], 1), round(s["x"][1], 1), round(s["y"][1], 1)]
    try:
        w = I.load_cards()["wells"]
        xs, ys = [c["x"] for c in w.values()], [c["y"] for c in w.values()]
        ex["wells"] = [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
    except Exception:
        pass
    return ex


def _extent_warnings(state, warns):
    ex = layer_extents(state)
    ref = ex.get("seismic") or ex.get("wells")
    if not ref:
        return
    rw, rh = max(ref[2] - ref[0], 1e-9), max(ref[3] - ref[1], 1e-9)
    for name, b in ex.items():
        if name in ("seismic", "wells"):
            continue
        ix = max(0.0, min(b[2], ref[2]) - max(b[0], ref[0]))
        iy = max(0.0, min(b[3], ref[3]) - max(b[1], ref[1]))
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
        if ix * iy / area < 0.5:
            warns.append(f"EXTENT: '{name}' bbox {b} mostly lies outside the seismic area {ref}; check coordinates/units or null values in that file")
        elif area > 25 * rw * rh:
            warns.append(f"EXTENT: '{name}' bbox {b} is much larger than the seismic area {ref}; the file probably has far-out or null points")


def build_inventory(state, rep, cfg, other_files):
    faults = [r for r in state["data"].values() if r["type"] == "shp" and r["kind"] == "fault"]
    polys = [r for r in state["data"].values() if r["type"] == "shp" and r["kind"] == "polygon"]
    hors = [r for r in state["data"].values() if r["type"] == "horizon"]

    def tr(rs):
        t = [r["time_ms"] for r in rs if r.get("time_ms") is not None]
        return [min(t), max(t)] if t else None
    summ = {"faults": {"n_slices": len(faults), "n_features": sum(len(r["feats"]) for r in faults), "time_range_ms": tr(faults)},
            "polygon_classes": {}, "horizons": [{"name": h["name"], "n_points": h["n"], "t_range_ms": h["t_range"], "notes": h["notes"]} for h in hors],
            "other_files_ignored": other_files}
    for cls in sorted({r["cls"] for r in polys}):
        rs = [r for r in polys if r["cls"] == cls]
        summ["polygon_classes"][cls] = {"n_slices": len(rs), "n_features": sum(len(r["feats"]) for r in rs), "time_range_ms": tr(rs)}
    return {"summary": summ, "warnings": rep["warnings"], "config": cfg.get("_path"),
            "files": [{"rel": r["rel"], "kind": r.get("kind", "horizon"), "cls": r.get("cls"), "time_ms": r.get("time_ms"),
                       "time_source": r.get("time_source"), "n": len(r["feats"]) if r["type"] == "shp" else r["n"],
                       "coord_mode": r["coord_mode"]} for r in state["data"].values()]}


# ------------------------------------------------------------------- loaders
def interpreted_ready():
    return (icache() / "manifest.pkl").exists()


@lru_cache(None)
def _state():
    mp = icache() / "manifest.pkl"
    return pickle.loads(mp.read_bytes()) if mp.exists() else {"files": {}, "data": {}}


def clear_caches():
    fault_csv._TREE.clear()
    _IL2XY.clear()
    _state.cache_clear()
    horizon.cache_clear()


def slices(kind, cls=None):
    out = [r for r in _state()["data"].values() if r["type"] == "shp" and r["kind"] == kind and (cls is None or r["cls"] == cls)]
    return sorted(out, key=lambda r: (r["time_ms"] if r.get("time_ms") is not None else 1e12, r["idx"] or 0, r["rel"]))


def polygon_classes():
    return sorted({r["cls"] for r in _state()["data"].values() if r["type"] == "shp" and r["kind"] == "polygon"})


@lru_cache(None)
def horizon(name=None):
    hs = [r for r in _state()["data"].values() if r["type"] == "horizon"]
    if not hs:
        return None
    h = next((r for r in hs if name and r["name"] == name), None) or max(hs, key=lambda r: r["n"])
    g = np.load(h["npz"])
    return {"name": h["name"], "x": g["x"], "y": g["y"], "t": g["t"], "gx": g["gx"], "gy": g["gy"], "H": g["H"],
            "cell": float(g["cell"]), "spacing": float(g["spacing"]), "notes": h["notes"]}


def diagnose():
    """Compact, shareable summary of what each interpreted file contains (no data values beyond ranges). Paste the output when something looks wrong."""
    import shapefile
    st = _state()
    print("INTERP_DIR:", interp_dir())
    print("seismic extent:", _seismic_extent())
    for k, r in st["data"].items():
        if r["type"] == "shp":
            allp = np.vstack([ring for f in r["feats"] for ring in f["rings"]])
            a0 = r["feats"][0]["attrs"] if r["feats"] else {}
            print(f"- {r['rel']} | {r['kind']}/{r['cls']} {r['geom']} | feats={len(r['feats'])} pts={len(allp)} | "
                  f"bbox=({allp[:,0].min():.2f},{allp[:,1].min():.2f})-({allp[:,0].max():.2f},{allp[:,1].max():.2f}) | time={r.get('time_ms')} ({r.get('time_source')}) | "
                  f"mode={r['coord_mode']} | fields={r.get('fields')} | attrs0={dict(list(a0.items())[:4])} | crs={str(r.get('crs'))[:60]!r}")
        else:
            print(f"- {r['rel']} | horizon | n={r['n']} t={r['t_range']} bbox={r['bbox']} notes={r['notes']}")
    print("georeference:", st.get("georef"))
