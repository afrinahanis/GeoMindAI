"""
ingest.py - builds the "data cards" and the well-to-volume cache the copilot reads.

    python ingest.py          # run once (and again whenever the data changes)

Expected layout (folder names are matched loosely; file names are matched to well names):
    data/
      <anything>.xlsx          well information (Well Name, RT, TD, X, Y, ...)
      deviations/*.txt         MD / inclination / azimuth
      checkshot/*.txt          depth / time
      las/*.las                logs
      inversion/*.sgy          P-imp, S-imp, Vp/Vs, sand probability
      stack/*.sgy              near / mid / far / ultra / full stacks
Override locations with the env vars DATA_DIR and CACHE_DIR.

Other modules import the loaders and the Volume class from here.
"""
import json
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache"))

# ---- ASSUMPTIONS: edit these if your data differs ---------------------------
SEGY_BYTE_OPTIONS = [(189, 193), (9, 21)]  # (inline byte, xline byte) tried in order
CHECKSHOT_TIME_IS_OWT = False              # True if checkshot time is one-way time
TRACK_STEP_M = 5.0                         # sampling step along the wellbore (m)
# Excel X/Y = surface location; RT = rotary table elevation above MSL (m);
# TVDSS = TVD - RT; seismic vertical axis = TWT in ms.
# -----------------------------------------------------------------------------

VOLUME_TAGS = {  # substring of normalised file name -> aliases the user/LLM may use
    "pimp": ["ip", "pimp", "pimpedance", "pimpedence", "acousticimpedance"],
    "simp": ["is", "simp", "simpedance", "simpedence", "shearimpedance"],
    "sand": ["sand", "sandprob", "sandprobability", "totalsandprobability", "tsp"],
    "vpvs": ["vpvs", "vpvsratio"],
    "near": ["near", "nearstack", "nearangle"],
    "mid": ["mid", "midstack", "midangle"],
    "ultra": ["ultra", "ultrastack", "ultrafar", "ultrafarstack"],  # before "far": UltraFar contains "far"
    "far": ["far", "farstack", "farangle"],
    "full": ["full", "fullstack", "fullangle"],
}
VOLUME_KEYS = {  # short, readable volume names used everywhere (long file names are only stored in the card)
    "pimp": "p_impedance", "simp": "s_impedance", "sand": "sand_probability", "vpvs": "vpvs",
    "near": "near_stack", "mid": "mid_stack", "far": "far_stack", "ultra": "ultrafar_stack",
    "full": "full_stack",
}
VOLUME_LABELS = {
    "pimp": "P-impedance", "simp": "S-impedance", "sand": "Total sand probability",
    "vpvs": "Vp/Vs ratio", "near": "Near-angle stack", "mid": "Mid-angle stack",
    "far": "Far-angle stack", "ultra": "Ultra-far-angle stack", "full": "Full-angle stack",
}


# ----------------------------------------------------------------- helpers
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_files(exts, folder_keys=None):
    out = []
    if not DATA_DIR.exists():
        return out
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$"):
            rel = str(p.relative_to(DATA_DIR).parent).lower()
            if folder_keys is None or any(k in rel for k in folder_keys):
                out.append(p)
    return sorted(out)


def match_well(path, wells):
    stem, best = norm(path.stem), None
    for w in wells:
        if norm(w) in stem and (best is None or len(norm(w)) > len(norm(best))):
            best = w
    return best


def _tok(line):
    return [t for t in re.split(r"[,\s;]+", line.strip()) if t and t not in ("#", "//")]


def _isnum(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def read_table(path):
    """Tolerant reader for whitespace/CSV text tables with optional header lines."""
    rows, heads = [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        t = _tok(line)
        if len(t) >= 2 and all(_isnum(x) for x in t):
            rows.append([float(x) for x in t])
        elif not rows:
            heads.append(t)
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    widths = [len(r) for r in rows]
    n = max(set(widths), key=widths.count)
    rows = [r for r in rows if len(r) == n]
    cols = None
    for h in reversed(heads):
        if len(h) == n and not any(x.startswith("(") for x in h):
            cols = [x.lower() for x in h]
            break
    return pd.DataFrame(rows, columns=cols or [f"c{i}" for i in range(n)])


def pick(df, keys, default_idx):
    for k in keys:
        for c in df.columns:
            if k in str(c).lower():
                return c
    return df.columns[min(default_idx, len(df.columns) - 1)]


def extrap(x, xp, fp):
    """np.interp with linear extrapolation at both ends."""
    x, xp, fp = (np.asarray(a, float) for a in (x, xp, fp))
    y = np.interp(x, xp, fp)
    if len(xp) >= 2:
        lo, hi = x < xp[0], x > xp[-1]
        if lo.any():
            y[lo] = fp[0] + (fp[1] - fp[0]) / (xp[1] - xp[0]) * (x[lo] - xp[0])
        if hi.any():
            y[hi] = fp[-1] + (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) * (x[hi] - xp[-1])
    return y


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


# ------------------------------------------------------------ well info/path
def load_well_info():
    files = find_files({".xlsx", ".xls"})
    if not files:
        raise FileNotFoundError(f"No Excel file found under {DATA_DIR}")
    raw = pd.read_excel(files[0], header=None)
    hdr = next(i for i, r in raw.iterrows()
               if r.astype(str).str.contains("well name", case=False).any())
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    col = {c.lower(): c for c in df.columns}
    df = df.dropna(subset=[col["well name"]])
    df["well_id"] = df[col["well name"]].astype(str).str.strip().str.lower()
    for k in ("rt", "wd", "wb", "td", "x", "y"):
        if k in col:
            df[col[k]] = pd.to_numeric(df[col[k]], errors="coerce")
    return df, col


def _flag(v):
    s = str(v).strip().lower()
    return s not in ("", "nan", "none", "x", "no", "n")


def min_curvature(md, inc, azi):
    inc, azi = np.radians(inc), np.radians(azi)
    n = len(md)
    tvd, north, east = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        dmd = md[i] - md[i - 1]
        i1, i2, a1, a2 = inc[i - 1], inc[i], azi[i - 1], azi[i]
        cosb = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))
        beta = np.arccos(np.clip(cosb, -1, 1))
        rf = 1.0 if beta < 1e-8 else 2 / beta * np.tan(beta / 2)
        north[i] = north[i - 1] + dmd / 2 * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
        east[i] = east[i - 1] + dmd / 2 * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
        tvd[i] = tvd[i - 1] + dmd / 2 * (np.cos(i1) + np.cos(i2)) * rf
    return tvd, north, east


def build_track(dev_path, x0, y0, rt, td):
    """Wellbore path (MD, TVD, TVDSS, X, Y) from a deviation survey; vertical if none."""
    if dev_path is None:
        md = np.array([0.0, td]); inc = np.zeros(2); azi = np.zeros(2)
    else:
        d = read_table(dev_path)
        cols = [pick(d, ("md", "meas"), 0), pick(d, ("incl", "inc", "devi"), 1),
                pick(d, ("azi", "az"), 2)]
        d = d[cols].dropna().sort_values(cols[0]).drop_duplicates(cols[0])
        md, inc, azi = (d.iloc[:, i].values.astype(float) for i in range(3))
        if md[0] > 0:
            md, inc, azi = np.r_[0, md], np.r_[0, inc], np.r_[azi[0], azi]
        if td and md[-1] < td:
            md, inc, azi = np.r_[md, td], np.r_[inc, inc[-1]], np.r_[azi, azi[-1]]
    tvd, n, e = min_curvature(md, inc, azi)
    rt = 0.0 if not np.isfinite(rt) else rt
    return pd.DataFrame({"md": md, "tvd": tvd, "tvdss": tvd - rt, "x": x0 + e,
                         "y": y0 + n, "inc": inc, "azi": azi})


def load_checkshot(path, track, rt):
    d = read_table(path)
    dcol = pick(d, ("tvd", "depth", "dept", "md"), 0)
    tcol = pick(d, ("twt", "owt", "time", "tt"), 1)
    if dcol == tcol:
        dcol, tcol = d.columns[0], d.columns[1]
    depth, t = d[dcol].values.astype(float), d[tcol].values.astype(float)
    if "md" in str(dcol) and "tvd" not in str(dcol):
        depth = np.interp(depth, track.md, track.tvd)
    if np.nanmax(t) < 15:        # seconds -> ms
        t = t * 1000
    if CHECKSHOT_TIME_IS_OWT or "owt" in str(tcol):
        t = t * 2
    cs = pd.DataFrame({"tvdss": depth - (0 if not np.isfinite(rt) else rt), "twt": t})
    cs = cs.dropna().sort_values("tvdss").drop_duplicates("tvdss")
    if cs.tvdss.iloc[0] > 0:
        cs = pd.concat([pd.DataFrame({"tvdss": [0.0], "twt": [0.0]}), cs])
    cs["twt"] = np.maximum.accumulate(cs.twt.values)
    return cs.drop_duplicates("tvdss").reset_index(drop=True)


# -------------------------------------------------------------------- logs
ROLE_RULES = [
    ("DTS", lambda m: m.startswith("DTS") or m in ("DTSM", "DTSH")),
    ("DT", lambda m: m in ("DT", "DTC", "DTCO", "DTP", "DT24", "AC") or m.startswith("DTC")),
    ("GR", lambda m: m.startswith("GR") or m in ("SGR", "CGR", "HCGR", "ECGR", "GAMMA")),
    ("RES", lambda m: m in ("RT", "ILD", "LLD", "RDEP", "RLLD", "RILD", "AT90", "AHT90", "RD",
                            "RLA5", "RLA4", "RT90") or m.startswith(("AT9", "AHT9", "RES", "LLD", "ILD"))),
    ("RHOB", lambda m: m in ("RHOB", "RHOZ", "DEN", "DENS", "ZDEN", "RHOM")),
    ("NPHI", lambda m: m in ("NPHI", "TNPH", "NPOR", "CNL", "NPHI_LS", "NPHI_SS")),
    ("PEF", lambda m: m in ("PEF", "PEFZ", "PE")),
    ("CALI", lambda m: m in ("CALI", "CAL", "HCAL", "CALS")),
    ("SP", lambda m: m == "SP"),
]


def detect_roles(df):
    roles = {}
    for role, rule in ROLE_RULES:
        cands = [c for c in df.columns if rule(c) and c not in roles.values()]
        if cands:
            roles[role] = max(cands, key=lambda c: df[c].notna().sum())
    return roles


def load_las(path):
    import lasio
    las = lasio.read(str(path), ignore_header_errors=True)
    df = las.df()
    unit = (las.curves[0].unit or "").strip().upper().rstrip(".")
    scale = 0.3048 if unit in ("F", "FT", "FEET") else 1.0
    df.index = df.index.astype(float) * scale
    df.index.name = "MD"
    df = df[~df.index.duplicated()].sort_index()
    df.columns = [str(c).upper() for c in df.columns]
    df = df.where(df > -990)
    units = {c.mnemonic.upper(): (c.unit or "") for c in las.curves[1:]}
    return df, units, ("converted ft->m" if scale != 1 else "m")


def curve_stats(s):
    v = s.dropna()
    if v.empty:
        return {"null_pct": 100.0}
    p = np.percentile(v, [10, 50, 90])
    return {"null_pct": 100 * (1 - len(v) / len(s)), "top_m": v.index.min(), "base_m": v.index.max(),
            "min": v.min(), "p10": p[0], "p50": p[1], "mean": v.mean(), "p90": p[2], "max": v.max()}


# ------------------------------------------------------------------- seismic
def _grid_ok(n_il, n_xl, n_tr):
    """A usable survey grid has several lines in both directions and roughly as many cells as traces."""
    return n_il >= 3 and n_xl >= 3 and n_il * n_xl <= 4 * n_tr


def regrid_from_xy(x, y):
    """Rebuild inline/xline numbers from trace coordinates when the SEG-Y headers are missing or degenerate.
    Finds the survey rotation from nearest-neighbour offsets, rotates, and bins to the median trace spacing."""
    from scipy.spatial import cKDTree
    pts = np.c_[x, y]
    sub = np.linspace(0, len(x) - 1, min(len(x), 20000)).astype(int)
    d, i = cKDTree(pts).query(pts[sub], k=3)
    v = np.vstack([pts[i[:, 1]] - pts[sub], pts[i[:, 2]] - pts[sub]])
    v = v[np.hypot(v[:, 0], v[:, 1]) > 0]
    ang = np.arctan2(v[:, 1], v[:, 0])
    th = np.angle(np.mean(np.exp(4j * ang))) / 4          # angle of the grid axes (mod 90 deg)
    c, s = np.cos(th), np.sin(th)
    u, w = pts[:, 0] * c + pts[:, 1] * s, -pts[:, 0] * s + pts[:, 1] * c
    def step(a):
        ua = np.unique(np.round(a, 3))
        df = np.diff(ua)
        df = df[df > 1e-6 * max(np.ptp(a), 1)]
        return float(np.percentile(df, 10)) if len(df) else 1.0
    du, dw = step(u), step(w)
    xl = np.round((u - u.min()) / du).astype(int) + 1
    il = np.round((w - w.min()) / dw).astype(int) + 1
    return il, xl, {"rotation_deg": float(np.degrees(th)), "spacing_xl": du, "spacing_il": dw}


def _open_segy_geom(path):
    import segyio
    for il_b, xl_b in SEGY_BYTE_OPTIONS:
        try:
            f = segyio.open(str(path), "r", iline=il_b, xline=xl_b, strict=False)
            if f.ilines is not None and f.xlines is not None and _grid_ok(len(f.ilines), len(f.xlines), f.tracecount):
                return f, (il_b, xl_b), True
            f.close()
        except Exception:
            continue
    return segyio.open(str(path), "r", ignore_geometry=True), None, False


def _headers(f, il_b, xl_b):
    sc = f.attributes(71)[:].astype(float)           # coordinate scalar
    sc = np.where(sc == 0, 1.0, np.where(sc < 0, 1.0 / np.abs(sc), sc))
    x, y = f.attributes(181)[:].astype(float), f.attributes(185)[:].astype(float)  # CDP X/Y
    if np.ptp(x) == 0 and np.ptp(y) == 0:
        x, y = f.attributes(73)[:].astype(float), f.attributes(77)[:].astype(float)  # source X/Y
    il, xl = f.attributes(il_b)[:].astype(int), f.attributes(xl_b)[:].astype(int)
    return x * sc, y * sc, il, xl


def _axis(a):
    u = np.unique(a)
    return {"min": int(u.min()), "max": int(u.max()), "count": int(len(u)),
            "step": int(np.median(np.diff(u))) if len(u) > 1 else 1}


def seismic_card(path):
    import segyio
    f, used, has_geom = _open_segy_geom(path)
    try:
        n = norm(path.stem)
        tag = next((t for t in VOLUME_TAGS if t in n), None)
        name = VOLUME_KEYS.get(tag, path.stem.lower())
        il_b, xl_b = used or (189, 193)
        ns = len(f.samples)
        dt_ms = segyio.tools.dt(f) / 1000.0
        t0 = float(f.header[0][segyio.TraceField.DelayRecordingTime])
        samples = t0 + dt_ms * np.arange(ns)
        x, y, il, xl = _headers(f, il_b, xl_b)
        regridded = None
        if not _grid_ok(len(np.unique(il)), len(np.unique(xl)), len(il)):
            il, xl, regridded = regrid_from_xy(x, y)
            has_geom = False
        np.savez(CACHE_DIR / f"geom_{name}.npz", x=x, y=y, il=il, xl=xl, samples=samples)
        step = max(1, f.tracecount // 300)
        a = np.array([f.trace[i] for i in range(0, f.tracecount, step)], dtype=float)
        pc = np.percentile(a, [1, 50, 99])
        try:
            text = " ".join(segyio.tools.wrap(f.text[0]).split())[:600]
        except Exception:
            text = ""
        kind = "inversion" if tag in ("pimp", "simp", "sand", "vpvs") else "stack"
        return {
            "name": name, "file": path.name, "path": str(path), "kind": kind,
            "label": VOLUME_LABELS.get(tag, name), "tags": VOLUME_TAGS.get(tag, []),
            "n_traces": int(f.tracecount), "n_samples": ns, "sample_rate_ms": dt_ms,
            "time_start_ms": float(samples[0]), "time_end_ms": float(samples[-1]),
            "inline": _axis(il), "xline": _axis(xl),
            "x_range": [x.min(), x.max()], "y_range": [y.min(), y.max()],
            "geometry_regular": has_geom,
            "geometry_rebuilt_from_xy": regridded, "segy_bytes": {"inline": il_b, "xline": xl_b},
            "amplitude": {"min": a.min(), "p1": pc[0], "median": pc[1], "p99": pc[2], "max": a.max(),
                          "mean": a.mean(), "std": a.std(), "zero_fraction": float((a == 0).mean())},
            "text_header_excerpt": text,
            "assumption": "vertical axis assumed TWT in ms",
        }
    finally:
        f.close()


class Volume:
    """Light wrapper: trace coordinates in a KD-tree + lazy trace access."""

    def __init__(self, name, path):
        from scipy.spatial import cKDTree
        g = np.load(CACHE_DIR / f"geom_{name}.npz")
        self.name, self.path = name, path
        self.x, self.y, self.il, self.xl, self.samples = (g[k] for k in ("x", "y", "il", "xl", "samples"))
        self.tree = cKDTree(np.c_[self.x, self.y])
        sub = np.linspace(0, len(self.x) - 1, min(2000, len(self.x))).astype(int)
        dd, _ = self.tree.query(np.c_[self.x[sub], self.y[sub]], k=2)
        self.max_dist = max(float(3 * np.median(dd[:, 1])), 1e-3)  # "inside volume" tolerance
        self._f = None

    def trace(self, i):
        import segyio
        if self._f is None:
            self._f = segyio.open(self.path, "r", ignore_geometry=True)
        return np.asarray(self._f.trace[int(i)], dtype=float)

    def sample_along(self, x, y, twt):
        """Values of the volume along a path; NaN where the path is outside the volume."""
        x, y, twt = (np.asarray(a, float) for a in (x, y, twt))
        d, idx = self.tree.query(np.c_[x, y])
        ok = (d <= self.max_dist) & (twt >= self.samples[0]) & (twt <= self.samples[-1])
        vals = np.full(len(x), np.nan)
        for i in np.unique(idx[ok]):
            m = ok & (idx == i)
            vals[m] = np.interp(twt[m], self.samples, self.trace(i))
        return vals, d


_VOLS = {}


def open_volume(name):
    v = resolve_volume(name)
    if v not in _VOLS:
        _VOLS[v] = Volume(v, load_cards()["volumes"][v]["path"])
    return _VOLS[v]


# ------------------------------------------------------------------ loaders
def cache_ready():
    return (CACHE_DIR / "cards.json").exists()


@lru_cache(None)
def load_cards():
    return json.loads((CACHE_DIR / "cards.json").read_text())


@lru_cache(None)
def load_track(well):
    return pd.read_csv(CACHE_DIR / "tracks" / f"{well}.csv")


@lru_cache(None)
def load_logs(well):
    return pd.read_pickle(CACHE_DIR / "logs" / f"{well}.pkl")


@lru_cache(None)
def load_extraction():
    p = CACHE_DIR / "extraction.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else {}


def clear_caches():
    for f in (load_cards, load_track, load_logs, load_extraction):
        f.cache_clear()
    _VOLS.clear()


def resolve_well(name):
    wells, k = load_cards()["wells"], norm(name)
    for w in wells:
        if norm(w) == k:
            return w
    for w in sorted(wells, key=len, reverse=True):
        if norm(w) in k and (k.startswith(norm(w)) or k.endswith(norm(w))):
            return w
    raise ValueError(f"Unknown well '{name}'. Available: {', '.join(wells)}")


def resolve_volume(name):
    vols, k = load_cards()["volumes"], norm(name)
    for v in vols:
        if norm(v) == k:
            return v
    for v, c in vols.items():
        if k in c["tags"]:
            return v
    for v in vols:
        if k and (k in norm(v) or norm(v) in k):
            return v
    raise ValueError(f"Unknown volume '{name}'. Available: {', '.join(vols)}")


# --------------------------------------------------------------------- main
def main():
    (CACHE_DIR / "tracks").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "logs").mkdir(exist_ok=True)
    cards = {"wells": {}, "volumes": {}, "warnings": []}
    warn = cards["warnings"].append

    info, col = load_well_info()
    wells = list(info.well_id)
    txt_files = find_files({".txt"})
    dev_files = [p for p in txt_files if "dev" in str(p.relative_to(DATA_DIR)).lower()]
    cs_files = [p for p in txt_files if p not in dev_files]
    las_files = find_files({".las"})
    print(f"{len(wells)} wells | {len(dev_files)} deviation | {len(cs_files)} checkshot | {len(las_files)} LAS")

    tracks, cs_tables = {}, {}
    for _, r in info.iterrows():
        w = r.well_id
        g = lambda k: r[col[k]] if k in col else np.nan
        rt, td, x0, y0 = g("rt"), g("td"), g("x"), g("y")
        c = {"well": w, "type": str(g("well types")), "rt_m": rt, "td_m": td, "wd_m": g("wd"),
             "x": x0, "y": y0, "has_deviation": _flag(g("deviations")),
             "has_checkshot": _flag(g("checkshots")), "has_markers": _flag(g("markers")),
             "has_las": _flag(g("las"))}
        dev = next((p for p in dev_files if match_well(p, wells) == w), None)
        if dev is None:
            warn(f"{w}: no deviation file found, assumed vertical")
        try:
            tr = build_track(dev, x0, y0, rt, td)
        except Exception as e:
            warn(f"{w}: deviation parse failed ({e}); assumed vertical")
            tr = build_track(None, x0, y0, rt, td)
        tracks[w] = tr
        c["max_inclination_deg"] = tr.inc.max()
        cs = next((p for p in cs_files if match_well(p, wells) == w), None)
        if cs is not None:
            try:
                cs_tables[w] = load_checkshot(cs, tr, rt)
            except Exception as e:
                warn(f"{w}: checkshot parse failed ({e})")
        elif c["has_checkshot"]:
            warn(f"{w}: Excel lists a checkshot but no checkshot file matched this well name")
        cards["wells"][w] = c

    # time-depth: own checkshot, else regional average of the checkshot wells (low confidence)
    max_tvdss = max(t.tvdss.max() for t in tracks.values())
    grid = np.arange(0, max_tvdss + 500, 10.0)
    avg = (np.mean([extrap(grid, c.tvdss, c.twt) for c in cs_tables.values()], axis=0)
           if cs_tables else None)
    for w, tr in tracks.items():
        c = cards["wells"][w]
        if w in cs_tables:
            tr["twt_ms"] = extrap(tr.tvdss, cs_tables[w].tvdss, cs_tables[w].twt)
            c["time_depth"] = {"source": "own checkshot", "confidence": "high"}
        elif avg is not None:
            tr["twt_ms"] = extrap(tr.tvdss, grid, avg)
            c["time_depth"] = {"source": f"average of {len(cs_tables)} checkshot wells",
                               "confidence": "low",
                               "note": "regional average curve; tie to seismic uncertain"}
        else:
            tr["twt_ms"] = np.nan
            c["time_depth"] = {"source": "none", "confidence": "none"}
        tr.to_csv(CACHE_DIR / "tracks" / f"{w}.csv", index=False)
        b = tr.iloc[-1]
        c["track_bottom"] = {"md_m": b.md, "tvd_m": b.tvd, "tvdss_m": b.tvdss, "x": b.x,
                             "y": b.y, "twt_ms": b.twt_ms}

    # logs
    for w in wells:
        c = cards["wells"][w]
        f = next((p for p in las_files if match_well(p, wells) == w), None)
        if f is None:
            warn(f"{w}: no LAS file matched")
            continue
        try:
            df, units, depth_note = load_las(f)
        except Exception as e:
            warn(f"{w}: LAS read failed ({e})")
            continue
        df.to_pickle(CACHE_DIR / "logs" / f"{w}.pkl")
        c["roles"] = detect_roles(df)
        c["las"] = {"file": f.name, "depth_unit_note": depth_note, "md_top_m": df.index.min(),
                    "md_base_m": df.index.max(), "n_curves": df.shape[1],
                    "curves": {m: {"unit": units.get(m, ""), **curve_stats(df[m])} for m in df.columns}}
        for need in ("GR", "RES", "RHOB", "DT"):
            if need not in c["roles"]:
                warn(f"{w}: no {need} curve recognised (see ROLE_RULES in ingest.py)")

    # seismic
    for p in find_files({".sgy", ".segy"}):
        print("reading", p.name)
        try:
            card = seismic_card(p)
            if card["name"] in cards["volumes"]:
                warn(f"{p.name}: duplicate volume type '{card['name']}', overwriting earlier file")
            cards["volumes"][card["name"]] = card
        except Exception as e:
            warn(f"{p.name}: SEG-Y read failed ({e})")

    # coverage + well-to-volume extraction
    ext = {}
    for vname, vc in cards["volumes"].items():
        vol = Volume(vname, vc["path"])
        if np.ptp(vol.x) == 0 and np.ptp(vol.y) == 0:
            warn(f"{vname}: coordinate headers empty, wells cannot be linked")
        for w, tr in tracks.items():
            c = cards["wells"][w]
            d, _ = vol.tree.query([c["x"], c["y"]])
            c.setdefault("volume_coverage", {})[vname] = {"nearest_trace_m": d, "inside": bool(d <= vol.max_dist)}
            if vc["kind"] != "inversion":
                continue
            md = np.arange(tr.md.min(), tr.md.max(), TRACK_STEP_M)
            I = lambda col: np.interp(md, tr.md, tr[col])
            vals, dist = vol.sample_along(I("x"), I("y"), I("twt_ms"))
            ok = np.isfinite(vals)
            if not ok.any():
                c.setdefault("extraction", {})[vname] = {"n_points": 0}
                continue
            ext.setdefault(w, {})[vname] = {"md": md[ok], "tvdss": I("tvdss")[ok],
                                           "twt": I("twt_ms")[ok], "val": vals[ok]}
            v = vals[ok]
            c.setdefault("extraction", {})[vname] = {
                "n_points": int(ok.sum()), "md_range": [md[ok].min(), md[ok].max()],
                "mean": v.mean(), "min": v.min(), "max": v.max(),
                "p10": np.percentile(v, 10), "p90": np.percentile(v, 90)}
        if vc["kind"] == "inversion" and not any(c["volume_coverage"][vname]["inside"]
                                                 for c in cards["wells"].values()):
            warn(f"{vname}: no well lies inside it - check coordinates/CRS")

    (CACHE_DIR / "extraction.pkl").write_bytes(pickle.dumps(ext))
    (CACHE_DIR / "cards.json").write_text(json.dumps(clean(cards), indent=1))
    clear_caches()
    print(f"done: {len(cards['wells'])} wells, {len(cards['volumes'])} volumes, "
          f"{len(cards['warnings'])} warnings")
    for m in cards["warnings"]:
        print("  warning:", m)
    return cards


if __name__ == "__main__":
    main()
