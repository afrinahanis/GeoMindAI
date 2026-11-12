"""
make_synthetic_interpreted.py - builds a demo data_export/ folder with the SAME layout as the real interpreted data:

  data_export/
    Fault (SHP)/timeslice_fault_01 ... 05/ fault.shp (+dbf/shx/prj)
    Polygon (SHP)/timeslice_channel_01 ... 03/, timeslice_overbank_01 ... 03/
    Horizon/horizon.xyt

Usage:  python make_synthetic_interpreted.py OUT_DIR [x0 y0 size]     (defaults match the demo raw data extent)
Only for testing the copilot without the real files.
"""
import sys
from pathlib import Path

import numpy as np
import shapefile


def write_lines(path, lines, attrs=None, z=None):
    w = shapefile.Writer(str(path), shapeType=shapefile.POLYLINEZ if z is not None else shapefile.POLYLINE)
    w.field("FAULT_ID", "N")
    w.field("NAME", "C", size=20)
    for i, ln in enumerate(lines):
        if z is not None:
            w.linez([[[x, y, z] for x, y in ln]])
        else:
            w.line([[list(p) for p in ln]])
        w.record(i + 1, f"F{i + 1}")
    w.close()
    Path(str(path) + ".prj").write_text('PROJCS["demo_UTM"]')


def write_polys(path, polys):
    w = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    w.field("ID", "N")
    for i, pg in enumerate(polys):
        ring = [list(p) for p in pg]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        # shapefile convention: outer rings clockwise
        a = 0.5 * sum(ring[k][0] * ring[k + 1][1] - ring[k + 1][0] * ring[k][1] for k in range(len(ring) - 1))
        if a > 0:
            ring = ring[::-1]
        w.poly([ring])
        w.record(i + 1)
    w.close()
    Path(str(path) + ".prj").write_text('PROJCS["demo_UTM"]')


def main(out, x0=1360000.0, y0=1797000.0, size=975.0, layout="A"):
    out = Path(out)
    fx = lambda f: x0 + f * size
    fy = lambda f: y0 + f * size
    rng = np.random.default_rng(5)
    # ---- horizon: anticline cut by a normal fault (east block dropped 35 ms)
    gx, gy = np.meshgrid(np.linspace(x0, x0 + size, 60), np.linspace(y0, y0 + size, 60))
    r2 = (gx - fx(0.45)) ** 2 + (gy - fy(0.5)) ** 2
    t = 900 - 130 * np.exp(-r2 / (2 * (0.22 * size) ** 2)) + 0.04 * (gy - y0)
    t = t + np.where(gx > fx(0.72), 35.0, 0.0)
    hd = out / "Horizon"
    hd.mkdir(parents=True, exist_ok=True)
    arr = np.c_[gx.ravel(), gy.ravel(), t.ravel() + rng.normal(0, 0.3, t.size)]
    np.savetxt(hd / "horizon.xyt", arr, fmt="%.2f")
    # ---- faults: main N-S fault at x = 0.72, second oblique fault in the last slices
    for k in range(1, 6):
        d = out / "Fault (SHP)" / f"timeslice_fault_{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        drift = (k - 3) * 0.004 * size
        main_f = [(fx(0.72) + drift + 8 * np.sin(s * 9), fy(0.05 + 0.9 * s)) for s in np.linspace(0, 1, 25)]
        lines = [main_f]
        if k >= 3:
            lines.append([(fx(0.1 + 0.3 * s), fy(0.15 + 0.25 * s)) for s in np.linspace(0, 1, 12)])
        write_lines(d / "fault", lines)
    # ---- polygons: sinuous channel + overbank belts, drifting a little per slice
    for k in range(1, 4):
        s = np.linspace(0, 1, 60)
        cx = fx(0.05 + 0.9 * s) + 12 * k
        cy = fy(0.85 - 0.6 * s + 0.06 * np.sin(s * 14)) + 10 * k
        nx, ny = -np.gradient(cy), np.gradient(cx)
        nn = np.hypot(nx, ny)
        nx, ny = nx / nn, ny / nn
        wch, wob = 30.0, 120.0
        chan = list(zip(cx + nx * wch, cy + ny * wch)) + list(zip((cx - nx * wch)[::-1], (cy - ny * wch)[::-1]))
        ob1 = list(zip(cx + nx * wob, cy + ny * wob)) + list(zip((cx + nx * wch)[::-1], (cy + ny * wch)[::-1]))
        ob2 = list(zip(cx - nx * wch, cy - ny * wch)) + list(zip((cx - nx * wob)[::-1], (cy - ny * wob)[::-1]))
        for cls, polys in (("channel", [chan]), ("overbank", [ob1, ob2])):
            d = out / "Polygon (SHP)" / f"timeslice_{cls}_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            write_polys(d / cls, polys)
    make_fault_csv(out, layout)
    print("synthetic interpreted data written to", out)


def make_fault_csv(out, layout="A", il0=100, xl0=200, n=40):
    """Fault (CSV) demo: IL_/CL_ point tables per slice + a TWT table. layout A: single columns, TWT 'slice,twt' with header;
    B: one column per fault (blank padded), TWT one value per slice, no header; C: TWT in seconds, one value per point (stacked)."""
    d = Path(out) / "Fault (CSV)"
    d.mkdir(parents=True, exist_ok=True)
    times = [800.0 + 20 * k for k in range(5)]
    stacked = []
    for k in range(1, 6):
        f1 = [(il0 + 3 + i, xl0 + 28 + round(0.1 * np.sin(i / 3.0) * 3)) for i in range(0, 34)]
        f2 = [(il0 + 5 + i, xl0 + 4 + i) for i in range(0, 12)] if k >= 3 else []
        if layout == "B":
            m = max(len(f1), len(f2))
            ilw = np.full((m, 2), np.nan)
            clw = np.full((m, 2), np.nan)
            ilw[:len(f1), 0], clw[:len(f1), 0] = [p[0] for p in f1], [p[1] for p in f1]
            if f2:
                ilw[:len(f2), 1], clw[:len(f2), 1] = [p[0] for p in f2], [p[1] for p in f2]
            np.savetxt(d / f"IL_timeslice_Fault_{k:02d}.csv", ilw[:, :2 if f2 else 1], delimiter=",", fmt="%g")
            np.savetxt(d / f"CL_timeslice_Fault_{k:02d}.csv", clw[:, :2 if f2 else 1], delimiter=",", fmt="%g")
        else:
            pts = f1 + f2
            hdr = "IL" if layout == "A" else ""
            np.savetxt(d / f"IL_timeslice_Fault_{k:02d}.csv", [p[0] for p in pts], fmt="%d", header=hdr, comments="")
            np.savetxt(d / f"CL_timeslice_Fault_{k:02d}.csv", [p[1] for p in pts], fmt="%d", header="CL" if layout == "A" else "", comments="")
        stacked += [times[k - 1] / 1000.0] * (len(f1) + len(f2))
    if layout == "A":
        np.savetxt(d / "TWT_timeslice_Fault.csv", np.c_[np.arange(1, 6), times], delimiter=",", fmt="%g", header="slice,twt", comments="")
    elif layout == "B":
        np.savetxt(d / "TWT_timeslice_Fault.csv", times, fmt="%g")
    else:
        np.savetxt(d / "TWT_timeslice_Fault.csv", stacked, fmt="%g")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], *(float(v) for v in a[2:5]), layout=(a[5] if len(a) > 5 else "A"))
