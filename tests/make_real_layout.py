"""
make_real_layout.py - synthetic data in the SAME layout as the team's real export (names and column layout), for testing:
  Fault (SHP)/1500_Fault_01/1500_Fault_01.shp   fault outlines as polygons in lon/lat (geographic CRS)
  Polygon (SHP)/0080_Channel_01/..., 0080_Overbank_01/...   lon/lat
  Fault (CSV)/TWT_1500_Fault_01.csv, IL_1800_Fault_01.csv, CL_2000_Fault_02.csv   columns FaultID,StickID,IL,CL,TWT (ms)
  Horizon/Horizon_1.xyt
Usage: python make_real_layout.py OUT_DIR CACHE_DIR(with raw geometry cache)
"""
import sys
from pathlib import Path

import numpy as np
import shapefile


def main(out, il0=100, xl0=200, n=40):
    import fault_csv as F
    import ingest as I
    out = Path(out)
    g = np.load(I.CACHE_DIR / next(p.name for p in I.CACHE_DIR.glob("geom_*.npz")))
    names = [p.stem[5:] for p in I.CACHE_DIR.glob("geom_*.npz")]
    gn = names[0]
    th = np.radians(25)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def to_lonlat(xy):
        d = (xy - np.array([1360500.0, 1797500.0])) @ R.T            # rotated survey grid -> true east/north metres
        return np.c_[122.2 + d[:, 0] / (111320.0 * np.cos(np.radians(16.2))), 16.2 + d[:, 1] / 110574.0]

    def xy(il, xl):
        return F.to_xy(np.asarray(il, float), np.asarray(xl, float), gn)[0]

    def write_poly(path, ring_ll):
        w = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
        w.field("ID", "N")
        r = [list(p) for p in ring_ll]
        r.append(r[0])
        a = 0.5 * sum(r[k][0] * r[k + 1][1] - r[k + 1][0] * r[k][1] for k in range(len(r) - 1))
        if a > 0:
            r = r[::-1]
        w.poly([r])
        w.record(1)
        w.close()
        Path(str(path) + ".prj").write_text('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984"]]')

    def buf(p, w=0.7):
        d = np.gradient(p, axis=0)
        nrm = np.c_[-d[:, 1], d[:, 0]]
        nrm /= np.hypot(nrm[:, 0], nrm[:, 1])[:, None] + 1e-12
        return np.vstack([p + nrm * w, (p - nrm * w)[::-1]])

    faults = {1500: [(np.c_[np.linspace(il0 + 4, il0 + 34, 18), xl0 + 28 + 1.5 * np.sin(np.linspace(0, 3, 18))])],
              2500: [(np.c_[np.linspace(il0 + 4, il0 + 34, 20), xl0 + 29 + np.linspace(0, 2, 20)]),
                     (np.c_[np.linspace(il0 + 6, il0 + 16, 10), xl0 + 5 + np.linspace(0, 10, 10)])],
              3500: [(np.c_[np.linspace(il0 + 4, il0 + 34, 15), xl0 + 31 + np.linspace(0, 3, 15)])]}
    (out / "Fault (CSV)").mkdir(parents=True, exist_ok=True)
    for t, fl in faults.items():
        for k, f in enumerate(fl, 1):
            np.savetxt(out / "Fault (CSV)" / f"TWT_{t}_Fault_{k:02d}.csv", np.c_[np.zeros(len(f)), np.zeros(len(f)), f, np.full(len(f), float(t))],
                       delimiter=",", fmt="%.2f", header="FaultID,StickID,IL,CL,TWT (ms)", comments="")
            d = out / "Fault (SHP)" / f"{t}_Fault_{k:02d}"
            d.mkdir(parents=True, exist_ok=True)
            write_poly(d / f"{t}_Fault_{k:02d}", to_lonlat(buf(xy(f[:, 0], f[:, 1]), 40.0)))
    # sections: IL_<n> sticks crossing the main fault at depth
    for line, k in ((il0 + 20, 1), (il0 + 30, 1)):
        tt = np.linspace(1500, 3500, 9)
        cl = xl0 + 28.5 + (tt - 1500) / 2000 * 3
        np.savetxt(out / "Fault (CSV)" / f"IL_{line}_Fault_{k:02d}.csv", np.c_[np.zeros(9), np.zeros(9), np.full(9, line), cl, tt],
                   delimiter=",", fmt="%.2f", header="FaultID,StickID,IL,CL,TWT (ms)", comments="")
    # channels / overbank polygons in lon/lat, folder number = unknown slice id
    s = np.linspace(0, 1, 40)
    chan = np.c_[il0 + 5 + 30 * s, xl0 + 8 + 20 * s + 3 * np.sin(s * 9)]
    for num in (50, 60, 80):
        ch = to_lonlat(buf(xy(chan[:, 0], chan[:, 1]), 60.0))
        d = out / "Polygon (SHP)" / f"{num:04d}_Channel_01"
        d.mkdir(parents=True, exist_ok=True)
        write_poly(d / f"{num:04d}_Channel_01", ch)
        d = out / "Polygon (SHP)" / f"{num:04d}_Overbank_01"
        d.mkdir(parents=True, exist_ok=True)
        write_poly(d / f"{num:04d}_Overbank_01", to_lonlat(buf(xy(chan[:, 0], chan[:, 1]), 180.0)))
    # horizon with a dome
    il, xl = np.meshgrid(np.linspace(il0 + 1, il0 + 38, 40), np.linspace(xl0 + 1, xl0 + 38, 40))
    p = xy(il.ravel(), xl.ravel())
    t = 2400 - 250 * np.exp(-(((il - il0 - 18) / 9) ** 2 + ((xl - xl0 - 15) / 9) ** 2)).ravel() * 0 + 0
    t = 2400 - 250 * np.exp(-(((il.ravel() - il0 - 18) / 9) ** 2 + ((xl.ravel() - xl0 - 15) / 9) ** 2))
    (out / "Horizon").mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "Horizon" / "Horizon_1.xyt", np.c_[p, t], fmt="%.2f")
    print("real-layout synthetic data written to", out)


if __name__ == "__main__":
    main(sys.argv[1])
