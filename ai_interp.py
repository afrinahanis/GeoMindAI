"""
ai_interp.py - deep-learning seismic interpretation (PyTorch) for the Exploration Copilot.

1) FAULTS: a small 3D U-Net trained on SYNTHETIC seismic (random reflectivity, folds, planar faults with
   throw; labels are the fault planes). This is the approach of FaultSeg3D (Wu et al., 2019): train on
   synthetic volumes with exact labels, then apply to field data. No labelled field data needed.
       python ai_interp.py train            # trains and writes models/fault_unet.pt
       python ai_interp.py selftest         # checks it finds planted faults
2) FACIES: unsupervised waveform + attribute clustering (k-means) on a horizon window - no training needed.

The Gemma tools `ai_fault_detection` and `ai_facies_map` (registered in tools.py) call these.
If torch or the weights are missing, ai_fault_detection falls back to the coherence method automatically.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

MODEL_PATH = Path(os.environ.get("MODEL_DIR", "models")) / "fault_unet.pt"
TILE = 64


# ============================ synthetic training data ============================
def _ricker(f, dt, n=41):
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_synthetic(rng, n=TILE):
    """One (volume, fault-label) pair, shape [n, n, n] = [inline, xline, time]."""
    L = n + 80
    refl = rng.normal(0, 1, L) * (rng.random(L) < rng.uniform(0.15, 0.5))
    refl = refl + 0.3 * rng.normal(0, 1, L)
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    # folding: smooth lateral warp of the time axis
    warp = ndi.gaussian_filter(rng.normal(0, 1, (n, n)), rng.uniform(6, 16))
    warp = warp / (np.abs(warp).max() + 1e-9) * rng.uniform(0, 10)
    shift = np.broadcast_to(warp[:, :, None], (n, n, n)).copy()
    label = np.zeros((n, n, n), np.float32)
    for _ in range(rng.integers(0, 4)):          # 0-3 planar faults
        strike = rng.uniform(0, math.pi)
        dip_from_vertical = math.radians(rng.uniform(0, 25))
        nx, ny = math.cos(strike), math.sin(strike)
        nz = math.tan(dip_from_vertical) * rng.uniform(-1, 1)
        nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
        c = rng.uniform(0.25, 0.75) * n
        d = (ii - n / 2) * nx + (jj - n / 2) * ny + (kk - n / 2) * nz - (c - n / 2)
        throw = rng.uniform(3, 9) * rng.choice([-1, 1])
        shift += np.where(d > 0, throw, 0.0)
        label = np.maximum(label, (np.abs(d) < 1.5).astype(np.float32))
    pos = kk + 40 + shift
    pos = np.clip(pos, 0, L - 1.001)
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    vol = refl[i0] * (1 - fr) + refl[i0 + 1] * fr
    w = _ricker(rng.uniform(0.05, 0.12), 1.0, 41)
    vol = ndi.convolve1d(vol, w, axis=2, mode="nearest")
    vol += rng.normal(0, rng.uniform(0.02, 0.35), vol.shape) * vol.std()
    if rng.random() < 0.5:                       # laterally smooth noise (like footprint/coloured noise)
        vol += ndi.gaussian_filter(rng.normal(0, 1, vol.shape), (2, 2, 0.5)) * vol.std() * rng.uniform(0, 0.4)
    vol *= (1 + 0.5 * ndi.gaussian_filter(rng.normal(0, 1, (n, n, n)), 12) * rng.uniform(0, 1))
    vol = (vol - vol.mean()) / (vol.std() + 1e-9)
    return vol.astype(np.float32), label.astype(np.float32)


# ================================== the model ====================================
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_unet(base=12):
    torch, nn, F = _torch()

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU(inplace=True),
                             nn.Conv3d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU(inplace=True))

    class UNet3D(nn.Module):
        def __init__(self):
            super().__init__()
            b = base
            self.e1, self.e2, self.e3 = block(1, b), block(b, 2 * b), block(2 * b, 4 * b)
            self.mid = block(4 * b, 8 * b)
            self.up3, self.d3 = nn.ConvTranspose3d(8 * b, 4 * b, 2, 2), block(8 * b, 4 * b)
            self.up2, self.d2 = nn.ConvTranspose3d(4 * b, 2 * b, 2, 2), block(4 * b, 2 * b)
            self.up1, self.d1 = nn.ConvTranspose3d(2 * b, b, 2, 2), block(2 * b, b)
            self.out = nn.Conv3d(b, 1, 1)
            self.pool = nn.MaxPool3d(2)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.d3(torch.cat([self.up3(m), e3], 1))
            d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.out(d1)          # logits

    return UNet3D()


def _loss(logits, y):
    torch, nn, F = _torch()
    pos = y.mean().clamp(1e-4, 0.5)
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=((1 - pos) / pos).clamp(max=50.0))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


def train(steps=1500, batch=4, lr=2e-3, seed=0, device=None, log_every=50, save=MODEL_PATH):
    """Train on freshly generated synthetic volumes (no dataset on disk). ~15-30 min on a GPU for 1500 steps."""
    torch, nn, F = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    net = build_unet().to(device)
    if Path(save).exists():
        net.load_state_dict(torch.load(save, map_location=device))
        print("resuming from", save)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0, run = time.time(), None
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    for s in range(1, steps + 1):
        xs, ys = zip(*[make_synthetic(rng) for _ in range(batch)])
        x = torch.from_numpy(np.stack(xs))[:, None].to(device)
        y = torch.from_numpy(np.stack(ys))[:, None].to(device)
        if rng.random() < 0.5:                      # flip augmentation (inline/xline)
            x, y = x.flip(2), y.flip(2)
        net.train()
        loss = _loss(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.95 * run + 0.05 * loss.item()
        if s % log_every == 0 or s == steps:
            print(f"step {s}/{steps}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)
            torch.save(net.state_dict(), save)
    return net


def available():
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "PyTorch is not installed"
    if not MODEL_PATH.exists():
        return False, f"no weights at {MODEL_PATH} (run: python ai_interp.py train)"
    return True, ""


_NET = {}


def load_net(device=None):
    torch, _, _ = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in _NET:
        net = build_unet().to(device)
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        net.eval()
        _NET[device] = net
    return _NET[device], device


def predict_volume(vol, stride=32, device=None):
    """Fault probability for a [inline, xline, time] volume of any size (tiled, overlap-averaged)."""
    torch, _, _ = _torch()
    net, device = load_net(device)
    v = (vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-9)
    v = np.nan_to_num(v).astype(np.float32)
    shp = v.shape
    pad = [(0, max(0, TILE - s) + ((-(max(s, TILE) - TILE)) % stride)) for s in shp]
    vp = np.pad(v, pad, mode="reflect")
    acc, cnt = np.zeros(vp.shape, np.float32), np.zeros(vp.shape, np.float32)
    w1 = np.hanning(TILE + 2)[1:-1].astype(np.float32) + 0.05
    w = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
    starts = [list(range(0, s - TILE + 1, stride)) for s in vp.shape]
    ntile = len(starts[0]) * len(starts[1]) * len(starts[2])
    print(f"AI fault detection: {ntile} tiles on {device} (cube {shp}); set AI_MAX_TRACES / AI_STRIDE env vars to trade speed for detail", flush=True)
    done = 0
    with torch.no_grad():
        for a in starts[0]:
            for b in starts[1]:
                for c in starts[2]:
                    t = torch.from_numpy(vp[a:a + TILE, b:b + TILE, c:c + TILE][None, None]).to(device)
                    p = torch.sigmoid(net(t))[0, 0].cpu().numpy()
                    acc[a:a + TILE, b:b + TILE, c:c + TILE] += p * w
                    cnt[a:a + TILE, b:b + TILE, c:c + TILE] += w
                    done += 1
                    if done % 25 == 0:
                        print(f"  tile {done}/{ntile}", flush=True)
    return (acc / np.maximum(cnt, 1e-6))[:shp[0], :shp[1], :shp[2]]


# ================================ Gemma-facing tools ===============================
def ai_fault_detection(twt_min_ms=None, twt_max_ms=None, threshold=0.5):
    """Detect faults with the 3D U-Net (AI) on the full stack; returns the number of faults, strike, length, location, time range and a map. Falls back to the coherence method if the AI model is unavailable."""
    import interpret as N
    import ingest as I
    ok, why = available()
    if not ok:
        r = N.count_faults()
        r["method"] = "coherence fallback"
        r["why_not_ai"] = why
        return r
    vn = N._vol_name(["full stack"])
    v = I.open_volume(vn)
    t0 = v.samples[0] if twt_min_ms is None else twt_min_ms
    t1 = v.samples[-1] if twt_max_ms is None else twt_max_ms
    seg = N._segments(vn, t0, t1, max_traces=int(os.environ.get("AI_MAX_TRACES", 10000)))
    cube, valid, edge = N.prep_cube_for_faults(seg["cube"])      # fill outside-footprint cells, flag survey-edge band
    dt = float(np.median(np.diff(v.samples)))
    prob = predict_volume(cube, stride=int(os.environ.get("AI_STRIDE", 48)))
    low = (prob > threshold)
    low_t = low.transpose(2, 0, 1)[::max(1, int(round(40 / dt)))]            # [windows, il, xl]
    pers = low_t.mean(0)
    pers = np.where(edge, 0.0, pers)                     # survey-edge steps are acquisition footprint, not faults
    inside = pers[~edge] if (~edge).any() else pers
    cand = pers >= max(0.15, float(np.percentile(inside, 90)))
    cand = ndi.binary_closing(cand, structure=np.ones((3, 3))) | cand
    cand &= ~edge
    from skimage.morphology import skeletonize
    skel = skeletonize(cand)
    nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junction = skel & (nb >= 3)
    branches = skel & ~ndi.binary_dilation(junction, structure=np.ones((3, 3)))
    lab, n = ndi.label(branches, structure=np.ones((3, 3)))
    segs = []
    starts = list(range(low_t.shape[0]))
    for b in range(1, n + 1):
        rr, cc = np.where(lab == b)
        if len(rr) < 6:
            continue
        for sg in N._split_straight(N._order_path(list(zip(rr.tolist(), cc.tolist()))), tol=2.5):
            sr, sc = np.array([p[0] for p in sg]), np.array([p[1] for p in sg])
            dr, dc = sr[-1] - sr[0], sc[-1] - sc[0]
            if len(sg) < 6 or math.hypot(dr, dc) < 6:
                continue
            seen = low_t[:, sr, sc].mean(1) > 0.4
            segs.append({"rr": sr, "cc": sc, "length": (math.hypot(dr, dc) + 1) * seg["spacing"],
                         "elong": math.hypot(dr, dc) / 2.0, "strike": math.degrees(math.atan2(dc, dr)) % 180,
                         "persist": float(pers[sr, sc].mean()), "win_idx": np.where(seen)[0], "starts": starts})
    fl = N._merge_collinear(segs, seg["spacing"], pers, low_t, starts)
    fl.sort(key=lambda f: -f["length"])
    faults = []
    step_ms = max(1, int(round(40 / dt))) * dt
    for i, f in enumerate(fl[:12]):
        wi = f["win_idx"]
        faults.append({"id": f"F{i+1}", "length_map_units": N._f(f["length"], 0), "strike_deg_from_inline_direction": N._f(f["strike"], 0),
                       "mean_fault_probability": N._f(float(prob[:, :, :][f["rr"], f["cc"]].mean()), 2),
                       "persistence_0_1": N._f(f["persist"], 2),
                       "centre_inline_xline": [int(seg["il"][int(f["rr"].mean())]), int(seg["xl"][int(f["cc"].mean())])],
                       "seen_between_twt_ms": [N._f(t0 + wi.min() * step_ms, 0), N._f(t0 + (wi.max() + 1) * step_ms, 0)] if len(wi) else None})
    mask = np.zeros(pers.shape, bool)
    for f in fl:
        mask[f["rr"], f["cc"]] = True
    N.LAST_FAULTS.clear()
    N.LAST_FAULTS.update({"seg": seg, "mask": mask, "method": "AI 3D U-Net"})
    img = N._plot("ai_fault_overview", seg, [{"kind": "image", "data": pers, "cmap": "gray_r", "label": "fraction of time slices with AI fault probability > %.2f" % threshold},
                                              {"kind": "mask", "data": mask}], f"AI fault detection (3D U-Net): {len(fl)} faults")
    return {"method": "3D U-Net trained on synthetic seismic", "volume": vn, "twt_window_ms": [float(t0), float(t1)],
            "n_faults": len(fl), "faults": faults, "image_path": img, "kind": "map", "index_used": "AI faults",
            "max_fault_probability": N._f(float(prob.max()), 2), "survey_edge_band_cells_excluded": int(edge.sum()),
            "caveat": "Network trained on synthetic data; check the map against sections before relying on it. Lateral grid decimated to <= 40000 traces."}


def _kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


def ai_facies_map(twt_min_ms=None, twt_max_ms=None, n_classes=5):
    """Unsupervised seismic facies: k-means clustering of the stack waveform plus sand probability, Vp/Vs and P-impedance in a TWT window; returns a class map, class statistics and the class at each well."""
    import interpret as N
    import ingest as I
    if twt_min_ms is None or twt_max_ms is None:
        twt_min_ms, twt_max_ms = N._auto_window(12.0)
    feats, names = [], []
    seg0 = None
    fs = N._vol_name(["full stack"])
    seg0 = N._segments(fs, twt_min_ms, twt_max_ms, max_traces=6000)
    cube = seg0["cube"]
    nil, nxl, ns = cube.shape
    wf = np.nan_to_num(cube.reshape(-1, ns))
    wf = wf / (np.linalg.norm(wf, axis=1, keepdims=True) + 1e-9)
    feats.append(wf[:, ::max(1, ns // 6)])
    grids = {}
    for key, nm in (("sand probability", "sand_probability"), ("vpvs", "vpvs"), ("p impedance", "p_impedance")):
        vn = N._vol_name([key])
        if vn:
            m = N._read_maps(vn, twt_min_ms, twt_max_ms, max_traces=6000)["grid"]
            if m.shape == (nil, nxl):
                grids[nm] = m
                z = (m - np.nanmean(m)) / (np.nanstd(m) + 1e-9)
                feats.append(np.nan_to_num(z).reshape(-1, 1) * 2.0)
    X = np.hstack(feats)
    lab, C = _kmeans(X, int(n_classes))
    lm = lab.reshape(nil, nxl)
    order = np.argsort([np.nanmean(grids["sand_probability"][lm == j]) if "sand_probability" in grids and (lm == j).any() else 0 for j in range(int(n_classes))])[::-1]
    remap = {int(o): i + 1 for i, o in enumerate(order)}
    lm2 = np.vectorize(remap.get)(lm)
    stats = []
    for j in range(1, int(n_classes) + 1):
        msk = lm2 == j
        row = {"class": j, "share_of_map": N._f(msk.mean(), 2)}
        for nm, g in grids.items():
            row[f"mean_{nm}"] = N._f(np.nanmean(g[msk]), 3) if msk.any() else None
        stats.append(row)
    wells = {}
    for w, (i, x) in N._well_ilxl(seg0["vol"]).items():
        r, c = int(np.searchsorted(seg0["il"], i)), int(np.searchsorted(seg0["xl"], x))
        if r < nil and c < nxl:
            wells[w] = int(lm2[r, c])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    I.CACHE_DIR.joinpath("images").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(lm2, origin="lower", aspect="auto", cmap="tab10", vmin=1, vmax=10,
                   extent=[seg0["xl"][0], seg0["xl"][-1], seg0["il"][0], seg0["il"][-1]])
    for w, (i, x) in N._well_ilxl(seg0["vol"]).items():
        ax.plot(x, i, "k^", ms=6)
        ax.text(x, i, " " + w, fontsize=8)
    ax.set_title(f"AI seismic facies (k-means, {n_classes} classes), {twt_min_ms:.0f}-{twt_max_ms:.0f} ms\nclass 1 = highest mean sand probability")
    ax.set_xlabel("xline"); ax.set_ylabel("inline")
    path = I.CACHE_DIR / "images" / f"ai_facies_{twt_min_ms:.0f}_{twt_max_ms:.0f}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"method": "unsupervised k-means on stack waveform + sand probability, Vp/Vs, P-impedance",
            "window_ms": [float(twt_min_ms), float(twt_max_ms)], "n_classes": int(n_classes), "class_statistics": stats,
            "class_at_wells": wells, "image_path": str(path), "kind": "map", "index_used": f"{twt_min_ms:.0f}-{twt_max_ms:.0f}ms",
            "caveat": "Classes are statistical groupings; they need a geological meaning from wells (e.g. sand-rich class near gas wells)."}


# ====================================== CLI ========================================
def selftest(n_vol=6, thr=0.5):
    """Quality check on fresh synthetic volumes: precision and recall of fault voxels (2-voxel tolerance)."""
    rng = np.random.default_rng(12345)
    tp_p = n_p = tp_r = n_r = 0
    for _ in range(n_vol):
        v, y = make_synthetic(rng)
        p = predict_volume(v) > thr
        y = y > 0
        tp_p += int((p & ndi.binary_dilation(y, iterations=2)).sum()); n_p += int(p.sum())
        tp_r += int((y & ndi.binary_dilation(p, iterations=2)).sum()); n_r += int(y.sum())
    return {"precision": round(tp_p / max(n_p, 1), 3), "recall": round(tp_r / max(n_r, 1), 3),
            "predicted_voxels": n_p, "true_fault_voxels": n_r}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    elif cmd == "selftest":
        print(selftest())
