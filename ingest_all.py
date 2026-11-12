"""
ingest_all.py - one entry point for BOTH data families.

  run_all()   first-time or full ingestion: raw (DATA_DIR) + interpreted (INTERP_DIR)
  refresh()   dynamic: detect new/changed/removed files in both folders and re-ingest only what changed
  status()    what is ingested and what is missing

Either family may be absent: the copilot works on raw only, interpreted only, or both.
Command line:  python ingest_all.py [--force]
"""
import hashlib
import json
import sys

import ingest as I
import interp_data as D


def _raw_fingerprint():
    h = hashlib.md5(b"ingest-v2-grid-rebuild")   # bump when ingestion logic changes so caches are rebuilt
    n = 0
    if I.DATA_DIR.exists():
        for p in sorted(I.DATA_DIR.rglob("*")):
            if p.is_file():
                s = p.stat()
                h.update(f"{p.relative_to(I.DATA_DIR)}|{s.st_size}|{int(s.st_mtime)}".encode())
                n += 1
    return h.hexdigest(), n


def _fp_path():
    return I.CACHE_DIR / "raw_fingerprint.json"


def _raw_changed():
    fp, n = _raw_fingerprint()
    old = json.loads(_fp_path().read_text()) if _fp_path().exists() else {}
    return (not I.cache_ready()) or old.get("fp") != fp, fp, n


def _ingest_raw(fp, n):
    if not I.DATA_DIR.exists() or n == 0:
        return {"status": "skipped", "reason": f"no raw data found at {I.DATA_DIR}"}
    try:
        I.main()
    except Exception as e:
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
    I.clear_caches()
    I.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _fp_path().write_text(json.dumps({"fp": fp, "n_files": n}))
    return {"status": "ingested", "n_files": n}


def _ingest_interp(force=False):
    if (D.interp_dir() is None or not D.interp_dir().exists()) and not D.extra_dirs():
        return {"status": "skipped", "reason": "INTERP_DIR not set or folder missing"}
    try:
        return {"status": "ok", **D.scan(force=force)}
    except Exception as e:
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}"}


def run_all(force=False):
    changed, fp, n = _raw_changed()
    raw = _ingest_raw(fp, n) if (changed or force) else {"status": "up to date"}
    interp = _ingest_interp(force=force)
    rep = {"raw": raw, "interpreted": interp}
    for k, v in rep.items():
        print(f"[{k}] {v.get('status')} {v.get('reason', '')}")
        if k == "interpreted" and v.get("inventory"):
            print("   ", v["inventory"])
            for w in v.get("warnings", [])[:8]:
                print("    warn:", w)
    return rep


def refresh(reingest_raw=False):
    """Incremental: only changed raw (if requested or detected) and changed interpreted files are re-read."""
    changed, fp, n = _raw_changed()
    raw = _ingest_raw(fp, n) if (changed or reingest_raw) else {"status": "no change"}
    interp = _ingest_interp()
    return {"raw": raw, "interpreted": interp,
            "note": "new files are available to all tools immediately; call the analysis tools again to reason on them"}


def status():
    return {"raw_ready": I.cache_ready(), "interpreted_ready": D.interpreted_ready(),
            "raw_dir": str(I.DATA_DIR), "interpreted_dir": str(D.interp_dir()) if D.interp_dir() else None}


if __name__ == "__main__":
    run_all(force="--force" in sys.argv)
