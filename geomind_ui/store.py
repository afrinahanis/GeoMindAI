"""UI persistence: settings (data paths, ingestion behaviour, LLM endpoint) and chat history. No analysis logic here."""
import json
import os
import time
import uuid
from pathlib import Path

HOME = Path(os.environ.get("GEOMIND_HOME", Path(__file__).resolve().parent.parent / ".geomind_ui"))
HOME.mkdir(parents=True, exist_ok=True)
SETTINGS = HOME / "settings.json"
CHATS = HOME / "chats.json"

DEFAULTS = {
    "raw_dir": "",            # folder with the raw data (SEG-Y stacks / inversions, LAS, deviations, checkshots)
    "interp_dir": "",         # folder with the interpreted data (Fault SHP/CSV, Polygon SHP, Horizon)
    "cache_dir": "",          # where ingested caches are stored
    "auto_ingest_interpreted": True,    # interpreted files are light: ingest incrementally right after upload
    "auto_ingest_raw": False,           # raw volumes are heavy: ask before ingesting
    "scan_on_open": True,               # look for new/changed interpreted files when the app opens
    "llm_base_url": "",
    "llm_model": "",
}


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS.read_text()))
    except Exception:
        pass
    # first run: take whatever the environment (or notebook settings) already defines
    s["raw_dir"] = s["raw_dir"] or os.environ.get("DATA_DIR", "")
    s["interp_dir"] = s["interp_dir"] or os.environ.get("INTERP_DIR", "")
    s["cache_dir"] = s["cache_dir"] or os.environ.get("CACHE_DIR", "")
    s["llm_base_url"] = s["llm_base_url"] or os.environ.get("LLM_BASE_URL", "")
    s["llm_model"] = s["llm_model"] or os.environ.get("LLM_MODEL", "")
    return s


def save_settings(s):
    SETTINGS.write_text(json.dumps(s, indent=1))


def apply_settings(s):
    """Point the existing modules at the configured folders. Must run before the data modules are used."""
    if s.get("raw_dir"):
        os.environ["DATA_DIR"] = s["raw_dir"]
    if s.get("interp_dir"):
        os.environ["INTERP_DIR"] = s["interp_dir"]
    if s.get("cache_dir"):
        os.environ["CACHE_DIR"] = s["cache_dir"]
    if s.get("llm_base_url"):
        os.environ["LLM_BASE_URL"] = s["llm_base_url"]
    if s.get("llm_model"):
        os.environ["LLM_MODEL"] = s["llm_model"]
    import ingest
    import interp_data
    if s.get("raw_dir"):
        ingest.DATA_DIR = Path(s["raw_dir"])
    if s.get("cache_dir"):
        ingest.CACHE_DIR = Path(s["cache_dir"])
    try:
        ingest.clear_caches()
        interp_data.clear_caches()
    except Exception:
        pass
    try:
        import agent
        import openai
        if s.get("llm_base_url"):
            agent.BASE_URL = s["llm_base_url"]
            agent.client = openai.OpenAI(base_url=s["llm_base_url"], api_key="token-not-needed")
        if s.get("llm_model"):
            agent.MODEL = s["llm_model"]
    except Exception:
        pass


# ------------------------------------------------------------------ chats
def load_chats():
    try:
        return json.loads(CHATS.read_text())
    except Exception:
        return {}


def save_chats(chats):
    slim = {}
    for cid, c in chats.items():
        # Retain empty conversations created with New chat.
        msgs = []
        for m in c["msgs"]:
            m = dict(m)
            if m.get("trace"):
                m["trace"] = [{"round": t["round"], "tool": t["tool"], "args": t["args"],
                               "result": (t["result"] if len(json.dumps(t["result"], default=str)) < 4000 else {"note": "result omitted (large)"})}
                              for t in m["trace"]]
            msgs.append(m)
        slim[cid] = {**c, "msgs": msgs}
    try:
        CHATS.write_text(json.dumps(slim, default=str))
    except Exception:
        pass


def new_chat(chats):
    cid = uuid.uuid4().hex[:8]
    chats[cid] = {"title": "New chat", "created": time.time(), "msgs": []}
    return cid
