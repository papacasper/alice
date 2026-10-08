"""Session persistence: ~/.alice/sessions/<cwd-slug>/<id>.json  (resume with --continue / --resume / /resume)."""
import json, os, re, time

ROOT = os.path.expanduser("~/.alice/sessions")

def _dir(cwd: str) -> str:
    d = os.path.join(ROOT, re.sub(r"[^A-Za-z0-9]+", "-", cwd).strip("-") or "root")
    os.makedirs(d, exist_ok=True)
    return d

def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def save(cwd: str, sid: str, messages: list, meta: dict | None = None):
    with open(os.path.join(_dir(cwd), sid + ".json"), "w") as f:
        json.dump({"id": sid, "cwd": cwd, "saved": time.time(), "meta": meta or {}, "messages": messages}, f, default=str)

def load(cwd: str, sid: str) -> dict | None:
    try:
        with open(os.path.join(_dir(cwd), sid + ".json")) as f: return json.load(f)
    except (OSError, ValueError): return None

def find(cwd: str, ref: str) -> str:
    """Session id for an id or a /rename name (exact, case-insensitive); "" if none."""
    for s in list_sessions(cwd):
        if ref == s["id"] or (s["name"] and s["name"].lower() == ref.lower()): return s["id"]
    return ""

def list_sessions(cwd: str) -> list[dict]:
    """Newest first: {id, preview, n}."""
    d, out = _dir(cwd), []
    for fn in sorted(os.listdir(d), reverse=True):
        if not fn.endswith(".json"): continue
        s = load(cwd, fn[:-5])
        if not s: continue
        first = next((m["content"] for m in s["messages"] if m["role"] == "user"), "")
        out.append({"id": s["id"], "preview": first.replace("\n", " ")[:70], "n": len(s["messages"]), "name": (s.get("meta") or {}).get("name", "")})
    return out
