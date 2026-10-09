"""Model picker (/model) and per-model settings.

settings.json:
  {"models": {"qwen3.5": {"num_ctx": 32768, "think": "auto", "description": "default, fast"},
              "hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M": {"num_ctx": 16384, "think": "off", "sampling": {"temperature": 0.7}}}}
A key applies to every model whose name contains it (case-insensitive); when several match, the longer key wins per setting.
Precedence for each setting: command-line flag > per-model > top-level setting > built-in default.
Keys: num_ctx, think (on|off|auto), sampling ({...}), llamaServerArgs, description. `/model set <key> <value>` saves one."""
import json, os, shlex, sys
from . import profiles, ui
from .fsutil import read_json, write_json

KEYS = ("num_ctx", "think", "sampling", "llamaServerArgs", "description")
USER_SETTINGS = "~/.alice/settings.json"

def model_conf(settings: dict, model: str) -> dict:
    """The per-model settings that apply to `model` (merged; the most specific key wins)."""
    table = settings.get("models")
    if not isinstance(table, dict): return {}
    out: dict = {}
    for k in sorted((k for k in table if isinstance(table[k], dict) and k.lower() in model.lower()), key=len):
        for kk, v in table[k].items():
            out[kk] = {**out[kk], **v} if isinstance(v, dict) and isinstance(out.get(kk), dict) else v
    return out

def effective(settings: dict, model: str, a=None) -> dict:
    """Context size, thinking mode, sampling and extra llama-server flags for `model`. `a`: parsed CLI args (flags win)."""
    mc = model_conf(settings, model)
    pick = lambda flag, key: (getattr(a, flag, None) if a is not None else None) or mc.get(key) or settings.get(key)
    return {"num_ctx": int(pick("num_ctx", "num_ctx") or profiles.DEFAULT_CTX),
            "think": str(pick("think", "think") or "auto"),
            "sampling": {**(settings.get("sampling") or {}), **(mc.get("sampling") or {}), **((getattr(a, "sampling", None) if a is not None else None) or {})},
            "llamaServerArgs": str(pick("llama_server_args", "llamaServerArgs") or ""),
            "description": str(mc.get("description", ""))}

def _loadable(name: str) -> bool:
    """A settings key that names an actual model (not a pattern like "qwen")."""
    return name.endswith(".gguf") or "/" in name or ":" in name

def size_of(model: str) -> str:
    from .llm import gguf_args
    try:
        args = gguf_args(model)
        return f"{os.path.getsize(args[1]) / 2**30:.1f} GB" if args[0] == "-m" else "download"
    except OSError: return ""

def choices(settings: dict, current: str, installed: list[str]) -> list[str]:
    """Current model, then models named in settings, then installed ones (no duplicates)."""
    out = [current]
    for n in [k for k in (settings.get("models") or {}) if _loadable(k)] + installed:
        if n not in out: out.append(n)
    return out

def describe(settings: dict, name: str) -> str:
    e = effective(settings, name)
    bits = [b for b in (size_of(name), f"{e['num_ctx'] // 1024}k ctx", f"think {e['think']}", e["description"]) if b]
    return " · ".join(bits)

def _tty() -> bool: return sys.stdin.isatty() and sys.stderr.isatty()

def pick(names: list[str], current: str, settings: dict) -> str | None:
    """Arrow-key list (Claude Code's /model): ↑/↓ or j/k move, 1-9 jump, Enter selects, Esc cancels. None = cancelled / no tty."""
    if not (_tty() and names): return None
    i, rows = (names.index(current) if current in names else 0), len(names)
    w = lambda s: print(s, file=sys.stderr, flush=True)
    def draw(first=False):
        if not first: sys.stderr.write(f"\x1b[{rows}F")
        for j, n in enumerate(names):
            mark = ui.acc("❯") if j == i else " "
            label = ui.bold(n) if j == i else n
            w(f"\x1b[2K {mark} {j + 1 if j < 9 else ' '}. {label}{' ✔' if n == current else ''}  {ui.dim(describe(settings, n))}")
    w(f" {ui.bold('Select model')}  " + ui.dim("↑/↓ to move · Enter to switch · Esc to cancel")); draw(True)
    with ui.ESC.paused():
        while True:
            try: k = ui.getkey()
            except (EOFError, KeyboardInterrupt): return None
            if k in ("\n", "\r"): return names[i]
            if k == "": return None
            if k in ("UP", "k"): i = (i - 1) % rows
            elif k in ("DOWN", "j"): i = (i + 1) % rows
            elif k.isdigit() and 0 < int(k) <= min(rows, 9): i = int(k) - 1; draw(); return names[i]
            else: continue
            draw()

def apply(app, name: str, reload: bool = False) -> str:
    """Switch `app` to model `name` with its per-model settings. Reloads the private llama-server when the model, context size or
    extra flags change. Returns "" on success or an error message (the old model is kept)."""
    from .llm import LLMError
    llm, e = app.llm, effective(app.settings, name, app.a)
    srv = getattr(llm, "server", None)
    extra = shlex.split(e["llamaServerArgs"])
    if srv and (reload or name != srv.model or e["num_ctx"] != srv.num_ctx or extra != srv.extra):
        try: srv.switch_model(name, say=lambda m: None, num_ctx=e["num_ctx"], extra=extra)
        except LLMError as err: return str(err).splitlines()[-1][:160]
    llm.model, llm.num_ctx = name, e["num_ctx"]
    if hasattr(llm, "think_mode"): llm.think_mode = e["think"]
    else: llm.think = e["think"] == "on"
    if hasattr(llm, "sampling"): llm.sampling = e["sampling"]
    app.refresh_system()
    return ""

def parse_value(key: str, raw: str):
    if key == "num_ctx":
        v = int(raw.lower().removesuffix("k")) * (1024 if raw.lower().endswith("k") else 1)
        if v < 2048: raise ValueError("num_ctx must be at least 2048")
        return v
    if key == "think":
        if raw not in ("on", "off", "auto"): raise ValueError("think is on, off or auto")
        return raw
    if key == "sampling":
        v = json.loads(raw)
        if not isinstance(v, dict): raise ValueError('sampling is a JSON object, e.g. {"temperature": 0.7}')
        return v
    return raw

def save(settings: dict, model: str, key: str, value, path: str = USER_SETTINGS) -> str:
    """Set (value is not None) or remove models[model][key] in the user settings file and in `settings`. Returns the file path."""
    f = os.path.expanduser(path)
    disk = read_json(f) if os.path.exists(f) else {}
    for d in (disk, settings):
        entry = d.setdefault("models", {}).setdefault(model, {})
        if value is None: entry.pop(key, None)
        else: entry[key] = value
        if not entry: d["models"].pop(model)
    os.makedirs(os.path.dirname(f), exist_ok=True); write_json(f, disk)
    return f
