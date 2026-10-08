"""Status line under the prompt, like Claude Code's. settings.json:
    {"statusLine": {"type": "command", "command": "~/.alice/statusline.sh", "padding": 0}}
The command gets Claude Code's status JSON on stdin; every stdout line is shown (ANSI colours kept). It runs off the render
path (a background thread), so a slow script never blocks typing. With no command, a built-in line shows model · context bar."""
import json, os, subprocess, threading, time

from . import sessions
from .fsutil import read_json, write_json

REFRESH = 1.0     # seconds between command runs while the prompt is showing (Claude Code: on change, throttled ~300 ms)
TIMEOUT = 5

def payload(app, ctx_used: int, version: str) -> dict:
    """Claude Code's statusLine stdin schema, filled with what alice knows (no cost: local model)."""
    llm, u = app.llm, getattr(app.llm, "usage", {}) or {}
    last = getattr(llm, "last", {}) or {}
    size = llm.num_ctx; pct = min(100, 100 * ctx_used // max(1, size))
    lines = getattr(getattr(app, "state", None), "lines", None) or [0, 0]
    return {
        "hook_event_name": "Status", "agent": "alice", "session_id": app.sid, "version": version,
        "transcript_path": os.path.join(sessions._dir(app.cwd), app.sid + ".json"),
        "cwd": app.cwd, "workspace": {"current_dir": app.cwd, "project_dir": app.cwd},
        "model": {"id": llm.model, "display_name": llm.model.split("/")[-1]},
        "output_style": {"name": getattr(app, "style", "default")}, "permission_mode": app.perms.mode,
        "cost": {"total_cost_usd": 0, "total_duration_ms": int((time.time() - getattr(app, "started", time.time())) * 1000),
                 "total_api_duration_ms": int(u.get("prompt_ms", 0) + u.get("gen_ms", 0)), "total_lines_added": lines[0], "total_lines_removed": lines[1]},
        "context_window": {"context_window_size": size, "used_percentage": pct, "remaining_percentage": 100 - pct,
                           "total_input_tokens": u.get("prompt_tokens", 0), "total_output_tokens": u.get("completion_tokens", 0),
                           "current_usage": {"input_tokens": max(0, last.get("prompt_tokens", 0) - last.get("cached_tokens", 0)),
                                             "output_tokens": last.get("completion_tokens", 0), "cache_creation_input_tokens": 0,
                                             "cache_read_input_tokens": last.get("cached_tokens", 0)} if last else None},
        "exceeds_200k_tokens": ctx_used > 200_000,
    }

def builtin(data: dict, width: int = 16) -> list[str]:
    """Default line: model (ctx) | [bar] pct% | used k / total k (left)."""
    c = data["context_window"]; pct = c["used_percentage"]; total = c["context_window_size"]
    used = pct * total // 100; fill = pct * width // 100
    return [f"{data['model']['display_name']} | [{'█' * fill}{'░' * (width - fill)}] {pct}% | "
            f"{used // 1000}k / {total // 1000}k ({(total - used) // 1000}k left)"]

class StatusLine:
    """Caches the command's output; refresh() starts at most one background run, on_update() repaints when it lands."""
    def __init__(self, on_update=lambda: None):
        self.lines: list[str] = []; self.at = 0.0; self.running = False; self.on_update = on_update; self.error = ""

    def run(self, cmd: str, data: dict) -> list[str]:
        try:
            p = subprocess.run(os.path.expanduser(cmd), shell=True, input=json.dumps(data), capture_output=True, text=True, timeout=TIMEOUT)
            self.error = "" if p.stdout.strip() or not p.returncode else f"statusLine exited {p.returncode}: {p.stderr.strip()[:80]}"
            return [l for l in p.stdout.rstrip("\n").splitlines()][:4]
        except subprocess.TimeoutExpired: self.error = f"statusLine timed out ({TIMEOUT}s)"
        except OSError as e: self.error = f"statusLine: {e}"
        return self.lines

    def refresh(self, cmd: str, data: dict):
        if self.running or time.time() - self.at < REFRESH: return
        self.running = True
        def go():
            try:
                new = self.run(cmd, data)
                changed = new != self.lines; self.lines = new
                if changed: self.on_update()
            finally: self.at = time.time(); self.running = False
        threading.Thread(target=go, daemon=True).start()

CLAUDE_SCRIPT = "~/.claude/statusline.sh"

def configure(settings: dict, arg: str, path: str = "~/.alice/settings.json") -> str:
    """/statusline [on|off|claude|<command>]: show or set settings.statusLine (saved to ~/.alice/settings.json). Returns a message."""
    arg = arg.strip(); conf = settings.get("statusLine")
    if not arg:
        cur = "off" if conf is False else (conf or {}).get("command") if isinstance(conf, dict) and conf.get("command") else "built-in (model | context bar | tokens)"
        return f"status line: {cur}\n  /statusline on | off | claude (reuse {CLAUDE_SCRIPT}) | <shell command reading Claude Code's status JSON on stdin>"
    if arg == "off": new = False
    elif arg in ("on", "default"): new = {}
    else:
        cmd = CLAUDE_SCRIPT if arg == "claude" else arg
        if arg == "claude" and not os.path.exists(os.path.expanduser(cmd)): return f"{cmd} not found"
        new = {"type": "command", "command": cmd, "padding": 0}
    f = os.path.expanduser(path)
    try:
        cur = read_json(f) if os.path.exists(f) else {}
        if new == {}: cur.pop("statusLine", None)
        else: cur["statusLine"] = new
        os.makedirs(os.path.dirname(f), exist_ok=True); write_json(f, cur)
    except (OSError, ValueError) as e: return f"could not save: {e}"
    if new == {}: settings.pop("statusLine", None)
    else: settings["statusLine"] = new
    return "status line: " + ("off" if new is False else new.get("command", "built-in") if new else "built-in")
