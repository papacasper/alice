"""Hooks (Claude Code style). settings.json:
  {"hooks": {"PreToolUse": [{"matcher": "Bash|Edit", "hooks": [{"type": "command", "command": "./check.sh"}]}]}}
Events and what "matcher" matches (a regex; "" or "*" = all):
  PreToolUse, PostToolUse, PermissionRequest  tool name
  Notification     notification_type (permission_prompt)
  PreCompact       trigger (manual | auto)
  SessionStart     source (startup | resume | clear)
  SessionEnd       reason (clear | prompt_input_exit | other)
  UserPromptSubmit, Stop, SubagentStop        (no matcher)
The command gets the event as JSON on stdin. Exit 0 = ok (stdout is passed back as extra context), exit 2 = block (stderr goes to
the model), other = warn only. Stdout may instead be JSON, as in Claude Code: {"decision": "block", "reason": ...},
{"hookSpecificOutput": {"permissionDecision": "allow|deny", ...}} (PreToolUse), {"hookSpecificOutput": {"decision": {"behavior":
"allow|deny", "message": ...}}} (PermissionRequest), and "additionalContext". PreCompact, Notification and SessionEnd cannot block."""
import json, os, re, subprocess

EVENTS = ("PreToolUse", "PostToolUse", "PermissionRequest", "Notification", "UserPromptSubmit", "Stop", "SubagentStop",
          "PreCompact", "SessionStart", "SessionEnd")
TOOL_EVENTS = ("PreToolUse", "PostToolUse", "PermissionRequest")
CANNOT_BLOCK = ("PreCompact", "Notification", "SessionEnd", "SessionStart")

def _parse_json(out: str):
    """Claude Code's structured hook output -> (decision 'allow'|'deny'|'block'|None, reason, extra context)."""
    try: d = json.loads(out)
    except ValueError: return None
    if not isinstance(d, dict): return None
    hso = d.get("hookSpecificOutput") or {}
    decision, reason = d.get("decision"), d.get("reason") or d.get("stopReason") or ""
    if hso.get("permissionDecision"): decision, reason = hso["permissionDecision"], hso.get("permissionDecisionReason", reason)
    if isinstance(hso.get("decision"), dict): decision, reason = hso["decision"].get("behavior"), hso["decision"].get("message", reason)
    if d.get("continue") is False: decision = "block"
    ctx = hso.get("additionalContext") or d.get("systemMessage") or ""
    return decision, reason, ctx

class Hooks:
    def __init__(self, config: dict | None, cwd: str, session_id: str = ""):
        self.config, self.cwd, self.session_id = config or {}, cwd, session_id
        self.decision: str | None = None     # 'allow' / 'deny' from the last run's JSON output (PermissionRequest, PreToolUse)

    def has(self, event: str) -> bool:
        return bool(self.config.get(event))

    def run(self, event: str, match: str = "", payload: dict | None = None) -> tuple[bool, str]:
        """-> (blocked, text). text is the blocking reason or the concatenated output of the hooks. `match` is what the
        entry's matcher is tested against (the tool name for tool events)."""
        outs, self.decision = [], None
        data = json.dumps({"hook_event_name": event, "session_id": self.session_id, "cwd": self.cwd,
                           **({"tool_name": match} if event in TOOL_EVENTS else {}), **(payload or {})}, default=str)
        for entry in self.config.get(event, []):
            m = entry.get("matcher", "*")
            if m not in ("", "*") and not re.fullmatch(m, match or ""): continue
            for h in entry.get("hooks", []):
                if h.get("type", "command") != "command": continue
                try:
                    r = subprocess.run(h["command"], shell=True, input=data, capture_output=True, text=True,
                                       timeout=h.get("timeout", 30), cwd=self.cwd,
                                       env={**os.environ, "ALICE_PROJECT_DIR": self.cwd, "CLAUDE_PROJECT_DIR": self.cwd})
                except subprocess.TimeoutExpired:
                    outs.append(f"[hook timed out: {h['command']}]"); continue
                if r.returncode == 2 and event not in CANNOT_BLOCK:
                    self.decision = "deny"
                    return True, (r.stderr.strip() or r.stdout.strip() or "blocked by hook")
                out = r.stdout.strip()
                if (j := _parse_json(out)) is not None:
                    decision, reason, ctx = j
                    if decision in ("allow", "approve"): self.decision = "allow"
                    if decision in ("block", "deny") and event not in CANNOT_BLOCK:
                        self.decision = "deny"; return True, reason or "blocked by hook"
                    out = ctx
                if out: outs.append(out)
        return False, "\n".join(outs)
