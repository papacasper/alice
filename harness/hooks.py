"""Hooks (Claude Code style). settings.json:
  {"hooks": {"PreToolUse": [{"matcher": "Bash|Edit", "hooks": [{"type": "command", "command": "./check.sh"}]}]}}
Events: PreToolUse, PostToolUse, UserPromptSubmit, Stop, SessionStart. The command gets the event as JSON on stdin.
Exit 0 = ok (stdout is passed back as extra context), exit 2 = block (stderr goes to the model), other = warn only."""
import json, os, re, subprocess

class Hooks:
    def __init__(self, config: dict | None, cwd: str, session_id: str = ""):
        self.config, self.cwd, self.session_id = config or {}, cwd, session_id

    def has(self, event: str) -> bool:
        return bool(self.config.get(event))

    def run(self, event: str, tool_name: str = "", payload: dict | None = None) -> tuple[bool, str]:
        """-> (blocked, text). text is the blocking reason or the concatenated stdout of the hooks."""
        outs = []
        data = json.dumps({"hook_event_name": event, "session_id": self.session_id, "cwd": self.cwd,
                           "tool_name": tool_name, **(payload or {})}, default=str)
        for entry in self.config.get(event, []):
            m = entry.get("matcher", "*")
            if event in ("PreToolUse", "PostToolUse") and m not in ("", "*") and not re.fullmatch(m, tool_name):
                continue
            for h in entry.get("hooks", []):
                if h.get("type", "command") != "command": continue
                try:
                    r = subprocess.run(h["command"], shell=True, input=data, capture_output=True, text=True,
                                       timeout=h.get("timeout", 30), cwd=self.cwd,
                                       env={**os.environ, "ALICE_PROJECT_DIR": self.cwd})
                except subprocess.TimeoutExpired:
                    outs.append(f"[hook timed out: {h['command']}]"); continue
                if r.returncode == 2:
                    return True, (r.stderr.strip() or r.stdout.strip() or "blocked by hook")
                if r.stdout.strip(): outs.append(r.stdout.strip())
        return False, "\n".join(outs)
