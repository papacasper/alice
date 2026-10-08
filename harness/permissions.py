"""Permission modes + allow/deny rules, modelled on Claude Code.

modes: default (ask for edits, shell, network) | acceptEdits | plan (read-only) | bypassPermissions (allow all)
rules: "Bash" | "Bash(git status:*)" | "Edit(/tmp/*)" | "mcp__server"  in settings `permissions.allow` / `.deny`
"""
import fnmatch

MODES = ["default", "acceptEdits", "plan", "bypassPermissions"]
CATEGORY = {"Write": "edit", "Edit": "edit", "Bash": "exec", "WebFetch": "net"}
PRIMARY_ARG = {"Bash": "command", "Write": "file_path", "Edit": "file_path", "Read": "file_path", "WebFetch": "url"}

def category(name: str) -> str:
    if name.startswith("mcp__"): return "exec"
    return CATEGORY.get(name, "read")

def _rule_matches(rule: str, name: str, args: dict) -> bool:
    tool, _, rest = rule.partition("(")
    if tool != name and not (tool.startswith("mcp__") and name.startswith(tool + "__")):
        return False
    if not rest: return True
    pat = rest.rstrip(")")
    val = str(args.get(PRIMARY_ARG.get(name, ""), ""))
    if pat.endswith(":*"): return val.startswith(pat[:-2])   # Bash(git status:*) = command prefix
    return fnmatch.fnmatch(val, pat)

class Permissions:
    def __init__(self, mode="default", allow=None, deny=None, ask=None):
        """ask(name, args) -> 'yes' | 'always' | 'no', or None when there is no terminal."""
        self.mode, self.allow, self.deny, self.ask = mode, list(allow or []), list(deny or []), ask
        self.session_allow: set[str] = set()

    def set_mode(self, mode: str):
        if mode not in MODES: raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def check(self, name: str, args: dict) -> str | None:
        """None = allowed; else the denial text to hand back to the model."""
        if any(_rule_matches(r, name, args) for r in self.deny):
            return f"error: {name} is denied by a permission rule. Do not retry; use another approach."
        cat = category(name)
        if cat == "read": return None
        if self.mode == "plan":
            return ("error: plan mode is active (read-only). Research with Read/Glob/Grep, then call ExitPlanMode "
                    "with your plan so the user can approve it before anything is changed.")
        if self.mode == "bypassPermissions": return None
        if self.mode == "acceptEdits" and cat == "edit": return None
        if name in self.session_allow or any(_rule_matches(r, name, args) for r in self.allow): return None
        answer = self.ask(name, args) if self.ask else None
        if answer is None:
            return (f"error: {name} needs permission and there is no terminal to ask. Re-run with "
                    "--permission-mode acceptEdits or bypassPermissions, or add an allow rule.")
        if answer == "always": self.session_allow.add(name)
        if answer in ("yes", "always"): return None
        return "error: the user denied this tool call. Do not retry it; ask what they want instead."
