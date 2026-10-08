"""Everything that goes into the model's context besides the chat: system prompt, CLAUDE.md memory,
environment block, skills, custom slash commands, and layered settings."""
import datetime, json, os, platform, re, subprocess

HOME = os.path.expanduser("~")
USER_DIRS = [os.path.join(HOME, ".claude"), os.path.join(HOME, ".alice")]   # later wins
MEMORY_CAP, MEMORY_TOTAL = 3000, 6000

def _load_prompt() -> str:
    """Global system prompt: ~/.alice/SYSTEM.md replaces the shipped harness/system_prompt.md when it exists."""
    for f in (os.path.join(HOME, ".alice", "SYSTEM.md"), os.path.join(os.path.dirname(__file__), "system_prompt.md")):
        try:
            t = open(f).read().strip()
            if t: return t
        except OSError: pass
    return "You are Alice, a concise CLI agent. Use your tools; never guess."

PROMPT = _load_prompt()

def read(p: str, cap: int | None = None) -> str:
    try:
        with open(p, errors="replace") as f: t = f.read()
    except OSError: return ""
    return t[:cap] + "\n[...truncated]" if cap and len(t) > cap else t

def frontmatter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m: return {}, text
    meta, lines = {}, m.group(1).splitlines()
    for i, l in enumerate(lines):
        k, sep, v = l.partition(":")
        if not sep or l.startswith((" ", "\t")): continue
        v = v.strip().strip("\"'")
        if v in ("", "|", ">", "|-", ">-") and i + 1 < len(lines): v = lines[i + 1].strip()
        meta[k.strip()] = v
    return meta, m.group(2)

# ---- settings (user < project < project-local; lists concatenate, dicts merge, scalars override)
def load_settings(cwd: str) -> dict:
    out: dict = {}
    files = [os.path.join(d, "settings.json") for d in (os.path.join(HOME, ".alice"),)]
    files += [os.path.join(cwd, ".alice", n) for n in ("settings.json", "settings.local.json")]
    for f in files:
        try: s = json.loads(read(f) or "{}")
        except ValueError: continue
        for k, v in s.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                for kk, vv in v.items():
                    out[k][kk] = out[k].get(kk, []) + vv if isinstance(vv, list) and isinstance(out[k].get(kk), list) else vv
            else: out[k] = v
    return out

# ---- memory (AGENTS.md / CLAUDE.md / ALICE.md): the global ~/.alice dir and the project dir only, no walking up
MEMORY_NAMES = ("AGENTS.md", "CLAUDE.md", "ALICE.md")

def memory_files(cwd: str) -> list[str]:
    dirs = [os.path.join(HOME, ".alice"), os.path.abspath(cwd)]
    out = []
    for d in dirs:
        for n in MEMORY_NAMES:
            f = os.path.join(d, n)
            if os.path.isfile(f) and f not in out: out.append(f)
    return out

def memory_target(cwd: str) -> str:
    """File that `# note` appends to: the project's existing memory file (ALICE.md first), else a new ALICE.md."""
    for n in ("ALICE.md", "AGENTS.md", "CLAUDE.md"):
        if os.path.isfile(os.path.join(cwd, n)): return os.path.join(cwd, n)
    return os.path.join(cwd, "ALICE.md")

def memory_text(cwd: str) -> str:
    parts, total = [], 0
    for f in reversed(memory_files(cwd)):          # nearest first, so the project's own file survives the total cap
        t = read(f, MEMORY_CAP)
        if total + len(t) > MEMORY_TOTAL: break
        total += len(t); parts.append(f"## {f}\n{t.strip()}")
    return "\n\n".join(reversed(parts))

def env_block(cwd: str, model: str) -> str:
    def git(*a):
        try: return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError): return ""
    is_git = git("rev-parse", "--is-inside-work-tree") == "true"
    lines = [f"Working directory: {cwd}", f"Is a git repository: {'yes' if is_git else 'no'}"]
    if is_git: lines.append(f"Git branch: {git('rev-parse', '--abbrev-ref', 'HEAD')}")
    lines += [f"Platform: {platform.system().lower()} ({platform.release()})",
              f"Today's date: {datetime.date.today().isoformat()} ({datetime.date.today():%A})", f"Model: {model}"]
    return "\n".join(lines)

# ---- skills and custom slash commands
def plugin_dirs(cwd: str) -> list[str]:
    """Plugins: ~/.alice/plugins/<name>/ and <project>/.alice/plugins/<name>/, each may hold skills/, commands/, agents/."""
    out = []
    for base in (os.path.join(HOME, ".alice", "plugins"), os.path.join(cwd, ".alice", "plugins")):
        if os.path.isdir(base): out += [os.path.join(base, n) for n in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, n))]
    return out

def _roots(cwd: str, kind: str) -> list[str]:
    return ([os.path.join(d, kind) for d in USER_DIRS] + [os.path.join(d, kind) for d in plugin_dirs(cwd)]
            + [os.path.join(cwd, d, kind) for d in (".claude", ".alice")])   # later wins

def discover_skills(cwd: str) -> dict:
    roots = _roots(cwd, "skills")
    out = {}
    for r in roots:
        try: names = sorted(os.listdir(r))
        except OSError: continue
        for n in names:
            p = os.path.join(r, n, "SKILL.md")
            if n == "synced" or not os.path.isfile(p): continue
            meta, _ = frontmatter(read(p, 2000))
            out[meta.get("name", n)] = (meta.get("description", "")[:100], p)
    return out

def discover_commands(cwd: str) -> dict:
    roots = _roots(cwd, "commands")
    out = {}
    for r in roots:
        try: names = sorted(os.listdir(r))
        except OSError: continue
        for n in names:
            if not n.endswith(".md"): continue
            meta, body = frontmatter(read(os.path.join(r, n)))
            out[n[:-3]] = (meta.get("description", body.strip().split("\n")[0][:80]), body.strip())
    return out

def discover_agents(cwd: str) -> dict:
    """Custom subagents: agents/*.md with frontmatter name/description/tools; the body is the subagent's system prompt."""
    roots = _roots(cwd, "agents")
    out = {}
    for r in roots:
        try: names = sorted(os.listdir(r))
        except OSError: continue
        for n in names:
            if not n.endswith(".md"): continue
            meta, body = frontmatter(read(os.path.join(r, n)))
            tools = [t.strip() for t in meta.get("tools", "").split(",") if t.strip()]
            out[meta.get("name", n[:-3])] = {"description": meta.get("description", "")[:100], "tools": tools, "prompt": body.strip()}
    return out

def build_system(cwd: str, model: str, skills: dict, use_memory: bool = True, extra: str = "", override: str = "", agents: dict | None = None) -> str:
    parts = [override or PROMPT, "# Environment\n" + env_block(cwd, model)]
    if use_memory and (m := memory_text(cwd)):
        parts.append("# Project and user instructions (follow them)\n" + m)
    if skills:
        parts.append("# Skills (call the Skill tool with the name to load one when it fits the task)\n" +
                     "\n".join(f"- {n}: {d}" for n, (d, _) in sorted(skills.items())))
    if agents:
        parts.append("# Custom subagents (use with Task, subagent_type=<name>)\n" +
                     "\n".join(f"- {n}: {a['description']}" for n, a in sorted(agents.items())))
    if extra: parts.append(extra)
    return "\n\n".join(parts)


STYLES = {"default": "",
          "concise": "Output style: concise. Answer in as few words as will do; no preamble, no recap of what you did unless asked.",
          "explanatory": "Output style: explanatory. After doing the work, add a short note on why you chose that approach and any trade-offs the user should know."}

def output_styles() -> list[str]:
    d = os.path.join(HOME, ".alice", "output-styles")
    custom = sorted(f[:-3] for f in os.listdir(d) if f.endswith(".md")) if os.path.isdir(d) else []
    return list(STYLES) + [n for n in custom if n not in STYLES]

def output_style(name: str) -> str:
    if name in STYLES: return STYLES[name]
    try: return open(os.path.join(HOME, ".alice", "output-styles", name + ".md")).read().strip()[:2000]
    except OSError: return ""
