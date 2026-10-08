"""Claude-Code-style tools: Read, Write, Edit, Bash, Glob, Grep, TodoWrite, WebFetch, Task, Skill,
AskUserQuestion, ExitPlanMode (+ calculate, today). Paths may be anywhere the user can access.

All tools are closures over one `State` (working dir, files-read ledger, todo list, hooks into the CLI),
so a subagent can share or fork it.
"""
from . import shell as shellmod
import difflib, fnmatch, html, json, os, re, shutil, signal, subprocess, urllib.request
from dataclasses import dataclass, field
from .bgtools import bg_tools, start_job
from .builtin import calculate, today
from .tools import Tool, Toolbox
from .fsutil import read_text

MARK = "__ALICE_CWD__"
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".cache"}
READ_TOOLS = ["Read", "Glob", "Grep"]
READ_BUDGET = 12000   # chars per Read page (~3k tokens of a 32k context)

@dataclass
class State:
    cwd: str
    read_files: dict = field(default_factory=dict)   # abs path -> mtime when last read/written
    todos: list = field(default_factory=list)
    skills: dict = field(default_factory=dict)       # name -> (description, SKILL.md path)
    spawn: object = None                             # callable(prompt, subagent_type) -> str   (Task tool)
    ask_user: object = None                          # callable(question) -> str                (AskUserQuestion)
    approve_plan: object = None                      # callable(plan) -> bool                   (ExitPlanMode)
    plan_mode: object = None                         # callable() -> bool
    ui_extra: str = ""                               # last diff, for the renderer to show (first 60 lines)
    ui_counts: tuple = (0, 0)                        # last diff's full (+added, -removed); ui_extra may be cut
    lines: list = field(default_factory=lambda: [0, 0])   # session totals (added, removed): status line cost.total_lines_*
    jobs: dict = field(default_factory=dict)         # background shells: id -> {proc, file, cmd, pos}
    turns: list = field(default_factory=list)        # per-turn undo log: {"n": msg index, "files": [(path, old|None)]}
    agents: dict = field(default_factory=dict)       # custom subagents: name -> {description, tools, prompt}

    def snap(self, p: str):
        """Remember a file's current content (or absence) so /rewind can restore it."""
        if self.turns and not any(p == q for q, _ in self.turns[-1]["files"]):
            self.turns[-1]["files"].append((p, read_text(p, errors="replace") if os.path.isfile(p) else None))

    def path(self, p: str) -> str:
        p = os.path.expanduser(p)
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(self.cwd, p))

def _mtime(p):
    try: return os.path.getmtime(p)
    except OSError: return None

def _is_binary(path: str) -> bool:
    with open(path, "rb") as f:
        return b"\0" in f.read(4096)

def _middle(s: str, limit: int = 5500) -> str:
    if len(s) <= limit: return s
    h = limit // 2
    return s[:h] + f"\n[... {len(s) - limit} chars cut ...]\n" + s[-h:]

def build_tools(st: State) -> list[Tool]:
    def Read(file_path: str, offset: int = 1, limit: int = 200) -> str:
        """Read a text file from disk with numbered lines. Long files are paged: pass offset (first line, 1-based) to continue. Always Read a file before you Edit or overwrite it.

        Args:
            file_path: path of the file (absolute, or relative to the working directory)
            offset: first line to return (1-based)
            limit: how many lines to return
        """
        p = st.path(file_path)
        if os.path.isdir(p): return f"error: {p} is a directory; use Bash ls or Glob"
        if not os.path.exists(p): return f"error: file does not exist: {p}"
        if _is_binary(p): return f"error: {p} looks like a binary file"
        with open(p, errors="replace") as f: lines = f.read().splitlines()
        st.read_files[p] = _mtime(p)
        a = max(offset, 1) - 1
        rows, size = [], 0
        for i, l in enumerate(lines[a:a + max(limit, 1)]):   # page by lines and by size, so the continuation hint is never cut off
            row = f"{a + i + 1:>5}\t{l[:600]}"
            if rows and size + len(row) > READ_BUDGET: break
            rows.append(row); size += len(row) + 1
        out, end = "\n".join(rows), a + len(rows)
        if end < len(lines):
            out += f"\n[showing lines {a + 1}-{end} of {len(lines)}; call Read again with offset={end + 1}]"
        return out or "(file is empty)"

    def Write(file_path: str, content: str) -> str:
        """Create a file or completely overwrite an existing one. To change part of an existing file use Edit instead. An existing file must be Read first.

        Args:
            file_path: path of the file (absolute, or relative to the working directory)
            content: the full new contents
        """
        p = st.path(file_path)
        existed = os.path.exists(p)
        if existed and p not in st.read_files:
            return f"error: {p} already exists and has not been Read in this session. Read it first, then Write."
        if existed and _mtime(p) != st.read_files.get(p):
            return f"error: {p} changed on disk since you read it. Read it again first."
        old = read_text(p, errors="replace") if existed else ""
        st.snap(p)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f: f.write(content)
        st.read_files[p] = _mtime(p)
        st.ui_extra = _diff(st, old, content, p) if existed else ""
        if not existed: st.lines[0] += len(content.splitlines())
        return f"{'Overwrote' if existed else 'Created'} {p} ({len(content.splitlines())} lines)"

    def Edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace an exact string in a file. old_string must match the file text exactly (including whitespace) and be unique unless replace_all is true. The file must be Read first.

        Args:
            file_path: path of the file (absolute, or relative to the working directory)
            old_string: the exact existing text to replace
            new_string: the replacement text (must differ from old_string)
            replace_all: replace every occurrence instead of requiring a unique match
        """
        p = st.path(file_path)
        if not os.path.isfile(p): return f"error: file does not exist: {p}"
        if p not in st.read_files: return f"error: {p} has not been Read in this session. Read it first, then Edit."
        if _mtime(p) != st.read_files[p]: return f"error: {p} changed on disk since you read it. Read it again first."
        if old_string == new_string: return "error: old_string and new_string are identical"
        if not old_string: return "error: old_string is empty; use Write to create or replace a whole file"
        text = read_text(p, errors="replace")
        n = text.count(old_string)
        if n == 0: return "error: old_string not found in the file. Copy it exactly from a fresh Read (watch whitespace/indentation)."
        if n > 1 and not replace_all:
            return f"error: old_string appears {n} times. Add more surrounding lines to make it unique, or set replace_all=true."
        new = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        st.snap(p)
        with open(p, "w") as f: f.write(new)
        st.read_files[p] = _mtime(p)
        st.ui_extra = _diff(st, text, new, p)
        return f"Edited {p}: replaced {n if replace_all else 1} occurrence(s)"

    def Bash(command: str, timeout: int = 120, description: str = "", run_in_background: bool = False) -> str:
        """Run a shell command (bash) and return its output and exit code. The working directory persists between calls. Use Read/Glob/Grep instead of cat/find/grep when possible. For servers and other long-running commands set run_in_background and poll with BashOutput.

        Args:
            command: the command line
            timeout: seconds before the command is killed
            description: short note on what the command does
            run_in_background: start it and return immediately with a shell id
        """
        if run_in_background: return start_job(st, command)
        script = f'{command}\n__ec=$?\nprintf "\\n{MARK}%s\\n" "$PWD"\nexit $__ec'
        p = subprocess.Popen(shellmod.argv(script), cwd=st.cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True)
        try:
            out, err = p.communicate(timeout=max(1, min(int(timeout), 600)))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.communicate()
            return f"error: command timed out after {timeout}s and was killed"
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL); raise
        if MARK in out:
            out, _, newcwd = out.rpartition(MARK)
            out = out.rstrip("\n")
            newcwd = newcwd.strip()
            if os.path.isdir(newcwd): st.cwd = newcwd
        res = _middle(out)
        err = shellmod.clean(err)
        if err.strip(): res += ("\n" if res else "") + "[stderr] " + _middle(err.strip(), 2000)
        if p.returncode: res = f"exit code {p.returncode}\n" + res
        return res or "(no output)"

    def Glob(pattern: str, path: str = "") -> str:
        """Find files by glob pattern (e.g. '**/*.py'), newest first. Use this to locate files by name.

        Args:
            pattern: glob pattern, e.g. 'src/**/*.ts'
            path: directory to search in (default: working directory)
        """
        base = st.path(path) if path else st.cwd
        if not os.path.isdir(base): return f"error: not a directory: {base}"
        hits = []
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), base)
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fn, pattern) or _globstar(rel, pattern):
                    hits.append(os.path.join(dp, fn))
        hits.sort(key=lambda f: -(_mtime(f) or 0))
        if not hits:
            last = pattern.rsplit("/", 1)[-1]   # '**/.tmp' matches only files literally named '.tmp'; the model meant the extension
            if last.startswith(".") and not any(c in last for c in "*?["):
                fix = pattern[: len(pattern) - len(last)] + "*" + last
                if "**" not in fix: fix = "**/" + fix
                if not (alt := Glob(fix, path)).startswith(("No files", "error")):
                    return f"No files named exactly '{last}'. Did you mean '{fix}'? It matches:\n{alt}"
            return "No files found"
        return "\n".join(hits[:100]) + (f"\n[showing 100 of {len(hits)}]" if len(hits) > 100 else "")

    def Grep(pattern: str, path: str = "", glob: str = "", output_mode: str = "files_with_matches",
             ignore_case: bool = False, context: int = 0, head_limit: int = 60) -> str:
        """Search file contents with a regex (ripgrep). output_mode: 'files_with_matches' (default), 'content' (matching lines with line numbers), or 'count'. Use this instead of grep/rg in Bash.

        Args:
            pattern: regular expression
            path: file or directory to search (default: working directory)
            glob: only search files matching this glob, e.g. '*.py'
            output_mode: files_with_matches, content or count
            ignore_case: case-insensitive match
            context: lines of context around each match (content mode)
            head_limit: max output lines
        """
        target = st.path(path) if path else st.cwd
        if not os.path.exists(target): return f"error: path does not exist: {target}"
        if output_mode not in ("files_with_matches", "content", "count"):
            return "error: output_mode must be files_with_matches, content or count"
        if shutil.which("rg"):
            cmd = ["rg", "--no-heading", "--color", "never", "-e", pattern]
            cmd += {"files_with_matches": ["-l"], "count": ["-c"], "content": ["-n"] + (["-C", str(context)] if context else [])}[output_mode]
            if ignore_case: cmd.append("-i")
            if glob: cmd += ["-g", glob]
            cmd.append(target)
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=st.cwd)
            if r.returncode == 2: return "error: " + (r.stderr.strip() or "rg failed")
            lines = r.stdout.splitlines()
        else:
            lines = _py_grep(pattern, target, glob, output_mode, ignore_case)
        if not lines: return "No matches found"
        more = f"\n[showing {head_limit} of {len(lines)} lines]" if len(lines) > head_limit else ""
        return "\n".join(l[:300] for l in lines[:head_limit]) + more

    def TodoWrite(todos: list) -> str:
        """Create or update your task list for a multi-step job. Pass the COMPLETE list each time. Each item: {"content": "...", "status": "pending"|"in_progress"|"completed"}. Keep exactly one item in_progress.

        Args:
            todos: the full list of todo items
        """
        if isinstance(todos, str):
            try: todos = json.loads(todos)
            except ValueError: todos = [todos]
        items = []
        for t in todos if isinstance(todos, list) else []:
            if isinstance(t, str): t = {"content": t}
            if isinstance(t, dict) and t.get("content"):
                s = t.get("status") if t.get("status") in ("pending", "in_progress", "completed") else "pending"
                items.append({"content": str(t["content"]), "status": s})
        st.todos = items
        return "Todos updated. Continue with the current task."

    def WebFetch(url: str, prompt: str = "") -> str:
        """Fetch a web page or text URL and return its readable text (HTML is stripped).

        Args:
            url: the http(s) URL
            prompt: what you are looking for (for your own reference)
        """
        if not re.match(r"https?://", url): return "error: url must start with http:// or https://"
        req = urllib.request.Request(url, headers={"User-Agent": "alice-harness/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace")
            ctype = r.headers.get_content_type()
        if "html" in ctype:
            raw = re.sub(r"(?is)<(script|style|noscript|svg)\b.*?</\1>", " ", raw)
            raw = re.sub(r"(?s)<[^>]+>", " ", raw)
            raw = re.sub(r"[ \t\r\f\v]+", " ", html.unescape(raw))
            raw = re.sub(r"\n\s*\n+", "\n", raw)
        return (f"[fetched {url}" + (f"; looking for: {prompt}" if prompt else "") + "]\n" + raw.strip())[:5800]

    def Task(description: str, prompt: str, subagent_type: str = "general-purpose") -> str:
        """Launch a subagent with a fresh context to do a self-contained job (research a codebase, search many files) and return its final report. subagent_type: 'general-purpose' (all tools) or 'explore' (read-only search). Give it a complete, standalone prompt.

        Args:
            description: 3-5 word summary of the job
            prompt: the full instructions for the subagent
            subagent_type: general-purpose or explore
        """
        if not st.spawn: return "error: subagents are not available here"
        return st.spawn(prompt, subagent_type)

    def Skill(name: str) -> str:
        """Load the full instructions of a named skill (listed in the system prompt) and then follow them.

        Args:
            name: the skill name
        """
        if name not in st.skills: return f"error: unknown skill '{name}'; available: {sorted(st.skills)}"
        path = st.skills[name][1]
        return f"[skill {name} from {os.path.dirname(path)}]\n" + read_text(path, errors="replace")[:5500]

    def AskUserQuestion(question: str) -> str:
        """Ask the user a clarifying question and wait for their typed answer. Use only when you are genuinely blocked on a choice only the user can make.

        Args:
            question: the question to ask
        """
        if not st.ask_user: return "error: no user is available to answer; make a reasonable assumption and say so"
        return "User answered: " + st.ask_user(question)

    def ExitPlanMode(plan: str) -> str:
        """In plan mode: present your finished implementation plan to the user for approval. Only call this when the plan is complete; plan mode ends if they approve.

        Args:
            plan: the plan, as a short numbered list
        """
        if not (st.plan_mode and st.plan_mode()): return "error: not in plan mode"
        if st.approve_plan and st.approve_plan(plan): return "User approved the plan. Plan mode is off; implement it now."
        return "User did not approve the plan. Revise it based on their feedback, or ask what they want."

    return [Tool(calculate), Tool(today), Tool(Read, max_output=READ_BUDGET + 200), *(Tool(f) for f in (Write, Edit, Bash, Glob, Grep, TodoWrite, WebFetch,
            Task, Skill, AskUserQuestion, ExitPlanMode)), *bg_tools(st)]

def cc_toolbox(st: State, only: list[str] | None = None, exclude: list[str] | None = None) -> Toolbox:
    tools = build_tools(st)
    if only is not None: tools = [t for t in tools if t.name in only]
    if exclude: tools = [t for t in tools if t.name not in exclude]
    return Toolbox(tools)

def _globstar(rel: str, pattern: str) -> bool:
    """'**/*.py' also matches top-level files; fnmatch's '*' already crosses '/', so only handle the '**/' prefix."""
    return pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])

def _diff(st: State, old: str, new: str, path: str) -> str:
    """Diff for display (first 60 lines); the full +/- counts go to st.ui_counts and the session totals in st.lines."""
    d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), os.path.basename(path), os.path.basename(path), lineterm="", n=2))
    body = [l for l in d[2:] if not l.startswith("@@")]
    st.ui_counts = (sum(l.startswith("+") for l in body), sum(l.startswith("-") for l in body))
    st.lines[0] += st.ui_counts[0]; st.lines[1] += st.ui_counts[1]
    return "\n".join(d[:60])

def _py_grep(pattern, target, glob, mode, icase):
    rx = re.compile(pattern, re.I if icase else 0)
    files = [target] if os.path.isfile(target) else [os.path.join(dp, f) for dp, dns, fns in os.walk(target)
             if not dns.__setitem__(slice(None), [d for d in dns if d not in SKIP_DIRS]) for f in fns]
    out = []
    for f in files:
        if glob and not fnmatch.fnmatch(os.path.basename(f), glob): continue
        try: lines = read_text(f, errors="strict").splitlines()
        except (UnicodeDecodeError, OSError): continue
        hits = [(i, l) for i, l in enumerate(lines, 1) if rx.search(l)]
        if not hits: continue
        out += [f] if mode == "files_with_matches" else [f"{f}:{len(hits)}"] if mode == "count" else [f"{f}:{i}:{l}" for i, l in hits]
    return out
