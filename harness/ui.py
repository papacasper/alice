"""Terminal rendering in the style of Claude Code: ● tool calls, ⎿ results, colored diffs, markdown answers,
a boxed permission prompt, Esc-to-interrupt and a welcome banner. rich is used when present; everything degrades to plain text."""
import contextlib, difflib, os, random, signal, sys, threading, time

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    CON = Console(highlight=False)
except ImportError:   # pragma: no cover
    CON = None

TTY = sys.stdout.isatty()
THEMES = {"terracotta": (173, "#d7875f"), "blue": (75, "#5fafff"), "green": (71, "#5faf5f"), "purple": (141, "#af87ff"),
          "amber": (214, "#ffaf00"), "mono": (250, "#bcbcbc")}
THEME = {"name": "terracotta"}
def accent_n() -> int: return THEMES[THEME["name"]][0]
def accent_hex() -> str: return THEMES[THEME["name"]][1]
def set_theme(name: str) -> bool:
    if name not in THEMES: return False
    THEME["name"] = name; return True
def c(code: str, s: str) -> str: return f"\033[{code}m{s}\033[0m" if TTY else s
def acc(s: str) -> str: return c(f"38;5;{accent_n()}", s)
bold, dim, red, green, yellow, cyan, blue = (lambda s, k=k: c(k, s) for k in ("1", "2", "31", "32", "33", "36", "34"))
CLEAR = "\r\033[K" if TTY else ""
VERBS = ["Thinking", "Pondering", "Cogitating", "Mulling", "Ruminating", "Reasoning", "Noodling", "Synthesizing", "Considering"]
TIPS = ["Start a message with ! to run a shell command, # to save a note to ALICE.md, @ to mention a file",
        "Shift+Tab cycles permission modes; /plan makes Alice read-only until you approve a plan",
        "Esc interrupts a running turn; Esc twice at an empty prompt rewinds the last turn (/rewind)",
        "Ctrl+O shows full tool output (/transcript); Ctrl+R searches prompt history; Ctrl+G edits in $EDITOR",
        "End a line with \\ (or press Alt+Enter) for a new line; /vim switches to vim keys",
        "/context shows how much of the context window is used; /compact frees space"]

def short_args(name: str, a: dict) -> str:
    key = {"Bash": "command", "Read": "file_path", "Write": "file_path", "Edit": "file_path", "Glob": "pattern",
           "Grep": "pattern", "WebFetch": "url", "WebSearch": "query", "Task": "description", "Skill": "name",
           "NotebookEdit": "notebook_path", "BashOutput": "bash_id", "KillShell": "shell_id"}.get(name)
    v = a.get(key) if key else None
    if v is None: v = ", ".join(f"{k}={str(x)[:40]!r}" for k, x in list(a.items())[:2])
    v = str(v).replace("\n", " ")
    return v if len(v) <= 90 else v[:87] + "..."

def render_diff(diff: str, indent: str) -> str:
    out = []
    for l in diff.splitlines():
        if l.startswith(("---", "+++")): continue
        out.append(indent + (green(l) if l.startswith("+") else red(l) if l.startswith("-") else dim(l)))
    return "\n".join(out)

def diff_counts(diff: str) -> tuple[int, int]:
    ls = [l for l in diff.splitlines() if not l.startswith(("---", "+++"))]
    return sum(l.startswith("+") for l in ls), sum(l.startswith("-") for l in ls)

def render_todos(todos: list, indent: str = "  ") -> str:
    icon = {"pending": "☐", "in_progress": "◐", "completed": "☒"}
    rows = []
    for t in todos:
        line = f"{indent}{icon[t['status']]} {t['content']}"
        rows.append(dim(line) if t["status"] == "completed" else bold(line) if t["status"] == "in_progress" else line)
    return "\n".join(rows)

TOOL_VERBS = {"Bash": "Running", "Grep": "Searching", "WebFetch": "Fetching", "WebSearch": "Searching the web", "Task": "Delegating"}

def elapsed(secs: float) -> str: return f"{int(secs)}s" if secs < 60 else f"{int(secs // 60)}m {int(secs % 60)}s"

def turn_summary(secs: float, tools: int, tokens: int) -> str:
    """'✻ Worked for 1m 12s · 3 tool calls · 1.2k tokens' — shown after turns long enough to matter."""
    t = elapsed(secs)
    tk = f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
    return f"✻ Worked for {t} · {plural(tools, 'tool call')} · {tk} tokens"

def plural(n: int, word: str) -> str: return f"{n} {word}" + ("" if n == 1 else "s")

def summarize(name: str, res: str) -> tuple[list[str], int]:
    """(lines to show under ⎿, how many more lines are hidden) -- Claude Code shows counts, not dumps."""
    lines = [l for l in res.splitlines() if l.strip()]
    if res.startswith("error"): return lines[:3] or ["error"], max(0, len(lines) - 3)
    if name == "Read": return [f"Read {plural(len(lines), 'line')}"], 0
    if name in ("Glob", "Grep"):
        none = not lines or lines[0].lower().startswith(("no ", "0 "))
        return [f"Found {plural(0 if none else len(lines), 'line' if name == 'Grep' else 'file')}"], 0
    if name == "WebSearch": return [f"Did 1 search · {plural(res.count(chr(10) + '   http'), 'result')}"], 0
    if name == "Bash": return (lines[:3] or ["(no output)"]), max(0, len(lines) - 3)
    return (lines[:1] or ["(no output)"]), max(0, len(lines) - 1)

# ---- Esc to interrupt -------------------------------------------------------------------------------------
class EscWatcher:
    """While a turn runs, a lone Esc keypress raises KeyboardInterrupt in the main thread (like Claude Code)."""
    def __init__(self):
        self._fd = None; self._old = None; self._stop = threading.Event(); self._t = None; self._off = threading.Event()
        self.buf = ""; self.queue: list[str] = []

    def __enter__(self):
        try:
            import termios, tty
            if not sys.stdin.isatty(): return self
            self._fd = sys.stdin.fileno(); self._old = termios.tcgetattr(self._fd); tty.setcbreak(self._fd)
        except Exception: self._fd = None; return self
        self._stop.clear(); self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self

    def _run(self):
        import select
        while not self._stop.is_set():
            if self._off.is_set(): time.sleep(0.05); continue
            if select.select([self._fd], [], [], 0.1)[0] and not self._off.is_set():
                data = os.read(self._fd, 256)
                if data == b"\x1b": os.kill(os.getpid(), signal.SIGINT)
                elif not data.startswith(b"\x1b"): self.feed(data.decode(errors="ignore"))   # typed ahead: queue it for after this turn

    def feed(self, text: str):
        """Type-ahead while a turn runs: Enter queues the line, Backspace edits, escape sequences are ignored."""
        for ch in text:
            if ch in "\r\n":
                if self.buf.strip(): self.queue.append(self.buf.strip())
                self.buf = ""
            elif ch in "\x7f\x08": self.buf = self.buf[:-1]
            elif ch == "\x15": self.buf = ""
            elif ch.isprintable(): self.buf += ch

    def status(self) -> str:
        bits = ([f"{len(self.queue)} queued"] if self.queue else []) + ([f"› {self.buf[-40:]}"] if self.buf else [])
        return " · " + " · ".join(bits) if bits else ""

    def __exit__(self, *a):
        if self._fd is None: return
        import termios
        self._stop.set(); self._t.join(1); self.buf = ""; termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old); self._fd = None

    @contextlib.contextmanager
    def paused(self):
        """Give the terminal back to cooked mode (for input()/prompts) while the watcher is live."""
        if self._fd is None: yield; return
        import termios, tty
        self._off.set(); time.sleep(0.12); termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        try: yield
        finally: tty.setcbreak(self._fd); self._off.clear()

ESC = EscWatcher()

def getkey() -> str:
    """One keypress without Enter ('' for a lone Esc); falls back to a line when there is no tty."""
    try:
        import termios, tty
        fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    except Exception: return (input() or " ")[:1]
    try:
        tty.setcbreak(fd); ch = os.read(fd, 1).decode(errors="ignore")
        if ch == "\x1b":
            import select
            if select.select([fd], [], [], 0.05)[0]: os.read(fd, 8); return "?"   # arrow keys etc.: ignore
            return ""
        if ch == "\x03": raise KeyboardInterrupt
        return ch
    finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ---- streaming renderer -----------------------------------------------------------------------------------
def answer_view(buf: str):
    t = Table.grid(padding=(0, 1)); t.add_column(width=1, style="bold"); t.add_column()
    t.add_row("●", Markdown(buf)); return t

class Renderer:
    """on_event handler + token sink. `state` supplies todos/ui_extra; `indent` marks subagent output."""
    def __init__(self, state, quiet: bool = False, indent: str = "", stream: bool = True):
        self.state, self.quiet, self.indent, self.stream = state, quiet, indent, stream
        self.show_answer = True   # one-shot mode prints the final answer itself (text or JSON), so it turns this off
        self._open = False; self._live = None; self.buf = ""
        self._stop = threading.Event(); self._thread = None
        self.log: list[tuple] = []     # (name, args, full result) for /transcript
        self.md = bool(CON and TTY and not indent)

    @property
    def _spin(self): return bool(self._thread and self._thread.is_alive())

    def _spin_start(self, verb: str = ""):
        self._stop = threading.Event(); t0 = time.time(); verb = verb or random.choice(VERBS)
        def run():
            frames = "✻✽✶✳"; i = 0
            while not self._stop.wait(0.25):
                print(f"\r{self.indent}{acc(frames[i % 4])} {dim(f'{verb}… ({elapsed(time.time() - t0)} · esc to interrupt){ESC.status()}')}\033[K",
                      end="", file=sys.stderr, flush=True); i += 1
        self._thread = threading.Thread(target=run, daemon=True); self._thread.start()

    def _spin_stop(self):
        if self._spin:
            self._stop.set(); self._thread.join(); print(CLEAR, end="", file=sys.stderr, flush=True)

    def token(self, t: str):
        self._spin_stop()
        if self.md:
            self.buf += t
            if not self._live:
                self._live = Live(answer_view(self.buf), console=CON, refresh_per_second=10, vertical_overflow="visible"); self._live.start()
            self._live.update(answer_view(self.buf)); return
        if not self._open and self.indent: print(self.indent, end="", flush=True)
        self._open = True
        print(t.replace("\n", "\n" + self.indent), end="", flush=True)

    def _end_text(self):
        if self._live: self._live.stop(); self._live = None; self.buf = ""
        if self._open: print(); self._open = False

    def event(self, e: dict):
        k = e["type"]
        if k == "llm_start" and not self.quiet and TTY:
            self._spin_start()
        elif k == "tool_batch":
            self._spin_stop(); self._end_text()
            if not self.quiet: print(self.indent + dim(f"  ⎿ {e['n']} tool calls in this step"), file=sys.stderr, flush=True)
        elif k == "tool_call":
            self._spin_stop(); self._end_text()
            self._cur = e
            if not self.quiet:
                print(f"{self.indent}{acc('●')} {bold(e['name'])}({dim(short_args(e['name'], e['args']))})", file=sys.stderr, flush=True)
                if TTY and e["name"] in ("Bash", "Task", "WebFetch", "WebSearch", "Grep"): self._spin_start(TOOL_VERBS.get(e["name"], "Working"))
        elif k == "tool":
            self._spin_stop()
            self.log.append((e["name"], e.get("args") or getattr(self, "_cur", {}).get("args", {}), e["result"]))
            if not self.quiet: self._tool_result(e)
            self.state.ui_extra = ""
        elif k == "answer":
            self._spin_stop()
            if self.stream: self._end_text()
            elif not self.show_answer: pass
            elif self.md: CON.print(answer_view(e["text"]))
            else: print(e["text"])

    def _tool_result(self, e):
        ind, res, name = self.indent + "  ", e["result"], e["name"]
        out = lambda s: print(s, file=sys.stderr)
        if name == "TodoWrite" and self.state.todos:
            out(render_todos(self.state.todos, ind))
        elif name in ("Edit", "Write", "NotebookEdit") and self.state.ui_extra and not res.startswith("error"):
            a, r = getattr(self.state, "ui_counts", None) or diff_counts(self.state.ui_extra)
            verb = "Wrote" if name == "Write" else "Updated"
            out(f"{ind}{dim('⎿')}  {verb} {dim(short_args(name, getattr(self, '_cur', {}).get('args', {})))} with "
                f"{green(plural(a, 'addition'))}{' and ' + red(plural(r, 'removal')) if r else ''}")
            out(render_diff(self.state.ui_extra, ind + "   "))
        else:
            shown, more = summarize(name, res)
            col = red if res.startswith("error") else dim
            for i, l in enumerate(shown):
                out(f"{ind}{dim('⎿') if i == 0 else ' '}  {col(l[:140])}")
            if more: out(f"{ind}   " + dim("… +" + plural(more, "line") + " (/expand or ctrl+o)"))

    def transcript(self) -> str:
        parts = []
        for name, args, res in self.log:
            parts.append(f"● {name}({short_args(name, args)})\n" + "\n".join("  ⎿ " + l if i == 0 else "    " + l for i, l in enumerate(res.splitlines() or [""])))
        return "\n\n".join(parts) or "(no tool calls yet)"

# ---- permission prompt ------------------------------------------------------------------------------------
def preview(name: str, args: dict) -> str:
    """What the tool is about to do, as a few lines: a diff for Edit, the head of the file for Write."""
    if name == "Edit":
        d = difflib.unified_diff(str(args.get("old_string", "")).splitlines(), str(args.get("new_string", "")).splitlines(), lineterm="", n=0)
        return render_diff("\n".join(list(d)[:14]), "")
    if name == "Write": return dim("\n".join(str(args.get("content", "")).splitlines()[:8]))
    if name == "Bash": return str(args.get("command", ""))[:600]
    return ""

def confirm(name: str, args: dict, indent: str = "") -> str | None:
    """Ask the user to approve a tool call. -> 'yes' | 'always' | 'no' | None (no terminal)."""
    if not (sys.stdin.isatty() and sys.stderr.isatty()): return None
    body = f"{bold(name)}({short_args(name, args)})"
    pv = preview(name, args)
    print(f"{indent}{acc('╭─')} {bold('Allow this tool?')}\n{indent}{acc('│')} {body}" +
          ("".join(f"\n{indent}{acc('│')}   {l}" for l in pv.splitlines())) +
          f"\n{indent}{acc('│')} {bold('1.')} Yes   {bold('2.')} Yes, and don't ask again for {name} this session   {bold('3.')} No (esc)"
          f"\n{indent}{acc('╰─')} ", end="", file=sys.stderr, flush=True)
    with ESC.paused():
        while True:
            try: k = getkey().lower()
            except (EOFError, KeyboardInterrupt): print(file=sys.stderr); return "no"
            ans = {"1": "yes", "y": "yes", "\n": "yes", "\r": "yes", "2": "always", "a": "always", "3": "no", "n": "no", "": "no"}.get(k)
            if ans: print(dim({"yes": "Yes", "always": "Yes, don't ask again", "no": "No"}[ans]), file=sys.stderr); return ans

def welcome(model: str, mode: str, cwd: str, extra: str = "") -> None:
    tip = random.choice(TIPS)
    if CON and TTY:
        CON.print(Panel(Text.assemble(("✻ ", f"color({accent_n()})"), ("Welcome to Alice!", "bold"), "\n\n", ("  /help for help, /status for your current setup\n\n", "dim"),
                                      (f"  cwd: {cwd}\n  model: {model} · {mode}{extra}", "dim")), border_style=f"color({accent_n()})", expand=False, padding=(0, 2)))
        CON.print(Text(f" ※ Tip: {tip}", style="dim")); CON.print()
    else:
        print(f"✻ Alice — Claude Code–style agent · /help for commands\n  {model} · {mode} · {cwd}\n")
