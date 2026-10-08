"""Interactive input in the style of Claude Code (prompt_toolkit): ruled prompt box, mode/model/context footer with
Shift+Tab mode cycling, slash and @-file completion menus, multiline input, history search. Falls back to input()."""
import glob, json, os, shutil, subprocess, sys, time

from . import statusline, ui

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.application import run_in_terminal
    from prompt_toolkit.styles import Style
    HAVE_PT = True
except ImportError:   # pragma: no cover
    HAVE_PT = False

SLASH_HELP = {
    "help": "show commands and shortcuts", "statusline": "status line under the prompt: on, off, claude, or a command", "exit": "quit", "clear": "start a new conversation", "compact": "summarize the conversation to free context",
    "model": "show or switch the model", "models": "list available models", "permissions": "show or set the permission mode",
    "plan": "toggle plan mode (read-only)", "tools": "list tools", "todos": "show the todo list", "cost": "token usage this session",
    "status": "session, model, context", "memory": "loaded memory files", "skills": "list skills", "agents": "list subagents",
    "mcp": "MCP servers and tools", "hooks": "configured hooks", "resume": "resume a past session", "init": "create an ALICE.md",
    "config": "show settings", "rewind": "undo the last turn", "context": "context window usage", "export": "write the conversation to a file",
    "doctor": "check the setup", "review": "review uncommitted changes", "transcript": "full tool output (ctrl+o)", "vim": "toggle vim keys", "security-review": "review changes for vulnerabilities", "release-notes": "recent changes to Alice", "usage": "token usage this session", "think": "thinking on, off or auto", "rename": "name this session", "output-style": "change how Alice writes", "add-dir": "add another working directory", "rc": "remote control from a phone or browser", "remote-control": "same as /rc", "selfedit": "let Alice change her own code (test-gated)", "restart": "relaunch into the current code", "expand": "full output of the last tool call", "diff": "files changed by the last turn", "copy": "copy the last answer", "retry": "redo the last prompt", "theme": "change the accent colour", "tasks": "background shells (ctrl+b)", "plugins": "installed plugins", "image": "attach an image (or ctrl+v)", "kill": "stop a background shell",
}
MODE_CYCLE = ["default", "acceptEdits", "plan", "bypassPermissions"]
MODE_LABEL = {"default": ("? for shortcuts", "ansidefault"), "acceptEdits": ("⏵⏵ accept edits on", "ansigreen"),
              "plan": ("⏸ plan mode on", "ansicyan"), "bypassPermissions": ("⏵⏵ bypass permissions on", "ansired")}
SHORTCUTS = """  ! shell    # note to ALICE.md    @ file    / commands      shift+tab  cycle permission mode
  esc  interrupt a running turn      esc esc  rewind (empty prompt) / clear (text)
  ctrl+o  full tool output           ctrl+t  todos  ctrl+b  shells  ctrl+s  stash  ctrl+v  paste image  ctrl+r  history search     ctrl+g  edit in $EDITOR
  alt+enter or \\ at line end  newline      ctrl+c  clear / exit      ctrl+d  exit      ctrl+l  redraw"""

def next_mode(mode: str) -> str:
    return MODE_CYCLE[(MODE_CYCLE.index(mode) + 1) % len(MODE_CYCLE)] if mode in MODE_CYCLE else MODE_CYCLE[0]

def complete_candidates(text: str, cwd: str, commands: dict | None = None, models=lambda: []) -> list[tuple[str, int, str]]:
    """(replacement, start_position, meta) for the word at the end of `text`; pure so it can be tested."""
    commands = commands or {}
    if text.startswith("/") and " " not in text:
        names = {**SLASH_HELP, **{n: "custom command" for n in commands}}
        return [("/" + n, -len(text), d) for n, d in sorted(names.items()) if ("/" + n).startswith(text)]
    if text.startswith("/think "):
        w = text.split(" ", 1)[1]; return [(m, -len(w), "") for m in ("on", "off", "auto") if m.startswith(w)]
    if text.startswith("/output-style "):
        from .context import output_styles
        w = text.split(" ", 1)[1]; return [(m, -len(w), "") for m in output_styles() if m.startswith(w)]
    if text.startswith("/model "):
        w = text.split(" ", 1)[1]; return [(m, -len(w), "") for m in models() if m.startswith(w) or w in m][:30]
    if text.startswith("/permissions "):
        w = text.split(" ", 1)[1]; return [(m, -len(w), "") for m in MODE_CYCLE if m.startswith(w)]
    word = text.rsplit(None, 1)[-1] if text.strip() and not text.endswith(" ") else ""
    if not word.startswith("@"): return []
    bare = os.path.expanduser(word[1:])
    base = bare if os.path.isabs(bare) else os.path.join(cwd, bare)
    out = []
    for m in sorted(glob.glob(glob.escape(base) + "*"), key=lambda p: (os.path.basename(p).startswith("."), p))[:60]:
        rel = m if os.path.isabs(bare) else os.path.relpath(m, cwd)
        d = os.path.isdir(m)
        out.append(("@" + rel + ("/" if d else ""), -len(word), "dir" if d else ""))
    return out

class Prompter:
    def __init__(self, app):
        self.app = app; self._git = ("", 0.0); self.session = None
        self.status = statusline.StatusLine(lambda: self.session and self.session.app.is_running and self.session.app.invalidate()); self.stash = ""; self.armed = (0.0, "")
        self.fancy = HAVE_PT and sys.stdin.isatty() and sys.stdout.isatty()
        if self.fancy: self.session = self._build()

    # ---- footer
    def branch(self) -> str:
        if time.time() - self._git[1] > 5:
            try: b = subprocess.run(["git", "-C", self.app.cwd, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=1).stdout.strip()
            except Exception: b = ""
            self._git = (b, time.time())
        return self._git[0]

    def status_lines(self) -> list[str]:
        """Claude Code-style status line rows: the statusLine command's output, else the built-in line; [] if statusLine is false."""
        conf = self.app.settings.get("statusLine", {})
        if conf is False or (isinstance(conf, dict) and conf.get("type") == "none"): return []
        from .cli import VERSION
        data = statusline.payload(self.app, self.ctx_tokens(), VERSION)
        cmd = conf.get("command") if isinstance(conf, dict) else None
        if not cmd: return statusline.builtin(data)
        self.status.refresh(cmd, data)
        return self.status.lines or ([self.status.error] if self.status.error else [])

    def model_names(self) -> list[str]:
        if not hasattr(self, "_models"):
            try:
                from .llm import installed_models
                self._models = installed_models() if getattr(self.app.llm, "server", None) else self.app.llm.models()
            except Exception: self._models = []
        return self._models

    def ctx_tokens(self) -> int:
        from ctxguard import est_tokens
        return sum(est_tokens(m.get("content") or "") + est_tokens(json.dumps(m.get("tool_calls") or "")) for m in self.app.agent.messages)

    def ctx_pct(self) -> int:
        return min(100, 100 * self.ctx_tokens() // max(1, self.app.llm.num_ctx))

    def footer(self) -> list:
        a = self.app; label, color = MODE_LABEL.get(a.perms.mode, (a.perms.mode, ""))
        has_status = bool(self.status_lines())
        right = " · ".join(x for x in (f"⎇ {self.branch()}" if self.branch() else "",) + (() if has_status else (a.llm.model.split("/")[-1][:34], f"{self.ctx_pct()}% ctx")) if x)
        w = shutil.get_terminal_size().columns
        left = f" {label}" + (" (shift+tab to cycle)" if a.perms.mode != "default" else "")
        if time.time() - self.armed[0] < 2: left = f" Press {self.armed[1]} again to exit"
        n = sum(j["proc"].poll() is None for j in a.state.jobs.values())
        if n: left += f" · {n} shell{'s' if n != 1 else ''} (ctrl+b)"
        if a.pending_images: left += f" · {len(a.pending_images)} image{'s' if len(a.pending_images) != 1 else ''}"
        if self.stash: left += " · stashed (ctrl+s)"
        if getattr(a, "sname", ""): left += f" · {a.sname}"
        if getattr(a, "style", "default") != "default": left += f" · {a.style} style"
        if getattr(a.rc, "server", None): left += " · remote control on"
        if len(left) + len(right) + 3 > w: left = left.replace(" (shift+tab to cycle)", "")
        if len(left) + len(right) + 3 > w: right = "" if has_status else f"{self.ctx_pct()}% ctx"   # narrow: keep the mode, drop model/branch
        room = w - len(right) - 3
        if len(left) > room: left = left[:max(room - 1, 0)] + "…"
        pad = max(1, w - len(left) - len(right) - 1)
        return [("fg:" + color if color else "", left), ("", " " * pad), ("class:dim", right)]

    # ---- session
    def _build(self):
        a = self.app; kb = KeyBindings()
        empty = Condition(lambda: not self.session.default_buffer.text)

        @kb.add("enter")
        def _(e):
            b = e.current_buffer
            if b.text.endswith("\\"): b.delete_before_cursor(1); b.insert_text("\n")
            else: b.validate_and_handle()
        for k in (("escape", "enter"), ("c-j",)):
            kb.add(*k)(lambda e: e.current_buffer.insert_text("\n"))

        @kb.add("s-tab")
        def _(e): a.perms.set_mode(next_mode(a.perms.mode)); e.app.invalidate()

        def exit_press(e, key, exc):
            """First press on an empty prompt arms the footer hint; a second within 2 s quits (like Claude Code)."""
            if time.time() - self.armed[0] < 2 and self.armed[1] == key: e.app.exit(exception=exc()); return
            self.armed = (time.time(), key); e.app.invalidate()
            try: e.app.loop.call_later(2.1, e.app.invalidate)
            except Exception: pass

        @kb.add("c-c")
        def _(e):
            if e.current_buffer.text: e.current_buffer.reset()
            else: exit_press(e, "Ctrl-C", KeyboardInterrupt)

        @kb.add("c-d", filter=empty)
        def _(e): exit_press(e, "Ctrl-D", EOFError)

        @kb.add("escape", "escape")
        def _(e):
            if e.current_buffer.text: e.current_buffer.reset(); return
            run_in_terminal(lambda: a.c_rewind(""))

        @kb.add("c-o")
        def _(e): run_in_terminal(a.c_transcript)

        @kb.add("c-s")
        def _(e):   # stash the draft (Claude Code): it comes back in the next prompt; Ctrl+S on an empty prompt restores it now
            b = e.current_buffer
            if b.text.strip(): self.stash = b.text; b.reset(); e.app.invalidate()
            elif self.stash: b.insert_text(self.stash); self.stash = ""; e.app.invalidate()

        @kb.add("c-v")
        def _(e):   # paste an image from the clipboard (text pastes with the terminal's own paste)
            def go():
                err = a.paste_image(); print("  " + (ui.dim(err) if err else ui.acc("●") + f" image attached ({len(a.pending_images)})"))
            run_in_terminal(go)

        @kb.add("c-b")
        def _(e): run_in_terminal(lambda: a.c_tasks(""))

        @kb.add("c-t")
        def _(e): run_in_terminal(lambda: a.c_todos(""))

        @kb.add("c-g")
        def _(e): e.current_buffer.open_in_editor(validate_and_handle=False)

        @kb.add("?", filter=empty)
        def _(e): run_in_terminal(lambda: print(ui.dim(SHORTCUTS)))

        outer = self
        class C(Completer):
            def get_completions(self, doc, ev):
                for text, start, meta in complete_candidates(doc.text_before_cursor, a.cwd, a.commands, outer.model_names):
                    yield Completion(text, start_position=start, display_meta=meta)

        def message():
            w = shutil.get_terminal_size().columns
            return [("class:rule", "─" * w + "\n"), ("class:prompt", "> ")]

        os.makedirs(os.path.expanduser("~/.alice"), exist_ok=True)
        s = PromptSession(message, history=FileHistory(os.path.expanduser("~/.alice/prompt_history")), completer=C(),
                          complete_while_typing=True, complete_style=CompleteStyle.COLUMN, auto_suggest=AutoSuggestFromHistory(),
                          key_bindings=kb, multiline=True, enable_open_in_editor=True,
                          prompt_continuation=lambda w, ln, wrap: "  ", reserve_space_for_menu=0,
                          style=Style.from_dict({"rule": "fg:#6c6c6c", "prompt": "bold fg:" + ui.accent_hex(), "dim": "fg:#808080",
                                                 "bottom-toolbar": "noreverse", "bottom-toolbar.text": "noreverse"}))
        # footer sits directly under the input (bottom_toolbar would pin it to the last terminal row)
        from prompt_toolkit.layout import Window, FormattedTextControl, ConditionalContainer
        from prompt_toolkit.filters import is_done, to_filter
        for w in s.layout.find_all_windows():
            if getattr(w.content, "buffer", None) is s.default_buffer: w.dont_extend_height = to_filter(True)
        from prompt_toolkit.formatted_text import ANSI, to_formatted_text
        def below():   # rule, status line rows (ANSI kept, like Claude Code), then the mode footer
            pad = " " * int((self.app.settings.get("statusLine") or {}).get("padding", 0) if isinstance(self.app.settings.get("statusLine"), dict) else 0)
            out = [("class:rule", "─" * shutil.get_terminal_size().columns + "\n")]
            for line in self.status_lines(): out += [("", " " + pad)] + to_formatted_text(ANSI(line)) + [("", "\n")]
            return out + self.footer()
        s.layout.container.children.append(ConditionalContainer(
            Window(FormattedTextControl(below), dont_extend_height=True), filter=~is_done))
        # reserve rows for the menu only while one can appear (/command or @file), so the footer hugs the input otherwise
        def _reserve(buf):
            t = buf.text; word = t.rsplit(None, 1)[-1] if t.strip() and not t.endswith(" ") else ""
            s.reserve_space_for_menu = 7 if (t.startswith("/") and " " not in t) or word.startswith("@") or t.startswith(("/permissions ", "/model ")) else 0
        s.default_buffer.on_text_changed += _reserve
        s.app.timeoutlen = 0.4; s.app.ttimeoutlen = 0.05
        return s

    def vim(self) -> str:
        if not self.fancy: return "vim mode needs an interactive terminal"
        on = self.session.editing_mode != EditingMode.VI
        self.session.editing_mode = EditingMode.VI if on else EditingMode.EMACS
        return "vim keys " + ("on" if on else "off")

    def read(self) -> str:
        """One submitted prompt. Raises EOFError (Ctrl+D) / KeyboardInterrupt (Ctrl+C on an empty prompt)."""
        if self.fancy:
            draft, self.stash = self.stash, ""
            return self.session.prompt(default=draft)
        line = input(ui.bold("> "))
        while line.endswith("\\"): line = line[:-1] + "\n" + input("  ")
        return line
