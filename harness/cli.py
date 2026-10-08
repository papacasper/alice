"""The Claude Code–style CLI: REPL with slash commands, permissions, hooks, MCP, sessions, subagents."""
import argparse, atexit, contextlib, glob, json, os, re, shutil, subprocess, sys, time, urllib.request
from . import profiles, context, sessions, ui
from .agent import Agent
from ctxguard import est_tokens
from .cctools import READ_TOOLS, State, cc_toolbox
from .hooks import Hooks
from .prompt import Prompter
from .llm import DEFAULT_MODEL, LLMError, LlamaServer, OllamaClient, OpenAIClient
from .mcp import McpServer
from .permissions import MODES, Permissions
from .tools import Toolbox

VERSION = "1.0"
INIT_PROMPT = ("Analyze this codebase with your tools (README, package manifests, entry points, tests) and create a concise "
               "ALICE.md in the working directory: build/test/run commands, architecture, and conventions. If one exists, improve it.")

def parse(argv=None):
    ap = argparse.ArgumentParser(prog="alice", description="Claude-Code-style agent on a local model (Ollama or llama.cpp)")
    ap.add_argument("task", nargs="?", help="one-shot prompt (non-interactive); omit for the interactive REPL")
    ap.add_argument("-p", "--print", action="store_true", help="non-interactive: print the answer and exit (reads stdin if no task)")
    ap.add_argument("--model"); ap.add_argument("--num-ctx", type=int); ap.add_argument("--max-steps", "--max-turns", type=int, default=0, help="0 = no limit")
    ap.add_argument("--root", "--cwd", dest="root", help="working directory (default: current)")
    ap.add_argument("--permission-mode", choices=MODES, help="default: bypassPermissions (everything allowed)")
    ap.add_argument("--dangerously-skip-permissions", action="store_true")
    ap.add_argument("--allowedTools", "--allowed-tools", action="append", default=[], metavar="RULE")
    ap.add_argument("--disallowedTools", "--disallowed-tools", action="append", default=[], metavar="RULE")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true", help="resume the latest session in this directory")
    ap.add_argument("-r", "--resume", nargs="?", const="", metavar="ID", help="resume a session by id (no id: list them)")
    ap.add_argument("--output-format", choices=["text", "json"], default="text")
    ap.add_argument("--system-prompt", "--system", dest="system", help="file replacing the default system prompt")
    ap.add_argument("--append-system-prompt", help="text appended to the system prompt")
    ap.add_argument("--tools-file", action="append", default=[], help="python file with @tool functions (repeatable)")
    ap.add_argument("--backend", choices=["ollama", "llama"], help="llama (default): private llama-server, no Ollama needed; or ollama")
    ap.add_argument("--base-url", help="use an already-running server: Ollama's URL, or an OpenAI-compatible URL with --backend llama")
    ap.add_argument("--llama-server-args", default="", help="extra arguments for the llama-server alice starts, e.g. '-ngl 20'")
    ap.add_argument("--think", nargs="?", const="on", default=None, choices=["on", "off", "auto"], help="thinking: on, off, or auto (default: first step of analytical prompts only)"); ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-memory", action="store_true"); ap.add_argument("--no-mcp", action="store_true")
    ap.add_argument("--sampling", type=json.loads, default=None, metavar="JSON", help='sampling overrides, e.g. \'{"temperature": 0.7}\' (beats settings.json "sampling")')
    ap.add_argument("-q", "--quiet", action="store_true", help="hide tool-call trace")
    ap.add_argument("--version", action="version", version=f"alice {VERSION}")
    ap.add_argument("--allow-write", action="store_true", help=argparse.SUPPRESS)   # legacy no-ops
    ap.add_argument("--allow-shell", action="store_true", help=argparse.SUPPRESS)
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, t in enumerate(argv):      # a bare --think (the old flag) must not swallow the prompt that follows it
        if t == "--think" and (i + 1 == len(argv) or argv[i + 1] not in ("on", "off", "auto")): argv[i] = "--think=on"
    return ap.parse_args(argv)

class App:
    def __init__(self, a, llm=None):
        self.a = a
        self.cwd = os.path.abspath(os.path.expanduser(a.root)) if a.root else os.getcwd()
        self.settings = context.load_settings(self.cwd)
        ui.set_theme(str(self.settings.get("theme", "terracotta")))
        self.interactive = not (a.task or a.print or not sys.stdin.isatty()) and a.output_format == "text"
        model = a.model or self.settings.get("model") or DEFAULT_MODEL
        self.llm = llm or self.make_llm(model, a.num_ctx or self.settings.get("num_ctx") or profiles.DEFAULT_CTX)
        self.sid = sessions.new_id()
        self.state = State(self.cwd); self.pending_images: list[str] = []
        self.state.skills = context.discover_skills(self.cwd)
        self.commands = context.discover_commands(self.cwd)
        self.state.agents = context.discover_agents(self.cwd)
        mode = "bypassPermissions" if a.dangerously_skip_permissions else (
            a.permission_mode or self.settings.get("permissionMode") or "bypassPermissions")
        p = self.settings.get("permissions", {})
        self.perms = Permissions(mode, p.get("allow", []) + a.allowedTools, p.get("deny", []) + a.disallowedTools,
                                 ask=lambda n, args: ui.confirm(n, args))
        self.prev_mode = mode if mode != "plan" else "bypassPermissions"
        self.hooks = Hooks(self.settings.get("hooks"), self.cwd, self.sid)
        self.mcp_servers: list[McpServer] = []
        self.toolbox = cc_toolbox(self.state)
        for f in a.tools_file:
            for t in Toolbox.from_file(f): self.toolbox.add(t)
        if not a.no_mcp: self._start_mcp()
        stream = not a.no_stream and a.output_format == "text"
        self.started = time.time(); self.rc = None; self.at_prompt = False; self.extra_dirs = []; self.sname = ""; self.style = str(self.settings.get("outputStyle", "default"))
        self.renderer = ui.Renderer(self.state, a.quiet or a.output_format == "json", stream=stream)
        self.agent = Agent(self.llm, self.toolbox, "", a.max_steps, self._on_event, gate=self.gate, post=self.post,
                           on_token=self.renderer.token if stream else None)
        self.state.spawn, self.state.plan_mode = self.spawn, lambda: self.perms.mode == "plan"
        self.state.ask_user = self.ask_user if self.interactive else None
        self.state.approve_plan = self.approve_plan if self.interactive else None
        self.refresh_system()
        self._resume()

    # ---- wiring
    def _start_mcp(self):
        for name, cfg in (self.settings.get("mcpServers") or {}).items():
            try:
                srv = McpServer(name, cfg["command"], cfg.get("args"), cfg.get("env"))
                for t in srv.tools(): self.toolbox.add(t)
                self.mcp_servers.append(srv); atexit.register(srv.close)
            except Exception as e:
                print(ui.yellow(f"warning: MCP server '{name}' failed: {e}"), file=sys.stderr)

    def refresh_system(self):
        override = open(os.path.expanduser(self.a.system)).read() if self.a.system else ""
        extra = self.a.append_system_prompt or ""
        if (st := context.output_style(self.style)): extra += "\n" + st
        if self.extra_dirs: extra += "\nAdditional working directories (you may read and edit files here too): " + ", ".join(self.extra_dirs)
        text = context.build_system(self.cwd, self.llm.model, self.state.skills, not self.a.no_memory,
                                    extra, override, self.state.agents)
        sysm = {"role": "system", "content": text}
        if self.agent.messages and self.agent.messages[0]["role"] == "system": self.agent.messages[0] = sysm
        else: self.agent.messages.insert(0, sysm)

    def _resume(self):
        a, sid = self.a, None
        if a.cont:
            ls = sessions.list_sessions(self.cwd); sid = ls[0]["id"] if ls else None
        elif a.resume: sid = a.resume
        elif a.resume == "":
            for s in sessions.list_sessions(self.cwd)[:15]: print(f"{s['id']}  {s['n']:>3} msgs  {s['preview']}")
            sys.exit(0)
        if sid:
            s = sessions.load(self.cwd, sid)
            if not s: print(f"no such session: {sid}", file=sys.stderr); sys.exit(1)
            self.agent.messages += [m for m in s["messages"] if m["role"] != "system"]
            self.sid = sid; self.hooks.session_id = sid
            print(ui.dim(f"resumed session {sid} ({len(s['messages'])} messages)"), file=sys.stderr)
            if (srv := getattr(self.llm, "server", None)) and srv.restore_slot(sid): print(ui.dim("restored the cached prompt (no re-read of the conversation)"), file=sys.stderr)

    def gate(self, name, args):
        if self.hooks.has("PreToolUse"):
            blocked, msg = self.hooks.run("PreToolUse", name, {"tool_input": args})
            if blocked: return f"error: blocked by hook: {msg}"
        return self.perms.check(name, args)

    def post(self, name, args, out):
        if not self.hooks.has("PostToolUse"): return None
        blocked, msg = self.hooks.run("PostToolUse", name, {"tool_input": args, "tool_response": out})
        return (f"[hook] {msg}" if msg else None)

    def spawn(self, prompt, kind):
        custom = self.state.agents.get(kind)
        if kind not in ("explore", "general-purpose") and not custom:
            return f"error: unknown subagent_type '{kind}'; available: general-purpose, explore, {', '.join(self.state.agents) or '(no custom agents)'}"
        only = READ_TOOLS if kind == "explore" else (custom["tools"] or None) if custom else None
        tb = cc_toolbox(self.state, only=only, exclude=["Task", "TodoWrite", "AskUserQuestion", "ExitPlanMode", "Skill"])
        base = custom["prompt"] if custom and custom["prompt"] else (
            "You are a subagent working for another agent. Do the task using your tools, then reply with a concise "
            "report: what you found, with file paths and line numbers. Do not ask questions.")
        sysm = base + "\n\n# Environment\n" + context.env_block(self.state.cwd, self.llm.model)
        r = ui.Renderer(self.state, self.a.quiet, indent="  │ ", stream=False)
        sub = Agent(self.llm, tb, sysm, 25, r.event, gate=self.gate, post=self.post)
        main_slot = getattr(self.llm, "slot", None)
        if main_slot is not None: self.llm.slot = main_slot + 1   # keep the main conversation's KV cache in its slot
        try: res = sub.ask(prompt)
        finally:
            if main_slot is not None: self.llm.slot = main_slot
        return res.answer + ("\n[subagent hit its step limit]" if res.status == "step_limit" else "")

    def ask_user(self, q):
        if not sys.stdin.isatty(): return "(no terminal; make a reasonable assumption)"
        print(f"{ui.yellow('?')} {q}", file=sys.stderr)
        with ui.ESC.paused():
            try: return input("  > ").strip()
            except (EOFError, KeyboardInterrupt): return "(no answer)"

    def approve_plan(self, plan):
        print(f"\n{ui.bold('Plan')}\n{plan}\n", file=sys.stderr)
        ok = ui.confirm("ExitPlanMode", {"plan": "approve and start implementing"})
        if ok in ("yes", "always"): self.perms.set_mode(self.prev_mode); return True
        return False

    # ---- running a turn
    IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp")

    def attach_image(self, path: str) -> str:
        """Queue an image for the next message. Returns "" on success or an error string."""
        import base64
        p = self.state.path(path)
        if not os.path.isfile(p) or not p.lower().endswith(self.IMG_EXT): return f"not an image file: {path}"
        if os.path.getsize(p) > 8_000_000: return "image is over 8 MB"
        self.pending_images.append(base64.b64encode(open(p, "rb").read()).decode()); return ""

    def paste_image(self) -> str:
        """Ctrl+V: attach an image from the clipboard (wl-paste or xclip). Returns "" on success or an error string."""
        import subprocess, tempfile
        cmds = [["wl-paste", "--type", "image/png"], ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]]
        for cmd in cmds:
            if not shutil.which(cmd[0]): continue
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            if r.returncode == 0 and r.stdout[:4] == b"\x89PNG":
                f = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f.write(r.stdout); f.close()
                return self.attach_image(f.name)
        return "no image on the clipboard (needs wl-paste or xclip)"

    def ensure_vision(self) -> str:
        """Make sure the backend can take images; restarts our llama-server with the projector if needed. "" = ready."""
        srv = getattr(self.llm, "server", None)
        if srv is None or srv.vision: return ""
        if not srv.can_see(): return "this model has no vision projector (mmproj); use a vision model"
        print(ui.dim("  restarting llama-server with vision enabled…"), file=sys.stderr)
        try: srv.restart_with_vision(say=lambda m: print(ui.dim(f"  {m}"), file=sys.stderr))
        except LLMError as e: return str(e)
        return ""

    def c_image(self, arg):
        err = self.attach_image(arg.strip()) if arg.strip() else "usage: /image <path>   (or ctrl+v to paste, or @file.png in a message)"
        print("  " + (ui.red(err) if err else ui.acc("●") + f" image attached ({len(self.pending_images)}); it goes with your next message"))

    def expand_mentions(self, text):
        extra = []
        for m in re.finditer(r"(?<!\S)@(\S+)", text):
            p = self.state.path(m.group(1).rstrip(",.;:"))
            if p.lower().endswith(self.IMG_EXT) and os.path.isfile(p): self.attach_image(p); continue
            if os.path.isfile(p): extra.append(f"[Contents of {p}]\n{context.read(p, 4000)}")
        return text + ("\n\n" + "\n\n".join(extra) if extra else "")

    def turn(self, text):
        if self.hooks.has("UserPromptSubmit"):
            blocked, msg = self.hooks.run("UserPromptSubmit", "", {"prompt": text})
            if blocked: print(ui.red(f"blocked by hook: {msg}"), file=sys.stderr); return None
            if msg: text += "\n\n" + msg
        if self.context_used() > 0.75 * self.llm.num_ctx:
            print(ui.dim("  (context is nearly full; compacting first)"), file=sys.stderr)
            try: self.compact_now()
            except LLMError: pass
        n, text = len(self.agent.messages), self.expand_mentions(text)
        self.state.turns.append({"n": n, "files": []})
        imgs, self.pending_images = self.pending_images, []
        t0, tok0 = time.time(), self._out_tokens(); self.last_prompt = text
        if imgs and (err := self.ensure_vision()):
            print(ui.red(f"  images dropped: {err}"), file=sys.stderr); imgs = []
        try:
            with (ui.ESC if self.interactive else contextlib.nullcontext()):
                r = self.agent.ask(text, imgs or None)
                if self.hooks.has("Stop"):
                    blocked, msg = self.hooks.run("Stop", "", {"answer": r.answer})
                    if blocked: r = self.agent.ask(f"[Stop hook] {msg}")
        except KeyboardInterrupt:
            self.renderer._spin_stop(); self.renderer._end_text()
            del self.agent.messages[n:]
            self.agent.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": "[Request interrupted by user]"}]
            print(ui.dim("\n  ⎿ Interrupted"), file=sys.stderr); return None
        except LLMError as e:
            self.renderer._spin_stop(); self.renderer._end_text()
            del self.agent.messages[n:]
            print(ui.red(f"LLM error: {e}"), file=sys.stderr); return None
        if r.status == "step_limit": print(ui.dim(f"  (stopped at the {self.a.max_steps}-step limit; say 'continue' to go on)"), file=sys.stderr)
        sessions.save(self.cwd, self.sid, self.agent.messages, {"model": self.llm.model, "name": self.sname})
        if self.interactive and time.time() - t0 >= 5:
            print(ui.dim("  " + ui.turn_summary(time.time() - t0, r.tool_calls, self._out_tokens() - tok0)), file=sys.stderr)
        self.last_answer = r.answer
        return r

    # ---- entry points
    def one_shot(self, task):
        self.renderer.show_answer = False   # printed once below; otherwise --output-format json got the answer text before the JSON
        r = self.turn(task)
        if r is None: return 1
        if self.a.output_format == "json":
            print(json.dumps({"result": r.answer, "status": r.status, "steps": r.steps, "tool_calls": r.tool_calls,
                              "session_id": self.sid, "usage": self.llm.usage}))
        elif not self.agent.on_token: print(r.answer)
        return 0 if r.status == "answered" else 2

    def repl(self):
        self.prompter = Prompter(self)
        if not self.prompter.fancy:
            try:
                import readline; hf = os.path.expanduser("~/.alice/history"); os.makedirs(os.path.dirname(hf), exist_ok=True)
                try: readline.read_history_file(hf)
                except OSError: pass
                atexit.register(readline.write_history_file, hf)
                readline.set_completer_delims(" \t\n"); readline.set_completer(self.complete); readline.parse_and_bind("tab: complete")
            except ImportError: pass
        ui.welcome(self.llm.model.split("/")[-1], self.perms.mode, self.cwd, f" · {len(self.toolbox.names())} tools")
        for e in (self.hooks.run("SessionStart", "")[1:] if self.hooks.has("SessionStart") else []): print(e)
        quit_armed = False
        while True:
            try:
                if ui.ESC.queue: line = ui.ESC.queue.pop(0); print(ui.dim("> ") + line)   # typed while the last turn ran
                else:
                    self.at_prompt = True
                    try: line = self.prompter.read()
                    finally: self.at_prompt = False
                quit_armed = False
            except EOFError: print(); return 0
            except KeyboardInterrupt:
                if quit_armed or getattr(self.prompter, "fancy", False): print(); return 0   # the fancy prompt already required two presses
                print(ui.dim("(Ctrl-C again, or /exit, to quit)")); quit_armed = True; continue
            line = line.strip()
            if not line: continue
            if line.startswith("!"): self.shell(line[1:]); continue
            if line.startswith("#"): self.remember(line.lstrip("#").strip()); continue
            if line.startswith("/"):
                if self.slash(line) == "exit": return 0
                continue
            self.turn(line); print()

    def make_llm(self, model, num_ctx):
        backend = self.a.backend or os.environ.get("ALICE_BACKEND") or self.settings.get("backend") or "llama"
        url = self.a.base_url or os.environ.get("ALICE_BASE_URL") or self.settings.get("baseUrl")
        if backend == "ollama":
            c = OllamaClient(model, host=url or "http://localhost:11434", num_ctx=num_ctx, think=self.think_arg() is True)
            ok, info = c.health()
            if not ok: raise LLMError(f"{info}. Start it (sudo systemctl start ollama) or use --backend llama (the default) or set ALICE_BACKEND=ollama only if Ollama is running.")
            return c
        if url:   # connect to a server that is already running
            c = OpenAIClient(model, host=url, num_ctx=num_ctx, think=self.think_arg(), api_key=os.environ.get("ALICE_API_KEY", ""))
            c.sampling = {**(self.settings.get("sampling") or {}), **(self.a.sampling or {})}
            return c
        import shlex
        srv = LlamaServer(model, num_ctx, extra=shlex.split(self.a.llama_server_args or self.settings.get("llamaServerArgs", "")),
                          auto=profiles.server_flags(self.settings), slot_dir=os.path.expanduser("~/.alice/slots"))
        srv.prune_slots()
        host = srv.start(say=lambda m: print(ui.dim(f"  {m}"), file=sys.stderr))
        c = OpenAIClient(model, host=host, num_ctx=num_ctx, think=self.think_arg()); c.server = srv; c.slot = 0
        c.sampling = {**(self.settings.get("sampling") or {}), **(self.a.sampling or {})}
        atexit.register(lambda: len(self.agent.messages) > 2 and srv.save_slot(self.sid))   # runs before the server is stopped
        return c

    def think_arg(self):
        """The --think value, else the "think" setting, else auto."""
        v = self.a.think or str(self.settings.get("think", "auto"))
        return {"on": True, "off": False}.get(v, "auto")

    def complete(self, text, i):
        """Tab completion: /commands, @paths and plain paths."""
        if text.startswith("/") and "/" not in text[1:]:
            names = ["help", "statusline", "exit", "clear", "compact", "model", "models", "permissions", "plan", "tools", "todos", "cost", "status",
                     "memory", "skills", "mcp", "hooks", "resume", "init", "config", "rewind", "context", "export", "doctor", "review",
                     "agents", *self.commands]
            opts = sorted({"/" + n for n in names if ("/" + n).startswith(text)})
        else:
            pre, bare = ("@", text[1:]) if text.startswith("@") else ("", text)
            base = self.state.path(bare) if bare else self.state.cwd + os.sep
            opts = [pre + (os.path.relpath(m, self.state.cwd) if not os.path.isabs(bare) and not bare.startswith("~") else m) + ("/" if os.path.isdir(m) else "")
                    for m in sorted(glob.glob(base + "*"))]
        return opts[i] if i < len(opts) else None

    def remember(self, note):
        """'# note' appends a bullet to the project's ALICE.md / AGENTS.md / CLAUDE.md (ALICE.md created if none), like Claude Code's # shortcut."""
        if not note: return
        path = context.memory_target(self.cwd)
        with open(path, "a") as f: f.write(f"- {note}\n")
        self.refresh_system(); print(ui.dim(f"  saved to {path}"))

    def shell(self, cmd):
        print(self.toolbox.call("Bash", {"command": cmd}))

    # ---- slash commands
    def slash(self, line):
        name, _, arg = line[1:].partition(" "); arg = arg.strip()
        h = {"help": self.c_help, "exit": lambda a: "exit", "quit": lambda a: "exit", "clear": self.c_clear, "compact": self.c_compact,
             "model": self.c_model, "models": self.c_models, "permissions": self.c_perms, "plan": self.c_plan, "tools": self.c_tools,
             "todos": self.c_todos, "cost": self.c_cost, "status": self.c_status, "memory": self.c_memory, "skills": self.c_skills,
             "mcp": self.c_mcp, "hooks": self.c_hooks, "resume": self.c_resume, "init": self.c_init, "config": self.c_config,
             "rewind": self.c_rewind, "context": self.c_context, "export": self.c_export, "doctor": self.c_doctor,
             "review": self.c_review, "agents": self.c_agents, "transcript": self.c_transcript, "copy": self.c_copy, "selfedit": self.c_selfedit, "rc": self.c_rc, "add-dir": self.c_add_dir, "think": self.c_think, "rename": self.c_rename, "output-style": self.c_output_style, "remote-control": self.c_rc, "restart": self.c_restart, "security-review": self.c_security_review, "release-notes": self.c_release_notes, "usage": self.c_cost, "expand": self.c_expand, "diff": self.c_diff, "retry": self.c_retry, "vim": self.c_vim, "theme": self.c_theme, "statusline": self.c_statusline,
             "plugins": self.c_plugins, "image": self.c_image, "tasks": self.c_tasks, "bashes": self.c_tasks, "jobs": self.c_tasks, "kill": self.c_kill}
        if name in h: return h[name](arg)
        if name in self.commands:
            body = self.commands[name][1].replace("$ARGUMENTS", arg)
            return self.turn(body) and print()
        print(ui.red(f"unknown command /{name}") + " — /help lists them")

    HELP_GROUPS = [("Conversation", "clear compact resume rename rewind retry diff copy export transcript expand"),
                   ("Session", "model models think output-style permissions plan status context cost usage theme statusline vim release-notes"),
                   ("Setup", "add-dir rc selfedit restart init memory config doctor tools skills agents plugins mcp hooks"),
                   ("Work", "todos tasks kill review security-review image")]

    def c_help(self, arg):
        """/help lists every command by group; /help <command> shows one."""
        from .prompt import SLASH_HELP, SHORTCUTS
        h = dict(SLASH_HELP); h.update({n: d for n, (d, _b) in self.commands.items()})
        if arg.strip():
            n = arg.strip().lstrip("/"); fn = getattr(self, "c_" + n, None)
            print(f"  /{n}  {ui.dim(h.get(n, ''))}" + (f"\n  {ui.dim((fn.__doc__ or '').strip())}" if fn and fn.__doc__ else "") if n in h or fn else ui.red(f"  no command /{n}")); return
        grouped = {n for _g, names in self.HELP_GROUPS for n in names.split()}
        groups = self.HELP_GROUPS + [("Custom", " ".join(sorted(self.commands)))] if self.commands else self.HELP_GROUPS
        for g, names in groups:
            names = [n for n in names.split() if n in h]
            if names: print(f"  {ui.bold(g)}\n    " + "  ".join(f"/{n}" for n in names))
        rest = sorted(n for n in h if n not in grouped and n not in self.commands and n not in ("help", "exit"))
        if rest: print(f"  {ui.bold('Other')}\n    " + "  ".join(f"/{n}" for n in rest))
        print(f"\n{SHORTCUTS}\n  {ui.dim('/help <command> for details')}")

    def c_clear(self, _):
        self.agent.reset(); self.state.todos.clear(); self.state.read_files.clear()
        self.sid = sessions.new_id(); self.hooks.session_id = self.sid; self.refresh_system(); print(ui.dim("  conversation cleared"))

    def context_used(self) -> int:
        """Estimated tokens the next request will carry: messages (including tool calls) plus the tool schemas."""
        from ctxguard import est_tokens
        return sum(est_tokens(m.get("content") or "") + est_tokens(json.dumps(m.get("tool_calls") or "")) for m in self.agent.messages) + est_tokens(json.dumps(self.toolbox.schemas()))

    def compact_now(self):
        """Summarize the history, on settings.helperModel if one is set (the private server swaps to it and back; costs two reloads).
        Measured 2026-10-07 (evals/live_checks.py helper-compact, gemma4-e2b vs the 9B): the helper was slower (7-9 s vs 5 s) and lost
        the key fact in 1 of 2 runs. Leave it unset on a single 8 GB GPU."""
        helper, srv, main = self.settings.get("helperModel"), getattr(self.llm, "server", None), self.llm.model
        if not (helper and srv and helper != main): return self.agent.compact()
        srv.switch_model(helper); self.llm.model = helper
        try: return self.agent.compact()
        finally: srv.switch_model(main); self.llm.model = main

    def c_think(self, arg):
        """/think [on|off|auto]: whether the model reasons before answering (auto: only on the first step of analytical or long prompts)."""
        a = arg.strip().lower()
        if a in ("on", "off", "auto"): self.llm.think_mode = a; self.settings["think"] = a
        elif a: print(ui.red("  use /think on, off or auto")); return
        print(f"  thinking: {getattr(self.llm, 'think_mode', 'n/a')}")

    def c_compact(self, _):
        try: self.compact_now(); print(ui.dim(f"  compacted to {len(self.agent.messages)} messages"))
        except LLMError as e: print(ui.red(f"LLM error: {e}"))

    def c_model(self, arg):
        """/model [name]: show the model, or switch to another one on the fly (the conversation is kept; a private llama-server reloads)."""
        from .llm import gguf_args
        name = arg.strip()
        if not name or name == self.llm.model: print(f"  model: {self.llm.model}" + ("" if name else ui.dim("   (/models lists installed ones, /model <name> switches)"))); return
        srv = getattr(self.llm, "server", None)
        if srv:
            if gguf_args(name)[0] == "-hf": print(ui.dim("  not installed locally; llama-server will download it"))
            print(ui.dim(f"  loading {name}…"))
            try: srv.switch_model(name, say=lambda m: None)
            except LLMError as e:
                print(ui.red(f"  could not load {name}; kept {srv.model}\n    " + str(e).splitlines()[-1][:160])); return
        self.llm.model = name; self.refresh_system()
        print(f"  {ui.acc('●')} switched to {ui.bold(name)}" + ui.dim("  (conversation kept)"))

    def c_models(self, _):
        from .llm import installed_models
        try:
            names = installed_models() if getattr(self.llm, "server", None) else self.llm.models()
            if getattr(self.llm, "server", None) and self.llm.model not in names: names.insert(0, self.llm.model)
            for m in names: print(f"  {ui.acc('●') if m == self.llm.model else ' '} {m}")
            if not names: print(ui.dim("  none found"))
        except Exception as e: print(ui.red(f"  cannot list models: {e}"))

    def c_perms(self, arg):
        if arg:
            try: self.perms.set_mode(arg)
            except ValueError as e: print(ui.red(f"  {e}")); return
        print(f"  mode: {self.perms.mode}   (modes: {', '.join(MODES)})")
        if self.perms.allow or self.perms.session_allow: print(f"  allow: {self.perms.allow + sorted(self.perms.session_allow)}")
        if self.perms.deny: print(f"  deny: {self.perms.deny}")

    def c_plan(self, _):
        if self.perms.mode == "plan": self.perms.set_mode(self.prev_mode)
        else: self.prev_mode = self.perms.mode; self.perms.set_mode("plan")
        print(f"  mode: {self.perms.mode}")

    def c_tools(self, _): print("  " + ", ".join(self.toolbox.names()))
    def c_todos(self, _): print(ui.render_todos(self.state.todos) if self.state.todos else "  (no todos)")
    def c_skills(self, _): [print(f"  {n}: {ui.dim(d)}") for n, (d, _p) in sorted(self.state.skills.items())] or print("  (none)")
    def c_memory(self, _): print("\n".join("  " + f for f in context.memory_files(self.cwd)) or "  (no AGENTS.md / CLAUDE.md / ALICE.md found)")
    def c_mcp(self, _): print("\n".join(f"  {s.name}: {len([n for n in self.toolbox.names() if n.startswith(f'mcp__{s.name}__')])} tools" for s in self.mcp_servers) or "  (no MCP servers)")
    def c_hooks(self, _): print(json.dumps(self.hooks.config, indent=2) if self.hooks.config else "  (no hooks)")
    def c_config(self, _): print(json.dumps(self.settings, indent=2) if self.settings else "  (no settings; ~/.alice/settings.json or ./.alice/settings.json)")

    def c_cost(self, _):
        u = self.llm.usage
        print(f"  {u['calls']} model calls · {u['prompt_tokens']} prompt + {u['completion_tokens']} completion tokens (local model: no cost)")
        if u.get("gen_ms"):
            pr = u["prompt_tokens"] or 1
            print(ui.dim(f"  prompt cache served {u['cached_tokens']} of {pr} prompt tokens ({100 * u['cached_tokens'] // pr}%) · generation {u['completion_tokens'] * 1000 / u['gen_ms']:.0f} tok/s"))

    def c_status(self, _):
        used = self.context_used()
        print(f"  session {self.sid} · {self.llm.model} · {self.perms.mode}\n  ~{used}/{self.llm.num_ctx} context tokens · {len(self.toolbox.names())} tools · cwd {self.state.cwd}")

    def c_resume(self, arg):
        if arg:
            sid = sessions.find(self.cwd, arg.strip()); s = sessions.load(self.cwd, sid) if sid else None
            if not s: print(ui.red("  no such session")); return
            self.agent.messages = self.agent.messages[:1] + [m for m in s["messages"] if m["role"] != "system"]; self.sid = sid
            self.sname = (s.get("meta") or {}).get("name", "")
            print(ui.dim(f"  resumed {sid}" + (f" ({self.sname})" if self.sname else "")))
        else:
            for s in sessions.list_sessions(self.cwd)[:15]: print(f"  {s['id']}  {s['n']:>3} msgs  {ui.bold(s['name']) + '  ' if s['name'] else ''}{ui.dim(s['preview'])}")

    def c_rename(self, arg):
        """/rename <name>: name this session so /resume <name> finds it."""
        name = arg.strip()
        if not name: print(f"  session name: {self.sname or ui.dim('(none)')}"); return
        if (other := sessions.find(self.cwd, name)) and other != self.sid: print(ui.red(f"  '{name}' is already used by session {other}")); return
        self.sname = name; sessions.save(self.cwd, self.sid, self.agent.messages, {"model": self.llm.model, "name": name})
        print(f"  {ui.acc('●')} session named {ui.bold(name)}")

    def c_output_style(self, arg):
        """/output-style [name]: change how Alice writes (default, concise, explanatory, or a file in ~/.alice/output-styles/<name>.md)."""
        names = context.output_styles(); name = arg.strip()
        if not name: print("\n".join(f"  {ui.acc('●') if n == self.style else ' '} {n}" for n in names)); return
        if name not in names: print(ui.red(f"  unknown style '{name}'") + ui.dim("  (" + ", ".join(names) + ")")); return
        self.style = name; self.refresh_system(); print(f"  {ui.acc('●')} output style: {ui.bold(name)}")

    def c_rewind(self, _):
        """Undo the last turn: restore files it changed via Edit/Write/NotebookEdit and drop its messages (Bash side effects are not undone)."""
        if not self.state.turns: print("  nothing to rewind"); return
        t = self.state.turns.pop()
        for p, old in reversed(t["files"]):
            if old is None:
                try: os.remove(p)
                except OSError: pass
            else:
                with open(p, "w") as f: f.write(old)
            self.state.read_files.pop(p, None)
        del self.agent.messages[t["n"]:]
        print(ui.dim(f"  rewound: {len(t['files'])} file(s) restored, conversation back to before that prompt"))

    def c_context(self, _):
        sysm = est_tokens(self.agent.messages[0]["content"]) if self.agent.messages else 0
        tools = est_tokens(json.dumps(self.toolbox.schemas()))
        msgs = sum(est_tokens(m.get("content") or "") + est_tokens(json.dumps(m.get("tool_calls") or "")) for m in self.agent.messages[1:])
        tot, cap = sysm + tools + msgs, self.llm.num_ctx
        print(f"  system prompt ~{sysm} · tools ~{tools} ({len(self.toolbox.names())}) · messages ~{msgs} ({len(self.agent.messages) - 1})\n"
              f"  total ~{tot}/{cap} tokens ({100 * tot // cap}%) — conservative estimate")

    def c_export(self, arg):
        path = arg or f"alice-{self.sid}.md"
        with open(self.state.path(path), "w") as f:
            for m in self.agent.messages[1:]:
                if m["role"] == "user": f.write(f"## You\n{m['content']}\n\n")
                elif m["role"] == "assistant":
                    for c in m.get("tool_calls") or []: f.write(f"**tool** `{c['function']['name']}` {json.dumps(c['function'].get('arguments'))}\n\n")
                    if m.get("content"): f.write(f"## Alice\n{m['content']}\n\n")
                else: f.write(f"```\n{m['content'][:1500]}\n```\n\n")
        print(f"  exported to {self.state.path(path)}")

    def c_doctor(self, _):
        import shutil
        ok = lambda b: ui.green("ok") if b else ui.red("MISSING")
        up, info = self.llm.health()
        try: has_model = any(m == self.llm.model or m.startswith(self.llm.model) for m in self.llm.models()) if up else False
        except Exception: has_model = False
        print(f"  backend {ok(up)} ({info}) · model {ok(has_model)} · rg {ok(shutil.which('rg'))} · git {ok(shutil.which('git'))} · curl {ok(shutil.which('curl'))}\n"
              f"  memory files: {len(context.memory_files(self.cwd))} · skills: {len(self.state.skills)} · commands: {len(self.commands)} · "
              f"agents: {len(self.state.agents)} · mcp: {len(self.mcp_servers)} · search: {'searxng' if os.environ.get('ALICE_SEARXNG_URL') else 'duckduckgo'}")

    def c_review(self, arg):
        self.turn("Review the current uncommitted changes" + (f" ({arg})" if arg else "") + ": run `git diff` (and `git diff --staged`), "
                  "read the touched code where needed, and report concrete bugs, risky edge cases and missing tests with file:line. Do not edit anything.") and print()

    def _on_event(self, e):
        self.renderer.event(e)
        if self.rc and self.rc.server: self.rc.record_event(e)

    def rc_submit(self, text):
        """A prompt from the remote page: injected into the open prompt if idle, else queued as type-ahead."""
        self.rc.record("user", text)
        app = getattr(getattr(self.prompter, "session", None), "app", None)
        if self.at_prompt and app and app.is_running:
            app.loop.call_soon_threadsafe(lambda: app.exit(result=text))
        else: ui.ESC.queue.append(text)

    def rc_interrupt(self):
        import _thread
        if not self.at_prompt: _thread.interrupt_main()      # same as pressing Esc during a turn

    def c_add_dir(self, arg):
        """/add-dir <path>: add another working directory the model may use (listed in its system prompt; no walk-up, no memory loading)."""
        path = os.path.abspath(os.path.expanduser(arg.strip()))
        if not arg.strip(): print(f"  directories: {', '.join([self.cwd, *self.extra_dirs])}"); return
        if not os.path.isdir(path): print(ui.red(f"  not a directory: {path}")); return
        if path != self.cwd and path not in self.extra_dirs: self.extra_dirs.append(path); self.refresh_system()
        print(f"  {ui.acc('●')} added {ui.bold(path)}")

    def c_rc(self, arg):
        """/rc: start remote control (web page to watch and drive this session; Tailscale address if available, else localhost). /rc stop, /rc local."""
        from .rc import RemoteControl
        arg = arg.strip()
        if arg == "stop":
            if self.rc: self.rc.stop()
            print(ui.dim("  remote control off")); return
        if not getattr(self.prompter, "fancy", False): print(ui.red("  remote control needs the interactive prompt")); return
        if self.rc is None: self.rc = RemoteControl(self); atexit.register(self.rc.stop)
        url = self.rc.start(local_only=arg == "local")
        print(f"  {ui.acc('✻')} remote control on: {ui.bold(url)}\n  {ui.dim('anyone with this URL can run commands as you; /rc stop ends it')}")
        if "127.0.0.1" in url: print(ui.dim("  (no Tailscale address found: reachable from this machine only)"))

    def c_selfedit(self, arg):
        """/selfedit <change>: Alice edits her own harness code. Snapshot first; tests must pass or everything is rolled back; then restart into the new code."""
        from . import selfedit
        if not arg: print(ui.red("  usage: /selfedit <what to change about Alice>")); return
        base = selfedit.snapshot()
        self.turn(f"Change your own harness code (the Python package at {selfedit.HARNESS}; tests in {selfedit.HARNESS}/tests) as follows: {arg}\n"
                  "Edit the source files, add or update a unit test for the change, keep every file under 500 lines, and do not touch anything outside that package. "
                  "Stop when done; the system runs the whole test suite and rolls the change back automatically if it fails.")
        files = selfedit.changed(base)
        if not files: print(ui.dim("  no files changed")); return
        print(ui.dim(f"  verifying {len(files)} changed file(s)…"))
        ok, tail = selfedit.verify()
        if not ok:
            selfedit.rollback(base); print(ui.red("  tests failed; rolled back to " + base[:8]) + "\n" + "\n".join("    " + l for l in tail.splitlines())); return
        print(f"  {ui.green('tests pass')}; committed {selfedit.commit(arg)} ({', '.join(os.path.relpath(f, selfedit.ROOT) for f in files[:5])}{' …' if len(files) > 5 else ''})")
        print(ui.dim("  /restart loads the new code (session is kept); /selfedit undo is: git revert that commit"))

    def c_restart(self, _):
        """Relaunch Alice into the current code, keeping this session."""
        from . import selfedit
        sessions.save(self.cwd, self.sid, self.agent.messages, {"model": self.llm.model, "name": self.sname})
        argv = selfedit.restart_argv(sys.argv[1:], self.sid)
        srv = getattr(self.llm, "server", None)
        if srv: srv.save_slot(self.sid); srv.stop()
        for j in self.state.jobs.values():
            if j["proc"].poll() is None:
                try: os.killpg(j["proc"].pid, 15)
                except OSError: pass
        print(ui.dim("  restarting…"), flush=True); os.execv(argv[0], argv)

    def c_security_review(self, arg):
        self.turn("Security-review the current uncommitted changes" + (f" ({arg})" if arg else "") + ": run `git diff`, read the touched code, and report concrete vulnerabilities "
                  "(injection, path traversal, unsafe deserialization, secrets, authz gaps) with file:line, severity and a fix. Do not edit anything.") and print()

    def c_release_notes(self, _):
        out = subprocess.run(["git", "log", "-15", "--format=%h %s"], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True).stdout
        print("\n".join("  " + l for l in out.splitlines()) or ui.dim("  (no history available)"))

    def c_transcript(self, _=""):
        """Full output of every tool call this session, in a pager (Ctrl+O)."""
        text = self.renderer.transcript()
        if ui.CON and sys.stdout.isatty():
            with ui.CON.pager(): print(text)
        else: print(text)

    def _out_tokens(self) -> int: return getattr(self.llm, "usage", {}).get("completion_tokens", 0)

    def c_diff(self, _):
        """Everything the last turn changed through Edit/Write/NotebookEdit, as a diff against the files' earlier contents."""
        import difflib
        if not self.state.turns or not self.state.turns[-1]["files"]: print(ui.dim("  the last turn changed no files")); return
        seen = {}
        for p, old in self.state.turns[-1]["files"]: seen.setdefault(p, old)   # first snapshot = state before the turn
        for p, old in seen.items():
            try: new = open(p, errors="replace").read()
            except OSError: new = ""
            d = "\n".join(difflib.unified_diff((old or "").splitlines(), new.splitlines(), lineterm="", n=2))
            a, r = ui.diff_counts(d)
            print(f"  {ui.bold(os.path.relpath(p, self.cwd))} {ui.green('+' + str(a))} {ui.red('-' + str(r))}{ui.dim(' (new file)') if old is None else ''}")
            print(ui.render_diff(d, "    "))

    def c_copy(self, _):
        """Copy the last answer to the clipboard (wl-copy or xclip)."""
        text = getattr(self, "last_answer", "")
        cmd = next((c for c in (["wl-copy"], ["xclip", "-selection", "clipboard"]) if shutil.which(c[0])), None)
        if not text: print(ui.dim("  nothing to copy yet")); return
        if not cmd: print(ui.red("  no clipboard tool (install wl-clipboard or xclip)")); return
        subprocess.run(cmd, input=text.encode(), timeout=5); print(ui.dim(f"  copied {len(text)} characters"))

    def c_retry(self, _):
        """Undo the last turn and run the same prompt again."""
        if not getattr(self, "last_prompt", ""): print(ui.dim("  nothing to retry")); return
        p = self.last_prompt; self.c_rewind(""); self.turn(p) and print()

    def c_expand(self, arg):
        """Print the full output of the last tool call (or of the Nth most recent: /expand 2)."""
        log = self.renderer.log
        try: n = int(arg or 1)
        except ValueError: print(ui.red("  usage: /expand [n]")); return
        if not log or not 1 <= n <= len(log): print(ui.dim("  no such tool call")); return
        name, args, res = log[-n]
        print(f"  {ui.acc('●')} {ui.bold(name)}({ui.dim(ui.short_args(name, args))})")
        print("\n".join("    " + l for l in res.splitlines() or [""]))

    def c_plugins(self, _):
        ds = context.plugin_dirs(self.cwd)
        if not ds: return print("  (no plugins; add a folder with skills/, commands/ or agents/ under ~/.alice/plugins/<name>/)")
        for d in ds:
            parts = [f"{len(os.listdir(os.path.join(d, k)))} {k}" for k in ("skills", "commands", "agents") if os.path.isdir(os.path.join(d, k))]
            print(f"  {ui.acc('●')} {ui.bold(os.path.basename(d))}  {ui.dim(', '.join(parts) or 'empty')}  {ui.dim(d)}")

    def c_tasks(self, _):
        """Background shells (Ctrl+B): id, status, command and the last line of output."""
        jobs = self.state.jobs
        if not jobs: return print("  (no background shells)")
        for jid, j in jobs.items():
            rc = j["proc"].poll()
            try: last = (open(j["file"], errors="replace").read().strip().splitlines() or [""])[-1][:70]
            except OSError: last = ""
            st = ui.green("running") if rc is None else (ui.dim("done") if rc == 0 else ui.red(f"exit {rc}"))
            print(f"  {ui.acc('●')} {ui.bold(jid)} {st}  {ui.dim(j['cmd'][:60])}" + (f"\n      {ui.dim(last)}" if last else ""))

    def c_kill(self, arg):
        from .bgtools import _kill
        j = self.state.jobs.get(arg.strip())
        if not j: return print(f"  usage: /kill <id>   running: {[k for k, v in self.state.jobs.items() if v['proc'].poll() is None]}")
        _kill(j["proc"]); print(f"  stopped {arg.strip()}")

    def c_theme(self, arg):
        """/theme [name]: accent colour for the bullets, welcome box and prompt; saved to ~/.alice/settings.json."""
        name = arg.strip().lower()
        if not name: return print("  themes: " + ", ".join(ui.THEMES) + f"  (current: {ui.THEME['name']})")
        if not ui.set_theme(name): return print(f"  unknown theme '{name}'; try: " + ", ".join(ui.THEMES))
        if getattr(self, "prompter", None) and self.prompter.fancy: self.prompter = type(self.prompter)(self)
        f = os.path.expanduser("~/.alice/settings.json")
        try:
            cur = json.load(open(f)) if os.path.exists(f) else {}
            cur["theme"] = name; os.makedirs(os.path.dirname(f), exist_ok=True); json.dump(cur, open(f, "w"), indent=2)
        except (OSError, ValueError) as e: print(f"  (could not save: {e})")
        print(f"  {ui.acc('●')} theme: {name}")

    def c_statusline(self, arg):
        from . import statusline
        print("  " + statusline.configure(self.settings, arg))

    def c_vim(self, _):
        print("  " + (self.prompter.vim() if getattr(self, "prompter", None) else "vim mode needs the interactive prompt"))

    def c_agents(self, _):
        print("  general-purpose, explore (built in)")
        for n, a in sorted(self.state.agents.items()): print(f"  {n}: {ui.dim(a['description'])}  tools={a['tools'] or 'all'}")

    def c_init(self, _): self.turn(INIT_PROMPT) and print()

def main(argv=None):
    a = parse(argv)
    if a.root and not os.path.isdir(os.path.expanduser(a.root)): print(f"no such directory: {a.root}", file=sys.stderr); return 1
    try: app = App(a)
    except LLMError as e: print(f"alice: {e}", file=sys.stderr); return 1
    task = a.task or (None if sys.stdin.isatty() or not a.print else sys.stdin.read().strip())
    if task is None and not sys.stdin.isatty(): task = sys.stdin.read().strip() or None
    if task: return app.one_shot(task)
    return app.repl()

if __name__ == "__main__":
    sys.exit(main())
