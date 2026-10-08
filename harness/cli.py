"""The Claude Code–style CLI: REPL with slash commands, permissions, hooks, MCP, sessions, subagents."""
import argparse, atexit, contextlib, glob, json, os, re, shutil, sys, time
from . import profiles, context, sessions, ui
from .agent import Agent
from .cctools import READ_TOOLS, State, cc_toolbox
from .hooks import Hooks
from .prompt import Prompter
from .llm import DEFAULT_MODEL, LLMError, LlamaServer, OllamaClient, OpenAIClient
from .mcp import McpServer
from .permissions import MODES, Permissions
from .tools import Toolbox
from .fsutil import read_bytes, read_text
from .commands import VERSION, Commands


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

class App(Commands):
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
        override = read_text(os.path.expanduser(self.a.system)) if self.a.system else ""
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
        self.pending_images.append(base64.b64encode(read_bytes(p)).decode()); return ""

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
