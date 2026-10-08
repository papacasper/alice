"""Slash commands (/help, /model, /compact, ...): a mixin for cli.App, which supplies llm, agent, state, toolbox, settings."""
import atexit, json, os, shutil, subprocess, sys
from . import context, sessions, ui
from ctxguard import est_tokens
from .llm import LLMError
from .permissions import MODES
from .fsutil import read_json, read_text, write_json

VERSION = "1.0"
INIT_PROMPT = ("Analyze this codebase with your tools (README, package manifests, entry points, tests) and create a concise "
               "ALICE.md in the working directory: build/test/run commands, architecture, and conventions. If one exists, improve it.")

class Commands:
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
            try: new = read_text(p, errors="replace")
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
            try: last = (read_text(j["file"], errors="replace").strip().splitlines() or [""])[-1][:70]
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
            cur = read_json(f) if os.path.exists(f) else {}
            cur["theme"] = name; os.makedirs(os.path.dirname(f), exist_ok=True); write_json(f, cur)
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
