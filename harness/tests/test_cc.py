import json, os, sys, tempfile, textwrap, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import Agent, context, sessions
from harness.cctools import State, cc_toolbox
from harness.cli import App, parse
from harness.hooks import Hooks
from harness.mcp import McpServer
from harness.permissions import Permissions
from test_harness import FakeLLM, call, say
from harness.fsutil import read_json, read_text, write_file, write_json

def tmp(): return os.path.realpath(tempfile.mkdtemp())

class TestCcTools(unittest.TestCase):
    def setUp(self):
        self.d = tmp(); self.st = State(self.d); self.tb = cc_toolbox(self.st)
        with open(os.path.join(self.d, "a.py"), "w") as f: f.write("x = 1\ny = 2\nx = 1\n")
    def c(self, _tool, **a): return self.tb.call(_tool, a)

    def test_read_numbered_and_paged(self):
        out = self.c("Read", file_path="a.py", limit=2)
        self.assertIn("1\tx = 1", out); self.assertIn("offset=3", out)
        self.assertIn("does not exist", self.c("Read", file_path="nope"))
    def test_edit_requires_read_then_unique(self):
        self.assertIn("has not been Read", self.c("Edit", file_path="a.py", old_string="y = 2", new_string="y = 3"))
        self.c("Read", file_path="a.py")
        self.assertIn("2 times", self.c("Edit", file_path="a.py", old_string="x = 1", new_string="x = 9"))
        self.assertIn("not found", self.c("Edit", file_path="a.py", old_string="zzz", new_string="q"))
        self.assertIn("identical", self.c("Edit", file_path="a.py", old_string="y", new_string="y"))
        self.assertIn("Edited", self.c("Edit", file_path="a.py", old_string="x = 1", new_string="x = 9", replace_all=True))
        self.assertEqual(read_text(os.path.join(self.d, "a.py")), "x = 9\ny = 2\nx = 9\n")
        self.assertIn("+x = 9", self.st.ui_extra)
    def test_line_counts_are_full_even_when_the_diff_is_cut(self):
        self.c("Write", file_path="big.txt", content="".join(f"l{i}\n" for i in range(100)))
        self.assertEqual(self.st.lines, [100, 0])
        self.c("Read", file_path="big.txt")
        self.c("Write", file_path="big.txt", content="".join(f"m{i}\n" for i in range(100)))
        self.assertEqual((self.st.ui_counts, self.st.lines), ((100, 100), [200, 100]))
        self.assertEqual(len(self.st.ui_extra.splitlines()), 60)
        self.c("Edit", file_path="big.txt", old_string="m5\n", new_string="")
        self.assertEqual((self.st.ui_counts, self.st.lines), ((0, 1), [200, 101]))
    def test_edit_detects_external_change(self):
        self.c("Read", file_path="a.py"); p = os.path.join(self.d, "a.py")
        write_file(p, "w", "changed\n"); os.utime(p, (1, 1))
        self.assertIn("changed on disk", self.c("Edit", file_path="a.py", old_string="changed", new_string="z"))
    def test_write_new_and_overwrite_rules(self):
        self.assertIn("Created", self.c("Write", file_path="sub/n.txt", content="hi"))
        self.assertIn("not been Read", self.c("Write", file_path="a.py", content="no"))
        self.c("Read", file_path="a.py"); self.assertIn("Overwrote", self.c("Write", file_path="a.py", content="new\n"))
    def test_bash_persists_cwd_and_reports_exit(self):
        os.mkdir(os.path.join(self.d, "sub"))
        self.c("Bash", command="cd sub"); self.assertEqual(self.st.cwd, os.path.join(self.d, "sub"))
        self.assertEqual(self.c("Bash", command="pwd"), os.path.join(self.d, "sub"))
        self.assertTrue(self.c("Bash", command="echo oops >&2; exit 3").startswith("exit code 3"))
        self.assertIn("timed out", self.c("Bash", command="sleep 5", timeout=1))
    def test_long_read_pages_with_hint_and_reaches_the_end(self):
        lines = [f"line {i}: the quick brown fox jumps over the lazy dog, entry {i * 7} of the ledger." for i in range(300)]
        lines[150] = "The vault codeword is PELICAN-4471."
        write_file(os.path.join(self.d, "big.txt"), "w", "\n".join(lines))
        r, seen, off = self.c("Read", file_path="big.txt"), "", 1
        for _ in range(10):
            seen += r; self.assertNotIn("output cut", r)              # never the generic mid-output cut
            m = __import__("re").search(r"offset=(\d+)\]$", r)
            if not m: break
            r = self.c("Read", file_path="big.txt", offset=int(m.group(1)))
        self.assertIn("PELICAN-4471", seen); self.assertIn("  300\t", seen)
    def test_glob_and_grep(self):
        write_file(os.path.join(self.d, "b.txt"), "w", "hello\n")
        self.assertIn("a.py", self.c("Glob", pattern="**/*.py")); self.assertEqual(self.c("Glob", pattern="*.zip"), "No files found")
        for bad in ("**/.py", ".py"):   # bare extension: point at the glob the model meant (it once concluded "no .tmp files" from '**/.tmp')
            r = self.c("Glob", pattern=bad); self.assertIn("Did you mean '**/*.py'", r); self.assertIn("a.py", r)
        self.assertEqual(self.c("Glob", pattern="**/.zip"), "No files found")
        self.assertIn("a.py", self.c("Grep", pattern="y = 2"))
        self.assertIn("a.py:2:y = 2", self.c("Grep", pattern="y = 2", output_mode="content"))
        self.assertEqual(self.c("Grep", pattern="nomatch"), "No matches found")
    def test_todowrite_and_skill_and_task(self):
        self.c("TodoWrite", todos=[{"content": "a", "status": "in_progress"}, "b", {"content": "c", "status": "bogus"}])
        self.assertEqual([t["status"] for t in self.st.todos], ["in_progress", "pending", "pending"])
        self.assertIn("unknown skill", self.c("Skill", name="x"))
        self.assertIn("not available", self.c("Task", description="d", prompt="p"))
        self.st.spawn = lambda p, k: f"{k}:{p}"; self.assertEqual(self.c("Task", description="d", prompt="p"), "general-purpose:p")

class TestPermissions(unittest.TestCase):
    def test_modes(self):
        p = Permissions("default"); self.assertIn("no terminal", p.check("Bash", {"command": "ls"}))
        self.assertIsNone(p.check("Read", {"file_path": "x"}))
        self.assertIsNone(Permissions("bypassPermissions").check("Bash", {"command": "ls"}))
        a = Permissions("acceptEdits"); self.assertIsNone(a.check("Edit", {})); self.assertIn("needs permission", a.check("Bash", {"command": "ls"}))
        self.assertIn("plan mode", Permissions("plan").check("Write", {}))
    def test_rules_and_ask(self):
        p = Permissions("default", allow=["Bash(git status:*)"], deny=["Bash(rm:*)"], ask=lambda n, a: "no")
        self.assertIsNone(p.check("Bash", {"command": "git status -s"}))
        self.assertIn("denied by a permission rule", p.check("Bash", {"command": "rm -rf x"}))
        self.assertIn("user denied", p.check("Bash", {"command": "ls"}))
        p.ask = lambda n, a: "always"; self.assertIsNone(p.check("Bash", {"command": "ls"})); self.assertIsNone(p.check("Bash", {"command": "pwd"}))
        self.assertIsNone(Permissions("default", allow=["mcp__srv"]).check("mcp__srv__t", {}))

class TestHooks(unittest.TestCase):
    def test_block_and_context(self):
        cfg = {"PreToolUse": [{"matcher": "Bash", "hooks": [{"command": "echo nope >&2; exit 2"}]}],
               "UserPromptSubmit": [{"hooks": [{"command": "echo extra-context"}]}]}
        h = Hooks(cfg, tmp())
        self.assertEqual(h.run("PreToolUse", "Bash", {}), (True, "nope")); self.assertEqual(h.run("PreToolUse", "Read", {}), (False, ""))
        self.assertEqual(h.run("UserPromptSubmit", "", {}), (False, "extra-context"))

SERVER = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        m = json.loads(line); i = m.get("id")
        if m["method"] == "initialize": r = {"protocolVersion": "2024-11-05", "capabilities": {}}
        elif m["method"] == "tools/list": r = {"tools": [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
        elif m["method"] == "tools/call": r = {"content": [{"type": "text", "text": "echo:" + m["params"]["arguments"]["text"]}]}
        else: continue
        print(json.dumps({"jsonrpc": "2.0", "id": i, "result": r}), flush=True)
''')

class TestMcp(unittest.TestCase):
    def test_roundtrip(self):
        p = os.path.join(tmp(), "srv.py"); write_file(p, "w", SERVER)
        s = McpServer("demo", sys.executable, [p], timeout=10)
        try:
            [t] = s.tools(); self.assertEqual(t.name, "mcp__demo__echo")
            self.assertEqual(t.schema()["function"]["parameters"]["required"], ["text"])
            self.assertEqual(t.call({"text": "hi"}), "echo:hi")
        finally: s.close()

class TestContext(unittest.TestCase):
    def test_frontmatter_memory_commands(self):
        meta, body = context.frontmatter("---\nname: x\ndescription: |\n  Does things\n---\nBody")
        self.assertEqual((meta["name"], meta["description"], body), ("x", "Does things", "Body"))
        d = tmp(); os.makedirs(os.path.join(d, ".alice", "commands"))
        write_file(os.path.join(d, "CLAUDE.md"), "w", "RULE-ONE")
        write_file(os.path.join(d, ".alice", "commands", "hi.md"), "w", "---\ndescription: greet\n---\nSay hi to $ARGUMENTS")
        self.assertIn("RULE-ONE", context.build_system(d, "m", {}))
        self.assertEqual(context.discover_commands(d)["hi"], ("greet", "Say hi to $ARGUMENTS"))
        self.assertIn(d, context.env_block(d, "m"))

class TestAgentGate(unittest.TestCase):
    def test_gate_denial_and_post(self):
        st = State(tmp()); tb = cc_toolbox(st)
        llm = FakeLLM([call("Bash", command="echo hi"), say("done")])
        r = Agent(llm, tb, gate=lambda n, a: "error: nope", post=lambda n, a, o: "x").run("go")
        self.assertEqual([m for m in r.messages if m["role"] == "tool"][0]["content"], "error: nope")
        llm = FakeLLM([call("Bash", command="echo hi"), say("done")])
        r = Agent(llm, tb, post=lambda n, a, o: "[extra]").run("go")
        self.assertEqual([m for m in r.messages if m["role"] == "tool"][0]["content"], "hi\n[extra]")
    def test_compact_and_reset(self):
        llm = FakeLLM([say("a"), say("SUMMARY")]); ag = Agent(llm, cc_toolbox(State(tmp())), "SYS"); ag.ask("one")
        ag.compact(); self.assertIn("SUMMARY", ag.messages[1]["content"]); self.assertEqual(len(ag.messages), 3)
        ag.reset(); self.assertEqual([m["role"] for m in ag.messages], ["system"])

class TestApp(unittest.TestCase):
    def make(self, script, *argv):
        d = tmp(); os.environ["HOME"] = os.environ.get("HOME", "")
        a = parse(["--root", d, "--no-mcp", "--no-stream", "-q", *argv, "task"]); llm = FakeLLM(script); llm.model, llm.usage = "fake", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        return App(a, llm), d
    def test_one_shot_edit_flow_and_session(self):
        app, d = self.make([call("Write", file_path="n.txt", content="hi"), say("created")])
        self.assertEqual(app.one_shot("make n.txt"), 0)
        self.assertEqual(read_text(os.path.join(d, "n.txt")), "hi")
        self.assertEqual(sessions.load(d, app.sid)["messages"][1]["content"], "make n.txt")
    def test_one_shot_stdout_is_only_the_answer(self):
        import io, contextlib
        for fmt, want in (("json", None), ("text", "the answer\n")):
            app, d = self.make([say("the answer")], "--output-format", fmt); buf = io.StringIO()
            with contextlib.redirect_stdout(buf): app.one_shot("q")
            if want is None: self.assertEqual(json.loads(buf.getvalue())["result"], "the answer")   # whole stdout must parse
            else: self.assertEqual(buf.getvalue(), want)                                           # printed once, not twice
    def test_subagent_runs_in_its_own_slot(self):
        slots = []
        def rec(v): return lambda m: (slots.append(app.llm.slot), v)[1]
        app, d = self.make([rec(call("Task", description="x", prompt="look", subagent_type="explore")), rec(say("found")), rec(say("ok"))])
        app.llm.slot = 0; app.one_shot("go")
        self.assertEqual(slots, [0, 1, 0]); self.assertEqual(app.llm.slot, 0)
    def test_default_mode_without_tty_denies_shell(self):
        app, d = self.make([call("Bash", command="touch x"), say("denied")], "--permission-mode", "default")
        app.one_shot("go"); self.assertFalse(os.path.exists(os.path.join(d, "x")))
    def test_plan_mode_blocks_then_slash_commands(self):
        app, d = self.make([call("Write", file_path="n.txt", content="hi"), say("blocked")], "--permission-mode", "plan")
        app.one_shot("go"); self.assertFalse(os.path.exists(os.path.join(d, "n.txt")))
        app.slash("/permissions acceptEdits"); self.assertEqual(app.perms.mode, "acceptEdits")
        app.slash("/plan"); self.assertEqual(app.perms.mode, "plan"); app.slash("/plan"); self.assertEqual(app.perms.mode, "acceptEdits")
        app.slash("/clear"); self.assertEqual(len(app.agent.messages), 1)
    def test_resume_loads_history(self):
        app, d = self.make([say("hello")]); app.one_shot("first")
        a2 = parse(["--root", d, "--no-mcp", "--no-stream", "-q", "--continue", "more"]); l2 = FakeLLM([]); l2.model = "fake"; app2 = App(a2, l2)
        self.assertEqual([m["content"] for m in app2.agent.messages if m["role"] == "user"], ["first"])
    def test_custom_command(self):
        app, d = self.make([say("hey bob")])
        os.makedirs(os.path.join(d, ".alice", "commands")); write_file(os.path.join(d, ".alice", "commands", "hi.md"), "w", "Say hi to $ARGUMENTS")
        app.commands = context.discover_commands(d); app.slash("/hi bob")
        self.assertEqual([m["content"] for m in app.agent.messages if m["role"] == "user"], ["Say hi to bob"])

class TestSystemPrompt(unittest.TestCase):
    def test_shipped_prompt_has_sections_and_fits_budget(self):
        from harness import context
        t = read_text(os.path.join(os.path.dirname(context.__file__), "system_prompt.md"))
        for h in ("# Doing what the user wants", "# Tone and style", "# Working method", "# Tool use", "# Care with the irreversible", "# Git and references"): self.assertIn(h, t)
        self.assertLess(len(t) / 3.5, 1100)   # tokens: it is resent every request on a 12k context
        self.assertEqual(context.PROMPT.split(".")[0][:15], "You are Alice, ")

    def test_user_override(self):
        import importlib
        from harness import context
        old = context.HOME; d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, ".alice"))
        write_file(os.path.join(d, ".alice", "SYSTEM.md"), "w", "CUSTOM GLOBAL PROMPT")
        try:
            context.HOME = d; self.assertEqual(context._load_prompt(), "CUSTOM GLOBAL PROMPT")
        finally: context.HOME = old


class TestMemoryFiles(unittest.TestCase):
    def test_project_dir_only_three_names_no_walk_up(self):
        from harness import context
        top = tempfile.mkdtemp(); sub = os.path.join(top, "proj"); os.mkdir(sub)
        write_file(os.path.join(top, "CLAUDE.md"), "w", "PARENT")                      # must NOT be loaded
        for n in ("AGENTS.md", "CLAUDE.md", "ALICE.md"): write_file(os.path.join(sub, n), "w", n)
        got = [os.path.basename(f) for f in context.memory_files(sub) if f.startswith(sub)]
        self.assertEqual(got, ["AGENTS.md", "CLAUDE.md", "ALICE.md"])
        self.assertFalse(any(f.startswith(top) and not f.startswith(sub) for f in context.memory_files(sub)))
        self.assertNotIn("PARENT", context.memory_text(sub))

    def test_note_target(self):
        from harness import context
        d = tempfile.mkdtemp(); self.assertEqual(os.path.basename(context.memory_target(d)), "ALICE.md")
        open(os.path.join(d, "AGENTS.md"), "w").close(); self.assertEqual(os.path.basename(context.memory_target(d)), "AGENTS.md")
        open(os.path.join(d, "ALICE.md"), "w").close(); self.assertEqual(os.path.basename(context.memory_target(d)), "ALICE.md")


class TestSlimSchemas(unittest.TestCase):
    def _schemas(self):
        from harness.cctools import State, cc_toolbox
        return {t["function"]["name"]: t["function"] for t in cc_toolbox(State(tempfile.mkdtemp())).schemas()}

    def test_every_slim_entry_matches_a_real_tool_and_params(self):
        from harness.slim import SLIM
        real = self._schemas()
        for name, (desc, params) in SLIM.items():
            if name in ("calculate", "today"): continue   # builtin tools live outside cc_toolbox
            self.assertIn(name, real); self.assertLessEqual(set(params), set(real[name]["parameters"]["properties"]), name)

    def test_slim_is_smaller_and_keeps_required(self):
        import json
        slim = self._schemas(); os.environ["ALICE_FULL_SCHEMAS"] = "1"
        try: full = self._schemas()
        finally: del os.environ["ALICE_FULL_SCHEMAS"]
        self.assertLess(len(json.dumps(slim)), 0.8 * len(json.dumps(full)))
        for n in slim: self.assertEqual(slim[n]["parameters"]["required"], full[n]["parameters"]["required"])
        self.assertNotIn("description", slim["Read"]["parameters"]["properties"]["file_path"])


class TestImages(unittest.TestCase):
    PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    def test_to_openai_image_parts(self):
        from harness.llm import to_openai
        m = to_openai([{"role": "user", "content": "what is this", "images": [self.PNG]}])[0]["content"]
        self.assertEqual(m[0], {"type": "text", "text": "what is this"})
        self.assertTrue(m[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_attach_and_mention(self):
        import base64
        d = tempfile.mkdtemp(); write_file(os.path.join(d, "a.png"), "wb", base64.b64decode(self.PNG)); write_file(os.path.join(d, "n.txt"), "w", "x")
        l = FakeLLM([]); l.model = "fake"; app = App(parse(["--root", d, "--no-mcp"]), l)
        self.assertEqual(app.attach_image("a.png"), ""); self.assertIn("not an image", app.attach_image("n.txt"))
        app.pending_images.clear(); out = app.expand_mentions("look @a.png please")
        self.assertEqual(len(app.pending_images), 1); self.assertNotIn("Contents of", out)

    def test_ask_stores_images_and_projector_lookup(self):
        from harness.agent import Agent
        from harness.llm import ollama_blob
        self.assertEqual(ollama_blob("hf.co/u/missing:Q4", "projector"), "")


class TestPlugins(unittest.TestCase):
    def test_plugin_commands_and_skills_discovered(self):
        from harness import context
        d = tempfile.mkdtemp(); pl = os.path.join(d, ".alice", "plugins", "demo")
        os.makedirs(os.path.join(pl, "commands")); os.makedirs(os.path.join(pl, "skills", "greet"))
        write_file(os.path.join(pl, "commands", "hi.md"), "w", "---\ndescription: say hi\n---\nSay hi to $ARGUMENTS")
        write_file(os.path.join(pl, "skills", "greet", "SKILL.md"), "w", "---\nname: greet\ndescription: greets\n---\nbody")
        self.assertEqual([os.path.basename(x) for x in context.plugin_dirs(d)], ["demo"])
        self.assertIn("hi", context.discover_commands(d)); self.assertIn("greet", context.discover_skills(d))


class TestTurnExtras(unittest.TestCase):
    def test_turn_summary_format(self):
        from harness.ui import turn_summary
        self.assertEqual(turn_summary(72, 3, 1234), "✻ Worked for 1m 12s · 3 tool calls · 1.2k tokens")
        self.assertEqual(turn_summary(9, 1, 80), "✻ Worked for 9s · 1 tool call · 80 tokens")

    def test_copy_and_retry(self):
        import io, contextlib
        from harness.cli import App, parse
        l = FakeLLM([say("first"), say("second")]); l.model = "fake"
        a = App(parse(["--root", tempfile.mkdtemp(), "--no-mcp", "--no-stream"]), l)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a.c_copy(""); a.c_retry("")                      # nothing yet
            a.turn("hello"); n = len(a.agent.messages); a.c_retry("")
        self.assertIn("nothing to copy", buf.getvalue()); self.assertIn("nothing to retry", buf.getvalue())
        self.assertEqual(a.last_prompt, "hello"); self.assertEqual(len(a.agent.messages), n)   # retry replaced, not appended
        self.assertEqual(a.last_answer, "second")


class TestModelSwitch(unittest.TestCase):
    FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_llama_server.py")

    def test_switch_keeps_port_and_rolls_back_on_failure(self):
        import urllib.request, json
        from harness.llm import LlamaServer, LLMError
        srv = LlamaServer("hf.co/x/one:Q4", 2048, binary=self.FAKE, log=os.path.join(tempfile.mkdtemp(), "l.log"))
        alias = lambda: json.load(urllib.request.urlopen(srv.host + "/v1/models"))["data"][0]["id"]
        try:
            host = srv.start(wait=20); self.assertEqual(alias(), "hf.co/x/one:Q4")
            self.assertEqual(srv.switch_model("hf.co/x/two:Q4"), host); self.assertEqual(alias(), "hf.co/x/two:Q4")
            with self.assertRaises(LLMError): srv.switch_model("hf.co/x/BAD:Q4")
            self.assertEqual(srv.model, "hf.co/x/two:Q4"); self.assertEqual(alias(), "hf.co/x/two:Q4")   # old model is back
        finally: srv.stop()

    def test_revive_after_server_dies(self):
        import urllib.request, signal
        from harness.llm import LlamaServer
        srv = LlamaServer("hf.co/x/one:Q4", 2048, binary=self.FAKE, log=os.path.join(tempfile.mkdtemp(), "l.log"))
        try:
            srv.start(wait=20); self.assertFalse(srv.revive())          # healthy: nothing to do
            srv.proc.kill(); srv.proc.wait()
            self.assertTrue(srv.revive()); self.assertEqual(urllib.request.urlopen(srv.host + "/health").status, 200)
        finally: srv.stop()

    def test_installed_models_reads_manifests(self):
        from harness.llm import installed_models
        st = tempfile.mkdtemp(); blob = os.path.join(st, "blobs", "sha256-aa"); os.makedirs(os.path.dirname(blob)); open(blob, "w").close()
        for rel in ("registry.ollama.ai/library/qwen3/8b", "hf.co/user/repo/Q4_K_M"):
            f = os.path.join(st, "manifests", rel); os.makedirs(os.path.dirname(f))
            write_json(f, {"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:aa"}]})
        old = os.environ.get("OLLAMA_MODELS"); os.environ["OLLAMA_MODELS"] = st
        try: got = installed_models()
        finally: os.environ.pop("OLLAMA_MODELS") if old is None else os.environ.update(OLLAMA_MODELS=old)
        self.assertIn("qwen3:8b", got); self.assertIn("hf.co/user/repo:Q4_K_M", got)

    def test_model_completion(self):
        from harness.prompt import complete_candidates
        got = complete_candidates("/model qw", "/", None, lambda: ["qwen3:8b", "gemma4:latest"])
        self.assertEqual([g[0] for g in got], ["qwen3:8b"])


class TestAddDir(unittest.TestCase):
    def test_add_dir_listed_in_system_prompt(self):
        import types
        from harness import cli
        app = types.SimpleNamespace(cwd="/w", extra_dirs=[], refreshed=0)
        app.refresh_system = lambda: setattr(app, "refreshed", app.refreshed + 1)
        d = tempfile.mkdtemp()
        cli.App.c_add_dir(app, d); cli.App.c_add_dir(app, d); cli.App.c_add_dir(app, "/nonexistent-zz")
        self.assertEqual(app.extra_dirs, [d]); self.assertEqual(app.refreshed, 1)


class TestRenameAndStyle(unittest.TestCase):
    def test_rename_find_and_styles(self):
        from harness import sessions
        cwd = tempfile.mkdtemp(); old = sessions.ROOT; sessions.ROOT = tempfile.mkdtemp()
        try:
            sessions.save(cwd, "20260101-000000", [{"role": "user", "content": "hi"}], {"name": "Auth Work"})
            self.assertEqual(sessions.find(cwd, "auth work"), "20260101-000000"); self.assertEqual(sessions.find(cwd, "20260101-000000"), "20260101-000000")
            self.assertEqual(sessions.find(cwd, "nope"), ""); self.assertEqual(sessions.list_sessions(cwd)[0]["name"], "Auth Work")
        finally: sessions.ROOT = old
        self.assertIn("concise", context.output_styles()); self.assertIn("as few words", context.output_style("concise"))
        self.assertEqual(context.output_style("default"), ""); self.assertEqual(context.output_style("missing-style"), "")

    def test_batch_event_only_for_multiple_calls(self):
        ev = []
        tb2 = cc_toolbox(State(tempfile.mkdtemp()))
        llm = FakeLLM([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "TodoWrite", "arguments": {"todos": []}}}] * 2}, say("ok")])
        Agent(llm, tb2, "", 5, ev.append).ask("go")
        self.assertEqual([e["n"] for e in ev if e["type"] == "tool_batch"], [2])


class TestEvalSuite(unittest.TestCase):
    def test_every_check_fails_untouched_and_passes_with_the_reference_solution(self):
        import subprocess
        run = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evals", "run.py")
        p = subprocess.run([sys.executable, run, "--selftest"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


class TestReasoningOnlyTurn(unittest.TestCase):
    def _serve(self, stream_chunks, plain):
        import http.server, threading
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n)); self.send_response(200)
                if body.get("stream"):
                    self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    for c in stream_chunks: self.wfile.write(("data: " + json.dumps({"choices": [{"delta": c}]}) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(plain).encode())
        srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_port}"

    def test_answer_that_only_exists_in_reasoning_is_returned(self):
        from harness.llm import OpenAIClient
        host = self._serve([{"reasoning_content": "x.log has "}, {"reasoning_content": "3 lines."}],
                           {"choices": [{"message": {"content": None, "reasoning_content": "3 lines."}}]})
        c = OpenAIClient("m", host=host)
        self.assertEqual(c.chat([{"role": "user", "content": "q"}], on_token=lambda t: None)["content"], "x.log has 3 lines.")
        self.assertEqual(c.chat([{"role": "user", "content": "q"}])["content"], "3 lines.")

    def test_real_content_wins_over_reasoning(self):
        from harness.llm import OpenAIClient
        host = self._serve([{"reasoning_content": "hmm"}, {"content": "42"}], {"choices": [{"message": {"content": "42", "reasoning_content": "hmm"}}]})
        self.assertEqual(OpenAIClient("m", host=host).chat([{"role": "user", "content": "q"}], on_token=lambda t: None)["content"], "42")


class TestBackendTuning(unittest.TestCase):
    FAKE = TestModelSwitch.FAKE

    def test_server_flags_and_settings(self):
        from harness import profiles
        f = profiles.server_flags({}); self.assertIn("q8_0", f); self.assertEqual(f[f.index("--spec-type") + 1], "ngram-mod")
        f = profiles.server_flags({"kvCache": "f16", "speculative": "none"}); self.assertNotIn("--spec-type", f); self.assertEqual(f[f.index("-ctk") + 1], "f16")
        f = profiles.server_flags({"draftModel": "/m/d.gguf"}); self.assertEqual(f[f.index("-md") + 1], "/m/d.gguf"); self.assertTrue(f[f.index("--spec-type") + 1].startswith("draft-simple"))

    def test_sampling_and_thinking_rules(self):
        from harness import profiles
        self.assertEqual(profiles.sampling("hf.co/x/Qwen3.5:Q4", False)["top_k"], 20); self.assertEqual(profiles.sampling("unknown", False), {})
        self.assertEqual(profiles.sampling("qwen3", False, {"temperature": 0.2})["temperature"], 0.2)
        W = profiles.want_thinking
        self.assertTrue(W("auto", "why is this test failing?", 1)); self.assertFalse(W("auto", "why is this test failing?", 2))
        self.assertFalse(W("auto", "list the files", 1)); self.assertTrue(W("auto", "x" * 500, 1)); self.assertTrue(W("on", "hi", 5)); self.assertFalse(W("off", "why", 1))

    def test_start_falls_back_when_tuned_flags_break_the_server(self):
        from harness.llm import LlamaServer
        srv = LlamaServer("m", 2048, binary=self.FAKE, log=os.path.join(tempfile.mkdtemp(), "l.log"), auto=["--refuse-me"])
        try:
            srv.start(wait=20, say=lambda m: None); self.assertEqual(srv.auto, [])
        finally: srv.stop()

    def test_slot_dir_is_created_if_missing(self):
        from harness.llm import LlamaServer
        d = os.path.join(tempfile.mkdtemp(), "deep", "slots"); LlamaServer("m", 2048, binary=self.FAKE, slot_dir=d)
        self.assertTrue(os.path.isdir(d))

    def test_slot_snapshot_roundtrip(self):
        from harness.llm import LlamaServer
        d = tempfile.mkdtemp(); srv = LlamaServer("m", 2048, binary=self.FAKE, log=os.path.join(d, "l.log"), slot_dir=os.path.join(d, "slots"))
        try:
            from harness.llm import OpenAIClient
            host = srv.start(wait=20); self.assertFalse(srv.restore_slot("s1"))
            size = lambda: os.path.getsize(os.path.join(d, "slots", srv.slot_name("s1")))
            c = OpenAIClient("m", host=host); c.chat([{"role": "user", "content": "hi"}])
            srv.save_slot("s1"); self.assertEqual(size(), 0)        # unpinned: ran in another slot, the snapshot of slot 0 is empty
            c.slot = 0; c.chat([{"role": "user", "content": "hi"}])  # what App sets for its own server
            self.assertTrue(srv.save_slot("s1")); self.assertGreater(size(), 0)
            self.assertTrue(srv.restore_slot("s1")); self.assertFalse(srv.restore_slot("other"))
        finally: srv.stop()

    def test_prune_slots_caps_total_size_oldest_first(self):
        from harness.llm import LlamaServer
        d = tempfile.mkdtemp(); srv = LlamaServer("m", 2048, binary=self.FAKE, slot_dir=d)
        for i in range(5):
            f = os.path.join(d, f"s{i}.bin"); write_file(f, "wb", b"x" * 100); os.utime(f, (1e9 + i, time.time() - 3600 + i))
        srv.prune_slots(max_bytes=250)
        self.assertEqual(sorted(os.listdir(d)), ["s3.bin", "s4.bin"])
    def test_recurrent_models_skip_slot_snapshots(self):
        from harness.llm import is_recurrent
        d = tempfile.mkdtemp(); hy, tf = os.path.join(d, "hy.gguf"), os.path.join(d, "tf.gguf")
        write_file(hy, "wb", b"GGUF....qwen35.ssm.state_size...."); write_file(tf, "wb", b"GGUF....qwen3.attention.head_count....")
        self.assertTrue(is_recurrent(hy)); self.assertFalse(is_recurrent(tf)); self.assertFalse(is_recurrent("hf.co/u/r:Q4"))
    def test_leaked_tool_calls_are_recovered(self):
        from harness.llm import leaked_tool_calls
        xml = "Let me look.\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/tmp/b.py\n</parameter>\n</function>\n</tool_call>"   # verbatim shape from a think=on eval
        calls, rest = leaked_tool_calls(xml)
        self.assertEqual(calls, [{"function": {"name": "Read", "arguments": {"file_path": "/tmp/b.py"}}}]); self.assertEqual(rest, "Let me look.")
        calls, _ = leaked_tool_calls('<tool_call>{"name": "Glob", "arguments": {"pattern": "*.py"}}</tool_call>')
        self.assertEqual(calls[0]["function"], {"name": "Glob", "arguments": {"pattern": "*.py"}})
        self.assertEqual(leaked_tool_calls("talking about <tool_call> tags"), ([], "talking about <tool_call> tags"))
    def test_tool_argument_repair(self):
        from harness.tools import tool, Toolbox
        @tool
        def put(file_path: str, items: list) -> str:
            """Store.

            Args:
                file_path: where
                items: what
            """
            return f"{file_path}:{len(items)}"
        tb = Toolbox([put])
        self.assertEqual(tb.call("put", {"file": "a.txt", "items": '["x","y"]'}), "a.txt:2")
        self.assertIn("unknown argument", tb.call("put", {"zzz": 1, "file_path": "a", "items": []}))
        self.assertIn("did you mean 'put'", tb.call("putt", {}))

    def test_read_only_calls_run_in_parallel_in_order(self):
        import time as _t
        from harness import Toolbox, tool
        @tool
        def Read(path: str) -> str:
            """Read a file."""
            _t.sleep(0.4); return "r:" + path
        ev = []
        llm = FakeLLM([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Read", "arguments": {"path": p}}} for p in "abc"]}, say("ok")])
        t0 = _t.time(); Agent(llm, Toolbox([Read]), "", 5, ev.append).ask("go"); took = _t.time() - t0
        self.assertLess(took, 1.0)       # three 0.4 s reads, sequentially 1.2 s
        self.assertEqual([e["result"] for e in ev if e["type"] == "tool"], ["r:a", "r:b", "r:c"])


class TestRemoteControl(unittest.TestCase):
    def _rc(self):
        from harness.rc import RemoteControl
        class A: at_prompt = True; cwd = "/x"; sent = []; llm = type("L", (), {"model": "m/qwen"})()
        A.rc_submit = lambda self, t: self.sent.append(t); A.rc_interrupt = lambda self: self.sent.append("<stop>")
        a = A(); return a, RemoteControl(a)

    def test_token_gate_state_and_send(self):
        import urllib.request, urllib.error, json
        a, rc = self._rc(); url = rc.start(local_only=True)
        try:
            self.assertTrue(url.startswith("http://127.0.0.1:"))
            base = url.rsplit("/", 2)[0]
            for bad in (base + "/", base + "/wrongtoken/", base + "/wrongtoken/state"):
                with self.assertRaises(urllib.error.HTTPError) as c: urllib.request.urlopen(bad)
                self.assertEqual(c.exception.code, 404)
            self.assertIn("Alice", urllib.request.urlopen(url).read().decode())
            rc.record("user", "hi"); rc.record_event({"type": "tool_call", "name": "Bash", "args": {"command": "ls"}}); rc.record_event({"type": "answer", "text": "done"})
            st = json.load(urllib.request.urlopen(url + "state?since=1"))
            self.assertEqual([e["kind"] for e in st["events"]], ["call", "answer"]); self.assertFalse(st["busy"]); self.assertEqual(st["model"], "qwen")
            post = lambda p, b: urllib.request.urlopen(urllib.request.Request(url + p, json.dumps(b).encode(), method="POST"))
            post("send", {"text": "  do it  "}); post("stop", {})
            with self.assertRaises(urllib.error.HTTPError) as c: post("send", {"text": " "})
            self.assertEqual(c.exception.code, 400); self.assertEqual(a.sent, ["do it", "<stop>"])
        finally: rc.stop()
        self.assertIsNone(rc.server)


class TestSelfEdit(unittest.TestCase):
    def _repo(self):
        import subprocess
        root = tempfile.mkdtemp(); h = os.path.join(root, "harness"); os.makedirs(os.path.join(h, "tests"))
        open(os.path.join(h, "__init__.py"), "w").close(); write_file(os.path.join(h, "cli.py"), "w", "VALUE = 1\n")
        write_file(os.path.join(h, "tests", "test_v.py"), "w", "import unittest\nfrom harness import cli\nclass T(unittest.TestCase):\n    def test(self): self.assertEqual(cli.VALUE, 1)\n")
        for c in (["init", "-q"], ["add", "-A"], ["-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"]): subprocess.run(["git", *c], cwd=root, check=True)
        return root, h

    def test_fail_rolls_back_pass_commits(self):
        from harness import selfedit
        root, h = self._repo(); os.environ.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="a@b", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="a@b")
        base = selfedit.snapshot(h)
        write_file(os.path.join(h, "cli.py"), "w", "VALUE = 2\n"); write_file(os.path.join(h, "new.py"), "w", "x = 1\n")
        self.assertEqual(sorted(os.path.basename(f) for f in selfedit.changed(base, h)), ["cli.py", "new.py"])
        ok, tail = selfedit.verify(root); self.assertFalse(ok)                                   # test pins VALUE == 1
        selfedit.rollback(base, h)
        self.assertEqual(read_text(os.path.join(h, "cli.py")), "VALUE = 1\n"); self.assertFalse(os.path.exists(os.path.join(h, "new.py")))
        write_file(os.path.join(h, "cli.py"), "w", "VALUE = 1\nOTHER = 3\n")
        ok, _ = selfedit.verify(root); self.assertTrue(ok)
        self.assertTrue(selfedit.commit("add OTHER", h)); self.assertEqual(selfedit.changed(selfedit.snapshot(h), h), [])

    def test_dirty_tree_is_snapshotted(self):
        from harness import selfedit
        root, h = self._repo(); os.environ.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="a@b", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="a@b")
        write_file(os.path.join(h, "cli.py"), "w", "VALUE = 1\n# wip\n"); base = selfedit.snapshot(h)
        selfedit.rollback(base, h); self.assertIn("# wip", read_text(os.path.join(h, "cli.py")))   # the snapshot, not the original, is the base

    def test_restart_argv(self):
        from harness.selfedit import restart_argv
        a = restart_argv(["-c", "--root", "/x", "-r", "old", "--permission-mode", "plan"], "NEW")
        self.assertEqual(a[1:], ["-m", "harness", "--root", "/x", "--permission-mode", "plan", "--resume", "NEW"])
        self.assertEqual(restart_argv(["-r", "--think"], "S")[-3:], ["--think", "--resume", "S"])


class TestGlobalShell(unittest.TestCase):
    def test_bash_tool_sees_bashrc_aliases_and_no_job_control_noise(self):
        from harness.cctools import State, cc_toolbox
        home = tempfile.mkdtemp(); write_file(os.path.join(home, ".bashrc"), "w", "alias hello_alice='echo from-rc'\nexport RC_VAR=yes\n")
        old = os.environ.get("HOME"); os.environ["HOME"] = home
        try:
            tb = cc_toolbox(State(tempfile.mkdtemp()))
            out = tb.call("Bash", {"command": "hello_alice; echo $RC_VAR; ls /var >/dev/null"})
            self.assertIn("from-rc", out); self.assertIn("yes", out); self.assertNotIn("job control", out); self.assertNotIn("[stderr]", out)
            os.environ["ALICE_PLAIN_BASH"] = "1"
            self.assertIn("not found", tb.call("Bash", {"command": "hello_alice"}))
        finally:
            os.environ.pop("ALICE_PLAIN_BASH", None)
            os.environ["HOME"] = old

    def test_bash_tool_finds_home_bin_commands_without_term_noise(self):
        """Launched with a bare env (cron, systemd, /rc): ~/bin comes from ~/.bashrc, and TERM-less tput stays quiet."""
        from harness.cctools import State, cc_toolbox
        home = tempfile.mkdtemp(); os.makedirs(os.path.join(home, "bin"))
        write_file(os.path.join(home, "bin", "mytool"), "w", "#!/bin/sh\necho mytool-ran\n"); os.chmod(os.path.join(home, "bin", "mytool"), 0o755)
        write_file(os.path.join(home, ".bashrc"), "w", 'PATH="$HOME/bin:$PATH"\nBOLD=$(tput bold)\n')
        saved = {k: os.environ.get(k) for k in ("HOME", "PATH", "TERM")}
        os.environ.update(HOME=home, PATH="/usr/bin:/bin"); os.environ.pop("TERM", None)
        try:
            out = cc_toolbox(State(tempfile.mkdtemp())).call("Bash", {"command": "mytool"})
            self.assertIn("mytool-ran", out); self.assertNotIn("TERM", out); self.assertNotIn("[stderr]", out)
        finally:
            for k, v in saved.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v


class TestSpinnerText(unittest.TestCase):
    def test_elapsed_and_verbs(self):
        from harness.ui import elapsed, TOOL_VERBS
        self.assertEqual((elapsed(5.9), elapsed(60), elapsed(125)), ("5s", "1m 0s", "2m 5s"))
        self.assertEqual(TOOL_VERBS["WebSearch"], "Searching the web")


class TestExpand(unittest.TestCase):
    def test_expand_prints_full_result(self):
        import io, contextlib
        from harness.cli import App, parse
        l = FakeLLM([]); l.model = "fake"; a = App(parse(["--root", tempfile.mkdtemp(), "--no-mcp"]), l)
        a.renderer.log += [("Bash", {"command": "seq 1 30"}, "\n".join(map(str, range(1, 31)))), ("Read", {"file_path": "x"}, "tiny")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): a.c_expand(""); a.c_expand("2"); a.c_expand("9"); a.c_expand("zz")
        out = buf.getvalue()
        self.assertIn("tiny", out); self.assertIn("    30", out); self.assertIn("no such tool call", out); self.assertIn("usage", out)


class TestHelp(unittest.TestCase):
    def test_help_grouped_and_per_command(self):
        import io, contextlib
        from harness.cli import App, parse
        from harness.prompt import SLASH_HELP
        d = tempfile.mkdtemp(); l = FakeLLM([]); l.model = "fake"; a = App(parse(["--root", d, "--no-mcp"]), l)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): a.c_help(""); a.c_help("diff"); a.c_help("nope")
        out = buf.getvalue()
        for g in ("Conversation", "Session", "Setup", "Work"): self.assertIn(g, out)
        self.assertIn("files changed by the last turn", out); self.assertIn("no command /nope", out)
        listed = {w.lstrip("/") for w in out.split() if w.startswith("/")}
        missing = set(SLASH_HELP) - listed - {"exit", "help"}
        self.assertFalse(missing, f"commands absent from /help: {missing}")


class TestDiffCommand(unittest.TestCase):
    def test_diff_shows_last_turn_changes(self):
        import io, contextlib
        from harness.cli import App, parse
        d = tempfile.mkdtemp(); f = os.path.join(d, "a.txt"); write_file(f, "w", "one\ntwo\n")
        l = FakeLLM([say("x")]); l.model = "fake"; a = App(parse(["--root", d, "--no-mcp", "--no-stream"]), l)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a.c_diff("")
            a.state.turns.append({"n": 0, "files": [(f, "one\ntwo\n")]}); write_file(f, "w", "one\nTWO\n"); a.c_diff("")
        out = buf.getvalue()
        self.assertIn("changed no files", out); self.assertIn("a.txt", out); self.assertIn("-two", out); self.assertIn("+TWO", out)


class TestTasksCommand(unittest.TestCase):
    def test_tasks_and_kill(self):
        import io, contextlib, time as _t
        from harness.bgtools import start_job
        from harness.cli import App, parse
        l = FakeLLM([]); l.model = "fake"; a = App(parse(["--root", tempfile.mkdtemp(), "--no-mcp"]), l)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a.c_tasks(""); start_job(a.state, "echo hi; sleep 30"); _t.sleep(0.5); a.c_tasks("")
            a.c_kill("bash_1"); _t.sleep(0.5); a.c_tasks("")
        out = buf.getvalue()
        self.assertIn("no background shells", out); self.assertIn("bash_1", out); self.assertIn("hi", out)
        self.assertIn("stopped bash_1", out); self.assertIn("exit", out.rsplit("bash_1", 1)[-1])


class TestUnloadOllama(unittest.TestCase):
    def test_unloads_loaded_models(self):
        import http.server, threading, json as _j
        from harness.llm import unload_ollama
        calls = []
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                b = _j.dumps({"models": [{"name": "m1"}]}).encode(); self.send_response(200); self.end_headers(); self.wfile.write(b)
            def do_POST(self):
                calls.append(_j.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); self.addCleanup(srv.server_close)
        self.assertEqual(unload_ollama(host=f"127.0.0.1:{srv.server_port}"), ["m1"])
        self.assertEqual(calls, [{"model": "m1", "keep_alive": 0}]); srv.shutdown()

    def test_no_ollama_is_silent(self):
        from harness.llm import unload_ollama
        self.assertEqual(unload_ollama(host="http://127.0.0.1:1"), [])


class TestServerCleanup(unittest.TestCase):
    def _run(self, sig):
        import subprocess, signal as sg, time, sys as _s
        fake = os.path.join(os.path.dirname(__file__), "fake_llama_server.py")
        code = ("import sys,time;from harness.llm import LlamaServer;s=LlamaServer('x.gguf',1024,binary=sys.argv[1]);"
                "s.start(wait=20);print(s.proc.pid,flush=True);time.sleep(60)")
        env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", ".."), "HOME": tempfile.mkdtemp()}
        p = subprocess.Popen([_s.executable, "-c", code, fake], stdout=subprocess.PIPE, env=env)
        child = int(p.stdout.readline()); p.send_signal(sig); p.wait(10); p.stdout.close()
        for _ in range(30):
            try: os.kill(child, 0)
            except ProcessLookupError: return True
            time.sleep(0.2)
        os.kill(child, 9); return False

    def test_sigterm_stops_server(self):
        import signal as sg; self.assertTrue(self._run(sg.SIGTERM))

    def test_sigkill_stops_server(self):
        import signal as sg; self.assertTrue(self._run(sg.SIGKILL))


if __name__ == "__main__":
    unittest.main()


class TestExtras(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.st = State(self.d)
        self.tb = cc_toolbox(self.st)

    def test_background_shell_lifecycle(self):
        out = self.tb.call("Bash", {"command": "echo hi; sleep 30", "run_in_background": True})
        self.assertIn("bash_1", out)
        import time; time.sleep(0.4)
        self.assertIn("hi", self.tb.call("BashOutput", {"bash_id": "bash_1"}))
        self.assertIn("Killed", self.tb.call("KillShell", {"shell_id": "bash_1"}))
        self.assertIsNotNone(self.st.jobs["bash_1"]["proc"].returncode)   # reaped, not a zombie
        self.assertIn("exited", self.tb.call("BashOutput", {"bash_id": "bash_1"}))
        self.assertIn("error", self.tb.call("BashOutput", {"bash_id": "nope"}))

    def test_notebook_edit(self):
        p = os.path.join(self.d, "n.ipynb")
        write_json(p, {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
        self.tb.call("NotebookEdit", {"notebook_path": "n.ipynb", "new_source": "x=1", "cell_number": 0, "edit_mode": "insert"})
        self.assertEqual(read_json(p)["cells"][0]["source"], ["x=1"])
        self.assertIn("error", self.tb.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_number": 5, "edit_mode": "delete"}))

    def test_write_snapshots_for_rewind(self):
        p = os.path.join(self.d, "f.txt"); write_file(p, "w", "old")
        self.tb.call("Read", {"file_path": "f.txt"})
        self.st.turns.append({"n": 0, "files": []})
        self.tb.call("Write", {"file_path": "f.txt", "content": "new"})
        self.assertEqual(self.st.turns[-1]["files"], [(p, "old")])

    def test_custom_agents_discovered(self):
        os.makedirs(os.path.join(self.d, ".alice", "agents"))
        write_file(os.path.join(self.d, ".alice", "agents", "rev.md"), "w", 
            "---\nname: reviewer\ndescription: reviews code\ntools: Read, Grep\n---\nYou review.\n")
        a = context.discover_agents(self.d)
        self.assertEqual(a["reviewer"]["tools"], ["Read", "Grep"])
        self.assertEqual(a["reviewer"]["prompt"], "You review.")


class TestLlamaBackend(unittest.TestCase):
    FAKE = os.path.join(os.path.dirname(__file__), "fake_llama_server.py")

    def setUp(self):
        from harness.llm import LlamaServer
        self.srv = LlamaServer("fake.gguf", 4096, binary=self.FAKE, log=os.path.join(tempfile.mkdtemp(), "l.log"))
        self.srv.bin = self.FAKE
        self.addCleanup(self.srv.stop)
        self.host = self.srv.start(wait=20)

    def client(self):
        from harness.llm import OpenAIClient
        return OpenAIClient("fake", host=self.host)

    def test_server_command_and_lifecycle(self):
        cmd = self.srv.command()
        for want in ("--jinja", "-c", "4096", "--port"): self.assertIn(want, cmd)
        self.srv.stop(); self.assertIsNotNone(self.srv.proc.poll())

    def test_gguf_args(self):
        from harness.llm import gguf_args
        self.assertEqual(gguf_args("hf.co/user/repo:Q4_K_M"), ["-hf", "user/repo:Q4_K_M"])
        self.assertEqual(gguf_args("user/repo"), ["-hf", "user/repo"])
        f = os.path.join(tempfile.mkdtemp(), "m.gguf"); open(f, "w").close()
        self.assertEqual(gguf_args(f), ["-m", f])

    def test_streaming_assembles_split_tool_call(self):
        seen = []
        m = self.client().chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}], on_token=seen.append)
        self.assertEqual(m["tool_calls"][0]["function"], {"name": "calculate", "arguments": {"expression": "2+2"}})

    def test_streaming_text_and_usage(self):
        c, seen = self.client(), []
        m = c.chat([{"role": "user", "content": "hi"}], on_token=seen.append)
        self.assertEqual((m["content"], seen), ("done", ["do", "ne"]))
        self.assertEqual({k: c.usage[k] for k in ("calls", "prompt_tokens", "completion_tokens")}, {"calls": 1, "prompt_tokens": 7, "completion_tokens": 3})

    def test_non_streaming_tool_call_and_history_translation(self):
        import urllib.request
        c = self.client()
        m = c.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        self.assertEqual(m["tool_calls"][0]["function"]["arguments"], {})
        c.chat([{"role": "user", "content": "hi"}, m, {"role": "tool", "tool_name": "today", "content": "Mon"}])
        sent = json.load(urllib.request.urlopen(self.host + "/_log"))[-1]
        a, t = sent["messages"][1], sent["messages"][2]
        self.assertIsInstance(a["tool_calls"][0]["function"]["arguments"], str)
        self.assertEqual(t["tool_call_id"], a["tool_calls"][0]["id"])
        self.assertIs(sent["chat_template_kwargs"]["enable_thinking"], False)

    def test_health_and_models(self):
        c = self.client(); self.assertTrue(c.health()[0]); self.assertEqual(c.models(), ["fake.gguf"])

    def test_missing_binary_message(self):
        from harness.llm import LlamaServer, LLMError
        s = LlamaServer("x", 1024); s.bin = ""
        with self.assertRaisesRegex(LLMError, "llama-server not found"): s.start()

    def test_app_connects_with_base_url_and_runs_a_turn(self):
        a = parse(["--backend", "llama", "--base-url", self.host, "--root", tempfile.mkdtemp(), "-q", "hello"])
        app = App(a)
        self.assertEqual(app.llm.kind, "openai")
        r = app.agent.ask("hello")
        self.assertEqual(r.answer, "done")

    def test_ollama_blob_resolution(self):
        from harness.llm import ollama_blob, gguf_args
        st = tempfile.mkdtemp(); os.environ["OLLAMA_MODELS"] = st
        self.addCleanup(os.environ.pop, "OLLAMA_MODELS", None)
        os.makedirs(os.path.join(st, "manifests/hf.co/u/r")); os.makedirs(os.path.join(st, "blobs"))
        blob = os.path.join(st, "blobs", "sha256-abc"); open(blob, "w").close()
        write_json(os.path.join(st, "manifests/hf.co/u/r/Q4"), {"layers": [{"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:zzz"},
                              {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc"}]})
        self.assertEqual(ollama_blob("hf.co/u/r:Q4"), blob)
        self.assertEqual(gguf_args("hf.co/u/r:Q4"), ["-m", blob])
        self.assertEqual(ollama_blob("hf.co/u/missing:Q4"), "")
