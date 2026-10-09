"""Model picker / per-model settings, HTTP MCP, stream-json, the extra hook events, and memory walk-up/imports."""
import contextlib, io, json, os, tempfile, unittest, urllib.request
from unittest import mock
from harness import models
from harness.cli import App, parse
from harness.fsutil import read_json
from test_harness import FakeLLM, say

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_llama_server.py")


class TestModelSettings(unittest.TestCase):
    S = {"num_ctx": 8192, "think": "auto", "sampling": {"temperature": 0.5, "top_k": 20},
         "models": {"qwen": {"num_ctx": 16384, "sampling": {"temperature": 0.6}},
                    "qwen3.5-9b": {"think": "off", "description": "fast"},
                    "hf.co/x/big:Q4": {"num_ctx": 4096, "think": "on"}}}

    def test_most_specific_key_wins_and_flags_beat_everything(self):
        e = models.effective(self.S, "hf.co/x/Qwen3.5-9B:Q4")
        self.assertEqual((e["num_ctx"], e["think"], e["description"]), (16384, "off", "fast"))
        self.assertEqual(e["sampling"], {"temperature": 0.6, "top_k": 20})       # per-model merges over the top-level sampling
        self.assertEqual(models.effective(self.S, "gemma")["num_ctx"], 8192)     # no per-model entry: top-level setting
        self.assertEqual(models.effective({}, "gemma")["num_ctx"], 32768)        # nothing set: built-in default
        a = parse(["--num-ctx", "2048", "--think", "on", "x"])
        e = models.effective(self.S, "qwen3.5-9b", a)
        self.assertEqual((e["num_ctx"], e["think"]), (2048, "on"))

    def test_choices_lists_named_models_not_patterns(self):
        self.assertEqual(models.choices(self.S, "cur", ["a.gguf", "hf.co/x/big:Q4"]), ["cur", "hf.co/x/big:Q4", "a.gguf"])

    def test_parse_and_save(self):
        self.assertEqual(models.parse_value("num_ctx", "16k"), 16384)
        for k, v in (("num_ctx", "100"), ("think", "maybe"), ("sampling", "[1]")):
            with self.assertRaises(ValueError): models.parse_value(k, v)
        f = os.path.join(tempfile.mkdtemp(), "s.json"); st = {}
        models.save(st, "m1", "think", "off", f); models.save(st, "m1", "num_ctx", 4096, f)
        self.assertEqual(read_json(f)["models"]["m1"], {"think": "off", "num_ctx": 4096}); self.assertEqual(st["models"]["m1"]["think"], "off")
        models.save(st, "m1", "think", None, f); models.save(st, "m1", "num_ctx", None, f)
        self.assertEqual(read_json(f)["models"], {}); self.assertNotIn("m1", st["models"])

    def test_picker_arrow_keys(self):
        keys = iter(["DOWN", "DOWN", "UP", "\r"])
        with mock.patch("harness.models._tty", return_value=True), mock.patch("sys.stderr", io.StringIO()) as err, \
             mock.patch("harness.ui.getkey", lambda: next(keys)):
            self.assertEqual(models.pick(["a", "b", "c"], "a", {}), "b")
            self.assertIn("Select model", err.getvalue())
        with mock.patch("harness.models._tty", return_value=True), mock.patch("sys.stderr", io.StringIO()), \
             mock.patch("harness.ui.getkey", lambda: ""):
            self.assertIsNone(models.pick(["a", "b"], "a", {}))                  # Esc cancels
        self.assertIsNone(models.pick(["a"], "a", {}))                           # no terminal: no picker


class TestModelSwitchApp(unittest.TestCase):
    def test_switch_applies_per_model_settings_and_reloads(self):
        home = tempfile.mkdtemp(); d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, ".alice"))
        with open(os.path.join(d, ".alice", "settings.json"), "w") as f:
            json.dump({"models": {"hf.co/x/two:Q4": {"num_ctx": 4096, "think": "off", "sampling": {"temperature": 0.3}}}}, f)
        with mock.patch.dict(os.environ, {"ALICE_LLAMA_SERVER": FAKE, "HOME": home}):
            app = App(parse(["--root", d, "--no-mcp", "--no-stream", "--model", "hf.co/x/one:Q4"]))
            srv = app.llm.server; args = lambda: json.load(urllib.request.urlopen(srv.host + "/_args"))
            try:
                self.assertEqual(args()[args().index("-c") + 1], "32768")
                with contextlib.redirect_stdout(io.StringIO()) as out: app.c_model("hf.co/x/two:Q4")
                self.assertIn("switched to", out.getvalue())
                self.assertEqual((app.llm.model, app.llm.num_ctx, app.llm.think_mode, app.llm.sampling), ("hf.co/x/two:Q4", 4096, "off", {"temperature": 0.3}))
                self.assertEqual(args()[args().index("-c") + 1], "4096")
                with contextlib.redirect_stdout(io.StringIO()): app.c_model("set num_ctx 8k")    # saved to the (fake) home, server reloaded
                self.assertEqual(args()[args().index("-c") + 1], "8192")
                self.assertEqual(read_json(os.path.join(home, ".alice", "settings.json"))["models"]["hf.co/x/two:Q4"], {"num_ctx": 8192})
                with contextlib.redirect_stdout(io.StringIO()) as out: app.c_model("hf.co/x/BAD:Q4")
                self.assertIn("could not load", out.getvalue()); self.assertEqual((app.llm.model, srv.num_ctx), ("hf.co/x/two:Q4", 8192))
            finally: srv.stop()


class TestStreamJson(unittest.TestCase):
    def run_app(self, argv, llm, stdin=None):
        from harness import cli
        out = io.StringIO()
        with mock.patch.object(cli.App, "make_llm", lambda self, m: llm), contextlib.redirect_stdout(out), \
             contextlib.redirect_stderr(io.StringIO()), mock.patch("sys.stdin", stdin or io.StringIO("")):
            code = cli.main(["--root", tempfile.mkdtemp(), "--no-mcp", *argv])
        return code, [json.loads(l) for l in out.getvalue().splitlines()]

    def test_output_events_match_claude_code_shape(self):
        from test_harness import call
        llm = FakeLLM([call("Glob", pattern="*.nothing"), say("all done")]); llm.model = "fake"
        code, ev = self.run_app(["-p", "--output-format", "stream-json", "--verbose", "find files"], llm)
        self.assertEqual(code, 0)
        self.assertEqual([e["type"] for e in ev], ["system", "assistant", "user", "assistant", "result"])
        self.assertEqual((ev[0]["subtype"], ev[0]["model"]), ("init", "fake")); self.assertIn("Glob", ev[0]["tools"])
        use = ev[1]["message"]["content"][0]
        self.assertEqual((use["type"], use["name"], use["input"]), ("tool_use", "Glob", {"pattern": "*.nothing"}))
        res = ev[2]["message"]["content"][0]
        self.assertEqual((res["type"], res["tool_use_id"]), ("tool_result", use["id"]))
        self.assertEqual(ev[3]["message"]["content"], [{"type": "text", "text": "all done"}])
        self.assertEqual((ev[4]["subtype"], ev[4]["is_error"], ev[4]["result"], ev[4]["num_turns"]), ("success", False, "all done", 2))
        self.assertEqual(len({e["session_id"] for e in ev}), 1)

    def test_input_stream_json_runs_each_message_as_a_turn(self):
        llm = FakeLLM([say("one"), say("two")]); llm.model = "fake"
        lines = "\n".join([json.dumps({"type": "user", "message": {"role": "user", "content": "first"}}), "not json",
                           json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "second"}]}})])
        code, ev = self.run_app(["--input-format", "stream-json", "--output-format", "stream-json"], llm, io.StringIO(lines))
        self.assertEqual(code, 0)
        self.assertEqual([e["result"] for e in ev if e["type"] == "result"], ["one", "two"])
        self.assertEqual([e["type"] for e in ev].count("system"), 1)
        self.assertEqual([m["content"] for m in llm.seen[1] if m["role"] == "user"], ["first", "second"])   # one conversation

    def test_input_stream_json_requires_stream_output(self):
        llm = FakeLLM([]); llm.model = "fake"
        self.assertEqual(self.run_app(["--input-format", "stream-json"], llm)[0], 1)


class TestHookEvents(unittest.TestCase):
    def app_with_hooks(self, hooks, llm, mode="default"):
        from harness import cli
        d = tempfile.mkdtemp(); self.log = os.path.join(d, "hook.log")
        rec = f"cat >> {self.log}; echo >> {self.log}"
        conf = {ev: [{"matcher": m, "hooks": [{"type": "command", "command": cmd or rec}]}] for ev, (m, cmd) in hooks.items()}
        os.makedirs(os.path.join(d, ".alice"))
        with open(os.path.join(d, ".alice", "settings.json"), "w") as f: json.dump({"hooks": conf}, f)
        llm.model = "fake"
        with mock.patch.object(cli.App, "make_llm", lambda self, m: llm), contextlib.redirect_stderr(io.StringIO()):
            return cli.App(parse(["--root", d, "--no-mcp", "--no-stream", "--permission-mode", mode]))

    def logged(self):
        if not os.path.exists(self.log): return []
        with open(self.log) as f: return [json.loads(l) for l in f.read().splitlines() if l.strip()]

    def test_permission_request_hook_can_allow_or_deny_and_notification_fires(self):
        allow = 'echo \'{"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": "allow"}}}\''
        app = self.app_with_hooks({"PermissionRequest": ("Write", allow), "Notification": ("permission_prompt", None)}, FakeLLM([]))
        with mock.patch("harness.ui.confirm", return_value="no") as conf:
            self.assertEqual(app.ask_permission("Write", {"file_path": "x"}), "yes")       # the hook answered; no prompt
            conf.assert_not_called()
            self.assertEqual(app.ask_permission("Bash", {"command": "ls"}), "no")          # matcher misses: Notification, then prompt
            conf.assert_called_once()
        self.assertEqual([(e["hook_event_name"], e["notification_type"]) for e in self.logged()], [("Notification", "permission_prompt")])
        app = self.app_with_hooks({"PermissionRequest": ("", "echo no >&2; exit 2")}, FakeLLM([]))
        with mock.patch("harness.ui.confirm", return_value="yes"): self.assertEqual(app.ask_permission("Bash", {"command": "ls"}), "no")

    def test_precompact_subagentstop_and_session_end(self):
        llm = FakeLLM([say("summary of the chat"), say("sub report"), say("sub retry")])
        app = self.app_with_hooks({"PreCompact": ("manual", None), "SessionEnd": ("", None),
                                   "SubagentStop": ("explore", "echo try harder >&2; exit 2")}, llm, mode="bypassPermissions")
        app.agent.messages += [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        with contextlib.redirect_stdout(io.StringIO()): app.c_compact("")
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()): out = app.spawn("look around", "explore")
        self.assertEqual(out, "sub retry")                                         # exit 2 made the subagent continue
        self.assertIn("try harder", llm.seen[-1][-1]["content"])
        with contextlib.redirect_stdout(io.StringIO()): app.c_clear("")
        ev = self.logged()
        self.assertEqual([(e["hook_event_name"], e.get("trigger") or e.get("reason")) for e in ev], [("PreCompact", "manual"), ("SessionEnd", "clear")])


class TestMemoryImports(unittest.TestCase):
    def test_at_imports_relative_recursive_and_not_in_code(self):
        from harness import context
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "docs"))
        files = {"CLAUDE.md": "Rules. See @docs/style.md and `@nope.md` and email me@example.com.\n```\n@also-nope.md\n```",
                 "docs/style.md": "STYLE; more in @deeper.md and loop back @../CLAUDE.md", "docs/deeper.md": "DEEPER",
                 "nope.md": "NOPE", "also-nope.md": "ALSO-NOPE"}
        for n, t in files.items():
            with open(os.path.join(d, n), "w") as f: f.write(t)
        t = context.memory_text(d)
        for want in ("STYLE", "DEEPER", "(imported)"): self.assertIn(want, t)
        for bad in ("NOPE", "ALSO-NOPE"): self.assertNotIn(bad, t)
        self.assertEqual(t.count("Rules."), 1)                                   # the cycle back to CLAUDE.md is not re-read

    def test_import_depth_limit(self):
        from harness import context
        d = tempfile.mkdtemp()
        for i in range(8):
            with open(os.path.join(d, f"m{i}.md"), "w") as f: f.write(f"LEVEL{i} @m{i + 1}.md")
        with open(os.path.join(d, "CLAUDE.md"), "w") as f: f.write("@m0.md")
        t = context.memory_text(d)
        self.assertIn("LEVEL4", t); self.assertNotIn("LEVEL5", t)


if __name__ == "__main__":
    unittest.main()
