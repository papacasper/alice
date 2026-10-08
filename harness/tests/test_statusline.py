import json, os, shutil, subprocess, tempfile, time, unittest
from types import SimpleNamespace
from harness import statusline


def fake_app(**over):
    llm = SimpleNamespace(model="hf.co/x/Qwen3.5-9B:Q4", num_ctx=32768, usage={"prompt_tokens": 900, "completion_tokens": 50, "prompt_ms": 10, "gen_ms": 20},
                          last={"prompt_tokens": 800, "completion_tokens": 40, "cached_tokens": 600})
    app = SimpleNamespace(llm=llm, sid="20261007-120000", cwd=tempfile.mkdtemp(), style="default", perms=SimpleNamespace(mode="default"), started=time.time())
    for k, v in over.items(): setattr(app, k, v)
    return app


class TestStatusLine(unittest.TestCase):
    def test_payload_matches_claude_code_schema(self):
        d = statusline.payload(fake_app(), 8192, "1.0")
        for k in ("hook_event_name", "session_id", "transcript_path", "cwd", "model", "workspace", "version", "output_style", "cost", "context_window", "exceeds_200k_tokens"):
            self.assertIn(k, d)
        c = d["context_window"]
        self.assertEqual((c["context_window_size"], c["used_percentage"], c["remaining_percentage"]), (32768, 25, 75))
        self.assertEqual(c["current_usage"], {"input_tokens": 200, "output_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 600})
        self.assertEqual(d["model"]["display_name"], "Qwen3.5-9B:Q4")
        self.assertIsNone(statusline.payload(fake_app(llm=SimpleNamespace(model="m", num_ctx=4096, usage={}, last={})), 0, "1")["context_window"]["current_usage"])

    def test_builtin_line(self):
        line = statusline.builtin(statusline.payload(fake_app(), 8192, "1.0"))[0]
        self.assertEqual(line, "Qwen3.5-9B:Q4 | [████░░░░░░░░░░░░] 25% | 8k / 32k (24k left)")

    def test_command_gets_json_and_keeps_every_line_and_ansi(self):
        s = statusline.StatusLine()
        cmd = """python3 -c "import json,sys; d=json.load(sys.stdin); print(d['model']['display_name']); print('\\033[31mred\\033[0m')" """
        self.assertEqual(s.run(cmd, statusline.payload(fake_app(), 0, "1")), ["Qwen3.5-9B:Q4", "\x1b[31mred\x1b[0m"])
        s.run("exit 3", {}); self.assertIn("exited 3", s.error)

    def test_refresh_never_blocks_the_prompt(self):
        hits = []; s = statusline.StatusLine(lambda: hits.append(1))
        t = time.time(); s.refresh("sleep 0.5; echo late", {}); self.assertLess(time.time() - t, 0.2)
        self.assertEqual(s.lines, [])
        for _ in range(40):
            if s.lines: break
            time.sleep(0.05)
        self.assertEqual((s.lines, hits), (["late"], [1]))

    def test_configure_saves_settings(self):
        f = os.path.join(tempfile.mkdtemp(), "settings.json"); st = {}
        self.assertIn("built-in", statusline.configure(st, "", f))
        statusline.configure(st, "echo hi", f)
        self.assertEqual(json.load(open(f))["statusLine"]["command"], "echo hi"); self.assertEqual(st["statusLine"]["command"], "echo hi")
        statusline.configure(st, "off", f); self.assertIs(json.load(open(f))["statusLine"], False)
        statusline.configure(st, "on", f); self.assertNotIn("statusLine", json.load(open(f))); self.assertNotIn("statusLine", st)

    @unittest.skipUnless(os.path.exists(os.path.expanduser(statusline.CLAUDE_SCRIPT)) and shutil.which("jq"), "no Claude Code statusline script")
    def test_users_claude_code_script_renders_alice_payload(self):
        out = statusline.StatusLine().run(statusline.CLAUDE_SCRIPT, statusline.payload(fake_app(), 8192, "1.0"))
        self.assertTrue(out and out[0].startswith("Qwen3.5-9B:Q4 (32768 ctx) | ["), out)
        self.assertIn("25%", out[0])


if __name__ == "__main__":
    unittest.main()
