import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import Agent, Toolbox, tool
from harness.builtin import calculate, default_toolbox
from harness.tools import Tool

class FakeLLM:
    """Scripted model: each item is a message dict, or a callable(messages)->dict."""
    def __init__(self, script, num_ctx=12288): self.script, self.num_ctx, self.seen = list(script), num_ctx, []
    def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        s = self.script.pop(0)
        return s(messages) if callable(s) else s

def call(name, **args): return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}
def say(t): return {"role": "assistant", "content": t}

@tool
def add(a: float, b: float = 0) -> str:
    """Add two numbers.

    Args:
        a: first number
        b: second number
    """
    return str(a + b)

class TestTools(unittest.TestCase):
    def test_schema(self):
        f = add.schema()["function"]
        self.assertEqual(f["name"], "add"); self.assertEqual(f["description"], "Add two numbers.")
        self.assertEqual(f["parameters"]["required"], ["a"])
        self.assertEqual(f["parameters"]["properties"]["a"], {"type": "number", "description": "first number"})
    def test_validation_and_coercion(self):
        self.assertEqual(add.call({"a": "2", "b": 3}), "5.0")
        self.assertIn("missing", add.call({}))
        self.assertIn("unknown argument", add.call({"a": 1, "zzz": 2}))
        self.assertIn("must be number", add.call({"a": "abc"}))
    def test_exceptions_become_errors(self):
        @tool
        def boom() -> str:
            """x"""
            raise RuntimeError("bad")
        self.assertEqual(boom.call({}), "error: RuntimeError: bad")
    def test_unknown_tool_and_duplicate(self):
        tb = Toolbox([add]); self.assertIn("unknown tool", tb.call("nope", {}))
        with self.assertRaises(ValueError): tb.add(add)
    def test_calculate_safe(self):
        self.assertEqual(calculate("120.5 + 75.25 * 2"), "271.0")
        for bad in ("__import__('os').system('id')", "2 ** 9999", "open('x')"):
            self.assertIn("error", Tool(calculate).call({"expression": bad}))
    def test_plugin_loading(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("from harness import tool\n@tool\ndef hi() -> str:\n    '''Say hi.'''\n    return 'hi'\n")
        self.assertEqual([t.name for t in Toolbox.from_file(f.name)], ["hi"]); os.unlink(f.name)

class TestFs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        with open(os.path.join(self.d, "a.log"), "w") as f: f.write("ERROR [x] 1\nINFO [y] 2\nERROR [x] 3\nERROR [y] 4\n")
        self.tb = default_toolbox(self.d)
    def test_sandbox_escape_blocked(self):
        self.assertIn("outside the sandbox", self.tb.call("read_file", {"path": "../../etc/passwd"}))
        self.assertIn("outside the sandbox", self.tb.call("list_dir", {"path": "/etc"}))
    def test_today_and_search_files(self):
        self.assertRegex(self.tb.call("today", {}), r"^\d{4}-\d\d-\d\d \(\w+\)")
        self.assertTrue(self.tb.call("search_files", {"pattern": "ERROR"}).startswith("3 matching lines"))

    def test_grep_and_count_by_group(self):
        self.assertTrue(self.tb.call("grep", {"pattern": "ERROR", "path": "a.log"}).startswith("3 matching lines"))
        self.assertEqual(self.tb.call("count_by_group", {"pattern": r"ERROR \[(\w+)\]", "path": "a.log"}), "x: 2\ny: 1")
    def test_write_and_shell_off_by_default(self):
        self.assertNotIn("write_file", self.tb.names()); self.assertNotIn("run_shell", self.tb.names())
        tb = default_toolbox(self.d, allow_write=True, allow_shell=True)
        self.assertIn("wrote", tb.call("write_file", {"path": "sub/n.txt", "content": "hi"}))
        self.assertIn("outside the sandbox", tb.call("write_file", {"path": "../evil", "content": "x"}))
        self.assertIn("exit 0", tb.call("run_shell", {"command": "true"}))

class TestAgent(unittest.TestCase):
    def test_tool_round_trip(self):
        llm = FakeLLM([call("add", a=2, b=3), say("5")])
        r = Agent(llm, Toolbox([add])).run("2+3?")
        self.assertEqual((r.answer, r.status, r.tool_calls), ("5", "answered", 1))
        tool_msg = [m for m in r.messages if m["role"] == "tool"][0]
        self.assertEqual((tool_msg["tool_name"], tool_msg["content"]), ("add", "5.0"))
    def test_tool_error_is_fed_back(self):
        llm = FakeLLM([call("add"), say("done")]); r = Agent(llm, Toolbox([add])).run("x")
        self.assertIn("missing", [m for m in r.messages if m["role"] == "tool"][0]["content"])
    def test_repeat_detection(self):
        llm = FakeLLM([call("add", a=1)] * 3 + [say("ok")]); r = Agent(llm, Toolbox([add])).run("x")
        self.assertIn("exact call", [m for m in r.messages if m["role"] == "tool"][2]["content"])
    def test_step_limit_forces_answer(self):
        llm = FakeLLM([call("add", a=i) for i in range(5)] + [say("best effort")])
        r = Agent(llm, Toolbox([add]), max_steps=5).run("x")
        self.assertEqual((r.status, r.answer), ("step_limit", "best effort"))
    def test_context_guard_keeps_task(self):
        big = "line\n" * 4000
        @tool
        def dump() -> str:
            """big output"""
            return big
        import harness.tools as T; T.MAX_OUTPUT = 10**6
        llm = FakeLLM([call("dump")] * 6 + [say("fin")], num_ctx=4096)
        Agent(llm, Toolbox([dump]), max_repeat=99).run("THE TASK")
        for sent in llm.seen: self.assertTrue(any(m.get("content") == "THE TASK" for m in sent))
        T.MAX_OUTPUT = 6000
    def test_history_persists(self):
        llm = FakeLLM([say("a"), say("b")]); ag = Agent(llm, Toolbox([]))
        ag.ask("one"); ag.ask("two")
        self.assertEqual([m["content"] for m in ag.messages if m["role"] == "user"], ["one", "two"])

if __name__ == "__main__":
    unittest.main()
