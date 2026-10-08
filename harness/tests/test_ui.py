import os, tempfile, unittest
from harness.prompt import next_mode, complete_candidates, MODE_CYCLE
from harness.ui import summarize, diff_counts, plural


class TestPromptHelpers(unittest.TestCase):
    def test_mode_cycle_wraps(self):
        m = MODE_CYCLE[0]
        for _ in MODE_CYCLE: m = next_mode(m)
        self.assertEqual(m, MODE_CYCLE[0])
        self.assertEqual(next_mode("bogus"), MODE_CYCLE[0])

    def test_slash_completion(self):
        names = [c[0] for c in complete_candidates("/he", ".")]
        self.assertIn("/help", names)
        self.assertEqual(complete_candidates("/he", ".", {"deploy": ""})[0][1], -3)
        self.assertTrue(any(c[0] == "/deploy" for c in complete_candidates("/dep", ".", {"deploy": ""})))

    def test_file_completion(self):
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "src")); open(os.path.join(d, "readme.md"), "w").close()
            self.assertEqual([c[0] for c in complete_candidates("see @re", d)], ["@readme.md"])
            self.assertEqual(complete_candidates("@s", d)[0][2], "dir")
            self.assertEqual(complete_candidates("no mention", d), [])


class TestSummaries(unittest.TestCase):
    def test_summaries(self):
        self.assertEqual(summarize("Read", "a\nb\nc")[0], ["Read 3 lines"])
        self.assertEqual(summarize("Glob", "")[0], ["Found 0 files"])
        self.assertEqual(summarize("Bash", "1\n2\n3\n4\n5"), (["1", "2", "3"], 2))
        self.assertEqual(summarize("Edit", "error: nope")[0], ["error: nope"])

    def test_diff_counts_and_plural(self):
        self.assertEqual(diff_counts("--- a\n+++ b\n+x\n+y\n-z\n ctx"), (2, 1))
        self.assertEqual(plural(1, "line"), "1 line"); self.assertEqual(plural(2, "line"), "2 lines")


class TestTypeAhead(unittest.TestCase):
    def test_queue_and_edit(self):
        from harness.ui import EscWatcher
        w = EscWatcher(); w.feed("helo\x7flo\rsecond\r  \rthird")
        self.assertEqual(w.queue, ["hello", "second"]); self.assertEqual(w.buf, "third")
        self.assertIn("2 queued", w.status()); self.assertIn("› third", w.status())
        w.feed("\x15"); self.assertEqual(w.buf, "")


class TestTheme(unittest.TestCase):
    def test_set_theme(self):
        from harness import ui
        try:
            self.assertTrue(ui.set_theme("blue")); self.assertEqual(ui.accent_n(), 75)
            self.assertFalse(ui.set_theme("nope")); self.assertEqual(ui.THEME["name"], "blue")
        finally: ui.set_theme("terracotta")


if __name__ == "__main__":
    unittest.main()
