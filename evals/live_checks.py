#!/usr/bin/env python3
"""Live checks on the real model for features the task evals can't see. Each prints PASS/FAIL and appends to live_results.jsonl.

  slot-restore   a resumed session reloads its saved KV cache: the first call is served from cache, the answer still knows the history
  helper-compact /compact on settings.helperModel swaps to the helper and back, and the summary keeps the facts (vs. compacting on the main model)

Usage: python3 evals/live_checks.py [slot-restore] [helper-compact] [--helper MODEL]
"""
import argparse, glob, json, os, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_results.jsonl")
SLOTS = os.path.expanduser("~/.alice/slots")
CODE = "PELICAN-4471"

def setup(d):
    # ~3k tokens of filler so a cache hit is measurable, with one fact to recall
    body = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog, entry {i * 7} of the ledger." for i in range(120))   # one Read page, so the check tests the restore and not paging
    open(os.path.join(d, "ledger.txt"), "w").write(body[: len(body) // 2] + f"\nThe vault codeword is {CODE}.\n" + body[len(body) // 2:])

MODEL = []   # --model for the checks, set from the command line

def alice(d, *args, timeout=600):
    p = subprocess.run([sys.executable, "-m", "harness", *MODEL, "--root", d, "--no-mcp", "--no-memory", "--output-format", "json", "-q", *args],
                       cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    try: return json.loads(p.stdout), p.stderr
    except ValueError: return {"result": p.stdout, "parse_error": True}, p.stderr

def slot_restore():
    d = tempfile.mkdtemp(prefix="live-slot-"); setup(d)
    j1, _ = alice(d, "-p", "Read ledger.txt and tell me the vault codeword.")
    sid = j1.get("session_id", "")
    files = glob.glob(os.path.join(SLOTS, f"{sid}-*.bin")); size = os.path.getsize(files[0]) if files else 0
    j2, err2 = alice(d, "-r", sid, "-p", "What was the vault codeword? Reply with just the codeword.")
    restored = "restored the cached prompt" in err2
    for f in files: os.remove(f)                           # control: same resume with no snapshot
    j3, _ = alice(d, "-r", sid, "-p", "What was the vault codeword? Reply with just the codeword.")
    hit, miss = (j2.get("usage") or {}).get("cached_tokens", 0), (j3.get("usage") or {}).get("cached_tokens", 0)
    info_model = MODEL[1] if MODEL else "default"
    ok = size > 1_000_000 and restored and CODE in j2.get("result", "") and hit > miss + 1000
    return ok, {"model": info_model, "slot_bytes": size, "restored_msg": restored, "cached_with_slot": hit, "cached_without": miss,
                "prompt_ms_with": round((j2.get("usage") or {}).get("prompt_ms", 0)), "prompt_ms_without": round((j3.get("usage") or {}).get("prompt_ms", 0)),
                "answer": j2.get("result", "")[:120]}

COMPACT = r"""
import json, sys, time
from harness.cli import App, parse
app = App(parse(["--root", sys.argv[1], "--no-mcp", "--no-memory", "-q", "--output-format", "json"]))
main = app.llm.model
app.one_shot("Read ledger.txt and tell me the vault codeword.")
t = time.time(); app.compact_now(); secs = time.time() - t
summary = app.agent.messages[1]["content"]
r = app.turn("What was the vault codeword? Reply with just the codeword.")
print("RESULT " + json.dumps({"secs": round(secs, 1), "back_on_main": app.llm.model == main, "summary_has_code": "%s" in summary,
                              "summary_chars": len(summary), "answer": (r.answer if r else "")[:120]}))
""" % CODE

def compact_run(helper):
    d = tempfile.mkdtemp(prefix="live-compact-"); setup(d)
    if helper:
        os.makedirs(os.path.join(d, ".alice")); json.dump({"helperModel": helper}, open(os.path.join(d, ".alice", "settings.json"), "w"))
    p = subprocess.run([sys.executable, "-c", COMPACT, d], cwd=ROOT, capture_output=True, text=True, timeout=900)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else {"error": p.stderr[-400:]}

def helper_compact(helper):
    h, m = compact_run(helper), compact_run("")
    ok = all(r.get("back_on_main") and CODE in r.get("answer", "") for r in (h, m))
    return ok, {"helper": helper, "with_helper": h, "main_only": m}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("checks", nargs="*", default=["slot-restore", "helper-compact"])
    ap.add_argument("--helper", default="unsloth-gemma4-e2b:latest")
    ap.add_argument("--slot-model", default="qwen3:8b", help="slot-restore needs a non-recurrent model; hybrid ones (qwen3.5) skip snapshots by design")
    a = ap.parse_args()
    for name in a.checks:
        t0 = time.time()
        MODEL[:] = ["--model", a.slot_model] if name == "slot-restore" else []
        ok, info = slot_restore() if name == "slot-restore" else helper_compact(a.helper)
        row = {"check": name, "ok": ok, "secs": round(time.time() - t0, 1), "ts": time.strftime("%F %T"), **info}
        open(OUT, "a").write(json.dumps(row) + "\n")
        print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(info))
