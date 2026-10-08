#!/usr/bin/env python3
"""Measure llama-server flag variants on this GPU: VRAM, prompt-eval and generation speed, and prompt-cache reuse.
    python3 evals/bench_server.py [--ctx 12288] [--model NAME] [--only baseline,kvq8]
Appends one JSON line per variant to evals/bench_results.jsonl."""
import argparse, json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from harness.llm import LlamaServer, DEFAULT_MODEL
from harness.fsutil import append_line

VARIANTS = {
    "baseline": [],
    "fa-on": ["-fa", "on"],
    "kvq8": ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"],
    "kvq4": ["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0"],
    "kvq8+ngram": ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "ngram-mod"],
    "kvq8+ngram-simple": ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "ngram-simple"],
}
CODE = "\n".join(f"def handler_{i}(request):\n    data = request.json()\n    return {{'id': {i}, 'ok': True, 'items': [x for x in data if x]}}\n" for i in range(14))
TASKS = [  # edit-style prompts: output largely repeats the input, where n-gram drafting shines
    f"Here is a file:\n```python\n{CODE}```\nRewrite the whole file with every `'ok': True` changed to `'ok': False`. Output only the code.",
    "Write a Python function that parses an ISO date string and returns the weekday name, with a docstring and two tests.",
]

def vram() -> int:
    try: return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
    except Exception: return -1

def post(host, body):
    req = urllib.request.Request(host + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)

def run(name, flags, model, ctx):
    srv = LlamaServer(model, ctx, extra=flags, log=f"/tmp/bench-{name}.log")
    row = {"variant": name, "flags": flags, "ctx": ctx, "model": model, "ts": time.strftime("%F %T")}
    try:
        t0 = time.time(); host = srv.start(wait=240); row["load_s"] = round(time.time() - t0, 1)
        post(host, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}})  # warm up
        row["vram_mib"] = vram(); gen = []
        for i, t in enumerate(TASKS):
            body = {"messages": [{"role": "user", "content": t}], "max_tokens": 500, "temperature": 0, "cache_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            r = post(host, body); tm = r.get("timings", {})
            gen.append({"task": i, "gen_tps": round(tm.get("predicted_per_second", 0), 1), "prompt_tps": round(tm.get("prompt_per_second", 0), 1),
                        "n_out": tm.get("predicted_n"), "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"), "text_hash": hash(r["choices"][0]["message"]["content"]) & 0xffff})
        r2 = post(host, {**body, "messages": [{"role": "user", "content": TASKS[1]}]}); tm = r2.get("timings", {})   # same prompt again: cache should serve it
        row["repeat_cache_n"], row["repeat_prompt_n"] = tm.get("cache_n"), tm.get("prompt_n")
        row["tasks"] = gen; row["vram_peak_mib"] = vram()
    except Exception as e: row["error"] = str(e)[-300:]
    finally: srv.stop(); time.sleep(2)
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ctx", type=int, default=12288); ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only", default=""); a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.jsonl")
    for name, flags in VARIANTS.items():
        if a.only and name not in a.only.split(","): continue
        row = run(name, flags, a.model, a.ctx); print(json.dumps(row)); append_line(out, json.dumps(row))
