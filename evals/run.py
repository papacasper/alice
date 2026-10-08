#!/usr/bin/env python3
"""Run the alice eval tasks against a model and grade them with deterministic checks.
    python3 evals/run.py                      # all tasks, default model/settings
    python3 evals/run.py --task fix-bug --model qwen3:8b --think off --tag kvq8
    python3 evals/run.py --selftest           # no model: every check fails on the untouched workspace and passes on the reference solution
Results append to evals/results.jsonl (one line per run) and a summary prints at the end."""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from tasks import TASKS

def selftest() -> int:
    bad = 0
    for t in TASKS:
        d = tempfile.mkdtemp(prefix="evalself-")
        try:
            t.setup(d); fails_first, _ = t.check(d, "")
            t.solution(d); ref = {"count-lines": "50", "grep-todo": "src/a.py src/lib/c.py", "run-script": "8555", "json-extract": "Grace",
                                  "csv-sum": "1660", "missing-file": "that file does not exist", "paged-read": "OSPREY-2093"}.get(t.name, "")
            passes, why = t.check(d, ref)
            if fails_first or not passes: bad += 1; print(f"BAD  {t.name}: passes-untouched={fails_first} passes-with-solution={passes} {why}")
            else: print(f"ok   {t.name}")
        finally: shutil.rmtree(d, ignore_errors=True)
    return bad

def run_one(t, a) -> dict:
    d = tempfile.mkdtemp(prefix=f"eval-{t.name}-"); t.setup(d)
    cmd = [sys.executable, "-m", "harness", "-p", "--root", d, "--no-mcp", "--no-memory", "--output-format", "json", f"--think={a.think}", "-q"]
    if a.model: cmd += ["--model", a.model]
    if a.num_ctx: cmd += ["--num-ctx", str(a.num_ctx)]
    if a.server_args: cmd += [f"--llama-server-args={a.server_args}"]   # "=" form: a value starting with "-" is otherwise read as a flag
    if a.max_steps: cmd += ["--max-steps", str(a.max_steps)]
    t0 = time.time(); row = {"task": t.name, "model": a.model or "default", "think": a.think, "tag": a.tag, "ts": time.strftime("%F %T")}
    if a.sampling: cmd += [f"--sampling={a.sampling}"]; row["sampling"] = a.sampling
    try:
        p = subprocess.run(cmd + [t.prompt], cwd=ROOT, capture_output=True, text=True, timeout=a.timeout)
        try: j = json.loads(p.stdout); answer = j.get("result") or j.get("answer") or ""; row["usage"] = j.get("usage")
        except ValueError: answer = p.stdout; row["parse_error"] = True
        ok, why = t.check(d, answer)
        if row.get("parse_error") and ok: ok, why = False, "stdout was not one JSON object (harness output bug)"
        row.update(ok=bool(ok), why="" if ok else why, answer=answer[:300], rc=p.returncode)
        if p.returncode and not ok: row["stderr"] = p.stderr[-300:]
    except subprocess.TimeoutExpired: row.update(ok=False, why=f"timeout after {a.timeout}s")
    row["secs"] = round(time.time() - t0, 1); shutil.rmtree(d, ignore_errors=True); return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", action="append", default=[]); ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=""); ap.add_argument("--think", default="auto", choices=["on", "off", "auto"]); ap.add_argument("--num-ctx", type=int, default=0)
    ap.add_argument("--sampling", default="", help='JSON passed to alice --sampling, e.g. \'{"temperature": 0.7}\'')
    ap.add_argument("--server-args", default=""); ap.add_argument("--max-steps", type=int, default=20); ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list: [print(f"{t.name:16} {','.join(t.tags):22} {t.prompt[:70]}") for t in TASKS]; sys.exit(0)
    if a.selftest: sys.exit(1 if selftest() else 0)
    todo = [t for t in TASKS if not a.task or t.name in a.task]; rows = []
    for _ in range(a.repeat):
        for t in todo:
            row = run_one(t, a); rows.append(row); open(os.path.join(HERE, "results.jsonl"), "a").write(json.dumps(row) + "\n")
            print(f"{'PASS' if row['ok'] else 'FAIL'}  {t.name:16} {row['secs']:6.1f}s  {row.get('why', '')[:80]}", flush=True)
    n = sum(r["ok"] for r in rows); print(f"\n{n}/{len(rows)} passed · {sum(r['secs'] for r in rows):.0f}s · model={a.model or 'default'} think={a.think} tag={a.tag or '-'}")
