import json, sys, os, re, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxguard import guard
M, ROOT, N, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
truth = json.load(open(f"{ROOT}/truth.json"))
def safe(p):
    f = os.path.realpath(os.path.join(ROOT, p))
    if not f.startswith(os.path.realpath(ROOT)): raise ValueError("outside sandbox")
    return f
flaky_seen = set()
def mk(flaky):
    def list_dir(path="."): return "\n".join(sorted(os.listdir(safe(path))))
    def read_file(path):
        if flaky and path not in flaky_seen:
            flaky_seen.add(path); return "error: temporary I/O error, please retry"
        t = open(safe(path)).read()
        return t if len(t) <= 3000 else t[:3000] + f"\n[truncated: {len(t)-3000} more chars; use grep to search this file]"
    def grep(pattern, path):
        ls = [l for l in open(safe(path)).read().splitlines() if re.search(pattern, l)]
        return f"{len(ls)} matching lines\n" + "\n".join(ls[:60]) + ("\n[showing first 60]" if len(ls) > 60 else "")
    def calculate(expression):
        if not all(c in "0123456789+-*/(). " for c in expression): return "error: invalid chars"
        return str(eval(expression, {"__builtins__": {}}))
    return {"list_dir": list_dir, "read_file": read_file, "grep": grep, "calculate": calculate}
P = lambda **k: {"type":"object","properties":{n:{"type":"string"} for n in k},"required":list(k)}
tools = [{"type":"function","function":{"name":n,"description":d,"parameters":P(**{a:1 for a in args})}} for n,d,args in [
 ("list_dir","List files in a directory (relative path)",["path"]),
 ("read_file","Read a text file (relative path); long files are truncated",["path"]),
 ("grep","Regex search lines in a file; returns match count and matching lines",["pattern","path"]),
 ("calculate","Evaluate an arithmetic expression",["expression"])]]
TASKS = {
 "T1_logs": ("Using the tools: find which service logged the most ERROR lines in logs/app.log, report its port from conf/services.toml, "
             "and how many of that service's ERROR lines have a failure code of 500 or higher. Report service, port, and count. Use the tools; do not guess.", False,
             lambda a: truth["top"] in a.lower() and str(truth["port"]) in a and re.search(rf"\b{truth['hi']}\b", a) is not None),
 "T2_flaky": ("Read data/sales.csv and report the total of the amount column, to 2 decimals. Use the calculate tool for the arithmetic.", True,
             lambda a: f"{truth['sales']:.2f}" in a.replace(",", "")),
 "T3_missing": ("What port does the 'payments' service listen on? Check conf/services.toml. If it isn't configured, say so plainly.", False,
             lambda a: re.search(r"not (configured|found|present|listed|defined|in)|no (entry|service|payments|configuration)|isn't|is not|doesn't|does not|cannot find|couldn't find|unable", a, re.I) is not None
                       and not re.search(r"payments[^.\n]{0,40}\b(8100|8137|8174|8211|8248)\b", a, re.I)),
}
def chat(msgs):
    sent = guard(msgs, 12288, tools) if os.environ.get("GUARD") else msgs
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps({"model":M,"messages":sent,"tools":tools,"stream":False,"think":False,"options":{"num_ctx":12288}}).encode())
    return json.load(urllib.request.urlopen(req, timeout=600))["message"]
results = []
for name, (prompt, flaky, check) in [(k,v) for k,v in TASKS.items() if k.startswith(os.environ.get('ONLY',''))]:
    for i in range(N):
        flaky_seen.clear(); fn = mk(flaky); msgs = ([{"role":"system","content":os.environ["SYS"]}] if os.environ.get("SYS") else []) + [{"role":"user","content":prompt}]
        calls = errs = 0; ans = ""; t0 = time.time(); status = "no_answer"
        try:
            for step in range(15):
                m = chat(msgs); msgs.append(m)
                if not m.get("tool_calls"): ans = m.get("content") or ""; status = "answered"; break
                for c in m["tool_calls"]:
                    f, a = c["function"]["name"], c["function"]["arguments"]; calls += 1
                    try: out = fn[f](**a)
                    except Exception as e: out = f"error: {e}"
                    if out.startswith("error"): errs += 1
                    msgs.append({"role":"tool","tool_name":f,"content":out})
            else: status = "step_limit"
        except Exception as e: status = f"crash:{e}"[:60]
        ok = status == "answered" and bool(check(ans))
        results.append({"task":name,"run":i,"ok":ok,"status":status,"calls":calls,"tool_errors":errs,"secs":round(time.time()-t0,1),"answer":ans[:400]})
        print(name, i, "PASS" if ok else "FAIL", status, f"calls={calls} errs={errs} {results[-1]['secs']}s", flush=True)
json.dump(results, open(OUT, "w"), indent=1)
