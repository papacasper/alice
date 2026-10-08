import json, sys, urllib.request
M, CTX, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
big = "\n".join(f"2026-10-01 ERROR [billing] failure code {i}" for i in range(60))
msgs = [{"role":"user","content":"Find the service with the most errors. Use tools."}]
for i in range(N):
    msgs.append({"role":"assistant","content":"","tool_calls":[{"function":{"name":"grep","arguments":{"pattern":"ERROR","path":"logs/app.log"}}}]})
    msgs.append({"role":"tool","tool_name":"grep","content":big})
tools=[{"type":"function","function":{"name":"grep","description":"search","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern","path"]}}}]
req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps({"model":M,"messages":msgs,"tools":tools,"stream":False,"think":False,"options":{"num_ctx":CTX,"num_predict":20}}).encode())
try:
    r = json.load(urllib.request.urlopen(req, timeout=300)); print(f"N={N} ctx={CTX}: OK prompt_tokens={r.get('prompt_eval_count')}")
except urllib.error.HTTPError as e: print(f"N={N} ctx={CTX}: HTTP {e.code}", e.read()[:120])
