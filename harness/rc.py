"""Remote control (/rc): a small web page + JSON API so the running session can be watched and driven from another
device (phone, laptop). Binds to the Tailscale address when there is one, else loopback only; every URL carries a
random token, and a wrong token is a plain 404. Prompts go through the same path as typed ones."""
import hmac, json, secrets, shutil, subprocess, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_EVENTS, MAX_BODY, MAX_RESULT = 600, 64_000, 2500

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Alice remote</title><style>
:root{--bg:#1a1816;--fg:#e8e3dc;--dim:#8a847b;--acc:#d97757;--box:#24211e}
@media(prefers-color-scheme:light){:root{--bg:#faf7f2;--fg:#2a2622;--dim:#7a746b;--box:#efe9df}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 ui-monospace,Menlo,monospace;display:flex;flex-direction:column;height:100dvh}
header{padding:10px 16px;border-bottom:1px solid var(--box);display:flex;gap:12px;justify-content:space-between;color:var(--dim)}
header b{color:var(--acc)}#log{flex:1;overflow:auto;padding:12px 16px}
.u{color:var(--acc);margin-top:14px;white-space:pre-wrap}.u:before{content:"> "}.a{margin-top:10px;white-space:pre-wrap}.a:before{content:"● ";color:var(--acc)}
.t{color:var(--dim);margin-top:8px}.t:before{content:"● ";color:var(--acc)}.r{color:var(--dim);white-space:pre-wrap;margin-left:16px;max-height:9em;overflow:auto}
form{display:flex;gap:8px;padding:10px 16px;border-top:1px solid var(--box)}
textarea{flex:1;background:var(--box);color:var(--fg);border:1px solid var(--box);border-radius:6px;padding:8px;font:inherit;resize:none;height:3.2em}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:0 14px;font:inherit}button.s{background:var(--box);color:var(--dim)}
</style><header><span><b>✻ Alice</b> <span id=m></span></span><span id=st>…</span></header><div id=log></div>
<form id=f><textarea id=x placeholder="message Alice (Enter sends, Shift+Enter newline)" autofocus></textarea><button>Send</button><button type=button class=s id=stop>Stop</button></form>
<script>
const B=location.pathname.replace(/\\/$/,"");let n=0,log=document.getElementById("log");
function add(c,t){const d=document.createElement("div");d.className=c;d.textContent=t;log.appendChild(d)}
async function poll(){try{const r=await fetch(B+"/state?since="+n);const s=await r.json();
 document.getElementById("m").textContent=s.model+" · "+s.cwd;document.getElementById("st").textContent=s.busy?"working…":"idle";
 const bottom=log.scrollTop+log.clientHeight>=log.scrollHeight-40;
 for(const e of s.events){n=e.i+1;add(e.kind==="user"?"u":e.kind==="answer"?"a":e.kind==="call"?"t":"r",e.text)}
 if(s.events.length&&bottom)log.scrollTop=log.scrollHeight}catch(e){document.getElementById("st").textContent="offline"}setTimeout(poll,1200)}
async function send(){const x=document.getElementById("x"),t=x.value.trim();if(!t)return;x.value="";
 await fetch(B+"/send",{method:"POST",body:JSON.stringify({text:t})})}
document.getElementById("f").onsubmit=e=>{e.preventDefault();send()};
document.getElementById("x").onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}};
document.getElementById("stop").onclick=()=>fetch(B+"/stop",{method:"POST"});poll();
</script>"""

def tailscale_ip() -> str:
    if not shutil.which("tailscale"): return ""
    try: r = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=4)
    except (OSError, subprocess.SubprocessError): return ""
    ip = r.stdout.split()[0] if r.returncode == 0 and r.stdout.split() else ""
    return ip if ip.startswith("100.") else ""

class RemoteControl:
    def __init__(self, app):
        self.app, self.token, self.events, self._n = app, secrets.token_urlsafe(24), [], 0
        self.server = None; self.url = ""; self._lock = threading.Lock()

    # ---- event log (what the remote page shows)
    def record(self, kind: str, text: str):
        with self._lock:
            self.events.append({"i": self._n, "kind": kind, "text": text[:MAX_RESULT]}); self._n += 1
            del self.events[:-MAX_EVENTS]

    def record_event(self, e: dict):
        from .ui import short_args
        k = e.get("type")
        if k == "tool_call": self.record("call", f"{e['name']}({short_args(e['name'], e['args'])})")
        elif k == "tool": self.record("result", e["result"])
        elif k == "answer" and e.get("text"): self.record("answer", e["text"])

    def state(self, since: int) -> dict:
        with self._lock: ev = [e for e in self.events if e["i"] >= since]
        return {"busy": not self.app.at_prompt, "model": self.app.llm.model.split("/")[-1], "cwd": self.app.cwd, "events": ev}

    # ---- lifecycle
    def start(self, local_only: bool = False) -> str:
        if self.server: return self.url
        host = "127.0.0.1" if local_only else (tailscale_ip() or "127.0.0.1")
        rc = self
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, body, ctype="application/json"):
                b = body if isinstance(body, bytes) else body.encode()
                self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(b)
            def _route(self):
                path, _, query = self.path.partition("?"); parts = path.strip("/").split("/")
                if not parts or not hmac.compare_digest(parts[0].encode(), rc.token.encode()): self._send(404, "not found", "text/plain"); return None
                return "/" + "/".join(parts[1:]), query
            def do_GET(self):
                r = self._route()
                if not r: return
                p, q = r
                if p == "/": self._send(200, PAGE, "text/html")
                elif p == "/state":
                    since = int(q.split("since=")[1].split("&")[0]) if "since=" in q and q.split("since=")[1].split("&")[0].isdigit() else 0
                    self._send(200, json.dumps(rc.state(since)))
                else: self._send(404, "not found", "text/plain")
            def do_POST(self):
                r = self._route()
                if not r: return
                p, _ = r
                n = min(int(self.headers.get("Content-Length") or 0), MAX_BODY); body = self.rfile.read(n)
                if p == "/send":
                    try: text = str(json.loads(body or b"{}").get("text", "")).strip()
                    except ValueError: text = ""
                    if not text: self._send(400, '{"error":"empty"}'); return
                    rc.app.rc_submit(text); self._send(200, '{"ok":true}')
                elif p == "/stop": rc.app.rc_interrupt(); self._send(200, '{"ok":true}')
                else: self._send(404, "not found", "text/plain")
        self.server = ThreadingHTTPServer((host, 0), H); self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://{host}:{self.server.server_address[1]}/{self.token}/"
        return self.url

    def stop(self):
        if self.server: self.server.shutdown(); self.server.server_close(); self.server = None; self.url = ""
