#!/usr/bin/env python3
"""Stand-in for llama-server: /health, /v1/models, /v1/chat/completions (streaming + tool calls). Records requests."""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
ALIAS = args[args.index("--alias") + 1] if "--alias" in args else "fake"
if "BAD" in ALIAS or "--refuse-me" in args: sys.exit(1)
if "--slot-save-path" in args and not os.path.isdir(args[args.index("--slot-save-path") + 1]): sys.exit(1)   # like the real server      # simulates a model or flag the server cannot start with
LOG, SLOTS = [], {}   # SLOTS: slot id -> requests run in it

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": self._json({"status": "ok"})
        elif self.path == "/v1/models": self._json({"data": [{"id": ALIAS}]})
        elif self.path == "/_log": self._json(LOG)
        elif self.path == "/_args": self._json(args)
        else: self._json({}, 404)
    def do_POST(self):
        if self.path.startswith("/slots/"):
            n = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["filename"]; d = args[args.index("--slot-save-path") + 1]
            act = self.path.split("action=")[1]
            slot = int(self.path.split("/slots/")[1].split("?")[0])
            if act == "save":
                with open(os.path.join(d, n), "w") as f: f.write("kv" * SLOTS.get(slot, 0))   # like the real server: an unused slot saves ~nothing
            elif not os.path.isfile(os.path.join(d, n)): return self._json({}, 404)
            return self._json({"ok": True})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); LOG.append(req)
        s = req.get("id_slot", 3); SLOTS[s] = SLOTS.get(s, 0) + 1   # unpinned requests land on slot 3, as the real 4-slot server's LRU did
        time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
        wants_tool = any(m["role"] == "tool" for m in req["messages"]) is False and bool(req.get("tools"))
        if not req.get("stream"):
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "today", "arguments": "{}"}}]} \
                if wants_tool else {"role": "assistant", "content": "done"}
            return self._json({"choices": [{"message": msg}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        ev = lambda d: self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
        if wants_tool:   # tool call split across chunks, as llama-server does
            ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "x", "function": {"name": "calcu", "arguments": ""}}]}}]})
            ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "late", "arguments": "{\"expr"}}]}}]})
            ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ession\": \"2+2\"}"}}]}}]})
        else:
            for w in ("do", "ne"): ev({"choices": [{"delta": {"content": w}}]})
        ev({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        self.wfile.write(b"data: [DONE]\n\n")

HTTPServer(("127.0.0.1", port), H).serve_forever()
