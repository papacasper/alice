"""MCP over HTTP: Streamable HTTP (JSON and SSE replies, session id) and legacy HTTP+SSE, against an in-process fake server."""
import json, os, queue, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from harness import mcp

TOOLS = [{"name": "echo", "description": "echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]

def reply(req):
    m, rid = req.get("method"), req.get("id")
    if m == "initialize": return {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": req["params"]["protocolVersion"], "capabilities": {"tools": {}}}}
    if m == "tools/list":
        if not req["params"].get("cursor"): return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS, "nextCursor": "p2"}}
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": [{**TOOLS[0], "name": "echo2"}]}}
    if m == "tools/call": return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "you said " + req["params"]["arguments"]["text"]}]}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no such method"}}

class Fake(BaseHTTPRequestHandler):
    mode = "json"          # json | sse | auth
    seen: list = []
    sse_q: "queue.Queue" = None
    def log_message(self, *a): pass
    def _body(self): return json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    def do_DELETE(self): Fake.seen.append(("DELETE", self.headers.get("Mcp-Session-Id"))); self.send_response(200); self.end_headers()
    def do_GET(self):   # legacy SSE stream
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(b"event: endpoint\ndata: /messages?sid=1\n\n"); self.wfile.flush()
        while (m := Fake.sse_q.get()) is not None:
            self.wfile.write(b"event: message\ndata: " + json.dumps(m).encode() + b"\n\n"); self.wfile.flush()
    def do_POST(self):
        req = self._body(); Fake.seen.append((self.path, dict(self.headers), req.get("method")))
        if self.path.startswith("/messages"):        # legacy: reply goes out on the GET stream
            self.send_response(202); self.end_headers()
            if "id" in req: Fake.sse_q.put(reply(req))
            return
        if Fake.mode == "auth": self.send_response(401); self.end_headers(); self.wfile.write(b"login first"); return
        if "id" not in req: self.send_response(202); self.end_headers(); return
        out = reply(req)
        if Fake.mode == "sse":
            note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
            b = (f"event: message\ndata: {json.dumps(note)}\n\n: keepalive\n\nevent: message\ndata: {json.dumps(out)}\n\n").encode()
            ctype = "text/event-stream"
        else: b, ctype = json.dumps(out).encode(), "application/json"
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        if req.get("method") == "initialize": self.send_header("Mcp-Session-Id", "sess-42")
        self.end_headers(); self.wfile.write(b)


class TestMcpHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake); cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        if Fake.sse_q: Fake.sse_q.put(None)
        cls.srv.shutdown(); cls.srv.server_close()

    def setUp(self): Fake.seen = []; Fake.mode = "json"

    def check(self, client):
        tools = client.tools()
        self.assertEqual([t.name for t in tools], ["mcp__x__echo", "mcp__x__echo2"])      # both pages of tools/list
        self.assertEqual(tools[0].call({"text": "hi"}), "you said hi")
        client.close()

    def test_streamable_http_json_reply_and_session(self):
        with mock.patch.dict(os.environ, {"MCP_TOKEN": "s3cret"}):
            c = mcp.connect("x", {"type": "http", "url": self.url + "/mcp", "headers": {"Authorization": "Bearer ${MCP_TOKEN}"}}, timeout=5)
        self.check(c)
        posts = [s for s in Fake.seen if s[0] == "/mcp"]
        self.assertEqual(posts[0][1].get("Authorization"), "Bearer s3cret")
        self.assertIsNone(posts[0][1].get("Mcp-Session-Id"))                              # none before initialize
        self.assertEqual({p[1].get("Mcp-Session-Id") for p in posts[1:]}, {"sess-42"})    # every later request carries it
        self.assertEqual(posts[2][1].get("Mcp-Protocol-Version"), "2025-06-18")
        self.assertIn(("DELETE", "sess-42"), Fake.seen)                                   # close() ends the session

    def test_streamable_http_sse_reply(self):
        Fake.mode = "sse"; self.check(mcp.connect("x", {"url": self.url + "/mcp"}, timeout=5))   # no type + url = http

    def test_legacy_sse(self):
        Fake.sse_q = queue.Queue()
        self.check(mcp.connect("x", {"type": "sse", "url": self.url + "/sse"}, timeout=5))
        self.assertTrue(any(s[0] == "/messages?sid=1" for s in Fake.seen))
        Fake.sse_q.put(None)

    def test_auth_error_names_the_fix(self):
        Fake.mode = "auth"
        with self.assertRaisesRegex(mcp.McpError, r"HTTP 401 \(the server wants a login"): mcp.connect("x", {"url": self.url + "/mcp"}, timeout=5)

    def test_expand_and_project_mcp_json_needs_approval(self):
        with mock.patch.dict(os.environ, {"A": "1"}, clear=False):
            os.environ.pop("NOPE", None)
            self.assertEqual(mcp.expand({"u": ["${A}", "${NOPE:-dflt}", "${NOPE}"]}), {"u": ["1", "dflt", ""]})
        d = tempfile.mkdtemp()
        with open(os.path.join(d, ".mcp.json"), "w") as f: json.dump({"mcpServers": {"p1": {"command": "x"}, "p2": {"command": "y"}}}, f)
        self.assertEqual(mcp.server_configs({}, d), ({}, ["p1", "p2"]))
        got, skipped = mcp.server_configs({"enabledMcpjsonServers": ["p2"], "mcpServers": {"s": {"command": "z"}}}, d)
        self.assertEqual((sorted(got), skipped), (["p2", "s"], ["p1"]))
        self.assertEqual(sorted(mcp.server_configs({"enableAllProjectMcpServers": True}, d)[0]), ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
