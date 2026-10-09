"""MCP client: tools/list + tools/call over the three transports Claude Code uses.
settings.json (or a project .mcp.json, same shape as Claude Code's):
  {"mcpServers": {
     "local":  {"command": "npx", "args": ["..."], "env": {}},                                  # stdio (newline-delimited JSON-RPC)
     "remote": {"type": "http", "url": "https://host/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}},   # Streamable HTTP
     "old":    {"type": "sse", "url": "https://host/sse"}}}                                     # legacy HTTP+SSE
${VAR} and ${VAR:-default} are expanded from the environment in command, args, env, url and headers.
Project .mcp.json servers only start when settings allow them (enableAllProjectMcpServers, or enabledMcpjsonServers: [names]),
because a cloned repo could otherwise run any command. No OAuth: servers that need a login want a token in "headers".
Tools are exposed to the model as mcp__<server>__<tool>."""
import http.client, json, os, queue, re, socket, subprocess, threading, urllib.error, urllib.parse, urllib.request
from .tools import Tool
from .fsutil import read_text

class McpError(RuntimeError): pass

CLIENT_INFO = {"name": "alice", "version": "1.0"}

class _Client:
    """Shared protocol: initialize handshake, tools, call. Subclasses implement _request(msg) -> response dict and _notify(msg)."""
    name: str
    timeout: int
    protocol = "2024-11-05"

    def _next_id(self) -> int:
        self._id = getattr(self, "_id", 0) + 1
        return self._id

    def _handshake(self):
        res = self._rpc("initialize", {"protocolVersion": self.protocol, "capabilities": {}, "clientInfo": CLIENT_INFO})
        self.negotiated = res.get("protocolVersion", self.protocol)
        self._notify({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        m = self._request({"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params or {}})
        if "error" in m: raise McpError(f"{self.name}: {m['error'].get('message')}")
        return m.get("result") or {}

    def tools(self) -> list["McpTool"]:
        out, cursor = [], None
        while True:          # tools/list is paginated (nextCursor)
            res = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            out += [McpTool(self, t) for t in res.get("tools", [])]
            cursor = res.get("nextCursor")
            if not cursor: return out

    def call(self, tool: str, args: dict) -> str:
        res = self._rpc("tools/call", {"name": tool, "arguments": args})
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if not text and res.get("structuredContent") is not None: text = json.dumps(res["structuredContent"])
        return ("error: " if res.get("isError") else "") + (text or json.dumps(res.get("content", "")))

    def close(self): pass


class McpServer(_Client):
    """stdio transport: a child process speaking newline-delimited JSON-RPC."""
    def __init__(self, name: str, command: str, args: list | None = None, env: dict | None = None, timeout: int = 30):
        self.name, self.timeout = name, timeout
        self.proc = subprocess.Popen([command, *(args or [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, env={**os.environ, **(env or {})})
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        try: self._handshake()
        except BaseException: self.close(); raise

    def _reader(self):
        for line in self.proc.stdout:
            try: self._q.put(json.loads(line))
            except ValueError: pass
        self._q.put(None)

    def _send(self, msg: dict):
        self.proc.stdin.write(json.dumps(msg) + "\n"); self.proc.stdin.flush()

    _notify = _send

    def _request(self, msg: dict) -> dict:
        self._send(msg)
        while True:
            try: m = self._q.get(timeout=self.timeout)
            except queue.Empty: raise McpError(f"{self.name}: timeout waiting for {msg['method']}")
            if m is None: raise McpError(f"{self.name}: server exited")
            if m.get("id") == msg["id"]: return m

    def close(self):
        try: self.proc.terminate(); self.proc.wait(2)
        except (OSError, subprocess.TimeoutExpired): pass
        for f in (self.proc.stdin, self.proc.stdout):
            try: f and f.close()
            except OSError: pass


def sse_events(stream):
    """Parse a text/event-stream: yields (event, data) for each complete event."""
    event, data = "message", []
    for raw in stream:
        line = raw.decode("utf-8", "replace").rstrip("\r\n") if isinstance(raw, bytes) else raw.rstrip("\r\n")
        if not line:
            if data: yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith(":"): continue
        else:
            k, _, v = line.partition(":"); v = v[1:] if v.startswith(" ") else v
            if k == "event": event = v
            elif k == "data": data.append(v)
    if data: yield event, "\n".join(data)


def _http_error(name: str, e: urllib.error.HTTPError) -> McpError:
    body = ""
    try: body = e.read()[:200].decode("utf-8", "replace")
    except OSError: pass
    finally: e.close()
    hint = " (the server wants a login: put a token in \"headers\")" if e.code in (401, 403) else ""
    return McpError(f"{name}: HTTP {e.code}{hint} {body}".strip())


class McpHttpServer(_Client):
    """Streamable HTTP transport (MCP 2025-03-26+): every message is a POST; the reply is JSON or an SSE stream."""
    protocol = "2025-06-18"

    def __init__(self, name: str, url: str, headers: dict | None = None, timeout: int = 30):
        self.name, self.url, self.headers, self.timeout, self.session = name, url, headers or {}, timeout, ""
        self._handshake()

    def _post(self, msg: dict):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.headers}
        if self.session: h["Mcp-Session-Id"] = self.session
        if getattr(self, "negotiated", None): h["MCP-Protocol-Version"] = self.negotiated
        req = urllib.request.Request(self.url, json.dumps(msg).encode(), h, method="POST")
        try: return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e: raise _http_error(self.name, e)
        except (urllib.error.URLError, OSError) as e: raise McpError(f"{self.name}: {getattr(e, 'reason', e)}")

    def _notify(self, msg: dict):
        with self._post(msg) as r: r.read()

    def _request(self, msg: dict) -> dict:
        with self._post(msg) as r:
            if sid := r.headers.get("Mcp-Session-Id"): self.session = sid
            if "text/event-stream" in r.headers.get("Content-Type", ""):
                for _ev, data in sse_events(r):
                    try: m = json.loads(data)
                    except ValueError: continue
                    if isinstance(m, dict) and m.get("id") == msg["id"] and ("result" in m or "error" in m): return m
                raise McpError(f"{self.name}: stream ended without a reply to {msg['method']}")
            try: m = json.loads(r.read() or b"{}")
            except ValueError: raise McpError(f"{self.name}: reply to {msg['method']} was not JSON")
            if isinstance(m, list): m = next((x for x in m if x.get("id") == msg["id"]), {})
            return m

    def close(self):
        if not self.session: return
        try:
            req = urllib.request.Request(self.url, headers={**self.headers, "Mcp-Session-Id": self.session}, method="DELETE")
            urllib.request.urlopen(req, timeout=5).close()
        except Exception: pass   # ending the session is a courtesy; the server may not support DELETE (405)


class McpSseServer(_Client):
    """Legacy HTTP+SSE transport (MCP 2024-11-05): a long GET stream carries replies; requests are POSTed to the endpoint it names."""
    def __init__(self, name: str, url: str, headers: dict | None = None, timeout: int = 30):
        self.name, self.url, self.headers, self.timeout = name, url, headers or {}, timeout
        self._q: queue.Queue = queue.Queue(); self._ready = threading.Event(); self.endpoint = ""; self._conn = self._sock = None; self._err = ""
        threading.Thread(target=self._reader, daemon=True).start()
        if not self._ready.wait(timeout): self.close(); raise McpError(f"{name}: no endpoint event from {url}")
        if self._err: raise McpError(self._err)
        self._handshake()

    def _reader(self):
        u = urllib.parse.urlsplit(self.url)
        conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        try:
            self._conn = conn_cls(u.netloc, timeout=self.timeout)
            self._conn.request("GET", (u.path or "/") + (f"?{u.query}" if u.query else ""), headers={"Accept": "text/event-stream", **self.headers})
            self._sock = self._conn.sock          # getresponse() may hand the socket to the response and clear conn.sock
            r = self._resp = self._conn.getresponse()
            if r.status != 200: self._err = f"{self.name}: HTTP {r.status} {r.read(200).decode('utf-8', 'replace')}".strip(); return
            self._sock.settimeout(None)    # an idle stream is normal; __init__ bounds the wait for the endpoint event
            for ev, data in sse_events(r):
                if ev == "endpoint": self.endpoint = urllib.parse.urljoin(self.url, data.strip()); self._ready.set()
                elif ev == "message":
                    try: self._q.put(json.loads(data))
                    except ValueError: pass
        except Exception as e:
            if not self._ready.is_set(): self._err = f"{self.name}: {e}"
        finally:
            self._ready.set(); self._q.put(None)
            for c in (getattr(self, "_resp", None), self._conn, self._sock):
                try: c and c.close()
                except Exception: pass

    def _post(self, msg: dict):
        req = urllib.request.Request(self.endpoint, json.dumps(msg).encode(), {"Content-Type": "application/json", **self.headers}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r: r.read()
        except urllib.error.HTTPError as e: raise _http_error(self.name, e)
        except (urllib.error.URLError, OSError) as e: raise McpError(f"{self.name}: {getattr(e, 'reason', e)}")

    _notify = _post

    def _request(self, msg: dict) -> dict:
        self._post(msg)
        while True:
            try: m = self._q.get(timeout=self.timeout)
            except queue.Empty: raise McpError(f"{self.name}: timeout waiting for {msg['method']}")
            if m is None: raise McpError(f"{self.name}: event stream closed")
            if m.get("id") == msg["id"]: return m

    def close(self):
        """Shut the socket down (closing the buffered stream would block on the reader thread's lock); the reader then exits."""
        try: self._sock and self._sock.shutdown(socket.SHUT_RDWR)
        except Exception: pass


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

def expand(v):
    """${VAR} / ${VAR:-default} in strings, recursively through lists and dicts (as Claude Code's .mcp.json allows)."""
    if isinstance(v, str): return _VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), v)
    if isinstance(v, list): return [expand(x) for x in v]
    if isinstance(v, dict): return {k: expand(x) for k, x in v.items()}
    return v

def connect(name: str, cfg: dict, timeout: int = 30) -> _Client:
    cfg = expand(cfg)
    kind = cfg.get("type") or ("http" if cfg.get("url") else "stdio")
    if kind in ("http", "streamable-http", "streamableHttp"): return McpHttpServer(name, cfg["url"], cfg.get("headers"), timeout)
    if kind == "sse": return McpSseServer(name, cfg["url"], cfg.get("headers"), timeout)
    if kind == "stdio": return McpServer(name, cfg["command"], cfg.get("args"), cfg.get("env"), timeout)
    raise McpError(f"{name}: unknown MCP transport type '{kind}'")

def server_configs(settings: dict, cwd: str) -> tuple[dict, list[str]]:
    """-> (servers to start, names in .mcp.json that were skipped because they are not approved). settings.json wins on a name clash."""
    try: project = json.loads(read_text(os.path.join(cwd, ".mcp.json"))).get("mcpServers") or {}
    except (OSError, ValueError, AttributeError): project = {}
    allowed = settings.get("enabledMcpjsonServers") or []
    ok = {n: c for n, c in project.items() if settings.get("enableAllProjectMcpServers") or n in allowed}
    skipped = [n for n in project if n not in ok and n not in (settings.get("mcpServers") or {})]
    return {**ok, **(settings.get("mcpServers") or {})}, skipped


class McpTool(Tool):
    """A Tool whose schema comes straight from the server (no Python signature)."""
    def __init__(self, server: _Client, spec: dict):
        self.server, self.remote = server, spec["name"]
        self.name = f"mcp__{server.name}__{spec['name']}"
        self.description = (spec.get("description") or spec["name"])[:300]
        self._schema = spec.get("inputSchema") or {"type": "object", "properties": {}}
        self.params, self.required = self._schema.get("properties", {}), self._schema.get("required", [])

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params, "required": self.required}}}

    def call(self, args: dict) -> str:
        try: out = self.server.call(self.remote, args if isinstance(args, dict) else {})
        except Exception as e: return f"error: {type(e).__name__}: {e}"
        return out if len(out) <= 6000 else out[:6000] + f"\n[truncated: {len(out) - 6000} more chars]"
