"""Minimal MCP client (stdio, newline-delimited JSON-RPC): tools/list + tools/call.
settings.json: {"mcpServers": {"name": {"command": "npx", "args": ["..."], "env": {}}}}
Tools are exposed to the model as mcp__<server>__<tool>."""
import json, os, queue, subprocess, threading
from .tools import Tool

class McpError(RuntimeError): pass

class McpServer:
    def __init__(self, name: str, command: str, args: list | None = None, env: dict | None = None, timeout: int = 30):
        self.name, self.timeout, self._id = name, timeout, 0
        self.proc = subprocess.Popen([command, *(args or [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, env={**os.environ, **(env or {})})
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "alice", "version": "1.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _reader(self):
        for line in self.proc.stdout:
            try: self._q.put(json.loads(line))
            except ValueError: pass
        self._q.put(None)

    def _send(self, msg: dict):
        self.proc.stdin.write(json.dumps(msg) + "\n"); self.proc.stdin.flush()

    def _rpc(self, method: str, params: dict | None = None):
        self._id += 1; rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            try: m = self._q.get(timeout=self.timeout)
            except queue.Empty: raise McpError(f"{self.name}: timeout waiting for {method}")
            if m is None: raise McpError(f"{self.name}: server exited")
            if m.get("id") == rid:
                if "error" in m: raise McpError(f"{self.name}: {m['error'].get('message')}")
                return m.get("result") or {}

    def tools(self) -> list["McpTool"]:
        return [McpTool(self, t) for t in self._rpc("tools/list").get("tools", [])]

    def call(self, tool: str, args: dict) -> str:
        res = self._rpc("tools/call", {"name": tool, "arguments": args})
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        return ("error: " if res.get("isError") else "") + (text or json.dumps(res.get("content", "")))

    def close(self):
        try: self.proc.terminate(); self.proc.wait(2)
        except (OSError, subprocess.TimeoutExpired): pass

class McpTool(Tool):
    """A Tool whose schema comes straight from the server (no Python signature)."""
    def __init__(self, server: McpServer, spec: dict):
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
