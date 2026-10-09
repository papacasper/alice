"""--output-format stream-json / --input-format stream-json: Claude Code's headless NDJSON protocol.

Out (one JSON object per line on stdout):
  {"type":"system","subtype":"init","session_id","cwd","model","tools":[...],"mcp_servers":[...],"permissionMode",...}
  {"type":"assistant","message":{"role":"assistant","content":[{"type":"text",...},{"type":"tool_use","id","name","input"}]},...}
  {"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id","content","is_error"}]},...}
  {"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":...}}}   (--include-partial-messages)
  {"type":"result","subtype":"success"|"error_max_turns"|"error_during_execution","is_error","result","num_turns","duration_ms",
   "usage":{"input_tokens","output_tokens"},"total_cost_usd":0,...}
In (--input-format stream-json): one {"type":"user","message":{"role":"user","content": "text" | [{"type":"text","text":...},
  {"type":"image","source":{"type":"base64","data":...}}]}} per line; each one is a turn and gets its own result line."""
import json, sys, time, uuid
from collections import deque

class StreamJson:
    def __init__(self, app, out=None):
        self.app, self.out = app, out or sys.stdout
        self.pending: deque = deque()     # tool_use ids waiting for their results, in call order

    def emit(self, obj: dict):
        obj.setdefault("session_id", self.app.sid)
        self.out.write(json.dumps(obj, default=str) + "\n"); self.out.flush()

    def init(self):
        a = self.app
        self.emit({"type": "system", "subtype": "init", "cwd": a.cwd, "model": a.llm.model, "tools": a.toolbox.names(),
                   "mcp_servers": [{"name": s.name, "status": "connected"} for s in a.mcp_servers],
                   "permissionMode": a.perms.mode, "slash_commands": sorted(a.commands), "apiKeySource": "none",
                   "output_style": a.style, "agents": sorted(a.state.agents), "skills": sorted(a.state.skills), "uuid": str(uuid.uuid4())})

    def token(self, text: str):
        """on_token sink for --include-partial-messages."""
        if text: self.emit({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                            "parent_tool_use_id": None, "uuid": str(uuid.uuid4())})

    def event(self, e: dict):
        k = e.get("type")
        if k == "assistant_message":
            m, blocks = e["message"], []
            if m.get("content"): blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                args = c["function"].get("arguments") or {}
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except ValueError: args = {"_raw": args}
                tid = c.get("id") or "toolu_" + uuid.uuid4().hex[:24]
                self.pending.append(tid)
                blocks.append({"type": "tool_use", "id": tid, "name": c["function"]["name"], "input": args})
            self.emit({"type": "assistant", "message": {"role": "assistant", "model": self.app.llm.model, "content": blocks},
                       "parent_tool_use_id": None, "uuid": str(uuid.uuid4())})
        elif k == "tool":
            out = str(e["result"])
            self.emit({"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": out,
                       "tool_use_id": self.pending.popleft() if self.pending else "", "is_error": out.startswith("error")}]},
                       "parent_tool_use_id": None, "uuid": str(uuid.uuid4())})

    def result(self, r, t0: float, usage0: dict):
        u = getattr(self.app.llm, "usage", {}) or {}
        sub = "error_during_execution" if r is None else ("error_max_turns" if r.status == "step_limit" else "success")
        self.pending.clear()
        self.emit({"type": "result", "subtype": sub, "is_error": sub != "success", "result": r.answer if r else "",
                   "num_turns": r.steps if r else 0, "duration_ms": int((time.time() - t0) * 1000),
                   "duration_api_ms": int(u.get("prompt_ms", 0) + u.get("gen_ms", 0) - usage0.get("prompt_ms", 0) - usage0.get("gen_ms", 0)),
                   "total_cost_usd": 0, "usage": {"input_tokens": u.get("prompt_tokens", 0) - usage0.get("prompt_tokens", 0),
                                                  "output_tokens": u.get("completion_tokens", 0) - usage0.get("completion_tokens", 0),
                                                  "cache_read_input_tokens": u.get("cached_tokens", 0) - usage0.get("cached_tokens", 0)},
                   "uuid": str(uuid.uuid4())})
        return sub == "success"

def read_inputs(stream):
    """Yield (text, images) for each user message line of --input-format stream-json; bad lines are reported on stderr and skipped."""
    for n, line in enumerate(stream, 1):
        if not line.strip(): continue
        try:
            m = json.loads(line)
            if m.get("type") != "user": continue
            content = m["message"]["content"]
        except (ValueError, KeyError, TypeError, AttributeError):
            print(f"alice: stream-json input line {n} is not a user message; skipped", file=sys.stderr); continue
        if isinstance(content, str): yield content, []; continue
        text = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        imgs = [b["source"]["data"] for b in content if isinstance(b, dict) and b.get("type") == "image" and (b.get("source") or {}).get("data")]
        yield text, imgs
