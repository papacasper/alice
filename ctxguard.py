"""ctxguard: keep an Ollama agent loop under num_ctx without ever losing the task.

Ollama silently drops the OLDEST messages when a chat exceeds num_ctx. The first user
message (the task) goes first; Qwen3.5's template then raises 'No user query found'
(HTTP 500) and other models just forget what they were asked. guard() compacts old tool
outputs (then old assistant/tool turns) on the client side so the task is always kept.
"""
import json

def est_tokens(s: str) -> int:
    return len(s) // 2 + 4  # conservative: log/number-heavy text is ~2 chars/token

def _msg_cost(m: dict) -> int:
    return est_tokens(m.get("content") or "") + est_tokens(json.dumps(m.get("tool_calls") or ""))

def guard(msgs: list[dict], num_ctx: int, tools: list | None = None,
          reserve: int = 1500, keep_recent_tools: int = 2) -> list[dict]:
    """Return a copy of msgs whose estimated size fits num_ctx-reserve-tools. Never drops
    system messages or the first user message; compacts oldest tool outputs first."""
    out = [dict(m) for m in msgs]
    budget = num_ctx - reserve - est_tokens(json.dumps(tools or []))
    total = lambda: sum(_msg_cost(m) for m in out)
    first_user = next((i for i, m in enumerate(out) if m["role"] == "user"), 0)
    tool_idx = [i for i, m in enumerate(out) if m["role"] == "tool"]
    for i in tool_idx[:-keep_recent_tools] if keep_recent_tools else tool_idx:  # 1) elide old tool output
        if total() <= budget: return out
        c = out[i].get("content") or ""
        if not c.startswith("[elided"):
            out[i]["content"] = f"[elided older tool output: {c.count(chr(10))+1} lines; call the tool again if needed]"
    while total() > budget:  # 2) drop oldest non-pinned turns
        drop = next((i for i in range(len(out) - 2) if i != first_user and out[i]["role"] != "system"), None)
        if drop is None: break
        out.pop(drop)
    return out
