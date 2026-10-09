"""The agent loop: model <-> tools, with context guard, loop detection, a step budget,
and optional permission gate / post-tool hook / token streaming (used by the CLI)."""
import itertools, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctxguard import guard  # scripts/local-llm/ctxguard.py (single source of truth)

from . import profiles

DEFAULT_SYSTEM = (
    "You are an agent with tools. If a tool call returns an error (for example a temporary I/O error), "
    "retry the same call up to 3 times before giving up. Never ask the user to retry. When counting or "
    "summing, use the tools' output rather than estimating; if output is long, use grep to narrow it. "
    "You do not know the current date, time, or any live or local data: never guess these, call the matching tool. "
    "If a tool exists for a part of the question, call it instead of answering that part from memory."
)

PARALLEL_SAFE = {"Read", "Glob", "Grep", "WebFetch"}   # read-only tools that can run side by side

@dataclass
class Result:
    answer: str
    status: str  # "answered" | "step_limit"
    steps: int
    tool_calls: int
    messages: list = field(default_factory=list)

class Agent:
    def __init__(self, llm, toolbox, system: str = DEFAULT_SYSTEM, max_steps: int = 15,
                 on_event: Callable[[dict], None] | None = None, max_repeat: int = 3,
                 gate: Callable[[str, dict], str | None] | None = None,
                 post: Callable[[str, dict, str], str | None] | None = None,
                 on_token: Callable[[str], None] | None = None):
        """gate(name,args) -> None to allow, or a string returned to the model as the tool result (denial).
        post(name,args,result) -> optional extra text appended to the result."""
        self.llm, self.toolbox, self.max_steps, self.max_repeat = llm, toolbox, max_steps, max_repeat
        self.on_event = on_event or (lambda e: None)
        self.gate, self.post, self.on_token = gate, post, on_token
        self.messages: list[dict] = [{"role": "system", "content": system}] if system else []

    def _guarded(self, tools):
        return guard(self.messages, getattr(self.llm, "num_ctx", 12288), tools)

    def _chat(self, msgs, tools):
        kw = {"on_token": self.on_token} if self.on_token else {}
        return self.llm.chat(msgs, tools, **kw)

    def _precheck(self, name, args, seen):
        """None if the call may run; otherwise the text to hand back to the model (repeat-loop stop or permission denial)."""
        key = (name, json.dumps(args, sort_keys=True, default=str))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= self.max_repeat:
            return (f"error: you already made this exact call {seen[key]} times with the same result. "
                    "Try a different approach, or answer now with what you have.")
        return self.gate(name, args) if self.gate else None

    def _after(self, name, args, out):
        extra = self.post(name, args, out) if self.post else None
        return out + "\n" + extra if extra else out

    def reset(self):
        """Forget the conversation (keeps the system prompt)."""
        self.messages = [m for m in self.messages[:1] if m["role"] == "system"]

    def compact(self) -> str:
        """Replace the history with a model-written summary (the /compact command)."""
        body = [m for m in self.messages if m["role"] != "system"]
        if not body:
            return ""
        ask = {"role": "user", "content": "Summarize this conversation so far for your own future use: the user's goals, "
               "decisions made, files read or changed, results, and what remains. Be concise and factual."}
        m = self.llm.chat(guard(self.messages + [ask], getattr(self.llm, "num_ctx", 12288), None), None)
        summary = m.get("content") or ""
        sysm = [x for x in self.messages[:1] if x["role"] == "system"]
        self.messages = sysm + [{"role": "user", "content": "Summary of our conversation so far:\n" + summary},
                                {"role": "assistant", "content": "Understood. I have the context; ready to continue."}]
        return summary

    def ask(self, text: str, images: list | None = None) -> Result:
        """Add a user message and run until the model answers (keeps history across calls)."""
        self.messages.append({"role": "user", "content": text, **({"images": images} if images else {})})
        seen, n_calls = {}, 0
        for step in (range(1, self.max_steps + 1) if self.max_steps else itertools.count(1)):  # 0 = unlimited
            tools = self.toolbox.schemas()
            self.on_event({"type": "llm_start", "step": step})
            if getattr(self.llm, "think_mode", None): self.llm.think = profiles.want_thinking(self.llm.think_mode, text, step)
            m = self._chat(self._guarded(tools), tools)
            calls = m.get("tool_calls") or []
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if calls:
                msg["tool_calls"] = calls
            self.messages.append(msg)
            self.on_event({"type": "assistant_message", "message": msg, "step": step})
            if not calls:
                self.on_event({"type": "answer", "text": m.get("content") or ""})
                return Result(m.get("content") or "", "answered", step, n_calls, self.messages)
            parsed = []
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments") or {}
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except ValueError: args = {}
                parsed.append((name, args))
            n_calls += len(parsed)
            if len(parsed) > 1: self.on_event({"type": "tool_batch", "n": len(parsed)})
            if len(parsed) > 1 and all(n in PARALLEL_SAFE for n, _ in parsed):   # independent read-only calls: run together
                pres = []
                for name, args in parsed:
                    self.on_event({"type": "tool_call", "name": name, "args": args}); pres.append(self._precheck(name, args, seen))
                with ThreadPoolExecutor(max_workers=min(4, len(parsed))) as ex:
                    futs = [None if pre is not None else ex.submit(self.toolbox.call, n, a) for pre, (n, a) in zip(pres, parsed)]
                    outs = [pre if pre is not None else self._after(n, a, f.result()) for pre, f, (n, a) in zip(pres, futs, parsed)]
                for (name, args), out in zip(parsed, outs):
                    self.on_event({"type": "tool", "name": name, "args": args, "result": out})
                    self.messages.append({"role": "tool", "tool_name": name, "content": out})
                continue
            for name, args in parsed:
                self.on_event({"type": "tool_call", "name": name, "args": args})
                out = self._precheck(name, args, seen)
                if out is None: out = self._after(name, args, self.toolbox.call(name, args))
                self.on_event({"type": "tool", "name": name, "args": args, "result": out})
                self.messages.append({"role": "tool", "tool_name": name, "content": out})
        # out of steps: force a best-effort answer without tools
        self.messages.append({"role": "user", "content": "Step limit reached. Answer now with what you have found so far, and say what is uncertain."})
        m = self._chat(self._guarded(None), None)
        self.messages.append({"role": "assistant", "content": m.get("content") or ""})
        self.on_event({"type": "answer", "text": m.get("content") or ""})
        return Result(m.get("content") or "", "step_limit", self.max_steps, n_calls, self.messages)

    def run(self, task: str) -> Result:
        return self.ask(task)
