"""LLM backends (stdlib only): OllamaClient (/api/chat) and OpenAIClient (/v1/chat/completions: llama.cpp llama-server, LM Studio, vLLM, ...)."""
from . import profiles
import glob, re, sys, atexit, ctypes, json, signal, os, shutil, socket, subprocess, time, urllib.error, urllib.request
from .fsutil import read_json, read_text

DEFAULT_MODEL = "hf.co/mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF:Q4_K_M"

class LLMError(RuntimeError):
    pass

class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = "http://localhost:11434",
                 num_ctx: int = 12288, think: bool = False, temperature: float | None = None,
                 timeout: int = 600, retries: int = 2):
        self.model, self.host, self.num_ctx, self.think = model, host.rstrip("/"), num_ctx, think
        self.temperature, self.timeout, self.retries = temperature, timeout, retries
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    kind = "ollama"

    def _get(self, path, timeout=3):
        with urllib.request.urlopen(self.host + path, timeout=timeout) as r: return json.load(r)

    def health(self) -> tuple[bool, str]:
        try: return True, "ollama " + (self._get("/api/version").get("version") or "?")
        except Exception as e: return False, f"ollama unreachable ({e})"

    def models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags", 5)["models"]]

    def _count(self, d: dict):
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += d.get("prompt_eval_count") or 0
        self.usage["completion_tokens"] += d.get("eval_count") or 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None, on_token=None) -> dict:
        """Return the assistant message. With on_token, streams and calls on_token(text) per chunk."""
        options = {"num_ctx": self.num_ctx}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        body = {"model": self.model, "messages": messages, "stream": bool(on_token),
                "think": self.think, "options": options}
        if tools:
            body["tools"] = tools
        data = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(f"{self.host}/api/chat", data, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if not on_token:
                        d = json.load(r); self._count(d)
                        return d["message"]
                    text, calls = [], []
                    for line in r:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        m = d.get("message") or {}
                        if m.get("content"):
                            text.append(m["content"]); on_token(m["content"])
                        calls.extend(m.get("tool_calls") or [])
                        if d.get("done"):
                            self._count(d)
                    out = {"role": "assistant", "content": "".join(text)}
                    if calls:
                        out["tool_calls"] = calls
                    return out
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace"); e.close()
                if e.code < 500 or attempt == self.retries:
                    raise LLMError(f"HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise LLMError(f"cannot reach Ollama at {self.host}: {e}") from None
            time.sleep(1.5 * (attempt + 1))
        raise LLMError("unreachable")


def to_openai(messages: list[dict]) -> list[dict]:
    """Ollama-shaped history -> OpenAI shape: JSON-string tool arguments, ids on tool calls, tool_call_id on results."""
    out, pending, n = [], [], 0
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                n += 1; cid = c.get("id") or f"call_{n}"; pending.append(cid)
                a = c["function"].get("arguments") or {}
                calls.append({"id": cid, "type": "function",
                              "function": {"name": c["function"]["name"], "arguments": a if isinstance(a, str) else json.dumps(a)}})
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "call_0", "content": m.get("content") or ""})
        else:
            if m["role"] == "user" and m.get("images"):
                parts = [{"type": "text", "text": m.get("content") or ""}] + [
                    {"type": "image_url", "image_url": {"url": f"data:{_mime(b)};base64,{b}"}} for b in m["images"]]
                out.append({"role": "user", "content": parts})
            else: out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def _mime(b64: str) -> str:
    return {"iVBOR": "image/png", "/9j/": "image/jpeg", "R0lGO": "image/gif", "UklGR": "image/webp"}.get(
        next((k for k in ("iVBOR", "/9j/", "R0lGO", "UklGR") if b64.startswith(k)), ""), "image/png")


class OpenAIClient:
    """OpenAI-compatible chat client. Same interface as OllamaClient, so the agent doesn't care which is behind it."""
    kind = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = "http://127.0.0.1:8080", num_ctx: int = 12288,
                 think: bool = False, temperature: float | None = None, timeout: int = 600, retries: int = 2,
                 api_key: str = ""):
        self.sampling, self.think_mode = {}, "auto" if think == "auto" else ("on" if think else "off")
        think = think is True
        self.model, self.host, self.num_ctx, self.think = model, host.rstrip("/"), num_ctx, think
        self.temperature, self.timeout, self.retries, self.api_key = temperature, timeout, retries, api_key
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "prompt_ms": 0.0, "gen_ms": 0.0}
        self.last: dict = {}   # the latest request's token counts (status line "current_usage")
        # llama-server slot to run in (None: the server picks). Alice's own server pins the conversation to slot 0, the one
        # save_slot/restore_slot use; with 4 slots the server otherwise picked any of them and the saved KV cache was empty.
        self.slot: int | None = None

    def _req(self, path, data=None):
        h = {"Content-Type": "application/json"}
        if self.api_key: h["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(self.host + path, data, h)

    def health(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(self._req("/v1/models"), timeout=3) as r: json.load(r)
            return True, f"openai-compatible server at {self.host}"
        except Exception as e: return False, f"no server at {self.host} ({e})"

    def models(self) -> list[str]:
        with urllib.request.urlopen(self._req("/v1/models"), timeout=5) as r:
            return [m["id"] for m in json.load(r).get("data", [])]

    def _timing(self, t):
        """llama-server's per-request timings: how much of the prompt came from the cache, and how long prompt/generation took."""
        if not t: return
        self.usage["cached_tokens"] += t.get("cache_n") or 0
        self.last["cached_tokens"] = t.get("cache_n") or 0
        self.usage["prompt_ms"] += t.get("prompt_ms") or 0
        self.usage["gen_ms"] += t.get("predicted_ms") or 0

    def _count(self, u):
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += (u or {}).get("prompt_tokens") or 0
        self.usage["completion_tokens"] += (u or {}).get("completion_tokens") or 0
        self.last.update(prompt_tokens=(u or {}).get("prompt_tokens") or 0, completion_tokens=(u or {}).get("completion_tokens") or 0)

    @staticmethod
    def _args(raw):
        if isinstance(raw, dict): return raw
        try: v = json.loads(raw or "{}"); return v if isinstance(v, dict) else {}
        except ValueError: return {}

    def chat(self, messages: list[dict], tools: list[dict] | None = None, on_token=None) -> dict:
        body = {"model": self.model, "messages": to_openai(messages), "stream": bool(on_token),
                "chat_template_kwargs": {"enable_thinking": self.think}}
        if on_token: body["stream_options"] = {"include_usage": True}
        body.update(profiles.sampling(self.model, self.think, self.sampling))
        if self.temperature is not None: body["temperature"] = self.temperature
        body["cache_prompt"] = True
        if self.slot is not None: body["id_slot"] = self.slot
        if tools: body["tools"] = tools
        data = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(self._req("/v1/chat/completions", data), timeout=self.timeout) as r:
                    if not on_token:
                        d = json.load(r); self._count(d.get("usage")); self._timing(d.get("timings"))
                        m = d["choices"][0]["message"]
                        out = {"role": "assistant", "content": m.get("content") or ""}
                        reasoning = m.get("reasoning_content") or ""
                        calls = [{"function": {"name": c["function"]["name"], "arguments": self._args(c["function"].get("arguments"))}}
                                 for c in m.get("tool_calls") or []]
                    else:
                        text, acc, reasoning = [], {}, []
                        for line in r:
                            line = line.decode(errors="replace").strip()
                            if not line.startswith("data:") or line == "data: [DONE]": continue
                            d = json.loads(line[5:])
                            if d.get("usage"): self._count(d["usage"])
                            if d.get("timings"): self._timing(d["timings"])
                            for ch in d.get("choices") or []:
                                dl = ch.get("delta") or {}
                                if dl.get("reasoning_content"): reasoning.append(dl["reasoning_content"])
                                if dl.get("content"): text.append(dl["content"]); on_token(dl["content"])
                                for tc in dl.get("tool_calls") or []:
                                    a = acc.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                                    f = tc.get("function") or {}
                                    a["name"] += f.get("name") or ""; a["args"] += f.get("arguments") or ""
                        out = {"role": "assistant", "content": "".join(text)}; reasoning = "".join(reasoning)
                        calls = [{"function": {"name": a["name"], "arguments": self._args(a["args"])}} for _, a in sorted(acc.items())]
                    if not calls and "<tool_call>" in out["content"]: calls, out["content"] = leaked_tool_calls(out["content"])
                    if calls: out["tool_calls"] = calls
                    elif not out["content"].strip() and reasoning.strip(): out["content"] = reasoning.strip()   # the model ended its turn inside its reasoning: that text is the answer
                    return out
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace"); e.close()
                if e.code < 500 or attempt == self.retries: raise LLMError(f"HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries: raise LLMError(f"cannot reach the model server at {self.host}: {e}") from None
            time.sleep(1.5 * (attempt + 1))
        raise LLMError("unreachable")


_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_FUNC = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)

def leaked_tool_calls(text: str) -> tuple[list[dict], str]:
    """Tool calls the model wrote as plain text instead of the server parsing them (seen with thinking on: Qwen's
    '<tool_call><function=Read><parameter=file_path>…' XML, or the older '<tool_call>{"name": …}' JSON). Returns (calls, text left over)."""
    calls = []
    for body in _CALL.findall(text):
        if (f := _FUNC.search(body)):
            args = {}
            for k, v in _PARAM.findall(f.group(2)):
                v = v.strip("\n")
                try: args[k] = json.loads(v) if v[:1] in "[{" else v
                except ValueError: args[k] = v
            calls.append({"function": {"name": f.group(1), "arguments": args}})
        else:
            try: j = json.loads(body)
            except ValueError: continue
            if isinstance(j, dict) and j.get("name"):
                a = j.get("arguments") or {}
                calls.append({"function": {"name": j["name"], "arguments": a if isinstance(a, dict) else OpenAIClient._args(a)}})
    return (calls, _CALL.sub("", text).strip()) if calls else ([], text)


def ollama_blob(model: str, kind: str = "model") -> str:
    """Path of the GGUF (kind="model") or vision projector (kind="projector") that Ollama already downloaded for `model`, or ""."""
    ref, _, tag = model.rpartition(":") if ":" in model.rsplit("/", 1)[-1] else (model, "", "latest")
    parts = ref.split("/")
    if "." not in parts[0]: parts = ["registry.ollama.ai"] + (["library"] if len(parts) == 1 else []) + parts
    stores = [os.environ.get("OLLAMA_MODELS", ""), "/var/lib/ollama/.ollama/models", os.path.expanduser("~/.ollama/models"),
              "/usr/share/ollama/.ollama/models"]
    for st in filter(None, stores):
        try:
            for l in read_json(os.path.join(st, "manifests", *parts, tag))["layers"] or []:
                f = os.path.join(st, "blobs", l["digest"].replace(":", "-"))
                if l["mediaType"].endswith("image." + kind) and os.access(f, os.R_OK): return f
        except (OSError, ValueError, KeyError): continue
    return ""


def gguf_args(model: str) -> list[str]:
    """llama-server arguments that load `model`: a .gguf path, an Ollama-store model (reused, no download), or a
    Hugging Face ref (hf.co/user/repo:QUANT is accepted; llama-server downloads it)."""
    path = os.path.expanduser(model)
    if path.endswith(".gguf") and os.path.isfile(path): return ["-m", path]
    if blob := ollama_blob(model): return ["-m", blob]
    return ["-hf", model[len("hf.co/"):] if model.startswith("hf.co/") else model]


_RECURRENT: dict[str, bool] = {}

def is_recurrent(model: str) -> bool:
    """True for hybrid/recurrent models (GGUF metadata has <arch>.ssm.* keys, e.g. qwen35). llama-server can't rewind their state
    to a shared prefix, so a saved slot never matched the next prompt: measured 0 cached tokens on resume (vs 2891/2923 on qwen3:8b)."""
    args = gguf_args(model)
    if args[0] != "-m": return False
    try:
        with open(args[1], "rb") as f: head = f.read(8 << 20)   # metadata keys sit at the start, before the tensor data
    except OSError: return False
    return b".ssm." in head


def installed_models() -> list[str]:
    """Models llama-server can load without downloading: Ollama-store manifests (as `name:tag`) and ~/.alice/models/*.gguf."""
    out = []
    for st in filter(None, [os.environ.get("OLLAMA_MODELS", ""), "/var/lib/ollama/.ollama/models", os.path.expanduser("~/.ollama/models"),
                            "/usr/share/ollama/.ollama/models"]):
        root = os.path.join(st, "manifests")
        for dp, _dn, fns in os.walk(root):
            for f in fns:
                parts = os.path.relpath(os.path.join(dp, f), root).split(os.sep)
                if len(parts) < 3: continue
                host, rest, tag = parts[0], parts[1:-1], parts[-1]
                name = "/".join(rest[1:] if (host, rest[0]) == ("registry.ollama.ai", "library") else ([*rest] if host == "registry.ollama.ai" else [host, *rest])) + ":" + tag
                if ollama_blob(name) and name not in out and not re.fullmatch(r"[0-9a-f]{40,}", tag): out.append(name)
    out += sorted(glob.glob(os.path.expanduser("~/.alice/models/*.gguf")))
    return out


def unload_ollama(say=lambda s: None, host: str = "") -> list[str]:
    """Ollama keeps models resident in VRAM for minutes; free them so llama-server can allocate. Returns unloaded names."""
    host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    if "://" not in host: host = "http://" + host
    done = []
    try:
        with urllib.request.urlopen(host + "/api/ps", timeout=2) as r: loaded = [m["name"] for m in json.load(r).get("models", [])]
        for name in loaded:
            req = urllib.request.Request(host + "/api/generate", json.dumps({"model": name, "keep_alive": 0}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15).read(); done.append(name)
    except Exception: pass  # no Ollama running / unreachable: nothing to free
    if done:
        say("unloaded from Ollama to free VRAM: " + ", ".join(done)); time.sleep(1)
    return done


def _die_with_parent():
    """Child pre-exec hook (Linux): SIGTERM the server if Alice dies by any means, including SIGKILL."""
    try: ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception: pass


class LlamaServer:
    """Starts a private llama-server for this session (so no Ollama is needed) and stops it at exit."""

    def __init__(self, model: str, num_ctx: int, binary: str = "", extra: list[str] | None = None, log: str = "", vision: bool = False,
                 auto: list[str] | None = None, slot_dir: str = ""):
        self.bin = binary or os.environ.get("ALICE_LLAMA_SERVER") or shutil.which("llama-server") or ""
        self.model, self.num_ctx, self.extra = model, num_ctx, extra or []
        self.log = log or os.path.expanduser("~/.alice/llama-server.log")
        self.proc = None; self.vision = vision
        self.auto, self.slot_dir = auto or [], slot_dir      # auto: tuned flags that are dropped if the server refuses to start with them
        if slot_dir:
            try: os.makedirs(slot_dir, exist_ok=True)
            except OSError: self.slot_dir = ""
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0)); self.port = sk.getsockname()[1]
        self.host = f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [self.bin, *gguf_args(self.model), "-c", str(self.num_ctx), "-ngl", "99", "--jinja",
                "--host", "127.0.0.1", "--port", str(self.port), "--alias", self.model,
                *(["--mmproj", proj] if self.vision and (proj := ollama_blob(self.model, "projector")) else []),
                *(["--slot-save-path", self.slot_dir] if self.slot_dir else []), *self.auto, *self.extra]

    def can_see(self) -> bool:
        """Is a vision projector available for this model? (reused from the Ollama store, or fetched by -hf)"""
        return bool(ollama_blob(self.model, "projector")) or not gguf_args(self.model)[0] == "-m"

    def revive(self) -> bool:
        """If the server process has died (crash, OOM kill, stray SIGTERM), start it again on the same port. True if restarted."""
        if self.proc is None or self.proc.poll() is None: return False
        self.proc = None
        print("\x1b[2m  model server stopped; restarting it…\x1b[0m", file=sys.stderr)
        try: self.start(say=lambda m: None); return True
        except LLMError: return False

    def switch_model(self, model: str, say=lambda s: None, num_ctx: int = 0, extra: list[str] | None = None) -> str:
        """Stop, load a different model (or the same one with a new context size / extra flags) on the same port (same URL, so the
        client keeps working); if it fails, bring the old one back with its old settings."""
        old = (self.model, self.vision, self.num_ctx, self.extra)
        self.stop(); self.proc = None; self.model, self.vision = model, False
        self.num_ctx = num_ctx or self.num_ctx
        if extra is not None: self.extra = extra
        try: return self.start(say=say)
        except LLMError:
            self.model, self.vision, self.num_ctx, self.extra = old; self.proc = None
            try: self.start(say=say)
            except LLMError: pass
            raise

    def restart_with_vision(self, say=lambda s: None) -> str:
        self.stop(); self.proc = None; self.vision = True
        return self.start(say=say)

    def start(self, wait: int = 600, say=lambda s: None) -> str:
        try: return self._start(wait, say)
        except LLMError as e:
            if not (self.auto or self.slot_dir) or "not found" in str(e): raise
            say("model server would not start with the tuned flags; retrying without them"); self.auto, self.slot_dir = [], ""
            return self._start(wait, say)

    # ---- KV-cache snapshots: keep a long conversation's processed prompt across /restart and relaunch
    def _slot(self, action: str, name: str) -> bool:
        if not (self.slot_dir and self.proc and self.proc.poll() is None): return False
        if (rec := _RECURRENT.get(self.model)) is None: rec = _RECURRENT[self.model] = is_recurrent(self.model)
        if rec: return False   # snapshot would be useless (see is_recurrent) and costs ~100-150 MB of disk and time at exit
        try:
            req = urllib.request.Request(f"{self.host}/slots/0?action={action}", json.dumps({"filename": name}).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r: return r.status == 200
        except Exception: return False

    def slot_name(self, sid: str) -> str:
        import hashlib
        return f"{sid}-{hashlib.sha1(self.model.encode()).hexdigest()[:8]}-{self.num_ctx}.bin"

    def save_slot(self, sid: str) -> bool:
        os.makedirs(self.slot_dir, exist_ok=True) if self.slot_dir else None
        return self._slot("save", self.slot_name(sid))

    def restore_slot(self, sid: str) -> bool:
        return bool(self.slot_dir) and os.path.isfile(os.path.join(self.slot_dir, self.slot_name(sid))) and self._slot("restore", self.slot_name(sid))

    def prune_slots(self, days: int = 7, max_bytes: int = 4 << 30):
        """Drop snapshots older than `days`, then the oldest until the total is under max_bytes (one is 150-550 MB; eval runs piled up 15 GB in an hour)."""
        files = []
        for f in glob.glob(os.path.join(self.slot_dir, "*.bin")) if self.slot_dir else []:
            try:
                if time.time() - os.path.getmtime(f) > days * 86400: os.remove(f)
                else: files.append((os.path.getmtime(f), os.path.getsize(f), f))
            except OSError: pass
        total = sum(sz for _, sz, _ in files)
        for _, sz, f in sorted(files):
            if total <= max_bytes: break
            try: os.remove(f); total -= sz
            except OSError: pass

    def _start(self, wait: int = 600, say=lambda s: None) -> str:
        if not self.bin:
            raise LLMError("llama-server not found. Install llama.cpp (Arch: sudo pacman -S llama-cpp ggml-cuda), "
                           "or set ALICE_LLAMA_SERVER to its path, or pass --base-url to use a server that is already running.")
        unload_ollama(say)
        os.makedirs(os.path.dirname(self.log), exist_ok=True)
        with open(self.log, "w") as logf:   # the child keeps its own copy of the fd; ours closes here
            self.proc = subprocess.Popen(self.command(), stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True, preexec_fn=_die_with_parent)
        atexit.register(self.stop)
        for sig in (signal.SIGTERM, signal.SIGHUP):  # default action skips atexit and would orphan the server (and its VRAM)
            signal.signal(sig, lambda n, f: sys.exit(128 + n))
        say(f"starting llama-server (port {self.port}; log {self.log})")
        end = time.time() + wait
        while time.time() < end:
            if self.proc.poll() is not None:
                tail = read_text(self.log, errors="replace")[-600:]
                raise LLMError(f"llama-server exited with code {self.proc.returncode}:\n{tail}")
            try:
                with urllib.request.urlopen(self.host + "/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok": return self.host
            except Exception: pass
            time.sleep(0.5)
        self.stop(); raise LLMError(f"llama-server did not become ready in {wait}s (see {self.log})")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(10)
            except subprocess.TimeoutExpired: self.proc.kill()
