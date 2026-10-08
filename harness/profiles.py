"""Per-model tuning: llama-server flags, sampling settings, and when to let the model think.

Flag choices come from evals/bench_server.py on the 8 GB RTX 4060 (see evals/bench_results.jsonl):
q8_0 KV cache saves ~160-740 MiB (more at long context) at identical speed, and n-gram speculative decoding
gives ~+35% tokens/s on edit-style output that repeats the input, with no loss elsewhere."""
import re

DEFAULT_CTX = 32768          # fits in ~5.9 GB with the q8_0 KV cache (measured), leaving room for the vision projector

BASE_FLAGS = ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "ngram-mod"]

# Conservative agentic sampling (lower temperature than chat defaults: tool calls must be well-formed). Not yet eval-tuned;
# override per user with settings.json {"sampling": {"temperature": 0.7, ...}}.
SAMPLING = [
    ("qwen", {"off": {"temperature": 0.6, "top_p": 0.9, "top_k": 20}, "on": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}}),
    ("gemma", {"off": {"temperature": 0.7, "top_p": 0.95, "top_k": 64}, "on": {"temperature": 0.7, "top_p": 0.95, "top_k": 64}}),
    ("deepseek-r1", {"off": {"temperature": 0.6, "top_p": 0.95}, "on": {"temperature": 0.6, "top_p": 0.95}}),
]

def sampling(model: str, thinking: bool, override: dict | None = None) -> dict:
    out = next((v["on" if thinking else "off"] for k, v in SAMPLING if k in model.lower()), {})
    return {**out, **(override or {})}

def server_flags(settings: dict | None = None) -> list[str]:
    """Flags added to every private llama-server start (a failed start retries without them). settings: kvCache, speculative, draftModel."""
    s, flags = settings or {}, list(BASE_FLAGS)
    kv = str(s.get("kvCache", "q8_0"))
    flags[flags.index("-ctk") + 1] = flags[flags.index("-ctv") + 1] = kv
    spec = str(s.get("speculative", "ngram-mod"))
    i = flags.index("--spec-type")
    if s.get("draftModel"): flags[i:i + 2] = ["--spec-type", "draft-simple," + spec if spec != "none" else "draft-simple", "-md", str(s["draftModel"])]
    elif spec == "none": del flags[i:i + 2]
    else: flags[i + 1] = spec
    return flags

HARD = re.compile(r"\b(why|debug|diagnos\w*|plan|design|architect\w*|refactor|investigat\w*|root cause|trade-?offs?|compare|optimi[sz]e|review|figure out|not working|fails?|broken)\b", re.I)

def want_thinking(mode: str, text: str, step: int) -> bool:
    """mode: on | off | auto. auto thinks only on the first step of a turn that looks like it needs reasoning (long or analytical)."""
    if mode == "on": return True
    if mode == "off" or step > 1: return False
    return len(text) > 400 or bool(HARD.search(text))
