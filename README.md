# Alice

[![test](https://github.com/papacasper/alice/actions/workflows/test.yml/badge.svg)](https://github.com/papacasper/alice/actions/workflows/test.yml)

A Claude Code–style coding agent for local models. It runs on one consumer GPU (built and tested on an 8 GB RTX 4060) with llama.cpp's `llama-server` or Ollama. The default model is an abliterated Qwen3.5-9B GGUF.

Alice has the Claude Code feel: Read/Edit/Write/Bash/Grep/Glob tools, permission modes, slash commands, subagents, hooks, MCP servers, skills, sessions you can resume, a status line, and a remote-control web page.

## Requirements

- Python 3.10+, standard library only. `prompt_toolkit` and `rich` are optional; they give the full prompt UI and Markdown rendering.
- A backend: `llama-server` from llama.cpp (Alice starts and manages its own private server), or a running Ollama.
- A GGUF model. Pull one with Ollama or point `--model` at your own.

## Run

```sh
git clone https://github.com/papacasper/alice && cd alice
python3 -m harness                         # interactive
python3 -m harness "fix the failing test"  # one-shot
python3 -m harness -p --output-format json "count the TODOs"   # headless, JSON out
```

A launcher on your PATH:

```sh
#!/bin/sh
PYTHONPATH="$HOME/src/alice" exec python3 -m harness "$@"
```

Settings live in `~/.alice/settings.json` and `./.alice/settings.json`. See [harness/README.md](harness/README.md) for every feature, flag, and setting.

## Tests and evals

```sh
python3 -m unittest discover -s harness/tests   # no model needed (fake LLM and a fake llama-server); CI runs this on 3.10 and latest
python3 evals/run.py --selftest                 # every eval check fails untouched and passes on its reference solution
python3 evals/run.py --repeat 3                 # 14 real tasks on the live model, graded by deterministic checks
python3 evals/live_checks.py                    # KV-cache restore and helper-model compaction
```

Results append to `evals/results.jsonl`. With the default model and thinking on `auto`, the suite passed 39/39 (13 tasks × 3), before the paged-read task was added.

## Layout

- `harness/`: the agent: loop, tools, CLI, prompt UI, backends.
- `evals/`: task suite, runner, and live checks.
- `ctxguard.py`: keeps the conversation inside the context window without dropping the task.
- `bench.py`, `repro.py`, `make_sandbox.py`: older benchmark and regression scripts (`make_sandbox.py <dir>` builds the benchmark sandbox).

## License

MIT. See [LICENSE](LICENSE).
