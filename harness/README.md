# harness — "Alice", a Claude Code–style agent for local Ollama models

Stdlib-only Python. Default model: abliterated Qwen3.5-9B (`harness/llm.py: DEFAULT_MODEL`).
Launcher: any `alice` script on your PATH that runs `PYTHONPATH=<repo> python3 -m harness "$@"` (see the top-level README).

```
alice                         # interactive REPL (streaming, slash commands, history)
alice "fix the failing test"  # one-shot; exit 0 answered, 2 step limit, 1 error
echo "summarize" | alice -p   # headless; --output-format json for scripts
alice -c  /  alice -r [ID]    # continue the latest / a specific session
python3 -m unittest discover -s harness/tests     # no Ollama needed (fake LLM)
```

## What mirrors Claude Code
| Feature | Here |
|---|---|
| Tools | `Read` `Write` `Edit` `Bash` `Glob` `Grep` `TodoWrite` `WebFetch` `Task` `Skill` `AskUserQuestion` `ExitPlanMode` (+ `calculate`, `today`) — `cctools.py` |
| Edit safety | must `Read` before `Edit`/`Write`; unique-match or `replace_all`; stale-file check; colored diff |
| Bash | persistent cwd between calls, timeout kill, middle-truncated output |
| Permissions | `--permission-mode default\|acceptEdits\|plan\|bypassPermissions`, allow/deny rules, `y/a/n` prompt — `permissions.py`. **Default here is bypassPermissions** (everything allowed); use `--permission-mode default` to be asked |
| Memory | `AGENTS.md`, `CLAUDE.md`, `ALICE.md` from `~/.alice/` and the project dir only (no walking up), capped; `--no-memory` |
| Slash commands | `/help /clear /compact /model /models /permissions /plan /tools /todos /cost /status /memory /skills /mcp /hooks /resume /init /config /exit`; custom ones from `.alice/commands/*.md` or `.claude/commands/*.md` (`$ARGUMENTS`) |
| Skills | `SKILL.md` dirs under `~/.claude/skills`, `~/.alice/skills`, `./.claude/skills`, `./.alice/skills`; model loads one with `Skill` |
| Subagents | `Task` (`general-purpose` or read-only `explore`) runs a fresh-context agent and returns its report |
| Hooks | `PreToolUse PostToolUse UserPromptSubmit Stop SessionStart` in `settings.json` (exit 2 blocks) — `hooks.py` |
| MCP | stdio servers from `settings.json` `mcpServers` -> `mcp__server__tool` — `mcp.py` |
| Sessions | autosaved to `~/.alice/sessions/`; `-c`, `-r`, `/resume` — `sessions.py` |
| Input niceties | `!cmd` shell, `@file` attach, `\` line continuation, Ctrl-C interrupts a turn |

Settings: `~/.alice/settings.json`, `./.alice/settings.json`, `./.alice/settings.local.json` (merged):
`{"model":..., "num_ctx":..., "permissionMode":..., "permissions":{"allow":["Bash(git status:*)"],"deny":["Bash(rm:*)"]}, "hooks":{...}, "mcpServers":{...}}`

## Not (yet) cloned
A full-screen TUI and plugin marketplaces.

## Modules (swap any one)
`llm.py` Ollama client (streaming, usage) · `tools.py` `@tool` registry · `cctools.py` Claude Code tools · `builtin.py` legacy sandboxed tools (kept for tests/plugins) · `agent.py` loop (ctxguard, repeat breaker, gate/post hooks) · `cli.py` REPL + slash commands · `ui.py` rendering · `context.py` system prompt/memory/skills/settings.

## Known limits
- 9B model, 12k context (`--num-ctx`): the system prompt + tool schemas cost ~3k tokens; `ctxguard` elides old tool output so the task is never dropped.
- Small models still guess when no tool is obviously relevant; the system prompt says never to guess dates/live data.
- Give the model exact tools for anything countable; it miscounts by eye.

## Added in the "more of a replica" pass
- Background shells: `Bash(run_in_background)`, `BashOutput`, `KillShell` (killed at exit).
- `NotebookEdit`, `WebSearch` (DuckDuckGo, fetched through `harness/fetch.mjs` on Node because DDG bot-challenges Python and curl; falls back to curl, then urllib; `ALICE_SEARXNG_URL` overrides).
- Custom subagents: `agents/*.md` with `name`, `description`, `tools` frontmatter, used via `Task(subagent_type=...)`.
- `/rewind` (restores files changed by Edit/Write/NotebookEdit; Bash side effects are not undone), `/context`, `/export`, `/doctor`, `/review`, `/agents`.
- `# note` saves to the project ALICE.md (or the existing AGENTS.md/CLAUDE.md), tab completion for `/commands` and paths, spinner with elapsed seconds, auto-compact at 75% of the context.
- Known cost: the 18 tool schemas take ~4.4k tokens of the 12k context (`/context`).

## Backend: llama.cpp by default (`--backend ollama` to use Ollama)
`alice --backend llama` starts a private `llama-server` (llama.cpp) for the session and stops it on exit; the log is `~/.alice/llama-server.log`.
- Install (Arch, NVIDIA): `sudo pacman -S llama-cpp ggml-cuda`. Or set `ALICE_LLAMA_SERVER=/path/to/llama-server`.
- The model is resolved in this order: a `.gguf` path, then the GGUF Ollama already downloaded (read from its manifest store, so no second download and no running Ollama needed), then a Hugging Face ref via `llama-server -hf`.
- Started with `-c <num_ctx> -ngl 99 --jinja` (`--jinja` is what makes tool calls come back structured). Extra flags: `--llama-server-args '-ngl 20'`.
- Already have a server (LM Studio, vLLM, a remote box, a running llama-server)? `alice --backend llama --base-url http://host:8080` connects instead of starting one (`ALICE_API_KEY` for auth).
- Settings keys: `backend`, `baseUrl`, `llamaServerArgs`; env: `ALICE_BACKEND`, `ALICE_BASE_URL`.
- Ollama stays the default until the real llama-server path has been run against the 9B model (tested so far only against a fake server).

## Claude Code-style UI (prompt_toolkit + rich)

- Welcome box, rotating tips, ruled `>` prompt with a footer directly beneath it (mode label + model + `% ctx`).
- Shift+Tab cycles permission mode; `/` and `@file` completion menus; `?` lists shortcuts; Alt+Enter or trailing `\` for newline.
- Esc interrupts a running turn (real SIGINT, so blocked reads stop); Esc Esc rewinds / clears; Ctrl+O full tool output; Ctrl+T todos; Ctrl+R history; Ctrl+G edit in `$EDITOR`.
- `●` tool bullets with `⎿` summaries ("Read 12 lines"), coloured diffs, numbered permission dialog, markdown answers, spinner with verbs.
- `/transcript` shows the full log, `/vim` toggles vim keys. Falls back to plain `input()` when not a TTY.
- `/theme [terracotta|blue|green|purple|amber|mono]` changes the accent colour live and saves it (`theme` in `~/.alice/settings.json`).
- Status line (like Claude Code): a row under the prompt, built-in `model | [context bar] pct% | used k / total k (left)`. `/statusline claude` reuses `~/.claude/statusline.sh`; `/statusline <command>` runs any script; `/statusline off` turns it off. Settings: `"statusLine": {"type": "command", "command": "...", "padding": 0}`. The command gets Claude Code's status JSON on stdin (plus `"agent": "alice"`). Every output line is shown with ANSI colours kept. It runs in a background thread so a slow script never blocks typing.
- Background shells: `/tasks` (or Ctrl+B) lists them, `/kill <id>` stops one, the footer shows how many are running.
- Ctrl+S stashes the draft (footer shows "stashed"); it returns in the next prompt, or press Ctrl+S on an empty prompt to restore it now.
- Type while Alice works: Enter queues the message (shown in the spinner line) and it runs when the turn ends.
- Footer hugs the prompt; menu space is reserved only for `/` and `@` completions.
- Images: `@photo.png` in a message, `/image path`, or Ctrl+V (clipboard via wl-paste/xclip). The first image restarts the private llama-server with the vision projector (`--mmproj`, reused from the Ollama store); the footer shows "N images" until sent.
- Plugins: `~/.alice/plugins/<name>/` (or `<project>/.alice/plugins/<name>/`) with `skills/`, `commands/`, `agents/` are discovered like user ones; `/plugins` lists them.
- `/selfedit <change>` lets Alice edit her own harness: git snapshot, the model edits, the full test suite runs, pass = commit, fail = automatic rollback (new files removed). `/restart` relaunches into the new code keeping the session.
- Ctrl-C or Ctrl-D on an empty prompt shows "Press Ctrl-C again to exit" for 2 s; a second press quits. `/expand [n]` prints the full output of a collapsed tool result. `/help` groups commands, `/help <cmd>` explains one. `/diff` shows what the last turn changed. `/copy` copies the last answer, `/retry` redoes the last prompt; turns over 5 s end with a "✻ Worked for …" summary line.
- Global system prompt: `harness/system_prompt.md` (tone, tasks, tool use, safety, git); `~/.alice/SYSTEM.md` replaces it, `--system-prompt FILE` overrides per run.
- Tool schemas are sent in compact form (`harness/slim.py`, -29% tokens: 2466 -> 1758 on the Qwen template); `ALICE_FULL_SCHEMAS=1` sends the full docstrings.
- Not cloned: a full-screen TUI (Claude Code itself is inline), emoji shortcodes.

## Remote control and model switching
- `/rc` starts a small web page + JSON API (`harness/rc.py`) to watch and drive the running session from a phone or another machine. It binds to the Tailscale IPv4 if there is one, else 127.0.0.1 (`/rc local` forces loopback); every URL carries a random token and a wrong token is a 404. Prompts sent from the page behave like typed ones (queued during a turn); Stop interrupts the turn. `/rc stop` ends it; the footer shows "remote control on".
- `/model <name>` switches models mid-session: the private llama-server is reloaded on the same port, the conversation is kept, and if the new model fails to load the old one is restored. `/models` lists installed Ollama-store models and `~/.alice/models/*.gguf`; `/model ` tab-completes them.
- `/add-dir <path>` adds another working directory (shown to the model in its system prompt). If the model server dies mid-session alice says so and restarts it.
- `/rename <name>` names a session (`/resume <name>`, shown in the list); `/output-style` switches between default/concise/explanatory or your own `~/.alice/output-styles/<name>.md`; when the model calls several tools in one step the UI prints "⎿ N tool calls in this step".

## Backend tuning
Measured on the 8 GB RTX 4060 with `evals/bench_server.py` (results in `evals/bench_results.jsonl`):
- **Server flags** (`harness/profiles.py`): `-fa on`, q8_0 KV cache and n-gram speculative decoding (`--spec-type ngram-mod`). q8 KV cuts VRAM 164 MiB at 12k context and 740 MiB at 64k with no speed change; n-gram drafting gave ~+37% tok/s on edit-style output (58 vs 42) and nothing worse elsewhere. If the server won't start with them, alice retries without. Settings: `kvCache` (`q8_0`/`f16`), `speculative` (`ngram-mod`/`none`), `draftModel` (a GGUF path; untested), `llamaServerArgs` (appended last, wins).
- **Default context is now 32768** (about 5.9 GB total with the q8 KV cache vs 5.6 GB at 12k before).
- **Prompt cache**: every request sends `cache_prompt`; `/cost` shows how many prompt tokens the cache served and the generation tok/s. The KV cache is snapshotted to `~/.alice/slots` on exit and `/restart`, and restored on `--resume`/`-c` (files older than 7 days are pruned).
- **Sampling** per model family (`profiles.SAMPLING`, conservative and not yet eval-tuned); override with `{"sampling": {...}}` in settings.
- **Thinking** is `auto` by default: only the first step of a long or analytical prompt reasons. `/think on|off|auto`, `--think=on|off|auto`, or the `think` setting.
- **Parallel tools**: when one step calls only Read/Glob/Grep/WebFetch they run side by side (results stay in order).
- **Tool-call repair**: a misspelled parameter (`file` for `file_path`) or a list passed as a JSON string is fixed; an unknown tool name suggests the closest real one.
- **Compaction** counts tool calls and tool schemas when deciding the context is nearly full; set `helperModel` to summarize with a different model (the private server swaps to it and back, so it costs two reloads).

## Evals
`python3 evals/run.py` runs 13 tasks (write/edit/search/shell/git/honesty) against the real model and grades each by inspecting files and command output, never by how the answer reads. `--selftest` proves every check fails on the untouched workspace and passes with its reference solution (no model needed; also in the unit tests). Options: `--model`, `--think`, `--num-ctx`, `--server-args`, `--tag`, `--repeat`. Results append to `evals/results.jsonl`.
