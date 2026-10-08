"""Compact tool schemas for the small local model. Every tool schema is re-sent on each request, so these
shorter descriptions buy back context (~35%). Docstrings in the tool sources stay the human-readable reference;
set ALICE_FULL_SCHEMAS=1 to send those instead. Param value None = leave the parameter undescribed (name says it all)."""
import os

SLIM = {
    "calculate": ("Evaluate an arithmetic expression exactly.", {"expression": None}),
    "today": ("Current local date, weekday and time.", {}),
    "Read": ("Read a file with line numbers (paged; pass offset to continue). Read before Edit/Write.",
             {"file_path": None, "offset": "1-based first line", "limit": None}),
    "Write": ("Create or overwrite a file. Existing files must be Read first; prefer Edit for changes.", {"file_path": None, "content": None}),
    "Edit": ("Replace an exact, unique string in a file (must be Read first).",
             {"file_path": None, "old_string": None, "new_string": None, "replace_all": "replace every match"}),
    "Bash": ("Run a bash command; returns output and exit code. Prefer Read/Glob/Grep over cat/find/grep. Long-running: run_in_background, then BashOutput.",
             {"command": None, "timeout": "seconds", "description": None, "run_in_background": None}),
    "Glob": ("Find files by glob pattern, newest first.", {"pattern": "e.g. src/**/*.ts", "path": None}),
    "Grep": ("Ripgrep file contents. output_mode: files_with_matches (default), content, count.",
             {"pattern": "regex", "path": None, "glob": "file filter, e.g. *.py", "output_mode": None, "ignore_case": None,
              "context": "context lines", "head_limit": None}),
    "TodoWrite": ("Update the task list; send the COMPLETE list each time: [{content, status: pending|in_progress|completed}]. One item in_progress.", {"todos": None}),
    "WebFetch": ("Fetch a URL and return its readable text.", {"url": None, "prompt": None}),
    "Task": ("Run a subagent with fresh context for a self-contained job and return its report. subagent_type: general-purpose or explore (read-only).",
             {"description": "3-5 words", "prompt": "complete standalone instructions", "subagent_type": None}),
    "Skill": ("Load a listed skill's full instructions, then follow them.", {"name": None}),
    "AskUserQuestion": ("Ask the user a question only they can answer (when genuinely blocked).", {"question": None}),
    "ExitPlanMode": ("Plan mode only: submit the finished plan for user approval.", {"plan": "short numbered list"}),
    "BashOutput": ("New output of a background shell, and whether it still runs.", {"bash_id": None, "filter": "regex"}),
    "KillShell": ("Stop a background shell.", {"shell_id": None}),
    "NotebookEdit": ("Edit a .ipynb cell: replace, insert at cell_number, or delete (0-based).",
                     {"notebook_path": None, "new_source": None, "cell_number": None, "cell_type": "code|markdown", "edit_mode": "replace|insert|delete"}),
    "WebSearch": ("Search the web; returns titles, URLs, snippets. Follow up with WebFetch.", {"query": None, "max_results": None}),
}

def apply(tool) -> None:
    if os.environ.get("ALICE_FULL_SCHEMAS"): return
    if tool.name in SLIM:
        tool.description, params = SLIM[tool.name]
        for k, spec in tool.params.items():
            if k in params: spec["description"] = params[k] or k
