"""Built-in tools. Filesystem tools are confined to a root directory; write/shell are opt-in."""
import ast, datetime, operator, os, re, subprocess
from .tools import Tool, Toolbox
from .fsutil import read_text

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        l, r = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(r) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](l, r)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

def _read(full: str) -> str:
    with open(full, errors="replace") as f:
        return f.read()

def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use this for all math.

    Args:
        expression: e.g. "120.5 + 75.25 * 2"
    """
    return str(_eval(ast.parse(expression.strip(), mode="eval").body))

def today() -> str:
    """Get the current local date, weekday and time. Use this for anything about 'today' or 'now'."""
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d (%A) %H:%M %Z")

def fs_tools(root: str) -> list[Tool]:
    root = os.path.realpath(root)

    def safe(p: str) -> str:
        full = os.path.realpath(os.path.join(root, p))
        if full != root and not full.startswith(root + os.sep):
            raise PermissionError(f"'{p}' is outside the sandbox")
        return full

    def list_dir(path: str = ".") -> str:
        """List files and folders in a directory (folders end with /).

        Args:
            path: directory relative to the root
        """
        d = safe(path)
        return "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(d, e)) else "") for e in os.listdir(d))) or "(empty)"

    def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
        """Read a text file (numbered lines). Long files are paged: use start_line to continue, grep to search.

        Args:
            path: file relative to the root
            start_line: first line to return (1-based)
            max_lines: how many lines to return
        """
        lines = _read(safe(path)).splitlines()
        a = max(start_line, 1) - 1
        chunk = lines[a:a + max_lines]
        out = "\n".join(f"{a + i + 1}: {l}" for i, l in enumerate(chunk))
        if a + max_lines < len(lines):
            out += f"\n[showing lines {a + 1}-{a + len(chunk)} of {len(lines)}; call again with start_line={a + max_lines + 1}]"
        return out or "(empty)"

    def grep(pattern: str, path: str, max_results: int = 40) -> str:
        """Regex search lines in a file. Returns the TOTAL match count plus the first matching lines.

        Args:
            pattern: Python regular expression
            path: file relative to the root
            max_results: how many matching lines to show
        """
        rx = re.compile(pattern)
        hits = [f"{i}: {l}" for i, l in enumerate(_read(safe(path)).splitlines(), 1) if rx.search(l)]
        return f"{len(hits)} matching lines\n" + "\n".join(hits[:max_results]) + (f"\n[showing first {max_results}]" if len(hits) > max_results else "")

    def count_by_group(pattern: str, path: str) -> str:
        """Count regex matches per captured group value (e.g. errors per service). Exact; prefer this to counting by eye.

        Args:
            pattern: regex with ONE capture group, e.g. 'ERROR \\[(\\w+)\\]'
            path: file relative to the root
        """
        rx = re.compile(pattern)
        if rx.groups != 1:
            raise ValueError("pattern must have exactly one capture group")
        counts: dict[str, int] = {}
        for l in _read(safe(path)).splitlines():
            m = rx.search(l)
            if m:
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        return "\n".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "0 matches"

    def search_files(pattern: str, path: str = ".", max_results: int = 40) -> str:
        """Regex search every text file under a directory (recursive). Returns file:line: text for the first matches plus the total.

        Args:
            pattern: Python regular expression
            path: directory relative to the root
            max_results: how many matches to show
        """
        rx, hits, total = re.compile(pattern), [], 0
        for dp, dns, fns in os.walk(safe(path)):
            dns[:] = [d for d in dns if d not in (".git", "node_modules", "__pycache__")]
            for fn in sorted(fns):
                full = os.path.join(dp, fn)
                try:
                    lines = read_text(full, errors="strict").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                for i, l in enumerate(lines, 1):
                    if rx.search(l):
                        total += 1
                        if len(hits) < max_results:
                            hits.append(f"{os.path.relpath(full, root)}:{i}: {l[:200]}")
        return f"{total} matching lines\n" + "\n".join(hits) + (f"\n[showing first {max_results}]" if total > max_results else "")

    return [Tool(f) for f in (list_dir, read_file, grep, search_files, count_by_group)]

def write_tools(root: str) -> list[Tool]:
    root = os.path.realpath(root)

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file.

        Args:
            path: file relative to the root
            content: full file contents
        """
        full = os.path.realpath(os.path.join(root, path))
        if not full.startswith(root + os.sep):
            raise PermissionError(f"'{path}' is outside the sandbox")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        return f"wrote {len(content)} chars to {path}"
    return [Tool(write_file)]

def shell_tools(root: str, timeout: int = 20) -> list[Tool]:
    def run_shell(command: str) -> str:
        """Run a shell command in the root directory (20s timeout). Returns exit code, stdout, stderr.

        Args:
            command: the command line
        """
        p = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout)
        return f"exit {p.returncode}\n{p.stdout}{('[stderr] ' + p.stderr) if p.stderr else ''}"
    return [Tool(run_shell)]

def default_toolbox(root: str, allow_write: bool = False, allow_shell: bool = False) -> Toolbox:
    tb = Toolbox([Tool(calculate), Tool(today), *fs_tools(root)])
    if allow_write:
        for t in write_tools(root): tb.add(t)
    if allow_shell:
        for t in shell_tools(root): tb.add(t)
    return tb
