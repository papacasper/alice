"""Safe self-modification: snapshot the harness in git, let the agent edit it, run the tests, then either commit
or roll back. Deterministic gate — the model's own "looks done" is never what decides."""
import os, subprocess, sys

HARNESS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HARNESS)          # the local-llm dir the tests run from

def git(*a, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=cwd or HARNESS, capture_output=True, text=True, timeout=60)

def snapshot(path: str = HARNESS) -> str:
    """Commit any uncommitted harness changes so there is a clean base to roll back to; return the base commit."""
    if git("status", "--porcelain", "--", path, cwd=path).stdout.strip():
        git("add", "-A", "--", path, cwd=path)
        git("commit", "-qm", "alice: pre-selfedit snapshot", "--", path, cwd=path)
    return git("rev-parse", "HEAD", cwd=path).stdout.strip()

def changed(base: str, path: str = HARNESS) -> list[str]:
    tracked = git("diff", "--name-only", base, "--", path, cwd=path).stdout.split()
    new = git("ls-files", "--others", "--exclude-standard", "--", path, cwd=path).stdout.split()
    return sorted(set(tracked + new))

def verify(root: str = ROOT, timeout: int = 240) -> tuple[bool, str]:
    """Imports cleanly and the whole suite passes."""
    env = {**os.environ, "PYTHONPATH": root}
    imp = subprocess.run([sys.executable, "-c", "import harness.cli"], cwd=root, env=env, capture_output=True, text=True)
    if imp.returncode: return False, "import failed:\n" + imp.stderr[-800:]
    try: r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "harness/tests"], cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired: return False, f"tests timed out after {timeout}s"
    tail = (r.stderr or r.stdout).strip().splitlines()[-12:]
    return r.returncode == 0, "\n".join(tail)

def rollback(base: str, path: str = HARNESS) -> None:
    git("checkout", base, "--", path, cwd=path)
    git("clean", "-fdq", "--", path, cwd=path)

def commit(msg: str, path: str = HARNESS) -> str:
    git("add", "-A", "--", path, cwd=path)
    git("commit", "-qm", "alice selfedit: " + msg[:70], "--", path, cwd=path)
    return git("rev-parse", "--short", "HEAD", cwd=path).stdout.strip()

def restart_argv(argv: list[str], sid: str) -> list[str]:
    """Command line that relaunches this session: original flags minus -c/-r, plus --resume <sid>."""
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip: skip = False; continue
        if a in ("-c", "--continue"): continue
        if a in ("-r", "--resume"):
            skip = i + 1 < len(argv) and not argv[i + 1].startswith("-"); continue
        if a.startswith("--resume="): continue
        out.append(a)
    return [sys.executable, "-m", "harness", *out, "--resume", sid]
