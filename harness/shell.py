"""How Alice launches bash: as the user's own interactive shell would, so global commands work —
aliases, functions and PATH entries from ~/.bashrc, not just binaries on the inherited PATH.
ALICE_PLAIN_BASH=1 falls back to a bare `bash -c`."""
import os, shutil

NOISE = ("bash: cannot set terminal process group", "bash: no job control in this shell")

def argv(script: str) -> list[str]:
    if os.environ.get("ALICE_PLAIN_BASH") or not os.path.isfile(os.path.expanduser("~/.bashrc")): return ["bash", "-c", script]
    return ["bash", "-ic", script]

def env() -> dict:
    """The environment for those shells. Without TERM (Alice run from cron, systemd or /rc), every `tput` in ~/.bashrc
    prints "No value for $TERM" into each Bash result; TERM=dumb silences it. A real terminal's TERM is kept."""
    return {**os.environ, "TERM": os.environ.get("TERM") or "dumb"}

def clean(stderr: str) -> str:
    """Drop the job-control warnings and the trailing `exit` an interactive bash prints when it has no terminal."""
    return "\n".join(l for l in stderr.splitlines() if not l.startswith(NOISE) and l != "exit")
