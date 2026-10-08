"""File helpers that always close their handle (bare open().read() leaks it until GC: ResourceWarning under -W error)."""
import json

def read_text(path: str, errors: str | None = None) -> str:
    with open(path, errors=errors) as f: return f.read()

def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f: return f.read()

def read_json(path: str):
    with open(path) as f: return json.load(f)

def write_json(path: str, data, indent: int = 2):
    with open(path, "w") as f: json.dump(data, f, indent=indent)

def write_file(path: str, mode: str, data):
    """open(path, mode).write(data), closed. mode: w, wb, a."""
    with open(path, mode) as f: f.write(data)

def append_line(path: str, line: str):
    with open(path, "a") as f: f.write(line + "\n")
