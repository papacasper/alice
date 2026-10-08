"""Tool registry: turn plain Python functions into validated, schema'd tools.

    from harness import tool
    @tool
    def add(a: float, b: float) -> str:
        '''Add two numbers.

        Args:
            a: first number
            b: second number
        '''
        return str(a + b)
"""
import inspect, re, typing

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
MAX_OUTPUT = 6000

class Tool:
    def __init__(self, fn, name: str | None = None, description: str | None = None, max_output: int = MAX_OUTPUT):
        self.fn, self.name, self.max_output = fn, name or fn.__name__, max_output
        doc = inspect.getdoc(fn) or ""
        self.description = description or doc.split("\n\n")[0].replace("\n", " ").strip() or self.name
        argdocs = {}
        if "Args:" in doc:
            for m in re.finditer(r"^\s+(\w+):\s*(.+)$", doc.split("Args:", 1)[1], re.M):
                argdocs[m.group(1)] = m.group(2).strip()
        hints = typing.get_type_hints(fn)
        self.params, self.required = {}, []
        for pname, p in inspect.signature(fn).parameters.items():
            t = hints.get(pname, str)
            if typing.get_origin(t) is not None:  # Optional[int], list[str], ... -> use the first concrete arg
                t = next((a for a in typing.get_args(t) if a in JSON_TYPES), typing.get_origin(t))
            self.params[pname] = {"type": JSON_TYPES.get(t, "string"), "description": argdocs.get(pname, pname)}
            if p.default is inspect.Parameter.empty:
                self.required.append(pname)
        self._types = {n: s["type"] for n, s in self.params.items()}

    def schema(self) -> dict:
        props = {k: {a: b for a, b in v.items() if not (a == "description" and b == k)} for k, v in self.params.items()}   # a description that just repeats the name is noise
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": self.required}}}

    def _coerce(self, key, value):
        want = self._types[key]
        try:
            if want == "integer" and not isinstance(value, bool): return int(float(value))
            if want == "number" and not isinstance(value, bool): return float(value)
            if want == "boolean" and isinstance(value, str): return value.strip().lower() in ("true", "1", "yes")
            if want == "string" and not isinstance(value, str): return str(value)
        except (TypeError, ValueError):
            raise ValueError(f"argument '{key}' must be {want}, got {value!r}")
        return value

    def _repair(self, args: dict) -> dict:
        """Fix the slips small models make: a misspelled or aliased parameter name that clearly means one real parameter
        (file -> file_path), and a list/object passed as a JSON string."""
        import difflib, json
        out = dict(args)
        for k in [k for k in args if k not in self.params]:
            free = [p for p in self.params if p not in out]
            near = difflib.get_close_matches(k, free, n=2, cutoff=0.6) or [p for p in free if p.startswith(k) or k.startswith(p)]
            if len(near) == 1: out[near[0]] = out.pop(k)
        for k, v in out.items():
            if self._types.get(k) in ("array", "object") and isinstance(v, str):
                try: out[k] = json.loads(v)
                except ValueError: pass
        return out

    def call(self, args: dict) -> str:
        """Validate, coerce and run. Never raises: failures come back as 'error: ...' for the model."""
        try:
            if not isinstance(args, dict):
                return "error: arguments must be an object"
            args = self._repair(args)
            unknown = sorted(set(args) - set(self.params))
            if unknown:
                return f"error: unknown argument(s) {unknown}; valid: {sorted(self.params)}"
            missing = [r for r in self.required if r not in args]
            if missing:
                return f"error: missing required argument(s) {missing}"
            out = self.fn(**{k: self._coerce(k, v) for k, v in args.items()})
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"
        out = out if isinstance(out, str) else str(out)
        if len(out) > self.max_output:   # cut at a line end and say so; a mid-line cut once hid Read's "call again with offset" hint
            cut = out.rfind("\n", 0, self.max_output) if "\n" in out[:self.max_output] else self.max_output
            out = out[:cut] + f"\n[output cut: {len(out) - cut} more chars not shown]"
        return out

def tool(fn=None, *, name: str | None = None, description: str | None = None):
    """Decorator: `@tool` or `@tool(name=...)`. Returns a Tool (still callable via .fn)."""
    make = lambda f: Tool(f, name, description)
    return make(fn) if fn else make

class Toolbox:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> "Toolbox":
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name: {t.name}")
        from . import slim; slim.apply(t)
        self._tools[t.name] = t
        return self

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, args: dict) -> str:
        t = self._tools.get(name)
        if t: return t.call(args)
        import difflib
        near = difflib.get_close_matches(name, self.names(), n=1, cutoff=0.6)
        return f"error: unknown tool '{name}'" + (f"; did you mean '{near[0]}'?" if near else "") + f" available: {self.names()}"

    @classmethod
    def from_file(cls, path: str) -> list[Tool]:
        """Load every `@tool`-decorated object from a Python file (plugin mechanism)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("harness_plugin_" + re.sub(r"\W", "_", path), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return [v for v in vars(mod).values() if isinstance(v, Tool)]
