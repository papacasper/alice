"""Background shells (BashOutput / KillShell), NotebookEdit and WebSearch."""
import atexit, html, json, os, re, shutil, signal, subprocess, tempfile, urllib.parse, urllib.request
from .tools import Tool
from . import shell as shellmod

def start_job(st, command: str) -> str:
    jid = f"bash_{len(st.jobs) + 1}"
    f = tempfile.NamedTemporaryFile("w+", prefix="alice-job-", suffix=".log", delete=False)
    proc = subprocess.Popen(shellmod.argv(command), cwd=st.cwd, stdin=subprocess.DEVNULL, stdout=f,
                            stderr=subprocess.STDOUT, start_new_session=True)
    st.jobs[jid] = {"proc": proc, "file": f.name, "cmd": command, "pos": 0}
    atexit.register(lambda: _kill(proc))
    return f"Started background shell {jid} (pid {proc.pid}). Poll it with BashOutput(bash_id='{jid}'); stop it with KillShell."

def _kill(proc):
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGTERM)
        except OSError: pass

def http_get(url: str, timeout: int = 20, limit: int = 600_000) -> str:
    """GET via Node's fetch (DuckDuckGo bot-challenges Python's and curl's TLS fingerprints), then curl, then urllib."""
    if shutil.which("node"):
        r = subprocess.run(["node", os.path.join(os.path.dirname(__file__), "fetch.mjs"), url, str(limit)],
                           capture_output=True, errors="replace", text=True, timeout=timeout + 5)
        if r.returncode == 0 and r.stdout: return r.stdout
    if shutil.which("curl"):
        r = subprocess.run(["curl", "-sSL", "-m", str(timeout), "-A", "Mozilla/5.0 (X11; Linux x86_64) alice-harness", url],
                           capture_output=True, errors="replace", text=True)
        if r.returncode == 0: return r.stdout[:limit]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 alice-harness"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read(limit).decode("utf-8", errors="replace")

def bg_tools(st) -> list[Tool]:
    def BashOutput(bash_id: str, filter: str = "") -> str:
        """Get the NEW output of a background shell since you last checked, plus whether it is still running.

        Args:
            bash_id: the id returned when you started it (e.g. bash_1)
            filter: optional regex; only matching lines are returned
        """
        j = st.jobs.get(bash_id)
        if not j: return f"error: no such shell '{bash_id}'; running: {sorted(st.jobs)}"
        with open(j["file"], errors="replace") as f:
            f.seek(j["pos"]); new = f.read(); j["pos"] = f.tell()
        if filter: new = "\n".join(l for l in new.splitlines() if re.search(filter, l))
        rc = j["proc"].poll()
        status = "running" if rc is None else f"exited with code {rc}"
        return f"[{bash_id}: {status}]\n" + (new[-5500:] or "(no new output)")

    def KillShell(shell_id: str) -> str:
        """Stop a background shell.

        Args:
            shell_id: the id of the background shell (e.g. bash_1)
        """
        j = st.jobs.get(shell_id)
        if not j: return f"error: no such shell '{shell_id}'; running: {sorted(st.jobs)}"
        if j["proc"].poll() is not None: return f"{shell_id} had already exited (code {j['proc'].returncode})"
        _kill(j["proc"]); return f"Killed {shell_id}"

    def NotebookEdit(notebook_path: str, new_source: str = "", cell_number: int = 0, cell_type: str = "code",
                     edit_mode: str = "replace") -> str:
        """Edit a Jupyter notebook (.ipynb) cell: replace its source, insert a new cell at cell_number, or delete it. Cell numbers are 0-based.

        Args:
            notebook_path: path of the .ipynb file
            new_source: the new cell source (ignored for delete)
            cell_number: 0-based index of the cell to change, or where to insert
            cell_type: code or markdown
            edit_mode: replace, insert or delete
        """
        p = st.path(notebook_path)
        if not os.path.isfile(p): return f"error: file does not exist: {p}"
        nb = json.load(open(p)); cells = nb.get("cells", [])
        if edit_mode not in ("replace", "insert", "delete"): return "error: edit_mode must be replace, insert or delete"
        if cell_type not in ("code", "markdown"): return "error: cell_type must be code or markdown"
        n = len(cells)
        if edit_mode == "insert":
            if not 0 <= cell_number <= n: return f"error: cell_number must be 0-{n} for insert"
            cell = {"cell_type": cell_type, "metadata": {}, "source": new_source.splitlines(True)}
            if cell_type == "code": cell.update(outputs=[], execution_count=None)
            cells.insert(cell_number, cell)
        else:
            if not 0 <= cell_number < n: return f"error: cell_number must be 0-{n - 1} (notebook has {n} cells)"
            if edit_mode == "delete": del cells[cell_number]
            else:
                c = cells[cell_number]; c["source"] = new_source.splitlines(True); c["cell_type"] = cell_type
                if cell_type == "code": c.setdefault("outputs", []); c["outputs"] = []; c["execution_count"] = None
                else: c.pop("outputs", None); c.pop("execution_count", None)
        st.snap(p); nb["cells"] = cells
        with open(p, "w") as f: json.dump(nb, f, indent=1)
        return f"{edit_mode} cell {cell_number} in {p} ({len(cells)} cells now)"

    def WebSearch(query: str, max_results: int = 5) -> str:
        """Search the web (DuckDuckGo) and return titles, URLs and snippets. Follow up with WebFetch to read a result.

        Args:
            query: the search query
            max_results: how many results to return
        """
        n = max(1, min(max_results, 10))
        if os.environ.get("ALICE_SEARXNG_URL"):   # e.g. http://mediaserver:8080 (JSON format must be enabled)
            try:
                d = json.loads(http_get(os.environ["ALICE_SEARXNG_URL"].rstrip("/") + "/search?" +
                                        urllib.parse.urlencode({"q": query, "format": "json"})))
            except ValueError: return "error: the SearXNG instance did not return JSON (enable the json format)"
            return "\n".join(f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {(r.get('content') or '')[:200]}"
                             for i, r in enumerate(d.get("results", [])[:n], 1)) or "No results"
        page = http_get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}))
        strip = lambda t: html.unescape(re.sub(r"<[^>]+>", "", t)).strip()
        out = []
        for blk in page.split('class="result__a"')[1:]:
            m = re.match(r'[^>]*href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
            if not m: continue
            sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', blk, re.S)
            url = m.group(1)
            if "uddg=" in url: url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
            out.append(f"{len(out) + 1}. {strip(m.group(2))}\n   {url}\n   {strip(sn.group(1) if sn else '')[:200]}")
            if len(out) >= n: break
        if not out and ("anomaly" in page or "challenge" in page):
            return ("error: DuckDuckGo served a bot challenge to this machine. Set ALICE_SEARXNG_URL to a SearXNG instance, "
                    "or use WebFetch on a URL you already know.")
        return "\n".join(out) or "No results"

    return [Tool(f) for f in (BashOutput, KillShell, NotebookEdit, WebSearch)]
