"""Local web demo for explainrec — stdlib only.

    .venv/bin/python demo/server.py                 # local Claude CLI backend
    .venv/bin/python demo/server.py --backend api   # Anthropic API backend

Starts a small HTTP server on http://localhost:8765 and opens the demo
page. The pipeline is built (and the baseline solved) once in the
background at startup, so each query costs one LP solve plus the LLM
calls. A browser page cannot call the Claude CLI itself; this server is
the bridge — the page POSTs the query here, the server runs
interpret -> re-solve -> compare -> explain and returns JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explainrec.pipeline import Pipeline  # noqa: E402
from explainrec.llm.backend import get_backend  # noqa: E402

STATE = {
    "pipeline": None,
    "backend": None,
    "status": "starting",
    "lock": threading.Lock(),
}


def build_pipeline() -> None:
    try:
        STATE["status"] = "loading data and fitting the rating model..."
        pipeline = Pipeline.build()
        STATE["status"] = "solving the baseline problem (~40 s)..."
        pipeline.base_solution
        STATE["pipeline"] = pipeline
        STATE["status"] = "ready"
    except Exception as e:  # surfaced via /status
        STATE["status"] = f"startup failed: {e}"


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>explainrec demo</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 46rem;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; background: #fafaf8; }
  h1 { font-size: 1.4rem; }
  .sub { color: #666; margin-top: -0.5rem; font-size: 0.9rem; }
  #status { font-size: 0.85rem; color: #666; margin: 0.8rem 0; }
  #status.ready { color: #2a7a2a; }
  textarea { width: 100%; box-sizing: border-box; font: inherit; padding: 0.6rem;
             border: 1px solid #ccc; border-radius: 6px; resize: vertical; }
  .row { display: flex; gap: 0.6rem; align-items: center; margin: 0.6rem 0 1rem; flex-wrap: wrap; }
  button { font: inherit; padding: 0.45rem 1rem; border: 1px solid #888;
           border-radius: 6px; background: #fff; cursor: pointer; }
  button:hover:not(:disabled) { background: #efefef; }
  button:disabled { opacity: 0.5; cursor: default; }
  button.example { font-size: 0.8rem; border-color: #ccc; color: #444; }
  label { font-size: 0.85rem; color: #444; }
  .card { background: #fff; border: 1px solid #e2e2de; border-radius: 8px;
          padding: 1rem 1.2rem; margin: 1rem 0; }
  .card h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
             color: #888; margin: 0 0 0.5rem; }
  pre { white-space: pre-wrap; word-break: break-word; font-size: 0.82rem;
        background: #f6f6f3; padding: 0.7rem; border-radius: 6px; overflow-x: auto; }
  #explanation { font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; }
  .error { color: #a02020; }
  .spinner { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>explainrec — what-if queries on a constrained recommender</h1>
<p class="sub">MovieLens 100k · allocation LP · the LLM edits the problem, the solver answers</p>
<div id="status">checking…</div>

<textarea id="query" rows="2" placeholder="e.g. What happens if we stop promoting cold items?"></textarea>
<div class="row">
  <button id="ask" disabled>Ask</button>
  <label><input type="checkbox" id="skip"> skip explanation (faster)</label>
</div>
<div class="row">
  <button class="example">What happens if we stop promoting cold items?</button>
  <button class="example">Would user 42 get the same movies if she were male?</button>
  <button class="example">What changes if we never recommend Horror movies?</button>
</div>

<div id="out" style="display:none">
  <div class="card"><h2>Modification</h2><div id="summary"></div><pre id="mod"></pre></div>
  <div class="card"><h2>Comparison report</h2><pre id="report"></pre></div>
  <div class="card" id="explCard"><h2>Explanation</h2><div id="explanation"></div></div>
</div>

<script>
const $ = (id) => document.getElementById(id);

async function poll() {
  try {
    const r = await (await fetch('/status')).json();
    $('status').textContent = r.status === 'ready' ? 'ready' : r.status;
    $('status').className = r.status === 'ready' ? 'ready' : '';
    $('ask').disabled = r.status !== 'ready';
    if (r.status !== 'ready') setTimeout(poll, 2000);
  } catch { setTimeout(poll, 2000); }
}
poll();

document.querySelectorAll('button.example').forEach(b =>
  b.addEventListener('click', () => { $('query').value = b.textContent; }));

$('ask').addEventListener('click', async () => {
  const query = $('query').value.trim();
  if (!query) return;
  $('ask').disabled = true;
  $('status').innerHTML = '<span class="spinner">&#9696;</span> interpreting, re-solving, comparing… (a minute or two)';
  $('status').className = '';
  $('out').style.display = 'none';
  try {
    const r = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, skip_explanation: $('skip').checked}),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    $('summary').textContent = data.modification.summary || '';
    $('mod').textContent = JSON.stringify(data.modification, null, 2);
    $('report').textContent = data.report_text || '(no change to the problem)';
    $('explanation').textContent = data.explanation || '';
    $('explCard').style.display = data.explanation ? '' : 'none';
    $('out').style.display = '';
    $('status').textContent = 'ready';
    $('status').className = 'ready';
  } catch (e) {
    $('status').innerHTML = '<span class="error">error: ' + e.message + '</span>';
  } finally {
    $('ask').disabled = false;
  }
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/status":
            self._send_json({"status": STATE["status"]})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self._send_json({"error": "not found"}, 404)
            return
        if STATE["pipeline"] is None:
            self._send_json({"error": "pipeline not ready yet"}, 503)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            query = str(payload["query"])
            skip = bool(payload.get("skip_explanation", False))
            with STATE["lock"]:  # one solve at a time
                result = STATE["pipeline"].ask(
                    query, backend=STATE["backend"], skip_explanation=skip,
                )
            from explainrec.compare import report_text
            self._send_json({
                "modification": result.modification.model_dump(exclude_defaults=True),
                "report_text": report_text(result.report) if result.report else "",
                "explanation": result.explanation or "",
            })
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep the terminal quiet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["api", "cli"], default="cli")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    kwargs = {"model": args.llm_model} if args.llm_model else {}
    STATE["backend"] = get_backend(args.backend, **kwargs)
    threading.Thread(target=build_pipeline, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"explainrec demo at {url}  (backend: {args.backend}; Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
