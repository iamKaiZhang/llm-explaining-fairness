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
from explainrec.llm.explainer import explain  # noqa: E402
from explainrec.eval.claims import check_attribution, check_claims  # noqa: E402
from explainrec.eval.faithfulness import check_faithfulness  # noqa: E402
from explainrec.eval.mechanism_grounding import (  # noqa: E402
    check_mechanism_grounding, tag_mechanism,
)
from explainrec.eval.projection import (  # noqa: E402
    QUERIES_PATH, find_entry_by_query, score_modification,
)

STATE = {
    "pipeline": None,
    "backend": None,
    "compare_backends": None,  # {label: Backend} for /ask-compare, or None if not configured
    "status": "starting",
    "lock": threading.Lock(),
}

ROLE_LABELS = {
    "end_user": "End user",
    "item_provider": "Item / content provider",
    "operator": "Platform operator",
    "regulator": "Regulator / auditor",
}


def _load_queries_by_role() -> dict:
    import yaml

    entries = yaml.safe_load(QUERIES_PATH.read_text())
    by_role: dict[str, list[dict]] = {}
    for e in entries:
        by_role.setdefault(e["role"], []).append(
            {"id": e["id"], "query": e["query"], "type": e["type"]}
        )
    return {
        "roles": [{"key": k, "label": ROLE_LABELS.get(k, k)} for k in by_role],
        "questions": by_role,
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


def _check_explanation(query: str, modification, report: dict, explanation: str) -> dict:
    """Faithfulness + mechanism-grounding + projection check for one
    (modification, report, explanation) triple.

    Faithfulness and mechanism grounding always run (they need only the
    report/modification and the explanation text actually produced this
    turn -- no gold label required). Mechanism grounding is skipped when
    the modification touches zero or more than one mechanism (see
    ``tag_mechanism``), since there is then no single ground truth to check
    the explanation's causal language against. Projection scoring only runs
    when this exact query text has a gold label in experiments/queries.yaml;
    otherwise there is nothing to score against, so we say so rather than
    guess.

    Split out from the single-backend ``/ask`` path so the same scoring can
    run once per model in the two-model comparison path -- both need
    identical checks, just applied to a different explanation of the same
    report.
    """
    check: dict = {}

    if explanation and report:
        f = check_faithfulness(report, explanation)
        check["faithfulness"] = {
            "claimed": f.claimed,
            "unmatched": f.unmatched,
            "unmatched_sentences": f.unmatched_sentences,
            "match_rate": round(f.match_rate, 3),
        }
        c = check_claims(report, explanation)
        check["claims"] = {
            "verified_rate": round(c.verified_rate, 3),
            "coverage": round(c.coverage, 3),
            "n_direction_sentences": c.n_direction_sentences,
            "n_claim_sentences": c.n_claim_sentences,
            "claims": [
                {
                    "sentence": cl.sentence, "metric": cl.metric,
                    "phrase": cl.phrase, "direction": cl.direction,
                    "negated": cl.negated, "actual": cl.actual,
                    "verdict": cl.verdict,
                }
                for cl in c.claims
            ],
        }
        a = check_attribution(report, explanation)
        check["attribution"] = {
            "rate": round(a.attribution_rate, 3),
            "items": [
                {
                    "sentence": at.sentence, "metric": at.metric,
                    "phrase": at.phrase, "number": at.number,
                    "verdict": at.verdict,
                }
                for at in a.attributions
            ],
        }
    else:
        check["faithfulness"] = None
        check["claims"] = None
        check["attribution"] = None

    mechanism = tag_mechanism(modification) if explanation else None
    if mechanism is None:
        check["mechanism_grounding"] = None
    else:
        m = check_mechanism_grounding(explanation, mechanism)
        check["mechanism_grounding"] = {
            "mechanism": m.mechanism,
            "hits": m.hits,
            "misses": m.misses,
            "grounded": m.grounded,
        }

    try:
        entry = find_entry_by_query(query)
    except FileNotFoundError:
        entry = None
    if entry is None:
        check["projection"] = None
    else:
        scored = score_modification(entry, modification)
        check["projection"] = {
            "expressible": scored.expressible,
            "correct": scored.correct,
            "field_matches": scored.field_matches,
            "note": scored.note,
        }
    return check


def _check_result(query: str, result) -> dict:
    return _check_explanation(query, result.modification, result.report, result.explanation)


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
  .pill { display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px;
          font-size: 0.78rem; font-weight: 600; white-space: nowrap; }
  .pill.ok { background: #e5f3e5; color: #2a7a2a; }
  .pill.bad { background: #fbe6e6; color: #a02020; }
  .pill.na { background: #eee; color: #777; }
  .checkDetail { font-size: 0.82rem; color: #333; margin-top: 0.35rem; line-height: 1.45; }
  .checkDetail b { color: #1a1a1a; }
  .checkIntro { font-size: 0.83rem; color: #666; line-height: 1.5; margin: 0 0 0.9rem; }
  .checkItem { border-top: 1px solid #eee; padding: 0.7rem 0 0.2rem; }
  .checkItem-last { padding-bottom: 0; }
  .checkHead { display: flex; justify-content: space-between; align-items: baseline;
               gap: 0.6rem; flex-wrap: wrap; }
  .checkName { font-size: 0.88rem; font-weight: 600; color: #222; }
  .checkMethod { font-size: 0.8rem; color: #777; line-height: 1.5; margin: 0.3rem 0 0; }
  .compareCols { display: flex; gap: 1rem; flex-wrap: wrap; }
  .compareCol { flex: 1; min-width: 18rem; border: 1px solid #e2e2de; border-radius: 8px;
                padding: 0.8rem 1rem; background: #fcfcfa; }
  .compareCol h3 { font-size: 0.82rem; margin: 0 0 0.5rem; color: #222; }
  .compareCol .explanation { font-size: 0.88rem; line-height: 1.45; white-space: pre-wrap;
                              margin-bottom: 0.7rem; }
  .compareCol .checkItem { padding: 0.5rem 0; }
  .compareCol .checkName { font-size: 0.78rem; }
  .spinner { display: inline-block; animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .roleTabs { display: flex; gap: 0.4rem; flex-wrap: wrap; margin: 0.6rem 0; }
  .roleTab { font-size: 0.82rem; padding: 0.35rem 0.8rem; border: 1px solid #ccc;
             border-radius: 999px; background: #fff; cursor: pointer; color: #444; }
  .roleTab:hover { background: #efefef; }
  .roleTab.active { background: #2a2a2a; color: #fff; border-color: #2a2a2a; }
  .questionBox { max-height: 9.5rem; overflow-y: auto; border: 1px solid #e2e2de;
                  border-radius: 6px; background: #fff; margin-bottom: 0.8rem; }
  .questionItem { padding: 0.5rem 0.7rem; font-size: 0.85rem; cursor: pointer;
                   border-bottom: 1px solid #f0f0ec; display: flex; gap: 0.5rem; align-items: baseline; }
  .questionItem:last-child { border-bottom: none; }
  .questionItem:hover { background: #f6f6f3; }
  .questionItem .type { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.03em;
                          color: #999; white-space: nowrap; }

  table.reportTable { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  table.reportTable th, table.reportTable td { text-align: right; padding: 0.3rem 0.5rem;
                          border-bottom: 1px solid #eee; }
  table.reportTable th:first-child, table.reportTable td:first-child { text-align: left; }
  table.reportTable th { color: #888; font-weight: 600; font-size: 0.72rem;
                          text-transform: uppercase; letter-spacing: 0.03em; }
  .metricRow { margin: 0.55rem 0; }
  .metricRow .metricLabel { font-size: 0.82rem; color: #333; margin-bottom: 0.2rem;
                              display: flex; justify-content: space-between; }
  .metricRow .metricLabel .delta { font-size: 0.78rem; font-weight: 600; }
  .delta.up { color: #2a7a2a; } .delta.down { color: #a02020; } .delta.flat { color: #888; }
  .barTrack { position: relative; height: 0.9rem; background: #f0f0ec; border-radius: 4px; }
  .barFill { position: absolute; top: 0; left: 0; height: 100%; border-radius: 4px; }
  .barFill.base { background: #b9b9ae; }
  .barFill.modified { background: #4a7fd6; }
  .barLegend { font-size: 0.72rem; color: #999; margin-top: 0.15rem; }
  .modList { font-size: 0.85rem; margin: 0 0 0.6rem; padding-left: 1.1rem; }
  .modList li { margin: 0.15rem 0; }
  details.rawToggle { margin-top: 0.6rem; }
  details.rawToggle summary { font-size: 0.78rem; color: #888; cursor: pointer; }
  .focalCard { border-top: 1px dashed #ddd; margin-top: 0.8rem; padding-top: 0.6rem; }
  .focalCols { display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: 0.82rem; }
  .focalCols > div { flex: 1; min-width: 10rem; }
  .focalCols h3 { font-size: 0.72rem; text-transform: uppercase; color: #888; margin: 0 0 0.3rem; }
  .focalCols ul { margin: 0; padding-left: 1.1rem; }
</style>
</head>
<body>
<h1>explainrec — what-if queries on a constrained recommender</h1>
<p class="sub">MovieLens 100k · allocation LP · the LLM edits the problem, the solver answers</p>
<div id="status">checking…</div>

<p class="sub" style="margin-top:1rem">Pick who's asking, then a question:</p>
<div class="roleTabs" id="roleTabs"></div>
<div class="questionBox" id="questionBox"></div>

<textarea id="query" rows="2" placeholder="e.g. What happens if we stop promoting cold items?"></textarea>
<div class="row">
  <button id="ask" disabled>Ask</button>
  <label><input type="checkbox" id="skip"> skip explanation (faster)</label>
  <label id="compareLabel" style="display:none">
    <input type="checkbox" id="compare"> compare two models' explanations
  </label>
</div>

<div id="out" style="display:none">
  <div class="card">
    <h2>Modification</h2>
    <ul class="modList" id="modList"></ul>
    <details class="rawToggle"><summary>raw JSON</summary><pre id="mod"></pre></details>
  </div>
  <div class="card">
    <h2>Comparison report</h2>
    <div id="reportMetrics"></div>
    <div id="reportTables"></div>
    <div id="focalArea"></div>
    <details class="rawToggle"><summary>raw text report</summary><pre id="report"></pre></details>
  </div>
  <div class="card" id="explCard"><h2>Explanation</h2><div id="explanation"></div></div>
  <div class="card" id="checkCard">
    <h2>Check</h2>
    <p class="checkIntro">
      Four independent, mechanical checks — none of them ask an LLM to grade the answer.
      Each compares the system's output against something it cannot have faked: a
      pre-written correct answer, the solver's own numbers, or the type of edit that was
      actually made.
    </p>

    <div class="checkItem">
      <div class="checkHead"><span class="checkName">1. Was the question translated correctly?</span><span id="checkProjection"></span></div>
      <p class="checkMethod">
        <b>Method:</b> some example questions come with a pre-written "gold" answer —
        the exact edit to the optimization problem a correct reading of the question
        should produce (e.g. "remove the cold-item-exposure constraint"). This check
        compares the system's actual edit to that gold answer, field by field. For
        questions the schema cannot express at all (e.g. "treat men and women
        equally"), the only correct behavior is to decline rather than quietly
        substitute a different, answerable edit — so the check instead looks for a
        no-op.
      </p>
      <div id="checkProjectionDetail" class="checkDetail"></div>
    </div>

    <div class="checkItem">
      <div class="checkHead"><span class="checkName">2. Are the numbers in the explanation real?</span><span id="checkFaithfulness"></span></div>
      <p class="checkMethod">
        <b>Method:</b> every number written in the explanation is extracted with a
        regular expression, then checked against every number in the comparison
        report above (allowing for rounding, sign flips like "-5%" vs. "a 5% drop",
        and percent/fraction rescaling like "66%" vs. 0.658). A number with no match
        anywhere in the report did not come from the solver — it is either invented or
        miscalculated. Numbers stated next to a named metric are held to a stricter
        standard: they must match that metric's <i>own</i> values in the report, so a
        real number quoted for the wrong quantity is also caught. This check needs no
        gold label, since it runs on whatever the system produced for <i>this</i>
        question.
      </p>
      <div id="checkFaithfulnessDetail" class="checkDetail"></div>
    </div>

    <div class="checkItem">
      <div class="checkHead"><span class="checkName">3. Do the numbers tell the right story?</span><span id="checkClaims"></span></div>
      <p class="checkMethod">
        <b>Method:</b> a real number can still be attached to a false story — "exposure
        rose to 3775" passes check 2 when 3775 is the <i>old</i> value and exposure
        actually fell. This check breaks the explanation into atomic directional claims
        ("&lt;metric&gt; went up / down / stayed the same") and verifies each one against the
        actual base&rarr;modified movement in the comparison report. The result is a
        per-claim scorecard rather than one verdict, so a mostly-right explanation with
        one wrong direction is distinguishable from one that is wrong everywhere.
        Claims about metrics the report does not carry are marked unverifiable and left
        out of the score.
      </p>
      <div id="checkClaimsDetail" class="checkDetail"></div>
    </div>

    <div class="checkItem checkItem-last">
      <div class="checkHead"><span class="checkName">4. Is the stated cause the real cause?</span><span id="checkMechanism"></span></div>
      <p class="checkMethod">
        <b>Method:</b> a number can be correct while the reason given for it is wrong —
        e.g. blaming "this user's taste changed" when the real cause was a constraint
        being removed. When the edit touches exactly one mechanism (a constraint, a
        gender counterfactual, or the slate size — known for certain, since we made the
        edit), the explanation's own wording is scanned for vocabulary that matches
        that mechanism versus vocabulary that belongs to a different one. Skipped when
        an edit mixes mechanisms or changes nothing, since there is then no single
        cause to check against.
      </p>
      <div id="checkMechanismDetail" class="checkDetail"></div>
    </div>
  </div>
  <div class="card" id="compareCard" style="display:none">
    <h2>Model comparison</h2>
    <p class="checkIntro">
      Same edit, same comparison report, explained separately by each model below —
      the interpreter ran once, so any difference is attributable to the explainer
      model, not to a different edit. Each column runs the same four checks as above.
    </p>
    <div class="compareCols" id="compareCols"></div>
  </div>
</div>

<script>
window.addEventListener('error', (ev) => {
  const el = document.createElement('pre');
  el.style.background = '#fbe6e6';
  el.style.color = '#a02020';
  el.style.padding = '0.6rem';
  el.style.whiteSpace = 'pre-wrap';
  el.textContent = 'JS error: ' + ev.message + ' (' + ev.filename + ':' + ev.lineno + ':' + ev.colno + ')';
  document.body.prepend(el);
});

const $ = (id) => document.getElementById(id);

let COMPARE_BACKENDS = [];

async function poll() {
  try {
    const r = await (await fetch('/status')).json();
    $('status').textContent = r.status === 'ready' ? 'ready' : r.status;
    $('status').className = r.status === 'ready' ? 'ready' : '';
    $('ask').disabled = r.status !== 'ready';
    COMPARE_BACKENDS = r.compare_backends || [];
    $('compareLabel').style.display = COMPARE_BACKENDS.length >= 2 ? '' : 'none';
    if (r.status !== 'ready') setTimeout(poll, 2000);
  } catch { setTimeout(poll, 2000); }
}
poll();

let QUESTIONS_BY_ROLE = {};
let ACTIVE_ROLE = null;

async function loadQueries() {
  try {
    const data = await (await fetch('/queries')).json();
    QUESTIONS_BY_ROLE = data.questions || {};
    const tabs = $('roleTabs');
    tabs.innerHTML = '';
    (data.roles || []).forEach((r, i) => {
      const b = document.createElement('button');
      b.className = 'roleTab' + (i === 0 ? ' active' : '');
      b.textContent = r.label;
      b.dataset.role = r.key;
      b.addEventListener('click', () => selectRole(r.key));
      tabs.appendChild(b);
      if (i === 0) ACTIVE_ROLE = r.key;
    });
    if (ACTIVE_ROLE) renderQuestionList(ACTIVE_ROLE);
  } catch (e) {
    $('questionBox').textContent = 'could not load example questions: ' + e.message;
  }
}

function selectRole(roleKey) {
  ACTIVE_ROLE = roleKey;
  document.querySelectorAll('.roleTab').forEach(b =>
    b.classList.toggle('active', b.dataset.role === roleKey));
  renderQuestionList(roleKey);
}

function renderQuestionList(roleKey) {
  const box = $('questionBox');
  box.innerHTML = '';
  (QUESTIONS_BY_ROLE[roleKey] || []).forEach(q => {
    const item = document.createElement('div');
    item.className = 'questionItem';
    item.innerHTML = '<span class="type">' + q.type + '</span><span>' + q.query + '</span>';
    item.addEventListener('click', () => { $('query').value = q.query; });
    box.appendChild(item);
  });
}
loadQueries();

function pill(text, cls) {
  return '<span class="pill ' + cls + '">' + text + '</span>';
}

function projectionCheck(proj) {
  if (proj === null || proj === undefined) {
    return { pill: pill('not scored', 'na'), detail:
      '<b>Why:</b> this exact question is not one of the pre-written examples with a ' +
      'known-correct answer, so there is nothing to compare the system’s edit against. ' +
      'Pick a question from the list above to see this check run.' };
  }
  if (!proj.expressible) {
    return proj.correct ? { pill: pill('correct', 'ok'), detail:
      '<b>Why:</b> this question asks for something the optimization problem’s schema ' +
      'cannot represent (see the method above). The system correctly recognized that ' +
      'and made no edit, rather than guessing at a nearby question it could answer.' }
    : { pill: pill('incorrect', 'bad'), detail:
      '<b>Why:</b> this question asks for something the schema cannot represent, so the ' +
      'only correct move was to decline. Instead the system silently made an edit ' +
      '(' + esc(proj.note || '') + ') and then explained ' +
      '<i>that</i> as if it were the answer to the question actually asked — a confident ' +
      'answer to a different question than the one posed.' };
  }
  if (proj.correct) {
    return { pill: pill('correct', 'ok'), detail:
      '<b>Why:</b> every field of the system’s edit (constraints added/removed, gender ' +
      'overrides, slate size, focal users) matches the pre-written correct answer for ' +
      'this question exactly.' };
  }
  const wrong = Object.entries(proj.field_matches || {}).filter(([, ok]) => !ok).map(([f]) => f);
  return { pill: pill('incorrect', 'bad'), detail:
    '<b>Why:</b> the system’s edit differs from the pre-written correct answer in ' +
    (wrong.length ? '<b>' + wrong.map(esc).join(', ') + '</b>' : 'at least one field') +
    '. Everything downstream (the re-solve, the comparison, the explanation) is then an ' +
    'answer to a different edit than the one the question actually called for.' };
}

function faithfulnessCheck(f, a) {
  if (f === null || f === undefined) {
    return { pill: pill('not applicable', 'na'), detail:
      '<b>Why:</b> the explanation step was skipped (or produced no report), so there is no ' +
      'text to check numbers in.' };
  }
  // field attribution: a number can exist in the report yet be stated for
  // the wrong metric ("the objective is 3775" when 3775 is the exposure)
  const misattributed = (a && a.items) ? a.items.filter(x => x.verdict === 'wrong_field') : [];
  const attrLines = misattributed.map(x =>
    '<li>“' + esc(x.sentence) + '” <span class="checkDetail" style="display:inline;color:#a02020">(' +
    fmt(x.number) + ' does exist in the report, but not under “' + esc(x.phrase) +
    '” — it belongs to a different metric)</span></li>').join('');
  if (f.unmatched.length === 0) {
    if (misattributed.length) {
      return { pill: pill('incorrect', 'bad'), detail:
        '<b>Why:</b> every number exists somewhere in the report, but the sentence(s) below ' +
        'attach a real number to the wrong metric:' +
        '<ul style="margin:0.3rem 0 0;padding-left:1.1rem">' + attrLines + '</ul>' };
    }
    return { pill: pill('correct', 'ok'), detail: f.claimed.length
      ? '<b>Why:</b> all ' + f.claimed.length + ' number(s) the explanation states (' +
        f.claimed.map(fmt).join(', ') + ') were traced back to a matching value in the ' +
        'comparison report above' +
        ((a && a.items && a.items.length)
          ? ', and each number stated next to a named metric matched that metric’s own values.'
          : '.')
      : '<b>Why:</b> the explanation made no numeric claims worth checking, so there is ' +
        'nothing that could be fabricated.' };
  }
  const sentences = f.unmatched_sentences || [];
  // group the flagged numbers by the exact sentence they came from, so a
  // sentence with two bad numbers is quoted once, not twice
  const bySentence = new Map();
  f.unmatched.forEach((n, i) => {
    const s = sentences[i] || '(sentence not found)';
    if (!bySentence.has(s)) bySentence.set(s, []);
    bySentence.get(s).push(n);
  });
  const quoted = Array.from(bySentence.entries()).map(([s, nums]) =>
    '<li>“' + esc(s) + '” <span class="checkDetail" style="display:inline;color:#a02020">' +
    '(flagged: ' + nums.map(fmt).join(', ') + ')</span></li>'
  ).join('');
  return { pill: pill('incorrect', 'bad'), detail:
    '<b>Why:</b> the sentence(s) below state a number with no close match anywhere in ' +
    'the comparison report above (even allowing for rounding and percent/fraction ' +
    'rescaling), so the solver did not produce it &mdash; the explanation is asserting a ' +
    'number it cannot have grounds for:' +
    '<ul style="margin:0.3rem 0 0;padding-left:1.1rem">' + quoted + attrLines + '</ul>' };
}

function claimsCheck(cl) {
  const dirWord = {up: 'went up', down: 'went down', flat: 'stayed the same'};
  const claimLine = (c) => {
    const said = (c.negated ? 'did not go ' + c.direction : dirWord[c.direction] || c.direction);
    const mark = c.verdict === 'verified' ? '&#10003;' : c.verdict === 'contradicted' ? '&#10007;' : '?';
    const cls = c.verdict === 'verified' ? 'ok' : c.verdict === 'contradicted' ? 'bad' : 'na';
    let line = pill(mark, cls) + ' “' + esc(c.phrase) + ' ' + esc(said) + '”';
    if (c.verdict === 'contradicted') {
      line += ' — the report shows it actually ' + esc(dirWord[c.actual] || c.actual) + '.';
    } else if (c.verdict === 'unverifiable') {
      line += ' — this metric is not in the report, so the claim cannot be checked.';
    }
    return '<div style="margin:0.25rem 0">' + line + '</div>';
  };
  if (cl === null || cl === undefined) {
    return { pill: pill('not applicable', 'na'), detail:
      '<b>Why:</b> the explanation step was skipped (or produced no report), so there are no ' +
      'directional claims to verify.' };
  }
  if (cl.claims.length === 0) {
    if ((cl.n_direction_sentences || 0) > 0) {
      return { pill: pill('unchecked', 'na'), detail:
        '<b>Why:</b> ' + cl.n_direction_sentences + ' sentence(s) talk about something ' +
        'moving up or down, but none name a metric the check’s lexicon recognizes — so ' +
        'those statements went <i>unchecked</i>, which is a coverage gap, not a pass.' };
    }
    return { pill: pill('no claims', 'na'), detail:
      '<b>Why:</b> the explanation states no recognizable “metric went up/down” claims, so ' +
      'there is nothing to verify at this level. (Only metrics in the check’s lexicon are ' +
      'extracted; a paraphrase outside it produces no claim rather than a wrong one.)' };
  }
  const nBad = cl.claims.filter(c => c.verdict === 'contradicted').length;
  const nOk = cl.claims.filter(c => c.verdict === 'verified').length;
  const nDir = cl.n_direction_sentences || 0;
  const nCov = cl.n_claim_sentences || 0;
  const coverageNote = (nDir > nCov)
    ? '<div style="margin-top:0.3rem"><b>Coverage:</b> ' + nCov + ' of ' + nDir +
      ' sentences with movement language yielded a checkable claim; the other ' +
      (nDir - nCov) + ' went unchecked (movement stated in words outside the lexicon).</div>'
    : '';
  return {
    pill: pill(nOk + '/' + (nOk + nBad) + ' claims correct', nBad === 0 ? 'ok' : 'bad'),
    detail: '<b>Why:</b> each directional claim found in the explanation, checked against the ' +
      'report’s actual movement:' + cl.claims.map(claimLine).join('') + coverageNote,
  };
}

function mechanismCheck(m) {
  const mechanismLabels = {
    'constraint-relaxation': 'a constraint being added or removed',
    'rating-reestimation': 'a user-attribute counterfactual (e.g. a gender override)',
    'slate-size-change': 'a change to the slate size',
  };
  if (m === null || m === undefined) {
    return { pill: pill('not applicable', 'na'), detail:
      '<b>Why:</b> this edit either changes nothing, or combines more than one mechanism ' +
      'at once, so there is no single ground-truth cause to check the explanation’s ' +
      'wording against.' };
  }
  if (m.grounded) {
    return { pill: pill('correct', 'ok'), detail:
      '<b>Why:</b> the true cause of this change is ' + (mechanismLabels[m.mechanism] || m.mechanism) +
      ' (we made that edit, so this is known for certain, not inferred). The explanation’s ' +
      'own wording matches that cause &mdash; it used language like ' +
      m.hits.map(h => '“' + esc(h) + '”').join(', ') + '.' };
  }
  return { pill: pill('incorrect', 'bad'), detail: m.misses.length
    ? '<b>Why:</b> the true cause of this change is ' + (mechanismLabels[m.mechanism] || m.mechanism) +
      ', but the explanation instead used language for a different mechanism (' +
      m.misses.map(mi => '“' + esc(mi) + '”').join(', ') +
      ') &mdash; it is telling a causal story that does not match what was actually edited.'
    : '<b>Why:</b> the true cause of this change is ' + (mechanismLabels[m.mechanism] || m.mechanism) +
      ', but the explanation used none of the language that would tie the change to it, ' +
      'so a reader cannot tell the real cause from the explanation alone.' };
}

function renderCheck(check) {
  const proj = projectionCheck(check.projection);
  $('checkProjection').innerHTML = proj.pill;
  $('checkProjectionDetail').innerHTML = proj.detail;

  const f = faithfulnessCheck(check.faithfulness, check.attribution);
  $('checkFaithfulness').innerHTML = f.pill;
  $('checkFaithfulnessDetail').innerHTML = f.detail;

  const cl = claimsCheck(check.claims);
  $('checkClaims').innerHTML = cl.pill;
  $('checkClaimsDetail').innerHTML = cl.detail;

  const m = mechanismCheck(check.mechanism_grounding);
  $('checkMechanism').innerHTML = m.pill;
  $('checkMechanismDetail').innerHTML = m.detail;
}

function checkPanelHtml(check) {
  const proj = projectionCheck(check.projection);
  const f = faithfulnessCheck(check.faithfulness, check.attribution);
  const cl = claimsCheck(check.claims);
  const m = mechanismCheck(check.mechanism_grounding);
  const row = (name, r) =>
    '<div class="checkItem">' +
      '<div class="checkHead"><span class="checkName">' + esc(name) + '</span>' + r.pill + '</div>' +
      '<div class="checkDetail">' + r.detail + '</div>' +
    '</div>';
  return (
    row('Question translated correctly?', proj) +
    row('Numbers real?', f) +
    row('Right story?', cl) +
    row('Right cause?', m)
  );
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function renderModification(mod) {
  const items = [];
  if (mod.remove_constraints && mod.remove_constraints.length) {
    items.push('Removed constraint' + (mod.remove_constraints.length > 1 ? 's' : '') +
      ': ' + mod.remove_constraints.map(esc).join(', '));
  }
  (mod.add_constraints || []).forEach(c => {
    items.push('Added constraint <b>' + esc(c.name) + '</b> (' + esc(c.type) + ')');
  });
  (mod.gender_overrides || []).forEach(g => {
    items.push('Set user ' + g.user_id + " gender to <b>" + esc(g.gender) + '</b> for this scenario');
  });
  if (mod.set_slate_size) {
    items.push('Changed slate size to <b>' + mod.set_slate_size + '</b>');
  }
  if (mod.focal_users && mod.focal_users.length) {
    items.push('Focused the report on user' + (mod.focal_users.length > 1 ? 's' : '') +
      ' ' + mod.focal_users.join(', '));
  }
  if (!items.length) items.push('No change to the problem (a no-op).');
  $('modList').innerHTML = items.map(t => '<li>' + t + '</li>').join('');
}

function fmt(n) {
  if (n === null || n === undefined) return '–';
  return (Math.round(n * 1000) / 1000).toLocaleString();
}

function deltaSpan(base, modified, higherIsBetter) {
  if (base === 0 && modified === 0) return '';
  const diff = modified - base;
  if (Math.abs(diff) < 1e-9) return '<span class="delta flat">no change</span>';
  const pct = base !== 0 ? (100 * diff / base) : null;
  const dir = diff > 0 ? 'up' : 'down';
  const good = higherIsBetter === undefined ? 'flat' : ((diff > 0) === higherIsBetter ? 'up' : 'down');
  const sign = diff > 0 ? '+' : '';
  const pctTxt = pct !== null ? ' (' + sign + fmt(pct) + '%)' : '';
  return '<span class="delta ' + good + '">' + sign + fmt(diff) + pctTxt + '</span>';
}

function barRow(label, base, modified, higherIsBetter) {
  const max = Math.max(Math.abs(base), Math.abs(modified), 1e-9);
  const baseW = Math.max(2, 100 * Math.abs(base) / max);
  const modW = Math.max(2, 100 * Math.abs(modified) / max);
  return (
    '<div class="metricRow">' +
      '<div class="metricLabel"><span>' + esc(label) + '</span>' + deltaSpan(base, modified, higherIsBetter) + '</div>' +
      '<div class="barTrack"><div class="barFill base" style="width:' + baseW + '%"></div></div>' +
      '<div class="barTrack" style="margin-top:2px"><div class="barFill modified" style="width:' + modW + '%"></div></div>' +
      '<div class="barLegend">baseline ' + fmt(base) + ' &nbsp;·&nbsp; modified ' + fmt(modified) + '</div>' +
    '</div>'
  );
}

function distTable(title, base, modified, rows) {
  const trs = rows.map(([key, label]) =>
    '<tr><td>' + label + '</td><td>' + fmt(base[key]) + '</td><td>' + fmt(modified[key]) + '</td></tr>'
  ).join('');
  return (
    '<div class="metricRow"><div class="metricLabel"><span>' + esc(title) + '</span></div>' +
    '<table class="reportTable"><thead><tr><th></th><th>baseline</th><th>modified</th></tr></thead>' +
    '<tbody>' + trs + '</tbody></table></div>'
  );
}

function renderReport(report) {
  const metrics = $('reportMetrics');
  const tables = $('reportTables');
  metrics.innerHTML = '';
  tables.innerHTML = '';
  if (!report || !report.objective) {
    metrics.innerHTML = '<p class="sub" style="margin:0">No change to the problem — nothing to compare.</p>';
    $('focalArea').innerHTML = '';
    return;
  }

  metrics.innerHTML =
    barRow('Total predicted rating (objective)', report.objective.base, report.objective.modified, true) +
    barRow('Mean predicted rating per recommendation',
      report.mean_predicted_rating.base, report.mean_predicted_rating.modified, true) +
    barRow('Users whose slate changed', 0, report.users_with_changed_slate.count) +
    barRow('Cold items shown (of ' + report.cold_item_exposure.n_cold_items + ')',
      report.cold_item_exposure.base_items_shown, report.cold_item_exposure.modified_items_shown, true) +
    barRow('Item exposure concentration (Gini, 0=even)',
      report.item_exposure_concentration.gini.base, report.item_exposure_concentration.gini.modified, false) +
    barRow("Top-10% items' share of all exposure",
      report.item_exposure_concentration.top_10pct_items_exposure_share.base,
      report.item_exposure_concentration.top_10pct_items_exposure_share.modified, false);

  const distRows = [['min', 'min (worst-off)'], ['p25', 'p25'], ['median', 'median'],
                     ['p75', 'p75'], ['max', 'max'], ['std', 'std dev']];
  tables.innerHTML =
    distTable('Per-user slate rating quality', report.per_user_slate_rating_distribution.base,
      report.per_user_slate_rating_distribution.modified, distRows) +
    distTable('Exploration burden (cold items per user slate)',
      report.exploration_burden_distribution.base, report.exploration_burden_distribution.modified,
      distRows.concat([['gini', 'gini']]));

  const focal = report.focal_users;
  const focalArea = $('focalArea');
  if (!focal) { focalArea.innerHTML = ''; return; }
  focalArea.innerHTML = Object.entries(focal).map(([uid, d]) => {
    const p = d.profile;
    return (
      '<div class="focalCard">' +
      '<div class="metricLabel"><span>User ' + esc(uid) + ' — age ' + p.age + ', ' + esc(p.gender) +
      ', ' + esc(p.occupation) + '</span></div>' +
      '<p class="checkDetail">' + p.n_ratings + ' ratings given, average ' + fmt(p.mean_rating_given) +
      '. Slate rating: ' + fmt(d.mean_predicted_slate_rating.base) + ' → ' + fmt(d.mean_predicted_slate_rating.modified) + '</p>' +
      '<div class="focalCols">' +
        '<div><h3>kept (' + d.kept.length + ')</h3><ul>' + d.kept.map(t => '<li>' + esc(t) + '</li>').join('') + '</ul></div>' +
        '<div><h3>removed (' + d.removed.length + ')</h3><ul>' + d.removed.map(t => '<li>' + esc(t) + '</li>').join('') + '</ul></div>' +
        '<div><h3>added (' + d.added.length + ')</h3><ul>' + d.added.map(t => '<li>' + esc(t) + '</li>').join('') + '</ul></div>' +
      '</div></div>'
    );
  }).join('');
}

const BACKEND_DISPLAY_NAMES = { cli: 'Claude', api: 'Claude', gemini: 'Gemini' };

function renderCompare(runs) {
  const cols = $('compareCols');
  cols.innerHTML = Object.entries(runs || {}).map(([label, run]) =>
    '<div class="compareCol"><h3>' + esc(BACKEND_DISPLAY_NAMES[label] || label) + '</h3>' +
    '<div class="explanation">' + esc(run.explanation || '(no explanation)') + '</div>' +
    checkPanelHtml(run.check || {}) +
    '</div>'
  ).join('');
}

$('ask').addEventListener('click', async () => {
  const query = $('query').value.trim();
  if (!query) return;
  const compareMode = $('compare').checked && COMPARE_BACKENDS.length >= 2;
  $('ask').disabled = true;
  $('status').innerHTML = '<span class="spinner">&#9696;</span> interpreting, re-solving, comparing… (a minute or two' +
    (compareMode ? ', twice over' : '') + ')';
  $('status').className = '';
  $('out').style.display = 'none';
  try {
    const r = await fetch(compareMode ? '/ask-compare' : '/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, skip_explanation: $('skip').checked}),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    renderModification(data.modification || {});
    $('mod').textContent = JSON.stringify(data.modification, null, 2);
    renderReport(data.report || {});
    $('report').textContent = data.report_text || '(no change to the problem)';

    if (compareMode) {
      $('explCard').style.display = 'none';
      $('checkCard').style.display = 'none';
      $('compareCard').style.display = '';
      renderCompare(data.runs || {});
    } else {
      $('compareCard').style.display = 'none';
      $('explanation').textContent = data.explanation || '';
      $('explCard').style.display = data.explanation ? '' : 'none';
      $('checkCard').style.display = '';
      renderCheck(data.check || {});
    }

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
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/status":
            self._send_json({
                "status": STATE["status"],
                "compare_backends": list(STATE["compare_backends"] or {}),
            })
        elif self.path == "/queries":
            try:
                self._send_json(_load_queries_by_role())
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/ask":
            self._handle_ask()
        elif self.path == "/ask-compare":
            self._handle_ask_compare()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_ask(self) -> None:
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
                "report": result.report or {},
                "report_text": report_text(result.report) if result.report else "",
                "explanation": result.explanation or "",
                "check": _check_result(query, result),
            })
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _handle_ask_compare(self) -> None:
        """Same interpret+solve as /ask, but explain the one resulting
        report with two different models and score each independently --
        the interpreter runs once, so both explanations answer the exact
        same edit; only the explainer model varies."""
        if STATE["pipeline"] is None:
            self._send_json({"error": "pipeline not ready yet"}, 503)
            return
        if STATE["compare_backends"] is None:
            self._send_json({"error": "no second backend configured for comparison"}, 400)
            return
        try:
            from explainrec.compare import report_text
            from explainrec.llm.interpreter import interpret
            from explainrec.scenario import Modification

            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            query = str(payload["query"])
            pipeline = STATE["pipeline"]

            with STATE["lock"]:  # one solve at a time
                try:
                    mod = interpret(query, pipeline.baseline, STATE["backend"])
                except Exception as e:
                    self._send_json({"error": f"could not interpret the query: {e}"}, 200)
                    return

                if mod.is_noop() and not mod.focal_users:
                    report, detail = {}, mod.summary
                else:
                    try:
                        report = pipeline.run_modification(mod)
                        detail = report_text(report)
                    except Exception as e:
                        report = {"problem_not_solvable": str(e)}
                        detail = f"The requested change could not be solved: {e}"

                runs = {}
                for label, backend in STATE["compare_backends"].items():
                    try:
                        explanation = explain(query, mod, detail, backend) if report else detail
                    except Exception as e:
                        explanation = f"(explanation unavailable from {label}: {e})"
                    runs[label] = {
                        "explanation": explanation,
                        "check": _check_explanation(query, mod, report, explanation),
                    }

            self._send_json({
                "modification": mod.model_dump(exclude_defaults=True),
                "report": report,
                "report_text": report_text(report) if report else "",
                "runs": runs,
            })
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep the terminal quiet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["api", "cli", "gemini"], default="cli")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--compare-backends", nargs="+", default=None, metavar="NAME",
        choices=["api", "cli", "gemini"],
        help="explain with two or more backends side by side (e.g. --compare-backends cli gemini); "
             "enables the 'compare models' toggle in the demo",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    kwargs = {"model": args.llm_model} if args.llm_model else {}
    STATE["backend"] = get_backend(args.backend, **kwargs)
    if args.compare_backends:
        STATE["compare_backends"] = {name: get_backend(name) for name in args.compare_backends}
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
