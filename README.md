# LLM Explanation

Can LLMs explain algorithmic resource-allocation decisions to people with different
backgrounds and priorities: counterfactuals, prioritized metrics, and accessible yet
correct explanations?

This repo contains `explainrec`, a MovieLens 100k simulation of a recommender
system whose recommendations come from a constrained optimization problem, plus
an LLM layer that answers natural-language what-if questions ("What if we stop
recommending cold items?", "Would I get the same movies if I were male?") by
editing the optimization problem, re-solving it, and explaining the comparison.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env         # add your ANTHROPIC_API_KEY (only needed for `ask`)

# solve the baseline problem (downloads MovieLens 100k on first run, ~5 MB)
.venv/bin/python -m explainrec baseline

# full LLM loop: interpret -> re-solve -> compare -> explain
.venv/bin/python -m explainrec ask "What happens if we stop promoting cold items?"

# same, but through the local Claude Code CLI (subscription; no API key needed)
.venv/bin/python -m explainrec ask --backend cli "What happens if we stop promoting cold items?"

# skip the explanation call and print the raw comparison report
.venv/bin/python -m explainrec ask --no-explain "Would user 12 get the same movies if they were female?"
```

Two LLM backends (`explainrec/llm/backend.py`):

| Backend | Auth | Schema enforcement |
| --- | --- | --- |
| `api` (default) | `ANTHROPIC_API_KEY` in `.env` | server-side structured outputs (`messages.parse`) |
| `cli` | local `claude` login (subscription) | prompt + Pydantic validation, one retry |

`--llm-model` overrides the model (API id like `claude-opus-5`, or a CLI alias
like `opus`/`sonnet`).

Solving one scenario takes ~40 s (a 943 x 1682 variable LP). Every solved
scenario — the baseline and each modified problem — is cached in
`data/cache/` (~13 MB per solution), keyed by the full problem definition,
so re-asking a question whose modification was solved before is instant and
only genuinely new problems pay the solve. Changing constraints, slate size,
or model hyperparameters yields a new key; delete the folder to force
re-solves. For interactive work, build the pipeline once and reuse it:

```python
from explainrec.pipeline import Pipeline
p = Pipeline.build()          # load data, fit rating model
p.base_solution               # solved once, then cached
r = p.ask("What if each cold movie only needed to reach 2 users?")
print(r.explanation)
```

## How it works

1. **Rating estimation** (`ratings.py`) — biases + an explicit gender-item
   pathway + ALS matrix factorization predict `r_hat[u, i]`.
2. **Allocation LP** (`problem.py`, `constraints.py`) — maximize total predicted
   rating subject to one slate per user and declarative fairness constraints
   (baseline: every cold item reaches at least 5 users).
3. **LLM interpreter** (`llm/interpreter.py`) — turns a stakeholder question
   into a structured `Modification` (add/remove constraints, gender
   counterfactuals) via structured outputs; invalid edits are rejected by
   schema validation before anything is solved.
4. **Compare + explain** (`compare.py`, `llm/explainer.py`) — both solutions
   are compared quantitatively; the explainer LLM only sees that report, so
   explanations stay grounded in solver output.

See [docs/architecture.md](docs/architecture.md) for the formulation, the
modification schema, and how to add constraint types. Research-level notes
live in the Obsidian vault (`05_Projects/LLM Explanation/`).

## Demos

- **Notebook**: `demo/notebooks/demo.ipynb` — executed walkthrough: baseline,
  the two flagship what-if queries via the CLI backend, and how to inject a
  `Modification` directly without the LLM.
- **Web demo**: `python demo/server.py` — starts a local page at
  `http://localhost:8765` where you type what-if questions. The server keeps
  the pipeline warm (baseline solved once at startup) and bridges the browser
  to the local Claude CLI; use `--backend api` for the API instead. Stdlib
  only, no extra dependencies.

## Repository layout

| Path | Purpose |
| --- | --- |
| `explainrec/` | The simulation + LLM pipeline (see architecture doc) |
| `tests/` | Pytest suite (small synthetic instances, no network/LLM) |
| `demo/notebooks/demo.ipynb` | Executed demo walkthrough |
| `demo/server.py` | Local web demo (stdlib HTTP server + single page) |
| `docs/architecture.md` | Formulation, module map, extension guide |
| `data/` | MovieLens 100k, downloaded on first run (git-ignored) |
| `archive/` | Earlier persona-based case-study notes (superseded) |

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

- Do not commit datasets or any human-subjects data.
- Put secrets in a local `.env` (git-ignored). Never commit API keys.
