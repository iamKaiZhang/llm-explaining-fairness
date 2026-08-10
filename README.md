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

# skip the explanation call and print the raw comparison report
.venv/bin/python -m explainrec ask --no-explain "Would user 12 get the same movies if they were female?"
```

Solving one scenario takes ~40 s (a 943 x 1682 variable LP); an `ask` from the
CLI solves both the baseline and the modified problem. For interactive work,
build the pipeline once and reuse it:

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

## Repository layout

| Path | Purpose |
| --- | --- |
| `explainrec/` | The simulation + LLM pipeline (see architecture doc) |
| `tests/` | Pytest suite (small synthetic instances, no network/LLM) |
| `docs/architecture.md` | Formulation, module map, extension guide |
| `data/` | MovieLens 100k, downloaded on first run (git-ignored) |
| `archive/` | Earlier persona-based case-study notes (superseded) |

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

- Do not commit datasets or any human-subjects data.
- Put secrets in a local `.env` (git-ignored). Never commit API keys.
