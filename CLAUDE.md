# llm-explaining-fairness — Agent Instructions

MovieLens 100k recommender simulation: constrained-optimization allocation +
LLM layer that answers what-if queries by editing the problem, re-solving, and
explaining the comparison. Read `docs/architecture.md` before changing the
optimization or the modification schema.

## Environment

- Python venv at `.venv/` (Python 3.14). Run everything through `.venv/bin/python`.
- Tests: `.venv/bin/python -m pytest tests/ -q` — fast, no network, no LLM.
- Two LLM backends (`explainrec/llm/backend.py`): `api` (Anthropic SDK,
  `ANTHROPIC_API_KEY` in git-ignored `.env`, model `claude-opus-5`) and `cli`
  (local `claude -p`, subscription auth, no key). The CLI backend strips
  `ANTHROPIC_*`/`CLAUDE*` vars from the subprocess env on purpose — keep that,
  it prevents API-key precedence and nested-session conflicts.

## Conventions

- Constraint types are declarative `ConstraintSpec`s compiled in
  `constraints.py`; follow the 4-step recipe in the architecture doc when
  adding one, and keep the interpreter system prompt in sync.
- The LLM must only interact with the solver through the `Modification`
  schema, and the explainer must only see the comparison report. Do not pass
  raw data or solver objects into prompts.
- A full scenario solve takes ~40 s; don't add tests that solve the real-size
  problem, use the tiny fixture in `tests/conftest.py`.
- `data/` is downloaded, git-ignored; never commit datasets.
- `archive/` holds the superseded persona-based case-study notes; don't edit.
