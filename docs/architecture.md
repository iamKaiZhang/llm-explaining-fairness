# `explainrec` architecture

## Problem

The simulated platform recommends a slate of $k$ movies to each of the $U = 943$
MovieLens users out of $I = 1682$ movies. Recommendations solve

$$\max_{x} \; \sum_{u,i} \hat r_{u,i} \, x_{u,i}
\quad \text{s.t.} \quad \sum_i x_{u,i} = k \;\; \forall u, \qquad 0 \le x_{u,i} \le 1,$$

plus a set of named, declarative constraints. $x_{u,i}$ is the LP relaxation of
the recommend/don't-recommend indicator. The base constraint matrix is the
incidence structure of a bipartite graph and hence totally unimodular, so with
integral bounds the solver (HiGHS via cvxpy) returns an integral vertex; added
constraints can break this, which `Solution.n_fractional` reports.

The baseline scenario carries one fairness constraint: every cold item (at most
20 ratings; 755 items) must be recommended to at least 5 users. Default slate
size is $k = 10$ (with $k = 1$ there are only 943 slots, so this constraint
would be infeasible).

## Rating model (`ratings.py`)

$$\hat r_{u,i} = \mu + b_u + b_i + \delta_{g(u), i} + p_u^\top q_i,$$

fit in stages: shrunk biases, then a shrunk gender-item effect
$\delta_{g,i}$ on the bias residuals, then rank-20 ALS matrix factorization on
the remainder. Train RMSE $\approx 0.83$.

The explicit $\delta$ pathway is what makes attribute counterfactuals well
defined: "if user $u$ were female" swaps $\delta_{M,i} \to \delta_{F,i}$ for
that user and re-solves. Caveat: $b_u$ and $p_u$ are learned from the user's
actual history, which may itself correlate with gender; the counterfactual
intervenes on the explicit demographic pathway only.

## Modules

```
explainrec/
├── data.py          MovieLens 100k download/load; Dataset context (cold items, genres)
├── ratings.py       RatingModel: biases + gender pathway + ALS MF
├── constraints.py   ConstraintSpec (declarative, JSON) -> cvxpy constraints
├── problem.py       solve_allocation: build LP, solve, package Solution
├── scenario.py      Scenario (immutable problem instance) + Modification schema
├── compare.py       quantitative comparison report between two Solutions
├── pipeline.py      Pipeline: load -> fit -> baseline; ask() and run_modification()
├── __main__.py      CLI: `baseline`, `ask`
└── llm/
    ├── interpreter.py   NL query -> Modification (structured outputs, claude-opus-5)
    └── explainer.py     comparison report -> NL explanation
```

## The what-if loop

```
query ──interpret──▶ Modification ──apply──▶ Scenario' ──solve──▶ Solution'
 (LLM, structured)      │                                             │
                        ▼                                             ▼
                  explanation ◀──explain(LLM)── report ◀──compare── (baseline Solution)
```

Two design decisions carry the faithfulness argument:

1. **The LLM never touches the solver.** It emits a `Modification` — a
   validated Pydantic object (`add_constraints`, `remove_constraints`,
   `gender_overrides`, `set_slate_size`, `focal_users`). Anything outside the
   schema is rejected at parse time; anything inconsistent (unknown constraint
   name, infeasible exposure demand) is rejected by the deterministic layer
   before solving.
2. **The explainer only sees the comparison report.** It cannot fabricate
   numbers that the solver did not produce; hallucination is limited to
   misreading the report, which is auditable.

## Modification schema (what the LLM can express)

| Field | Meaning |
| --- | --- |
| `add_constraints` | new `ConstraintSpec`s: `min_item_exposure`, `max_item_exposure`, `forbid_items`, `force_assign`, over item selectors `cold` / `ids` / `genre` / `popular` / `all` |
| `remove_constraints` | names of active constraints to drop (e.g. `cold-item-exposure` for "no exploration") |
| `gender_overrides` | per-user attribute counterfactuals (re-estimates $\hat r$) |
| `set_slate_size` | change $k$ |
| `focal_users` | users the query is about; the report includes their profile (recorded age/gender/occupation, rating history) and slate diff (kept/removed/added titles, cold items labeled `[cold]`, per-slate cold counts, predicted slate rating) |

The comparison report also carries distributional fairness metrics computed
per solve: the per-user slate-rating distribution (min/quartiles/max/std; min
is the worst-off user), the per-user exploration-burden distribution (cold
items per slate, with Gini), and item-exposure concentration (Gini and the
top-10% exposure share). Adding a per-user metric means computing it in
`solve_allocation` (it usually needs $\hat r$, which differs per scenario),
storing it on `Solution`, summarizing it in `compare.py`, and bumping
`solution_version` in the pipeline cache key.

## Adding a constraint type

1. Add the literal to `ConstraintSpec.type` and any new parameters as optional
   fields; extend the `_check_params` table.
2. Add a branch in `build_constraint` returning cvxpy constraints over `X`.
3. Add a line to `ConstraintSpec.describe` (the interpreter prompt lists active
   constraints through it) and mention the new type in the interpreter's
   system prompt if the LLM should use it.
4. Test it on the tiny fixture in `tests/`.

## Known limitations / next steps

- One scenario solve is ~40 s; the CLI re-solves the baseline on every
  invocation. Cache the fitted model and baseline solution to disk if this
  becomes annoying.
- Counterfactuals cover gender only; age/occupation pathways would follow the
  same pattern in `RatingModel`.
- The objective is fixed (total predicted rating). Objective modifications
  (e.g. welfare weights) would extend `Modification` and `solve_allocation`.
- No evaluation of explanation quality yet - that is what the metrics notes in
  the vault (`LLME - Metrics`) are for.
