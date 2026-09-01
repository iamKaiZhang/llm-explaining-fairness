# Measurement-query datasets

Evaluation datasets testing whether the LLM maps a stakeholder query to the
**correct measurement tool** (plus parameters / metric choice) and returns
the **correct value**. These are *measurement* queries about the current
solution — unlike `experiments/queries.yaml`, they are not counterfactual
reconstructions and cannot be expressed by the `Modification` schema; they
anticipate a measurement-tool schema as the next capability of the pipeline.

## Design (question-set pattern)

Each question set = a **template** with placeholders + an **executable gold
definition**. Placeholder values instantiate distinct questions; each
question can carry **paraphrases** whose gold answer is, by construction,
unchanged. Ground truth is *computed, never hand-typed*: every `answer`
comes from a function in `explainrec/metrics.py` evaluated on the baseline
scenario (cold-item exposure >= 5, slate size 10).

Question surface forms live in `question_forms_v1.json`, deliberately mixing
two styles: the **main form** is clean and literal (the canonical phrasing a
grader would quote), while the **paraphrases** carry the realistic variety —
LLM-generated persona rewrites plus curated creative seed forms that always
survive regeneration. Paraphrase carryover is keyed on (question id, gold
params), so editing a main form never discards the paraphrases.

## Format (`llm_eval_dataset/*.json`)

```jsonc
{
  "set_id": "coverage-disparity-v1",
  "definition": "...",                 // the concept being measured
  "template": "What fraction of {{ALPHA}}-similar item pairs ...",
  "expected_tool": "coverage_disparity",       // gold tool choice
  "implementation": "explainrec.metrics.coverage_disparity",
  "counterfactual": false,
  "context": { "scenario": "...", "solution_cache": "...", ... },
  "questions": [{
    "id": "cd-01",
    "params": {"alpha": 0.5, "epsilon": 0.05}, // gold parameters
    "expected_metric": "MAE",                  // only where a metric must be chosen
    "question": "...",                         // instantiated template
    "paraphrases": ["...", "..."],             // same gold answer
    "answer": 0.8873,                          // computed ground truth
    "tolerance": 0.01,                         // |value - answer| bound
    "answer_details": { ... }                  // provenance (counts, means)
  }]
}
```

Grade a system's answer with `explainrec.eval.measurement.grade_answer(set,
question_id, value, tool=..., metric=...)` — checks value (within tolerance),
tool choice, and metric choice (where required) separately.

## Question sets (v1)

| Set | Concept | Questions | Gold |
| --- | --- | --- | --- |
| `coverage_disparity` | Individual fairness over items: any two alpha-similar recommended items should receive similar coverage. Similarity = genre Jaccard; gap = relative exposure gap. | 10 (alpha x epsilon grid) + 40 paraphrases | fraction of alpha-similar recommended pairs with gap <= epsilon |
| `coverage` | Fraction of catalog items recommended at least once. | 1 + 4 paraphrases | 0.4958 (834/1682) |
| `benefit_inequality` | Generalized cross-entropy (beta = 0.5) of per-user benefit vs uniform; 0 = equal, more negative = more unequal. Half the questions name the metric (MAE/RMSE/nDCG/Precision/Recall), most describe it indirectly — metric choice is part of the test. | 10 + 40 paraphrases | GCE per metric |
| `group_coverage_gap` | Provider-side group fairness: within-group coverage (fraction of the group's items recommended at least once) vs the complement. | 10 (genre / era / cold groups) + 40 paraphrases | absolute gap; both fractions in details |
| `demographic_parity` | User-side group fairness: mean over a group's users of the proportion of their top-k recommendations predicted at >= stars; parity holds if the two groups' gap <= bound. | 10 (group pairs x k x stars x bound) + 40 paraphrases | gap (`answer`) + parity verdict (`answer_bool`) |
| `worst_group` | Worst-group / maximin fairness: per-group utility of the top-k recommendations (mean predicted rating, or fraction predicted >= 4); which group is worst served, and is it within the bound of the best? | 8 (group sets x k x utility x bound) + 32 paraphrases | worst group (`expected_choice`) + best-worst gap (`answer`) + verdict (`answer_bool`) |

Total: 49 questions, 196 paraphrases -> 245 test inputs.

For human review, `python datasets/export_review.py` renders everything to
`llm_eval_dataset_review.xlsx` (Overview sheet + one sheet per set, one row
per question with its paraphrases and gold side by side). Re-run after
regenerating the dataset.

## Caveats (v1)

- Per-user accuracy metrics are **in-sample** (no train/test split yet);
  ranking metrics use each user's observed items at k=10, relevance =
  rating >= 4. A future split would change the gold values, not the format.
- "Would rate at least S stars" for *recommended* items is defined on the
  **model's predicted ratings** (recommended items have no observed ratings).
- MovieLens 100k has no director-gender or primary-language metadata, so the
  provider-side groups use genre, release era (parsed from titles), and cold
  status instead of the literature's example groups.
- Top-k of a slate = the k slate items with the highest predicted rating.
- Ground truth depends on the baseline solution; if the baseline definition
  changes, regenerate (`python datasets/build_llm_eval_dataset.py
  [--paraphrase]`). Regeneration preserves existing paraphrases (matched by
  question id + gold params) and only asks the LLM for missing ones. The manifest
  records the solution cache key used.
