"""Generate datasets/llm_eval_dataset/ — measurement-query sets with computed
ground truth.

    .venv/bin/python datasets/build_llm_eval_dataset.py                # gold only
    .venv/bin/python datasets/build_llm_eval_dataset.py --paraphrase   # + LLM rewrites

Ground truth is computed, never hand-typed: every answer comes from the
functions in ``explainrec/metrics.py`` evaluated on the baseline scenario
(cold-item exposure >= 5, slate size 10), whose solution is loaded from the
pipeline's disk cache. Re-running the script reproduces the same numbers as
long as the baseline definition is unchanged.

``--paraphrase`` asks the local Claude CLI to rewrite each question a few
ways (the image-in-the-lab-notes pattern: gold stays fixed, surface form
varies). Paraphrases never change answers, so the step is optional and
re-runnable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from explainrec.metrics import (  # noqa: E402
    benefit_inequality, coverage, coverage_disparity, demographic_parity,
    group_coverage_gap, worst_group,
)
from explainrec.pipeline import Pipeline  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "llm_eval_dataset"
# Question surface forms: {id: {"question": main form, "seed_paraphrases":
# [...]}}. The main form is the clean canonical phrasing; seed paraphrases
# are curated creative variants that always survive regeneration.
FORMS_PATH = Path(__file__).resolve().parent / "question_forms_v1.json"
FORMS: dict = json.loads(FORMS_PATH.read_text())

# (alpha, epsilon) grid for the coverage-disparity set. Question surface
# forms live in question_forms_v1.json.
CD_PARAMS = [(a, e) for a in (0.5, 1.0) for e in (0.05, 0.1, 0.2, 0.3, 0.5)]

# --- set 4: provider-side item group specs (gc-01..gc-10). MovieLens 100k
# has no director-gender or language metadata, so genre / era / cold-status
# groups play the same structural role. Surface forms: question_forms_v1.json.
GC_GROUPS = [
    {"kind": "genre", "genre": "Drama"},
    {"kind": "genre", "genre": "Comedy"},
    {"kind": "genre", "genre": "Horror"},
    {"kind": "genre", "genre": "Documentary"},
    {"kind": "genre", "genre": "Children's"},
    {"kind": "genre", "genre": "Action"},
    {"kind": "genre", "genre": "Musical"},
    {"kind": "era", "before": 1980},
    {"kind": "era", "before": 1990},
    {"kind": "cold"},
]

# user-group vocabulary: label -> spec
USER_GROUPS = {
    "male users": {"attribute": "gender", "value": "M"},
    "female users": {"attribute": "gender", "value": "F"},
    "users under 25": {"attribute": "age", "min": 0, "max": 24},
    "users 25-34": {"attribute": "age", "min": 25, "max": 34},
    "users 35-44": {"attribute": "age", "min": 35, "max": 44},
    "users 45 and older": {"attribute": "age", "min": 45, "max": 200},
    "users under 30": {"attribute": "age", "min": 0, "max": 29},
    "users 30 and older": {"attribute": "age", "min": 30, "max": 200},
    "students": {"attribute": "occupation", "value": "student"},
    "engineers": {"attribute": "occupation", "value": "engineer"},
    "educators": {"attribute": "occupation", "value": "educator"},
    "programmers": {"attribute": "occupation", "value": "programmer"},
    "writers": {"attribute": "occupation", "value": "writer"},
    "administrators": {"attribute": "occupation", "value": "administrator"},
    "artists": {"attribute": "occupation", "value": "artist"},
    "retired users": {"attribute": "occupation", "value": "retired"},
}

# --- set 5: (group_x, group_y, k, stars, bound); forms in question_forms_v1.json
DP_PARAMS = [
    ("male users", "female users", 10, 4, 0.05),
    ("male users", "female users", 10, 4, 0.01),
    ("male users", "female users", 5, 4, 0.05),
    ("male users", "female users", 10, 3, 0.02),
    ("female users", "male users", 10, 5, 0.01),
    ("users under 30", "users 30 and older", 10, 4, 0.05),
    ("users under 25", "users 45 and older", 10, 4, 0.05),
    ("students", "engineers", 10, 4, 0.05),
    ("students", "retired users", 10, 4, 0.1),
    ("educators", "programmers", 10, 4, 0.05),
]

# --- set 6: (group labels, k, utility, bound)
AGE_BANDS = ["users under 25", "users 25-34", "users 35-44", "users 45 and older"]
OCC_A = ["students", "engineers", "educators", "programmers", "writers"]
OCC_B = ["administrators", "artists", "retired users", "writers", "students"]
# (groups, k, utility, bound); forms in question_forms_v1.json
WG_PARAMS = [
    (["male users", "female users"], 10, "mean_predicted_rating", 0.05),
    (["male users", "female users"], 10, "fraction_predicted_at_least_4", 0.05),
    (["male users", "female users"], 5, "mean_predicted_rating", 0.05),
    (AGE_BANDS, 10, "mean_predicted_rating", 0.1),
    (AGE_BANDS, 10, "fraction_predicted_at_least_4", 0.1),
    (AGE_BANDS, 5, "mean_predicted_rating", 0.1),
    (OCC_A, 10, "mean_predicted_rating", 0.1),
    (OCC_B, 10, "fraction_predicted_at_least_4", 0.1),
]

# expected metric per question id bi-01..bi-10; the question surface forms
# (mostly not naming the metric) live in question_forms_v1.json.
BI_METRICS = ["MAE", "MAE", "RMSE", "RMSE", "nDCG", "nDCG",
              "Precision", "Precision", "Recall", "Recall"]


def build(paraphrase: bool) -> None:
    pipeline = Pipeline.build()
    solution = pipeline.base_solution
    data = pipeline.baseline.data
    model = pipeline.baseline.model

    context = {
        "scenario": "baseline: maximize total predicted rating; every cold "
                    "item (<=20 ratings) reaches >=5 users; slate size 10",
        "solution_cache": pipeline._cache_path(pipeline.baseline).name,
        "n_users": data.n_users,
        "n_items": data.n_items,
    }

    # --- set 1: coverage disparity ------------------------------------
    cd_questions = []
    for n, (alpha, epsilon) in enumerate(CD_PARAMS, 1):
        res = coverage_disparity(solution, data, alpha, epsilon)
        cd_questions.append({
            "id": f"cd-{n:02d}",
            "params": {"alpha": alpha, "epsilon": epsilon},
            "question": FORMS[f"cd-{n:02d}"]["question"],
            "paraphrases": [],
            "answer": round(res["fraction"], 4),
            "tolerance": 0.01,
            "answer_details": {k: v for k, v in res.items() if k != "fraction"},
        })
    cd_set = {
        "set_id": "coverage-disparity-v1",
        "definition": (
            "Violation of coverage disparity, on the individual-fairness "
            "reading: any two alpha-similar recommended items should receive "
            "similar coverage. Measured as the fraction of alpha-similar "
            "recommended item pairs whose relative exposure gap "
            "|e_i - e_j| / max(e_i, e_j) is at most epsilon. Similarity is "
            "genre Jaccard; 'recommended' means exposure >= 1 user."
        ),
        "template": ("What fraction of {{ALPHA}}-similar item pairs have at "
                     "most {{EPSILON}} gaps in exposure?"),
        "expected_tool": "coverage_disparity",
        "implementation": "explainrec.metrics.coverage_disparity",
        "counterfactual": False,
        "context": context,
        "questions": cd_questions,
    }

    # --- set 2: coverage ------------------------------------------------
    cov = coverage(solution, data)
    cov_set = {
        "set_id": "coverage-v1",
        "definition": "Coverage: fraction of catalog items recommended to at "
                      "least one user in the current solution.",
        "template": "What fraction of items are being recommended at least once?",
        "expected_tool": "coverage",
        "implementation": "explainrec.metrics.coverage",
        "counterfactual": False,
        "context": context,
        "questions": [{
            "id": "cov-01",
            "params": {},
            "question": FORMS["cov-01"]["question"],
            "paraphrases": [],
            "answer": round(cov["fraction"], 4),
            "tolerance": 0.01,
            "answer_details": {k: v for k, v in cov.items() if k != "fraction"},
        }],
    }

    # --- set 3: benefit inequality (GCE) --------------------------------
    bi_questions = []
    for n, metric in enumerate(BI_METRICS, 1):
        res = benefit_inequality(model, data, metric)
        bi_questions.append({
            "id": f"bi-{n:02d}",
            "question": FORMS[f"bi-{n:02d}"]["question"],
            "params": {"metric": metric},
            "expected_metric": metric,
            "paraphrases": [],
            "answer": round(res["gce"], 5),
            "tolerance": max(0.002, abs(res["gce"]) * 0.05),
            "answer_details": {k: v for k, v in res.items() if k != "gce"},
        })
    bi_set = {
        "set_id": "benefit-inequality-v1",
        "definition": (
            "Generalized cross-entropy (GCE, Deldjoo et al. 2019): system- "
            "level inequality in benefit across consumers with respect to a "
            "per-user accuracy metric. beta = 0.5, fair distribution = "
            "uniform; 0 = perfectly equal, more negative = more unequal. "
            "Per-user metrics are in-sample (no train/test split yet): "
            "MAE/RMSE over each user's observed ratings; nDCG/Precision/"
            "Recall at k=10 over each user's observed items ranked by "
            "predicted rating, relevance = rating >= 4."
        ),
        "template": ("How much inequality is there across users with respect "
                     "to {{USER_METRIC}}?"),
        "expected_tool": "benefit_inequality",
        "implementation": "explainrec.metrics.benefit_inequality",
        "counterfactual": False,
        "context": context,
        "questions": bi_questions,
    }

    r_hat = model.predict_matrix()

    # --- set 4: per-group utility gaps (provider side) -------------------
    gc_questions = []
    for n, spec in enumerate(GC_GROUPS, 1):
        res = group_coverage_gap(solution, data, spec)
        gc_questions.append({
            "id": f"gc-{n:02d}",
            "question": FORMS[f"gc-{n:02d}"]["question"],
            "params": {"group": spec},
            "paraphrases": [],
            "answer": round(res["gap"], 4),
            "tolerance": 0.01,
            "answer_details": {k: v for k, v in res.items() if k != "gap"},
        })
    gc_set = {
        "set_id": "group-coverage-gap-v1",
        "definition": (
            "Per-group utility gaps, provider side: within-group coverage is "
            "the fraction of a group's items recommended to at least one "
            "user; the answer is the absolute gap between the group and its "
            "complement. MovieLens 100k carries no director-gender or "
            "language metadata, so genre, release-era, and cold-status "
            "groups stand in for the paper's example groups."
        ),
        "template": ("Do movies {{GROUP}} have different fractions of items "
                     "within that group being recommended?"),
        "expected_tool": "group_coverage_gap",
        "implementation": "explainrec.metrics.group_coverage_gap",
        "counterfactual": False,
        "context": context,
        "questions": gc_questions,
    }

    # --- set 5: demographic parity (user side) ---------------------------
    dp_questions = []
    for n, (gx, gy, k, stars, bound) in enumerate(DP_PARAMS, 1):
        res = demographic_parity(solution, r_hat, data,
                                 USER_GROUPS[gx], USER_GROUPS[gy],
                                 k=k, stars=stars, bound=bound)
        dp_questions.append({
            "id": f"dp-{n:02d}",
            "question": FORMS[f"dp-{n:02d}"]["question"],
            "params": {"group_x": USER_GROUPS[gx], "group_y": USER_GROUPS[gy],
                       "k": k, "stars": stars, "bound": bound},
            "paraphrases": [],
            "answer": round(res["gap"], 4),
            "answer_bool": res["parity_holds"],
            "tolerance": 0.005,
            "answer_details": {k_: v for k_, v in res.items()
                               if k_ not in ("gap", "parity_holds")},
        })
    dp_set = {
        "set_id": "demographic-parity-v1",
        "definition": (
            "Demographic parity, user side: for each user, the proportion of "
            "their top-k recommendations with predicted rating >= stars "
            "('would rate' is defined on the model's predictions, since "
            "recommended items have no observed ratings); group value = mean "
            "over the group's users; parity holds if the absolute gap "
            "between the two groups is at most the bound. Top-k = the k "
            "slate items with the highest predicted rating."
        ),
        "template": ("Do users {{GROUP-X}} and users {{GROUP-Y}} get the same "
                     "proportion of their {{K}} recommendations that they "
                     "would rate at least {{STARS}} stars, allowing a "
                     "difference of at most {{BOUND}}?"),
        "expected_tool": "demographic_parity",
        "implementation": "explainrec.metrics.demographic_parity",
        "counterfactual": False,
        "context": context,
        "questions": dp_questions,
    }

    # --- set 6: worst-group / maximin fairness ---------------------------
    wg_questions = []
    for n, (labels, k, utility, bound) in enumerate(WG_PARAMS, 1):
        res = worst_group(solution, r_hat, data,
                          [(lb, USER_GROUPS[lb]) for lb in labels],
                          k=k, utility=utility, bound=bound)
        wg_questions.append({
            "id": f"wg-{n:02d}",
            "question": FORMS[f"wg-{n:02d}"]["question"],
            "params": {"groups": labels, "k": k, "utility": utility,
                       "bound": bound},
            "paraphrases": [],
            "expected_choice": res["worst_group"],
            "answer": round(res["gap"], 4),
            "answer_bool": res["within_bound"],
            "tolerance": 0.02 if utility == "mean_predicted_rating" else 0.005,
            "answer_details": {k_: v for k_, v in res.items()
                               if k_ not in ("gap", "worst_group", "within_bound")},
        })
    wg_set = {
        "set_id": "worst-group-v1",
        "definition": (
            "Worst-group / maximin fairness: per-group utility of the top-k "
            "recommendations (mean predicted rating, or the fraction of "
            "recommendations predicted at >= 4 stars); the answer names the "
            "worst-served group and reports the best-worst gap and whether "
            "it is within the bound."
        ),
        "template": ("Among {{GROUPS}}, which user group gets the least "
                     "enjoyable {{K}} recommendations according to "
                     "{{UTILITY}}, and is its utility within {{BOUND}} of "
                     "the best-served group?"),
        "expected_tool": "worst_group",
        "implementation": "explainrec.metrics.worst_group",
        "counterfactual": False,
        "context": context,
        "questions": wg_questions,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = {
        "coverage_disparity.json": cd_set,
        "coverage.json": cov_set,
        "benefit_inequality.json": bi_set,
        "group_coverage_gap.json": gc_set,
        "demographic_parity.json": dp_set,
        "worst_group.json": wg_set,
    }
    # carry over paraphrases from an existing dataset (matched by question id
    # + identical gold params — paraphrase validity is tied to the params, not
    # to the main form's wording), then make sure every curated seed
    # paraphrase from question_forms_v1.json is present. Regeneration is
    # therefore stable and --paraphrase only fills the gaps.
    for name, content in sets.items():
        path = OUT_DIR / name
        old = {}
        if path.exists():
            old = {q["id"]: q for q in json.loads(path.read_text())["questions"]}
        for q in content["questions"]:
            prev = old.get(q["id"])
            if prev and json.dumps(prev["params"], sort_keys=True) == \
                    json.dumps(q["params"], sort_keys=True):
                q["paraphrases"] = [p for p in prev.get("paraphrases", [])
                                    if p != q["question"]]
            for seed in FORMS[q["id"]].get("seed_paraphrases", []):
                if seed not in q["paraphrases"] and seed != q["question"]:
                    q["paraphrases"].append(seed)
    if paraphrase:
        for s in sets.values():
            if any(not q["paraphrases"] for q in s["questions"]):
                add_paraphrases(s)
    for name, content in sets.items():
        (OUT_DIR / name).write_text(json.dumps(content, indent=2) + "\n")
        print(f"wrote {OUT_DIR / name}  ({len(content['questions'])} questions)")

    manifest = {
        "dataset": "llm_eval_dataset",
        "purpose": "Evaluate whether the LLM maps a stakeholder query to the "
                   "correct measurement tool (+ parameters/metric) and "
                   "returns the correct value.",
        "sets": [s["set_id"] for s in sets.values()],
        "grading": "explainrec.eval.measurement.grade_answer",
        "context": context,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_DIR / 'manifest.json'}")


def add_paraphrases(question_set: dict, n: int = 3) -> None:
    """Ask the local Claude CLI for n paraphrases per question (gold fixed)."""
    from explainrec.llm.backend import CliBackend, extract_json

    backend = CliBackend(model="opus")
    numbered = {q["id"]: q["question"] for q in question_set["questions"]
                if not q["paraphrases"]}
    system = (
        "You rewrite evaluation questions for a recommender-system study. "
        "For each question, produce paraphrases that real, different "
        "stakeholders might type: vary the persona (end user, content "
        "provider, product manager, auditor), the tone (casual, formal, "
        "annoyed, curious), and the sentence structure - do not just "
        "shuffle words. Hard constraints: the meaning must stay identical "
        "(same quantities, same implied metric, same groups); every "
        "parameter must remain unambiguously recoverable from the text "
        "(numbers may switch between percent and decimal form, e.g. 5% vs "
        "0.05, but must not be dropped or changed); if the original "
        "describes a metric indirectly, the paraphrases must keep implying "
        "that same metric without ever naming it. Reply with a single JSON "
        "object mapping each question id to a list of paraphrase strings, "
        "and nothing else."
    )
    user = (f"Produce {n} paraphrases for each of these questions:\n"
            f"{json.dumps(numbered, indent=2)}")
    raw = backend.text(system, user)
    mapping = json.loads(extract_json(raw))
    for q in question_set["questions"]:
        q["paraphrases"] = list(mapping.get(q["id"], []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphrase", action="store_true",
                        help="add LLM-rewritten paraphrases via the local Claude CLI")
    args = parser.parse_args()
    build(paraphrase=args.paraphrase)
