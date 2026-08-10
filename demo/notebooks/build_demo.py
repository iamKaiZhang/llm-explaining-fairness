"""Generate demo.ipynb next to this script (run once; the notebook is the artifact)."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md(
    "# explainrec demo\n"
    "\n"
    "End-to-end walkthrough of the LLM-explained recommender:\n"
    "estimate ratings -> solve the allocation LP -> ask natural-language\n"
    "what-if questions -> the LLM edits the problem, we re-solve, compare,\n"
    "and the LLM explains the comparison.\n"
    "\n"
    "LLM calls go through the **local Claude Code CLI** (subscription auth,\n"
    "no API key). Switch to `ApiBackend()` if you prefer the API.\n"
))

cells.append(code(
    "# make the repo root importable regardless of where Jupyter was launched\n"
    "import sys\n"
    "from pathlib import Path\n"
    "try:\n"
    "    import explainrec  # noqa: F401\n"
    "except ModuleNotFoundError:\n"
    "    root = next(p for p in Path.cwd().resolve().parents\n"
    "                if (p / \"explainrec\").is_dir())\n"
    "    sys.path.insert(0, str(root))\n"
    "\n"
    "from explainrec.pipeline import Pipeline\n"
    "from explainrec.llm.backend import CliBackend\n"
    "from explainrec.compare import report_text\n"
    "\n"
    "backend = CliBackend(model=\"opus\")   # or ApiBackend() with ANTHROPIC_API_KEY\n"
    "p = Pipeline.build()                 # downloads ML-100k on first run, fits the rating model\n"
    "data = p.baseline.data\n"
    "print(f\"train RMSE: {p.baseline.model.train_rmse:.3f}\")\n"
    "print(data.summary())"
))

cells.append(md(
    "## Baseline\n"
    "Total-rating maximization with one fairness constraint: every cold item\n"
    "must reach at least 5 users. Solving the full LP takes ~40 s; the solution\n"
    "is cached on the pipeline afterwards.\n"
))

cells.append(code(
    "sol = p.base_solution\n"
    "cold = data.cold_items\n"
    "print(f\"objective: {sol.objective:.1f}   mean predicted rating: {sol.mean_predicted_rating:.3f}\")\n"
    "print(f\"cold items shown: {int((sol.exposure[cold] > 1e-6).sum())}/{len(cold)}\")\n"
    "print(f\"solve time: {sol.solve_seconds:.1f}s   fractional entries: {sol.n_fractional}\")\n"
    "print(\"\\nslate of user 42:\")\n"
    "for i in sol.recs[42]:\n"
    "    print(f\"  {data.title(i)}\")"
))

cells.append(md(
    "## What-if 1: stop exploring\n"
    "A content-strategy question: what does the cold-start promotion cost us?\n"
))

cells.append(code(
    "r1 = p.ask(\"What happens if we stop promoting cold items?\", backend=backend)\n"
    "print(\"modification:\", r1.modification.summary)\n"
    "print()\n"
    "print(report_text(r1.report))"
))

cells.append(code("print(r1.explanation)"))

cells.append(md(
    "## What-if 2: individual gender counterfactual\n"
    "A user-facing recourse question. Note the *non-locality*: because the\n"
    "exposure constraints couple users, flipping one user's gender also\n"
    "perturbs other users' slates.\n"
))

cells.append(code(
    "r2 = p.ask(\"Would user 42 get the same movies if she were male?\", backend=backend)\n"
    "print(\"modification:\", r2.modification.summary)\n"
    "print()\n"
    "print(report_text(r2.report))"
))

cells.append(code("print(r2.explanation)"))

cells.append(md(
    "## Bypassing the LLM\n"
    "For experiments, inject a `Modification` directly - the deterministic\n"
    "core (`run_modification`) is separate from the LLM layer.\n"
))

cells.append(code(
    "from explainrec.scenario import Modification\n"
    "from explainrec.constraints import ConstraintSpec, ItemSelector\n"
    "\n"
    "mod = Modification(\n"
    "    summary=\"relax cold exposure from 5 to 2 users per item\",\n"
    "    remove_constraints=[\"cold-item-exposure\"],\n"
    "    add_constraints=[ConstraintSpec(\n"
    "        name=\"cold-item-exposure-relaxed\", type=\"min_item_exposure\",\n"
    "        items=ItemSelector(kind=\"cold\"), min_users=2,\n"
    "    )],\n"
    ")\n"
    "print(report_text(p.run_modification(mod)))"
))

cells.append(md(
    "Further reading: `README.md` (usage), `docs/architecture.md`\n"
    "(formulation, modification schema, extension guide), and the web demo\n"
    "(`python demo/server.py`).\n"
))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3",
}

out_path = Path(__file__).resolve().parent / "demo.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}")
