# Metrics: Explanation Evaluation (working draft)

How we judge an explanation produced by the system. This is a living table, not a final
spec: the metrics are still being refined through our literature review, so expect rows
to change, split, or be dropped. The detailed, citation-backed synthesis is kept in the
local literature review; this table is the shared, evolving decision on what we will
actually measure.

Status legend: **candidate** (under consideration), **adopted** (committed for the next
experiment), **dropped** (considered and set aside, with a reason).

| Metric | What it captures | How we would measure | Basis | Status |
| --- | --- | --- | --- | --- |
| Faithfulness / correctness | The explanation reflects what the mechanism actually did, not a plausible fiction | Automatable: compare against the simulator's ground-truth decision path | Co-12 Correctness; Ichmoukhamedov et al. | candidate |
| Self-consistency | Same decision explained twice yields the same reasons | Automatable: repeat generations and compare | Co-12 Consistency / Continuity | candidate (brainstormed) |
| Counterfactual robustness | The "what would change the outcome" claims hold and are stable | Automatable: perturb inputs, re-check the counterfactual | Co-12 Continuity; CF validity | candidate (brainstormed) |
| Conciseness | Length matches the decision's complexity, no filler | Length heuristic plus human judgement | Co-12 Compactness (tension with completeness) | candidate (brainstormed) |
| Sufficiency and necessity | The reasons motivate the decision, and each cited reason is load-bearing | Human plus ablation (drop a reason, test if the case still holds) | Co-12 Completeness (sufficiency), Contrastivity (necessity) | candidate (brainstormed) |
| Relevance to the user | Pitched to the persona's background and priorities | Human study | Co-12 Context and Coherence | candidate (brainstormed) |
| Comprehension | The user can accurately restate why the decision was made | Human study | Martens et al. (XAIstories) | candidate |
| Trust / cognitive load | Downstream effect on user trust and effort | Human study | Martens et al.; XAI eval literature | candidate (needs experiments) |
| Convincingness | Whether the user finds the explanation persuasive | Human study | Martens et al. | dropped as a target (project forbids persuasion); keep only as a diagnostic, a rise without matching comprehension is a warning sign |

Sources for the basis column: Nauta et al. 2023 (Co-12, arXiv:2201.08164), Martens et al.
2025 (arXiv:2309.17057), Ichmoukhamedov et al. 2024 (arXiv:2412.10220). Co-12 groups
properties into content, presentation, and user aspects.
