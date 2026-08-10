"""CLI entry point.

    python -m explainrec baseline
    python -m explainrec ask "What if we stop recommending cold items?"
    python -m explainrec ask --no-explain "..."   # interpret + solve only
"""

from __future__ import annotations

import argparse
import json

from .compare import report_text
from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="explainrec")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("baseline", help="solve and describe the baseline problem")
    p_base.add_argument("--slate-size", type=int, default=10)

    p_ask = sub.add_parser("ask", help="answer a what-if query via the LLM")
    p_ask.add_argument("query")
    p_ask.add_argument("--slate-size", type=int, default=10)
    p_ask.add_argument("--no-explain", action="store_true",
                       help="skip the explanation call; print the raw report")

    args = parser.parse_args()
    pipeline = Pipeline.build(slate_size=args.slate_size)

    if args.command == "baseline":
        sol = pipeline.base_solution
        data = pipeline.baseline.data
        print(f"train RMSE: {pipeline.baseline.model.train_rmse:.3f}")
        print(f"status: {sol.status}  objective: {sol.objective:.1f}  "
              f"solve: {sol.solve_seconds:.1f}s  fractional entries: {sol.n_fractional}")
        print(f"mean predicted rating of recommendations: {sol.mean_predicted_rating:.3f}")
        cold = data.cold_items
        print(f"cold items shown: {int((sol.exposure[cold] > 1e-6).sum())}/{len(cold)}"
              f"  (total cold exposure {sol.exposure[cold].sum():.0f})")
        print("\nactive constraints:")
        print(pipeline.baseline.describe_constraints())
        return

    result = pipeline.ask(args.query, skip_explanation=args.no_explain)
    print(f"modification: {result.modification.summary}")
    print(json.dumps(result.modification.model_dump(exclude_defaults=True), indent=2))
    if result.report:
        print("\ncomparison report:")
        print(report_text(result.report))
    if result.explanation:
        print("\nexplanation:")
        print(result.explanation)


if __name__ == "__main__":
    main()
