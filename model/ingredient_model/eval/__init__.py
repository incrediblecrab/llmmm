"""Evaluation: metrics, the control gate, and reporting.

Every metric is scored against labels no model trained on — the substitution
catalogue, a held-out slice of co-occurrence edges, and held-out whole recipes.
Cosine-similarity eyeballing ("does this look tasty") is not evidence and is not
reported here.
"""
from .completion import recipe_completion
from .harness import EvalContext, build_context, control_gate, evaluate
from .metrics import all_but_top, unit
from .report import collect, leaderboard, render_one, report

__all__ = [
    "EvalContext", "build_context", "control_gate", "evaluate",
    "recipe_completion", "all_but_top", "unit",
    "collect", "leaderboard", "render_one", "report",
]
