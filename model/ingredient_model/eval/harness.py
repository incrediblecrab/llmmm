"""Evaluation driver.

The context (labels, held-out edges, degree-matched negatives) is built once per
split and cached, then reused for every model scored against it. That is not
only a speed concern: rebuilding it per model would re-sample the negatives, and
two models' scores would then differ by sampling noise as well as by quality.
"""
from __future__ import annotations

import functools

import numpy as np

from ..config import PATHS, SEED
from ..data.graphs import GRAPH_FULL, load_ii_graph
from ..data.labels import TIERS, load_substitutions
from ..data.splits import DEFAULT_SPLIT, Split, get_split
from . import metrics as M
from .completion import recipe_completion


class EvalContext:
    def __init__(self, split: Split, n: int, itos: list[str],
                 unigram: np.ndarray, degree: np.ndarray, edge_set: set,
                 subs, held):
        self.split = split
        self.n = n
        self.itos = itos
        self.unigram = unigram
        self.degree = degree
        self.edge_set = edge_set
        self.subs = subs
        self.held = held
        self.link_negatives = None
        self.stats: dict = {}


@functools.lru_cache(maxsize=4)
def build_context(split_name: str = DEFAULT_SPLIT) -> EvalContext:
    """Assemble labels and controls for one protocol.

    The *full* graph defines the edge set, deliberately: a link-prediction
    negative must be a true non-edge in reality, not merely absent from a
    training split, or a held-out edge could be handed back as a negative.
    """
    split = get_split(split_name)
    g = load_ii_graph(GRAPH_FULL)
    held = None
    hp = PATHS.graphs / split.heldout
    if hp.exists():
        z = np.load(hp, allow_pickle=True)
        held = (z["src"].astype(np.int64), z["dst"].astype(np.int64))

    ctx = EvalContext(
        split=split, n=g.n_vocab, itos=g.itos,
        unigram=np.asarray(g.unigram, np.float64), degree=g.degree(),
        edge_set=g.edge_key_set(), subs=load_substitutions(tuple(g.itos)),
        held=held)
    if held is not None:
        ctx.link_negatives = M.sample_degree_matched_negatives(ctx, seed=SEED)
    ctx.stats = {
        "split": split.name, "vocab": ctx.n,
        "isolated": int((ctx.degree == 0).sum()),
        "heldout_edges": 0 if held is None else len(held[0]),
        "labels": ctx.subs.summary(),
    }
    return ctx


def evaluate(W: np.ndarray, ctx: EvalContext | None = None,
             whiten: bool = False, completion_corpus=None,
             n_completion: int = 20_000, scorer=None) -> dict:
    ctx = ctx or build_context()
    if W.shape[0] != ctx.n:
        raise ValueError(f"embedding has {W.shape[0]} rows, vocabulary is {ctx.n}")
    if whiten:
        W = M.all_but_top(W, k=3)
    U = M.unit(W)
    auc, auc_ci = M.m4_link_auc(U, ctx)
    out = {
        "split": ctx.split.name,
        "n": int(W.shape[0]), "d": int(W.shape[1]), "whitened": bool(whiten),
        "M1_participation_ratio": M.m1_participation_ratio(W),
        "M4_link_auc": auc, "M4_link_auc_ci95": auc_ci,
    }
    for tier in TIERS:
        acc, ci = M.m2_triplet_accuracy(U, ctx, tier)
        out[f"M2_triplet_accuracy_{tier}"] = acc
        out[f"M2_triplet_accuracy_{tier}_ci95"] = ci
        out[f"M3_recall_at_10_{tier}"] = M.m3_recall_at_10(U, ctx, tier)
    out.update({f"M5_{k}": v for k, v in M.m5_popularity(W, ctx).items()})
    if completion_corpus is not None:
        out.update(recipe_completion(W, completion_corpus, n_test=n_completion,
                                     unigram=ctx.unigram, scorer=scorer))
        # Re-score with the embedding centroid removed. This changes glove's
        # recall@10 from 0.2509 to 0.4919 and never costs any family more than
        # sampling noise, so it is reported alongside rather than instead of the
        # raw number. The mechanism is *not* established — translation gauge,
        # popularity-in-the-mean and centroid size were each tested and each
        # fails to predict which families move. See eval/completion.py.
        centred = recipe_completion(W - W.mean(0, keepdims=True), completion_corpus,
                                    n_test=n_completion, unigram=ctx.unigram)
        out.update({f"M6_centred_{k[3:]}": v for k, v in centred.items()
                    if k.startswith("M6_") and not k.startswith("M6_popularity")
                    and k != "M6_n"})
    return out


def control_gate(ctx: EvalContext | None = None, d: int = 300) -> dict:
    """Score pure noise. **This blocks everything.**

    A metric that rates random vectors above chance is measuring an artefact,
    and every number computed with it afterwards is void. It is a gate, not a
    formality.
    """
    ctx = ctx or build_context()
    rng = np.random.default_rng(SEED)
    results = {
        "random_gaussian": evaluate(rng.normal(size=(ctx.n, d)), ctx),
        "collapsed_rank1": evaluate(
            rng.normal(size=(ctx.n, 1)) @ rng.normal(size=(1, d)), ctx),
    }
    g = results["random_gaussian"]
    checks = {f"M2_{t}": g[f"M2_triplet_accuracy_{t}"] for t in TIERS}
    checks["M4"] = g["M4_link_auc"]
    failed = {k: v for k, v in checks.items() if abs(v - 0.5) >= 0.02}
    return {"results": results, "checks": checks, "failed": failed,
            "passed": not failed}
