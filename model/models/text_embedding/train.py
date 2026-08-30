"""Name-based embeddings, and their alignment to the learned space."""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from ingredient_model.artifacts import load_embedding
from ingredient_model.config import PATHS
from ingredient_model.data.graphs import load_ii_graph
from ingredient_model.registry import register
from ingredient_model.spec import TrainContext, TrainResult

EMBED_DEFAULTS = dict(model="text-embedding-3-small", d_model=0, batch=256,
                      prompt="{name}", cache=True)
ALIGN_DEFAULTS = dict(model="text-embedding-3-small", source_run="",
                      d_model=300, ridge=1.0, cache=True)

CACHE = PATHS.data / "cache" / "text_embeddings"


def _readable(name: str) -> str:
    """`green_bell_pepper` -> `green bell pepper`.

    Underscored tokens are out of distribution for a text encoder; it will
    happily embed them, but as an unfamiliar string rather than as the phrase
    they represent.
    """
    return name.replace("_", " ").strip()


def embed_names(names: list[str], model: str, batch: int = 256,
                prompt: str = "{name}", cache: bool = True) -> np.ndarray:
    """Embed ingredient names via Azure Foundry / OpenAI, with a disk cache.

    The cache key covers the model, the prompt template and the exact name list,
    so any change that would alter the vectors produces a miss while a re-run of
    the same configuration costs nothing.
    """
    key = hashlib.sha256(
        json.dumps([model, prompt, names], sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"{key}.npy"
    if cache and path.exists():
        print(f"  cache hit {path.name}")
        return np.load(path)

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "text embeddings need credentials — set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY (or OPENAI_API_KEY). This family is the only "
            "one that calls a hosted service; every other model trains locally.")
    try:
        from openai import AzureOpenAI, OpenAI
    except ModuleNotFoundError as e:
        raise RuntimeError("pip install 'ingredient-model[foundry]'") from e

    client = (AzureOpenAI(azure_endpoint=endpoint, api_key=api_key,
                          api_version="2024-10-21")
              if endpoint else OpenAI(api_key=api_key))
    texts = [prompt.format(name=_readable(n)) for n in names]
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model, input=chunk)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                # Rate limits are the expected failure, not an exceptional one.
                wait = 2 ** attempt
                print(f"    retry in {wait}s ({type(e).__name__})")
                time.sleep(wait)
        print(f"    {min(i + batch, len(texts)):>5}/{len(texts)}", flush=True)

    W = np.asarray(out, np.float32)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, W)
    return W


@register(name="text-embed", family="text_embedding", cost_hint="cheap",
          defaults=EMBED_DEFAULTS, tags=("cold-start", "hosted", "oov"),
          requires=("ii_graph_train",),
          description="Embed ingredient names with a hosted text model")
def train_text_embed(ctx: TrainContext) -> TrainResult:
    """Pure name semantics, with nothing learned from recipes.

    Expected to score poorly on M4 and M2, and that is the point: it measures
    how much of the ingredient space is recoverable from language alone. Whatever
    it gets is free for ingredients that have never been cooked with in this
    corpus, and the gap to the learned models is the value of the corpus.
    """
    p = {**EMBED_DEFAULTS, **dict(ctx.params)}
    g = load_ii_graph(ctx.graph)
    W = embed_names(list(g.itos), str(p["model"]), int(p["batch"]),
                    str(p["prompt"]), bool(p["cache"]))
    d = int(p["d_model"])
    if d and d < W.shape[1]:
        # Truncation, not PCA: OpenAI's embedding models are Matryoshka-trained,
        # so a prefix of the vector is itself a valid lower-dimensional
        # embedding and needs no fitting.
        W = W[:, :d]
    return TrainResult(embedding=np.ascontiguousarray(W),
                       metadata={"native_dim": int(W.shape[1]), **p})


@register(name="text-aligned", family="text_embedding", cost_hint="cheap",
          defaults=ALIGN_DEFAULTS, tags=("cold-start", "hybrid", "oov"),
          requires=("ii_graph_train",),
          description="Ridge map from name space into a trained recipe space")
def train_text_aligned(ctx: TrainContext) -> TrainResult:
    """Learn ``text -> learned`` so an unseen name can be placed in the good space.

    This is the family's actual product use. A ridge regression fitted on the
    1,790 ingredients that exist in both spaces gives a projection that maps any
    new name into the recipe-trained geometry. The reconstruction is worse than
    the real thing for known ingredients, which is why it is only used where
    there is no real thing.

    Reported metrics are for the *reconstructed* vectors, so the numbers state
    plainly how much of the learned space survives the trip from language —
    which is exactly the question a cold-start user cares about.
    """
    p = {**ALIGN_DEFAULTS, **dict(ctx.params)}
    if not p["source_run"]:
        raise ValueError(
            "text-aligned needs a trained space to align to: "
            "--set source_run=<run-id>. Aligning to nothing is text-embed.")

    target = load_embedding(PATHS.run_dir(str(p["source_run"])))
    g = load_ii_graph(ctx.graph)
    X = embed_names(list(g.itos), str(p["model"]), 256, "{name}",
                    bool(p["cache"])).astype(np.float64)
    if target.shape[0] != X.shape[0]:
        raise ValueError(f"source run has {target.shape[0]} rows, "
                         f"vocabulary has {X.shape[0]}")

    X = np.hstack([X, np.ones((len(X), 1))])
    lam = float(p["ridge"])
    A = X.T @ X + lam * np.eye(X.shape[1])
    M = np.linalg.solve(A, X.T @ target)
    recon = X @ M

    resid = float(np.linalg.norm(recon - target) / np.linalg.norm(target))
    cos = float(np.mean(np.sum(
        recon / np.maximum(np.linalg.norm(recon, axis=1, keepdims=True), 1e-12)
        * target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12),
        axis=1)))
    print(f"  relative residual {resid:.4f}   mean cosine to target {cos:.4f}")
    return TrainResult(
        embedding=recon.astype(np.float32),
        metadata={"relative_residual": resid, "mean_cosine_to_target": cos, **p},
        extra_arrays={"projection": M.astype(np.float32)})
