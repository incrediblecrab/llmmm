"""The complete native ingredient predictor in a Hugging Face-compatible package."""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from torch import nn

from models.set_transformer.train import DEFAULTS, _build


class IngredientPredictor(
        nn.Module, PyTorchModelHubMixin,
        repo_url="https://github.com/incrediblecrab/llmmm",
        library_name="pytorch",
        tags=["ingredient-completion", "set-transformer"]):
    def __init__(self, vocabulary: list[str], architecture: dict):
        super().__init__()
        if (not vocabulary or not all(isinstance(name, str) and name for name in vocabulary)
                or len(set(vocabulary)) != len(vocabulary)):
            raise ValueError("vocabulary must contain unique nonempty ingredient names")
        self.vocabulary = list(vocabulary)
        self.architecture = dict(architecture)
        self._index = {name: i for i, name in enumerate(vocabulary)}
        self.network = _build(len(vocabulary), {**DEFAULTS, **architecture}, "cpu")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Unnormalized candidate logits; recommend() excludes visible ingredients."""
        if input_ids.ndim != 2 or not input_ids.shape[0] or not input_ids.shape[1]:
            raise ValueError("input_ids must be a nonempty [batch, context] matrix")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer ingredient IDs")
        n = len(self.vocabulary)
        if ((input_ids < 0) | (input_ids >= n)).any():
            raise ValueError("an input ingredient ID is outside the vocabulary")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape or (
                (attention_mask != 0) & (attention_mask != 1)).any():
            raise ValueError("attention_mask must be an aligned zero/one matrix")
        mask = attention_mask.bool()
        if (mask.sum(1) < 2).any():
            raise ValueError("at least two visible ingredients are required per context")
        counts = torch.zeros((len(input_ids), n), device=input_ids.device, dtype=torch.int64)
        counts.scatter_add_(1, input_ids.long(), mask.long())
        if (counts > 1).any():
            raise ValueError("a context must be a set of unique ingredient IDs")
        ids = torch.cat([input_ids, torch.full(
            (len(input_ids), 1), n, device=input_ids.device, dtype=input_ids.dtype)], dim=1)
        padding = torch.cat([~mask, torch.zeros(
            (len(input_ids), 1), device=input_ids.device, dtype=torch.bool)], dim=1)
        position = torch.full(
            (len(input_ids),), input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        return self.network(ids, padding, position)

    @torch.inference_mode()
    def recommend(self, ingredients: list[str], top_k: int = 10) -> list[dict]:
        names = list(dict.fromkeys(name.strip().lower().replace(" ", "_") for name in ingredients))
        unknown = [name for name in names if name not in self._index]
        if unknown:
            raise ValueError(f"unknown canonical ingredient names: {', '.join(unknown)}")
        if len(names) < 2:
            raise ValueError("at least two distinct canonical ingredients are required")
        if type(top_k) is not int or not 1 <= top_k <= len(self.vocabulary) - len(names):
            raise ValueError("top_k must be within the number of unseen candidates")
        device = self.network.tok.weight.device
        ids = torch.tensor([[self._index[name] for name in names]], device=device)
        was_training = self.training
        try:
            self.eval()
            logits = self(ids)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("the predictor produced non-finite scores")
            logits[ids[0]] = -torch.inf
            ranking = torch.argsort(logits, descending=True, stable=True)[:top_k]
            return [{"ingredient": self.vocabulary[int(i)], "score": float(logits[i])}
                    for i in ranking]
        finally:
            self.train(was_training)

    @classmethod
    def _from_pretrained(
            cls, *, model_id, revision, cache_dir, force_download,
            local_files_only, token, map_location="cpu", strict=True, **model_kwargs):
        if strict is not True:
            raise ValueError("native predictor loading must be strict")
        model = cls(**model_kwargs)
        path = Path(model_id)
        weights = (path / "model.safetensors" if path.is_dir() else Path(hf_hub_download(
            repo_id=model_id, filename="model.safetensors", revision=revision,
            cache_dir=cache_dir, force_download=force_download,
            local_files_only=local_files_only, token=token)))
        model.load_state_dict(load_file(weights, device="cpu"), strict=True)
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("native predictor weights contain non-finite values")
        return model.to(map_location).eval()
