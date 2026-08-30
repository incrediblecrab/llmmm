"""Azure configuration, with the subscription's real limits as defaults."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

#: 2 vCPUs x 3 nodes = 6, which is the entire dedicated quota. A larger SKU
#: means fewer nodes, not more compute.
DEFAULT_SKU = "Standard_F2s_v2"
MAX_VCPUS = 6

#: A curated image, deliberately. Handing Azure ML a conda file triggers a
#: hidden image-build cluster that eats the quota this job needs.
DEFAULT_IMAGE = ("azureml://registries/azureml/environments/"
                 "acpt-pytorch-2.2-cuda12.1/versions/55")


@dataclass
class AzureConfig:
    """Defaults match the live `cookbook-regression` workspace as verified via
    the ARM REST API — `cpu-cluster` already exists as Standard_F2s_v2 x 3.

    Note `az ml` commands hang against this subscription (>120s with no
    output). `cloud/setup.py` times out rather than blocking; the REST API
    answers the same questions in under a second.
    """

    subscription_id: str = ""
    resource_group: str = "cookbook-regression"
    workspace_name: str = "cookbook-regression"
    location: str = "westus2"
    compute_name: str = "cpu-cluster"
    vm_size: str = DEFAULT_SKU
    max_nodes: int = 3
    environment: str = DEFAULT_IMAGE
    storage_account: str = "cookingrecipes"
    experiment_name: str = "ingredient-model"
    idle_seconds_before_scaledown: int = 300
    tags: dict = field(default_factory=dict)

    @property
    def vcpus_per_node(self) -> int:
        # Encoded rather than parsed: the SKU name is not a reliable source, and
        # exceeding the quota fails at submission with an unhelpful message.
        return {"Standard_F2s_v2": 2, "Standard_F4s_v2": 4,
                "Standard_D2s_v3": 2, "Standard_D4s_v3": 4}.get(self.vm_size, 2)

    def validate(self) -> list[str]:
        problems = []
        if not self.subscription_id:
            problems.append(
                "subscription_id is unset — run `python -m cloud.setup --init`")
        total = self.vcpus_per_node * self.max_nodes
        if total > MAX_VCPUS:
            problems.append(
                f"{self.vm_size} x {self.max_nodes} needs {total} vCPUs but the "
                f"regional dedicated quota is {MAX_VCPUS}. The cluster will be "
                f"created and then never allocate a node.")
        if "conda" in self.environment or self.environment.endswith((".yml", ".yaml")):
            problems.append(
                "a conda environment file triggers a hidden 4-vCPU image-build "
                "cluster that consumes the quota this job needs. Use a curated "
                "image.")
        return problems

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AzureConfig":
        data = json.loads(path.read_text()) if path.exists() else {}
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        # Environment wins over the file so CI and a laptop can share one repo
        # without the file being edited back and forth.
        cfg.subscription_id = (os.environ.get("AZURE_SUBSCRIPTION_ID")
                               or cfg.subscription_id)
        cfg.resource_group = os.environ.get("AZURE_RESOURCE_GROUP") or cfg.resource_group
        cfg.workspace_name = os.environ.get("AZURE_WORKSPACE") or cfg.workspace_name
        return cfg


def load_config(path: Path = CONFIG_PATH) -> AzureConfig:
    return AzureConfig.load(path)
