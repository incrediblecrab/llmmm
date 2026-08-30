"""Submit a sweep to Azure ML.

The unit of submission is an *experiment file*, the same one `im sweep` runs
locally. That is deliberate: a job that can only be described in Azure is a job
that cannot be reproduced on a laptop, and the fastest way to waste a fixed
compute budget is to debug a distributed configuration remotely.

Workflow: run the sweep locally with `--dry-run`, then submit the identical
file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AzureConfig, load_config

ROOT = Path(__file__).resolve().parent.parent


def _client(cfg: AzureConfig):
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ModuleNotFoundError as e:
        raise SystemExit("pip install 'ingredient-model[azure]'") from e
    return MLClient(DefaultAzureCredential(), cfg.subscription_id,
                    cfg.resource_group, cfg.workspace_name)


def ensure_compute(ml, cfg: AzureConfig, verbose: bool = True):
    from azure.ai.ml.entities import AmlCompute

    try:
        existing = ml.compute.get(cfg.compute_name)
        if verbose:
            print(f"  compute {cfg.compute_name}: {existing.provisioning_state} "
                  f"({existing.size}, max {existing.max_instances} nodes)")
        return existing
    except Exception:
        pass

    if verbose:
        print(f"  creating {cfg.compute_name} ({cfg.vm_size} x {cfg.max_nodes})")
    cluster = AmlCompute(
        name=cfg.compute_name, size=cfg.vm_size, min_instances=0,
        max_instances=cfg.max_nodes,
        idle_time_before_scale_down=cfg.idle_seconds_before_scaledown,
        # Dedicated, not LowPriority: this subscription has zero low-priority
        # quota, so a spot request is rejected rather than merely queued.
        tier="Dedicated")
    return ml.compute.begin_create_or_update(cluster).result()


def submit(experiment: Path, cfg: AzureConfig | None = None,
           dry_run: bool = False) -> int:
    cfg = cfg or load_config()
    problems = cfg.validate()
    if problems:
        print("azure configuration is invalid:")
        for p in problems:
            print(f"  {p}")
        return 2

    exp_path = Path(experiment)
    if not exp_path.exists():
        print(f"no experiment file at {exp_path}")
        return 2

    rel = exp_path.relative_to(ROOT) if exp_path.is_absolute() else exp_path
    command = f"python -m ingredient_model.cli sweep {rel}"

    print(f"workspace   {cfg.workspace_name} ({cfg.resource_group}, {cfg.location})")
    print(f"compute     {cfg.compute_name}  {cfg.vm_size} x {cfg.max_nodes}"
          f"  = {cfg.vcpus_per_node * cfg.max_nodes}/{6} vCPU quota")
    print(f"image       {cfg.environment.rsplit('/', 3)[-3:][0]}...")
    print(f"command     {command}")

    if dry_run:
        print("\ndry run — nothing submitted.")
        return 0

    from azure.ai.ml import command as ml_command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.entities import Environment

    ml = _client(cfg)
    ensure_compute(ml, cfg)

    job = ml_command(
        code=str(ROOT),
        command=command,
        environment=cfg.environment,
        compute=cfg.compute_name,
        experiment_name=cfg.experiment_name,
        display_name=exp_path.stem,
        tags={"experiment": exp_path.stem, **cfg.tags},
        outputs={
            # UPLOAD, not RW_MOUNT. The FUSE write-mount fails with a 409 that
            # is reported as an authentication error, which sends you debugging
            # credentials that were never the problem.
            "results": {"type": AssetTypes.URI_FOLDER,
                        "mode": InputOutputModes.UPLOAD}
        },
    )
    created = ml.jobs.create_or_update(job)
    print(f"\nsubmitted {created.name}")
    print(f"  {created.studio_url}")
    print(f"  python -m cloud.status {created.name}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cloud.submit", description=__doc__)
    ap.add_argument("experiment", help="path to an experiments/*.yaml file")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return submit(Path(a.experiment), dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
