"""Recovery must work before installing scientific packages."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


MODEL = Path(__file__).resolve().parents[1]


def test_recovery_module_runs_without_site_packages():
    environment = {**os.environ, "PYTHONPATH": str(MODEL)}
    result = subprocess.run(
        [sys.executable, "-S", "-m", "ingredient_model.recovery", "--help"],
        cwd=MODEL, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "{pack,restore,verify}" in result.stdout


def test_package_import_does_not_load_scientific_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "import sys, ingredient_model; from ingredient_model import recovery; "
         "assert not {'numpy', 'scipy', 'torch'} & sys.modules.keys()"],
        cwd=MODEL, env={**os.environ, "PYTHONPATH": str(MODEL)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_package_exports_remain_available():
    import ingredient_model as package
    from ingredient_model import registry, spec

    assert package.ModelSpec is spec.ModelSpec
    assert package.TrainContext is spec.TrainContext
    assert package.TrainResult is spec.TrainResult
    assert package.register is registry.register
    assert package.get("ease").name == "ease"
    assert package.all_specs is registry.all_specs
    assert package.families is registry.families
    with pytest.raises(AttributeError):
        getattr(package, "not_a_package_export")
