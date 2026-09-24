from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RETIRED_DATASET = "incrediblecrab/llmmm-recipe-" + "sample"


@pytest.fixture
def publisher(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("huggingface_hub")
    monkeypatch.syspath_prepend(str(ROOT / "model/scripts"))
    return importlib.import_module("publish_recipe_demo")


def test_browser_report_walker_counts_tests_not_nested_fields(publisher):
    report = {"suites": [{"suites": [{"specs": [{"tests": [
        {"status": "expected", "results": [{"status": "passed"}]},
        {"status": "unexpected", "results": [{"status": "failed"}]},
    ]}]}]}]}
    assert len(publisher._browser_tests(report)) == 2


def _tracked_files() -> list[str]:
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout to list the tracked files")
    return [name for name in listing.stdout.decode().split("\0") if name]


def test_no_code_can_republish_the_retired_sample_dataset(publisher):
    assert not hasattr(publisher, "main") and not hasattr(publisher, "publish_files")
    code = [name for name in _tracked_files() if name.endswith((".py", ".js", ".html", ".mjs", ".cjs", "Makefile"))]
    assert "model/scripts/publish_recipe_demo.py" in code
    assert [name for name in code if RETIRED_DATASET in (ROOT / name).read_text(encoding="utf-8")] == []


def test_no_current_document_links_to_the_retired_sample_dataset():
    # Release receipts under model/results are historical records and keep the name they published under.
    documents = [name for name in _tracked_files() if not name.startswith("model/results/")
                 and name.endswith((".md", ".json", ".txt", ".yml", ".yaml"))]
    assert "README.md" in documents
    link = "huggingface.co/datasets/" + RETIRED_DATASET
    assert [name for name in documents if link in (ROOT / name).read_text(encoding="utf-8", errors="replace")] == []
