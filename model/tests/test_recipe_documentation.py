from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_docs")


def test_recipe_evidence_is_part_of_the_documentation_registry(checker, monkeypatch, tmp_path):
    results = tmp_path / "model" / "results"
    results.mkdir(parents=True)
    names = {"recipe_catalog_build.json", "recipe_ranker_training.json",
             "recipe_search_live.json", "recipe_search_export.json",
             "huggingface_recipe_demo_release.json"}
    for name in names:
        (results / name).write_text(json.dumps({"verified_count": 1234}))
    monkeypatch.setattr(checker, "MODEL", results.parent)
    assert {path.name for path, _ in checker.artefact_json_files()} == names


def test_doi_destinations_are_not_metrics_but_link_labels_still_are(
        checker, monkeypatch, tmp_path, capsys):
    doc = tmp_path / "README.md"
    doc.write_text("There are 1,234 rows. [Paper](https://doi.org/10.1007/BF00992696).\n")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "DOCS", [doc])
    monkeypatch.setattr(checker, "BASELINE", tmp_path / "absent-baseline.json")
    monkeypatch.setattr(checker, "known_values", lambda: ({1234.0}, {"fixture": 1}))
    monkeypatch.setattr(sys, "argv", ["check_docs.py"])
    assert checker.main() == 0
    capsys.readouterr()
    doc.write_text("[9,876 rows](https://example.test/1,234)\n")
    assert checker.main() == 1


def test_documentation_gate_exits_nonzero_for_a_planted_count(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("Verified recipe count: 9,876.\n")
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    code = (
        "import sys; from pathlib import Path; import check_docs as c; "
        "c.ROOT=Path(sys.argv[1]); c.DOCS=[c.ROOT/'README.md']; "
        "c.BASELINE=c.ROOT/'absent.json'; c.known_values=lambda:({1234.0},{'fixture':1}); "
        "sys.argv=['check_docs.py']; raise SystemExit(c.main())")
    process = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=scripts,
        capture_output=True, text=True)
    assert process.returncode == 1
    assert "9,876" in process.stdout
