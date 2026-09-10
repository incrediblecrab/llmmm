from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def publisher(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("huggingface_hub")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("publish_recipe_demo")


def test_new_space_creation_can_only_request_the_free_static_sdk(publisher, monkeypatch):
    calls = {}

    class Api:
        created = False

        def repo_info(self, *_args, **_kwargs):
            if not self.created:
                raise publisher.RepositoryNotFoundError(
                    "missing", response=publisher.httpx.Response(
                        404, request=publisher.httpx.Request("GET", "https://huggingface.co/api/spaces/test")))
            return SimpleNamespace(sha="a" * 40, siblings=[
                SimpleNamespace(rfilename=name)
                for name in (".gitattributes", "README.md", "index.html", "style.css")])

        def create_repo(self, repository, **kwargs):
            calls["create"] = (repository, kwargs)
            self.created = True

        def create_commit(self, repository, **kwargs):
            calls["commit"] = (repository, kwargs)
            return SimpleNamespace(oid="b" * 40)

    monkeypatch.setattr(publisher, "public_files", lambda *args: calls.update(verified=args))
    result = publisher.publish_files(Api(), publisher.SPACE_REPOSITORY, "space",
                                     {"index.html": b"test fixture", "README.md": b"test card",
                                      "styles.css": b"test styles"}, update=False)
    assert result == "b" * 40
    assert calls["create"][1] == {
        "repo_type": "space", "private": False, "exist_ok": False, "space_sdk": "static",
    }
    assert calls["commit"][1]["parent_commit"] == "a" * 40
    operations = calls["commit"][1]["operations"]
    assert [item.path_in_repo for item in operations if isinstance(item, publisher.CommitOperationDelete)] == ["style.css"]
    assert calls["verified"][3] == "b" * 40


@pytest.mark.parametrize("private,gated,sdk,extra", [
    (True, False, "static", None),
    (False, "auto", "static", None),
    (False, False, "docker", None),
    (False, False, "gradio", None),
    (False, False, "static", "private.sqlite"),
])
def test_publication_refuses_to_repurpose_other_repositories(publisher, private, gated, sdk, extra):
    names = ["index.html"] + ([extra] if extra else [])
    api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(
        private=private, gated=gated, sdk=sdk, siblings=[SimpleNamespace(rfilename=name) for name in names]))
    with pytest.raises(ValueError):
        publisher.publish_files(api, publisher.SPACE_REPOSITORY, "space",
                                {"index.html": b"test fixture"}, update=True)


def test_browser_report_walker_counts_tests_not_nested_fields(publisher):
    report = {"suites": [{"suites": [{"specs": [{"tests": [
        {"status": "expected", "results": [{"status": "passed"}]},
        {"status": "unexpected", "results": [{"status": "failed"}]},
    ]}]}]}]}
    assert len(publisher._browser_tests(report)) == 2
