from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def publisher(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("publish_ingredient_demo")


def test_only_free_static_hardware_can_be_created(publisher, monkeypatch, tmp_path):
    calls = {}
    for name in ("README.md", "index.html", "styles.css"):
        (tmp_path / name).write_text("fixture")
    files = publisher.inventory(tmp_path)

    class Api:
        created = False

        def repo_info(self, *_args, **_kwargs):
            if not self.created:
                raise publisher.RepositoryNotFoundError(
                    "missing", response=publisher.httpx.Response(
                        404, request=publisher.httpx.Request("GET", "https://huggingface.co/api/spaces/test")))
            return SimpleNamespace(
                sha="a" * 40, private=False, gated=False, sdk="static",
                siblings=[SimpleNamespace(rfilename=name)
                          for name in (".gitattributes", "README.md", "index.html", "style.css")])

        def create_repo(self, _repository, **kwargs):
            calls["create"] = kwargs
            self.created = True

        def create_commit(self, _repository, **kwargs):
            calls["commit"] = kwargs
            return SimpleNamespace(oid="b" * 40)

    monkeypatch.setattr(publisher, "verify_public_tree", lambda *args: calls.update(verified=args))
    assert publisher.publish_tree(Api(), "fixture/space", "space", tmp_path, files, update=False) == "b" * 40
    assert calls["create"] == {
        "repo_type": "space", "private": False, "exist_ok": False, "space_sdk": "static",
    }
    assert calls["commit"]["parent_commit"] == "a" * 40
    assert [op.path_in_repo for op in calls["commit"]["operations"]
            if isinstance(op, publisher.CommitOperationDelete)] == ["style.css"]
    assert calls["verified"][3] == "b" * 40


@pytest.mark.parametrize("private,gated,sdk,extra", [
    (True, False, "static", None), (False, "auto", "static", None),
    (False, False, "gradio", None), (False, False, "docker", None),
    (False, False, "static", "private.sqlite"),
])
def test_other_remote_repositories_cannot_be_repurposed(publisher, tmp_path, private, gated, sdk, extra):
    names = ["index.html"] + ([extra] if extra else [])
    api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(
        private=private, gated=gated, sdk=sdk,
        siblings=[SimpleNamespace(rfilename=name) for name in names]))
    with pytest.raises(ValueError):
        publisher.publish_tree(api, "fixture/space", "space", tmp_path, {"index.html": {}}, update=True)


def test_anonymous_bytes_are_verified_not_only_hub_metadata(publisher, monkeypatch, tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"fixture")
    files = publisher.inventory(tmp_path)
    calls = []

    def repo_info(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(private=False, gated=False,
                               siblings=[SimpleNamespace(rfilename="data.parquet")])

    monkeypatch.setattr(publisher, "downloaded_file", lambda *args: path)
    api = SimpleNamespace(repo_info=repo_info)
    publisher.verify_public_tree(api, "fixture/dataset", "dataset", "a" * 40, files)
    assert calls[0]["token"] is False
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="anonymous download"):
        publisher.verify_public_tree(api, "fixture/dataset", "dataset", "a" * 40, files)


def test_existing_tags_are_never_moved(publisher):
    api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    publisher.immutable_tag(api, "fixture/dataset", "dataset", "v1", "a" * 40)
    with pytest.raises(ValueError, match="must not move"):
        publisher.immutable_tag(api, "fixture/dataset", "dataset", "v1", "b" * 40)


def test_dataset_inventory_rejects_an_extra_private_file(publisher, monkeypatch, tmp_path):
    for name in ("README.md", "index/ingredient-index.json", "index/safe.gz",
                 "data/train-00000-of-00001.parquet"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setattr(publisher, "load_ingredient_catalog", lambda directory: (
        {"arrays": {"ingredients": {"file": "safe.gz"}}, "url_shards": []}, {}))
    (tmp_path / "dataset-manifest.json").write_text(json.dumps({"files": publisher.inventory(tmp_path)}))
    assert len(publisher.verified_dataset_files(tmp_path)) == 5
    (tmp_path / "private.sqlite").write_text("not part of the release")
    with pytest.raises(ValueError, match="outside its verified inventory"):
        publisher.verified_dataset_files(tmp_path)


def test_viewer_must_cover_the_full_population_not_just_matching_first_rows(publisher, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "index").mkdir()
    (tmp_path / "index/ingredient-index.json").write_text(json.dumps({"n_recipes": 3}))
    rows = [
        {"id": value, "ingredient_ids": [0], "ingredients": ["egg"], "source": "fixture",
         "language": "en", "total_minutes": None, "servings": None, "source_url": None}
        for value in range(3)
    ]
    publisher.pq.write_table(publisher.pa.Table.from_pylist(rows),
                             tmp_path / "data/train-00000-of-00001.parquet")
    count = 3

    class Client:
        def get(self, url, **kwargs):
            if url.endswith("first-rows"):
                body = {"rows": [{"row": row, "truncated_cells": []} for row in rows]}
            else:
                body = {"partial": False, "pending": [], "failed": [], "size": {
                    "dataset": {"num_rows": count},
                    "splits": [{"config": "default", "split": "train", "num_rows": count, "num_columns": 8}],
                }}
            return publisher.httpx.Response(200, json=body)

    assert publisher.verify_dataset_viewer(Client(), tmp_path)["total_rows"] == 3
    count = 2
    with pytest.raises(ValueError, match="full declared population"):
        publisher.verify_dataset_viewer(Client(), tmp_path)
