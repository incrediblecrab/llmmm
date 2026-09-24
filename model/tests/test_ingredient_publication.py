from __future__ import annotations

import gzip
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_ingredient_dataset import REVISION, ingredient_index  # noqa: F401 - shared real-index fixture


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
            return SimpleNamespace(sha="a" * 40, private=False, gated=False, sdk="static")

        def list_repo_tree(self, _repository, **kwargs):
            calls["tree"] = kwargs
            return [publisher.RepoFile(path=name, size=7, oid="0" * 40)
                    for name in (".gitattributes", "README.md", "index.html", "style.css")]

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
    assert calls["tree"]["revision"] == calls["commit"]["parent_commit"] == "a" * 40
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
    api = SimpleNamespace(
        repo_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40, private=private, gated=gated, sdk=sdk),
        list_repo_tree=lambda *args, **kwargs: [publisher.RepoFile(path=name, size=1, oid="0" * 40)
                                                for name in names])
    with pytest.raises(ValueError):
        publisher.publish_tree(api, "fixture/space", "space", tmp_path, {"index.html": {}}, update=True)


def test_anonymous_bytes_are_verified_not_only_hub_metadata(publisher, monkeypatch, tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"fixture")
    files = publisher.inventory(tmp_path)
    calls = []

    def repo_info(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(private=False, gated=False)

    def list_repo_tree(*args, **kwargs):
        calls.append(kwargs)
        return [publisher.RepoFile(path="data.parquet", size=7, oid="0" * 40)]

    monkeypatch.setattr(publisher, "downloaded_file", lambda *args: path)
    api = SimpleNamespace(repo_info=repo_info, list_repo_tree=list_repo_tree)
    publisher.verify_public_tree(api, "fixture/dataset", "dataset", "a" * 40, files)
    assert len(calls) == 2 and all(call["token"] is False for call in calls)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="anonymous download"):
        publisher.verify_public_tree(api, "fixture/dataset", "dataset", "a" * 40, files)


def test_large_updates_commit_only_changes_in_chained_batches_with_control_files_last(
        publisher, monkeypatch, tmp_path):
    names = ["README.md", "dataset-manifest.json", "index/ingredient-index.json", "data/same-lfs.parquet",
             "data/changed.parquet", "data/same-regular.json", "index/text-shards.json.gz",
             "index/text/0000.json.gz", "index/text/0001.json.gz"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(name))
    files = publisher.inventory(tmp_path)
    regular = (tmp_path / "data/same-regular.json").read_bytes()

    def remote(name, **fields):
        return publisher.RepoFile(**{"path": name, "size": files[name]["bytes"], "oid": "0" * 40, **fields})

    tree = [
        publisher.RepoFile(path=".gitattributes", size=1, oid="0" * 40),
        remote("README.md"), remote("dataset-manifest.json"), remote("index/ingredient-index.json"),
        remote("data/same-lfs.parquet", lfs={"size": files["data/same-lfs.parquet"]["bytes"],
                                             "oid": files["data/same-lfs.parquet"]["sha256"], "pointerSize": 1}),
        remote("data/changed.parquet", lfs={"size": files["data/changed.parquet"]["bytes"],
                                            "oid": "f" * 64, "pointerSize": 1}),
        remote("data/same-regular.json", oid=hashlib.sha1(b"blob %d\0" % len(regular) + regular).hexdigest()),
    ]
    commits, verified = [], []

    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="a" * 40, private=False, gated=False)

        def list_repo_tree(self, *_args, **kwargs):
            assert kwargs["revision"] == "a" * 40 and kwargs["recursive"]
            return tree

        def create_commit(self, _repository, **kwargs):
            commits.append(kwargs)
            return SimpleNamespace(oid=f"{len(commits):040d}")

    monkeypatch.setattr(publisher, "MAX_COMMIT_FILES", 3)
    monkeypatch.setattr(publisher, "downloaded_file", lambda *args: tmp_path / "dataset-manifest.json")
    monkeypatch.setattr(publisher, "verify_public_tree", lambda *args: verified.append(args[3]))
    revision = publisher.publish_tree(Api(), "fixture/dataset", "dataset", tmp_path, files, update=True)
    assert [[operation.path_in_repo for operation in commit["operations"]] for commit in commits] == [
        ["index/text-shards.json.gz", "index/text/0000.json.gz", "index/text/0001.json.gz"],
        ["data/changed.parquet"],
        ["README.md", "dataset-manifest.json", "index/ingredient-index.json"],
    ]
    assert all(isinstance(operation, publisher.CommitOperationAdd)
               for commit in commits for operation in commit["operations"])
    assert [commit["parent_commit"] for commit in commits] == ["a" * 40, f"{1:040d}", f"{2:040d}"]
    assert revision == verified[0] == f"{3:040d}"


@pytest.mark.parametrize("listed,deleted", [(["index/source-links.u8.gz"], ["index/source-links.u8.gz"]), ([], None)])
def test_updates_delete_only_remote_files_the_previous_release_listed(
        publisher, monkeypatch, tmp_path, listed, deleted):
    release = tmp_path / "release"
    (release / "index").mkdir(parents=True)
    (release / "dataset-manifest.json").write_text("{}")
    (release / "index/link-status.u8.gz").write_text("new")
    files = publisher.inventory(release)
    previous = tmp_path / "previous-manifest.json"
    previous.write_text(json.dumps({"files": {name: {} for name in ["dataset-manifest.json", *listed]}}))
    commits = []
    api = SimpleNamespace(
        repo_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40, private=False, gated=False),
        list_repo_tree=lambda *args, **kwargs: [publisher.RepoFile(path=name, size=1, oid="0" * 40)
                                                for name in ("dataset-manifest.json", "index/source-links.u8.gz")],
        create_commit=lambda _repository, **kwargs: commits.append(kwargs) or SimpleNamespace(oid="b" * 40))
    monkeypatch.setattr(publisher, "downloaded_file", lambda *args: previous)
    monkeypatch.setattr(publisher, "verify_public_tree", lambda *args: None)
    if deleted is None:
        with pytest.raises(ValueError, match="unrelated remote files"):
            publisher.publish_tree(api, "fixture/dataset", "dataset", release, files, update=True)
        assert commits == []
    else:
        assert publisher.publish_tree(api, "fixture/dataset", "dataset", release, files, update=True) == "b" * 40
        assert [operation.path_in_repo for operation in commits[0]["operations"]
                if isinstance(operation, publisher.CommitOperationDelete)] == deleted


@pytest.mark.parametrize("headers,waits", [
    (['"resolvers";r=2999;t=216'], []),
    (['"resolvers";r=100;t=216', '"resolvers";r=2999;t=300'], [217]),
    ([None], []),
    (['"resolvers";r=100;t=5'] * 3, None),
])
def test_browser_checks_wait_for_the_anonymous_hub_quota(publisher, headers, waits):
    responses, requests, slept = iter(headers), [], []

    def head(url, **kwargs):
        requests.append(kwargs)
        value = next(responses)
        return SimpleNamespace(headers={} if value is None else {
            "ratelimit": value, "ratelimit-policy": '"fixed window";"resolvers";q=3000;w=300'})

    client = SimpleNamespace(head=head)
    if waits is None:
        with pytest.raises(RuntimeError, match="did not recover"):
            publisher.wait_for_resolver_quota(client, "https://hub.test/resolve/r/f", sleep=slept.append)
        assert slept == [6, 6, 6]
    else:
        publisher.wait_for_resolver_quota(client, "https://hub.test/resolve/r/f", sleep=slept.append)
        assert slept == waits
    assert requests == [{"follow_redirects": False}] * len(headers)


def test_release_tags_must_be_new_and_existing_tags_must_survive(publisher):
    refs = {publisher.INGREDIENT_DATASET_REPOSITORY: {"v0.1.0": "a" * 40},
            publisher.SPACE_REPOSITORY: {"v0.1.0-sample": "b" * 40}}
    api = SimpleNamespace(list_repo_refs=lambda repository, **kwargs: SimpleNamespace(tags=[
        SimpleNamespace(name=name, target_commit=target) for name, target in refs[repository].items()]))
    before = publisher.unused_release_tags(api, "v0.2.0", "v0.3.0")
    for dataset_tag, space_tag in (("v0.1.0", "v0.3.0"), ("v0.2.0", "v0.1.0-sample")):
        with pytest.raises(ValueError, match="never moved or reused"):
            publisher.unused_release_tags(api, dataset_tag, space_tag)
    refs[publisher.SPACE_REPOSITORY]["v0.3.0"] = "c" * 40
    publisher.check_tags_preserved(api, before)
    refs[publisher.INGREDIENT_DATASET_REPOSITORY]["v0.1.0"] = "d" * 40
    with pytest.raises(ValueError, match="moved or disappeared"):
        publisher.check_tags_preserved(api, before)


def test_existing_tags_are_never_moved(publisher):
    api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    publisher.immutable_tag(api, "fixture/dataset", "dataset", "v1", "a" * 40)
    with pytest.raises(ValueError, match="must not move"):
        publisher.immutable_tag(api, "fixture/dataset", "dataset", "v1", "b" * 40)


def test_dataset_inventory_rejects_an_extra_private_file_or_missing_card_text(publisher, monkeypatch, tmp_path):
    for name in ("README.md", "index/ingredient-index.json", "index/safe.gz", "index/text-shards.json.gz",
                 "index/text/0000.json.gz", "data/train-00000-of-00001.parquet"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setattr(publisher, "load_ingredient_catalog", lambda directory: (
        {"arrays": {"ingredients": {"file": "safe.gz"}}, "url_shards": [], "link_shards": [],
         "text_manifest": {"file": "text-shards.json.gz"}}, {}))
    monkeypatch.setattr(publisher, "load_text_shards", lambda directory, metadata: [{"file": "text/0000.json.gz"}])

    def write_manifest():
        (tmp_path / "dataset-manifest.json").unlink(missing_ok=True)
        (tmp_path / "dataset-manifest.json").write_text(json.dumps({"files": publisher.inventory(tmp_path)}))

    write_manifest()
    assert len(publisher.verified_dataset_files(tmp_path)) == 7
    (tmp_path / "private.sqlite").write_text("not part of the release")
    with pytest.raises(ValueError, match="outside its verified inventory"):
        publisher.verified_dataset_files(tmp_path)
    (tmp_path / "private.sqlite").unlink()
    (tmp_path / "index/text/0000.json.gz").unlink()
    write_manifest()
    with pytest.raises(ValueError, match="missing declared index files"):
        publisher.verified_dataset_files(tmp_path)


def test_released_card_text_must_equal_the_verified_index(publisher, ingredient_index, tmp_path):
    from ingredient_model.ingredient_dataset import build_ingredient_dataset

    release = tmp_path / "release"
    build_ingredient_dataset(ingredient_index, release, source_revision=REVISION)
    check = publisher.verify_dataset_content(release, ingredient_index)
    assert check["records_compared"] == 7 and check["text_shards_compared"] == 4
    assert check["cooking_instructions_exported"] is False and "private_prose_exported" not in check
    shard = release / "index/text/0001.json.gz"
    shard.write_bytes(gzip.compress(json.dumps({"first_id": 2, "titles": ["Changed", None],
                                                "ingredient_lines": [None, None]}).encode()))
    with pytest.raises(ValueError, match="card text differs"):
        publisher.verify_dataset_content(release, ingredient_index)


def _viewer_rows(publisher, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "index").mkdir()
    (tmp_path / "index/ingredient-index.json").write_text(json.dumps({"n_recipes": 3}))
    rows = [
        {"id": value, "ingredient_ids": [0], "ingredients": ["egg"], "source": "fixture",
         "language": "en", "total_minutes": None, "servings": None, "source_url": None,
         "recipe_link": None, "link_status": "none"}
        for value in range(3)
    ]
    publisher.pq.write_table(publisher.pa.Table.from_pylist(rows),
                             tmp_path / "data/train-00000-of-00001.parquet")
    return rows


def _viewer_size(publisher, count):
    return {"partial": False, "pending": [], "failed": [], "size": {
        "dataset": {"num_rows": count},
        "splits": [{"config": "default", "split": "train", "num_rows": count,
                    "num_columns": len(publisher.PARQUET_COLUMNS)}],
    }}


def test_viewer_must_cover_the_full_population_not_just_matching_first_rows(publisher, tmp_path):
    rows = _viewer_rows(publisher, tmp_path)
    count = 3

    class Client:
        def get(self, url, **kwargs):
            if url.endswith("first-rows"):
                body = {"rows": [{"row": row, "truncated_cells": []} for row in rows]}
            else:
                body = _viewer_size(publisher, count)
            return publisher.httpx.Response(200, json=body)

    assert publisher.verify_dataset_viewer(Client(), tmp_path)["total_rows"] == 3
    count = 2
    with pytest.raises(ValueError, match="full declared population"):
        publisher.verify_dataset_viewer(Client(), tmp_path)


def test_viewer_waits_while_it_still_serves_the_previous_release(publisher, monkeypatch, tmp_path):
    rows = _viewer_rows(publisher, tmp_path)
    previous = [{**row, "source_url": "https://www.cookbooks.test/old"} for row in rows]
    served = [previous, previous, rows]
    sleeps = []

    class Client:
        def get(self, url, **kwargs):
            if url.endswith("first-rows"):
                current = served.pop(0) if len(served) > 1 else served[0]
                return publisher.httpx.Response(200, json={
                    "rows": [{"row": row, "truncated_cells": []} for row in current]})
            return publisher.httpx.Response(200, json=_viewer_size(publisher, 3))

    clock = SimpleNamespace(monotonic=publisher.time.monotonic, sleep=sleeps.append)
    monkeypatch.setattr(publisher, "time", clock)
    assert publisher.verify_dataset_viewer(Client(), tmp_path)["total_rows"] == 3
    assert len(sleeps) == 2
    served[:] = [previous]
    clock.monotonic = iter([0, 0, 1_000]).__next__
    with pytest.raises(TimeoutError, match="differ from this release"):
        publisher.verify_dataset_viewer(Client(), tmp_path, timeout=10)
