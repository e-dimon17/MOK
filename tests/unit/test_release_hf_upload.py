"""release/hf_upload.py — dry-run plan, model card rendering, fake-hub upload path."""

from __future__ import annotations

import sys
import types

import pytest

from release.hf_upload import (
    MODEL_CARD_TEMPLATE,
    PlannedOp,
    UploadError,
    card_placeholders,
    plan_release,
    render_model_card,
    upload_release,
)

# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #


def test_card_placeholders():
    assert card_placeholders(MODEL_CARD_TEMPLATE) == {
        "model_name",
        "benchmarks_table",
        "provenance_root_hash",
        "manifest_hash",
        "replay_instructions",
    }


def test_render_model_card_fills_everything():
    card = render_model_card(
        model_name="MoK-54B-chat",
        benchmarks_table="| Task | Metric | Value |\n|---|---|---:|\n| mmlu | acc | 0.6500 |",
        provenance_root_hash="ab" * 32,
        manifest_hash="cd" * 32,
    )
    assert "# MoK-54B-chat" in card
    assert "| mmlu | acc | 0.6500 |" in card
    assert f"`{'ab' * 32}`" in card
    assert f"`{'cd' * 32}`" in card
    assert "python -m release.replay_window" in card          # default replay instructions
    assert "python -m release.verify_bundle" in card
    assert "license: apache-2.0" in card
    assert "{" not in card.replace("{}", "")            # no unfilled placeholders remain


def test_render_model_card_rejects_unknown_placeholder():
    with pytest.raises(UploadError, match="unknown placeholders"):
        render_model_card(
            model_name="x",
            benchmarks_table="t",
            provenance_root_hash="r",
            manifest_hash="m",
            card_template="# {model_name} {surprise_field}",
        )


# --------------------------------------------------------------------------- #
# Dry-run planning
# --------------------------------------------------------------------------- #


def _dirs(tmp_path):
    weights = tmp_path / "hf_export"
    prov = tmp_path / "provenance"
    weights.mkdir()
    prov.mkdir()
    (weights / "model.safetensors").write_bytes(b"w")
    (prov / "index.json").write_bytes(b"{}")
    return {"": weights, "provenance": prov}


def test_dry_run_plan(tmp_path):
    ops = upload_release(
        "mok-subnet/MoK-54B-chat",
        _dirs(tmp_path),
        provenance_root_hash="ab" * 32,
        manifest_hash="cd" * 32,
        benchmarks_table="(table)",
        dry_run=True,
    )
    assert ops == [
        PlannedOp("create_repo", "mok-subnet/MoK-54B-chat", source="if absent (private)"),
        PlannedOp("upload_folder", ".", source=str(tmp_path / "hf_export")),
        PlannedOp("upload_folder", "provenance", source=str(tmp_path / "provenance")),
        PlannedOp("upload_file", "README.md", source="<rendered model card>"),
    ]


def test_dry_run_does_not_import_huggingface_hub(tmp_path, monkeypatch):
    # Poison the import: if the dry run tried `import huggingface_hub` it would raise.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    ops = upload_release("org/model", _dirs(tmp_path), dry_run=True)
    assert [op.op for op in ops] == ["create_repo", "upload_folder", "upload_folder", "upload_file"]


@pytest.mark.parametrize("repo", ["nomodel", "a/b/c", "/x", "x/"])
def test_bad_repo_id_rejected(tmp_path, repo):
    with pytest.raises(UploadError, match="org/name"):
        upload_release(repo, _dirs(tmp_path), dry_run=True)


def test_missing_dir_rejected(tmp_path):
    with pytest.raises(UploadError, match="does not exist"):
        upload_release("org/model", {"weights": tmp_path / "absent"}, dry_run=True)


def test_empty_dirs_rejected():
    with pytest.raises(UploadError, match="at least one"):
        upload_release("org/model", {}, dry_run=True)


def test_illegal_path_in_repo_rejected(tmp_path):
    d = _dirs(tmp_path)
    with pytest.raises(UploadError, match="illegal path_in_repo"):
        upload_release("org/model", {"../escape": d["provenance"]}, dry_run=True)


def test_plan_release_public_visibility(tmp_path):
    ops = plan_release("org/model", _dirs(tmp_path), private=False)
    assert ops[0].source == "if absent (public)"


# --------------------------------------------------------------------------- #
# Real path against a fake huggingface_hub
# --------------------------------------------------------------------------- #


class _FakeApi:
    instances: list[_FakeApi] = []

    def __init__(self, token=None):
        self.token = token
        self.calls: list[tuple] = []
        _FakeApi.instances.append(self)

    def repo_exists(self, repo_id, repo_type):
        self.calls.append(("repo_exists", repo_id, repo_type))
        return False

    def create_repo(self, repo_id, repo_type, private):
        self.calls.append(("create_repo", repo_id, repo_type, private))

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        self.calls.append(("upload_folder", repo_id, folder_path, path_in_repo))

    def upload_file(self, repo_id, repo_type, path_or_fileobj, path_in_repo, commit_message):
        self.calls.append(("upload_file", repo_id, path_in_repo, path_or_fileobj))


def test_upload_executes_against_hub_api(tmp_path, monkeypatch):
    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)
    _FakeApi.instances.clear()

    dirs = _dirs(tmp_path)
    ops = upload_release(
        "mok-subnet/MoK-54B-chat",
        dirs,
        provenance_root_hash="ab" * 32,
        manifest_hash="cd" * 32,
        token="hf_secret",
        dry_run=False,
    )
    assert len(ops) == 4
    (api,) = _FakeApi.instances
    assert api.token == "hf_secret"
    kinds = [c[0] for c in api.calls]
    assert kinds == ["repo_exists", "create_repo", "upload_folder", "upload_folder", "upload_file"]
    # root dir uploads to '.', provenance to 'provenance'
    assert api.calls[2][3] == "."
    assert api.calls[3][3] == "provenance"
    readme = api.calls[4][3]
    assert b"MoK-54B-chat" in readme
    assert ("ab" * 32).encode() in readme
