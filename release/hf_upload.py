"""HuggingFace release uploader — weights + provenance bundle + model card.

`upload_release` pushes the release artifacts (converted HF weights from step
F/G, the provenance bundle from release/provenance.py) to a hub repo and writes a
provenance-forward model card. huggingface_hub is imported lazily and only on
a real upload; dry_run=True computes and returns the exact operation plan
without touching the network (unit-tested).
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

README_FILENAME = "README.md"
DEFAULT_COMMIT_MESSAGE = "MoK release upload"

#: Provenance-forward model card. Placeholders are filled by render_model_card.
MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- mixture-of-experts
- bittensor
- verified-pretraining
- provenance
---

# {model_name}

{model_name} is a 54B-total / 5.5B-active Mixture-of-Experts language model
(32 layers, 128 routed experts top-8 + 1 shared, MXFP8 routed / BF16 dense)
pretrained **decentralized on a Bittensor subnet with bitwise replay
verification**: every training window's weights hash was committed on-chain
before the next window began, and independent auditors reproduced sampled
windows bit-for-bit.

Load with `trust_remote_code=True` (custom MoK-MoE modeling class).

## Benchmarks

{benchmarks_table}

## Provenance

This release ships with a verifiable provenance bundle — the run manifest,
every window's certified state root and gradient payload hashes, the signed
audit log, and a replay script. Nothing in the training lineage can be
altered without breaking these hashes.

- provenance bundle root hash (committed on-chain): `{provenance_root_hash}`
- run manifest hash: `{manifest_hash}`

Verify the bundle offline:

```bash
pip install mok-subnet
python -m release.verify_bundle ./provenance
```

## Replay a training window yourself

{replay_instructions}

## License

Apache 2.0 — weights, code, and provenance bundle.
"""

DEFAULT_REPLAY_INSTRUCTIONS = """\
Any window of the run can be re-derived bitwise from public inputs:

```bash
python -m release.replay_window --bundle ./provenance \\
    --window <N> --miner-uid <U> \\
    --theta-start ./checkpoints/w<N padded to 8> \\
    --config ./configs/bulk.yaml --out report.json
```

Exit code 0 means the replayed weights hash equals the hash the miner
committed on-chain during training — the window is exactly reproducible."""


class UploadError(ValueError):
    """The release upload inputs are invalid."""


@dataclass(frozen=True)
class PlannedOp:
    """One hub operation, in execution order (the dry_run return unit)."""

    op: str        # 'create_repo' | 'upload_file' | 'upload_folder'
    target: str    # repo id or path_in_repo destination
    source: str = ""

    def to_json(self) -> dict[str, str]:
        return {"op": self.op, "target": self.target, "source": self.source}


def card_placeholders(template: str = MODEL_CARD_TEMPLATE) -> set[str]:
    """The set of {placeholder} field names a card template requires."""
    return {field for _, field, _, _ in string.Formatter().parse(template) if field}


def render_model_card(
    *,
    model_name: str,
    benchmarks_table: str,
    provenance_root_hash: str,
    manifest_hash: str,
    replay_instructions: str = DEFAULT_REPLAY_INSTRUCTIONS,
    card_template: str = MODEL_CARD_TEMPLATE,
) -> str:
    """Fill the card template; raises UploadError on unknown/missing placeholders."""
    fields = {
        "model_name": model_name,
        "benchmarks_table": benchmarks_table,
        "provenance_root_hash": provenance_root_hash,
        "manifest_hash": manifest_hash,
        "replay_instructions": replay_instructions,
    }
    missing = card_placeholders(card_template) - set(fields)
    if missing:
        raise UploadError(f"card template requires unknown placeholders: {sorted(missing)}")
    return card_template.format(**fields)


def plan_release(hf_repo: str, dirs: dict[str, Path], *, private: bool = True) -> list[PlannedOp]:
    """The exact ordered operation list an upload will perform (pure)."""
    if hf_repo.count("/") != 1 or not all(hf_repo.split("/")):
        raise UploadError(f"hf_repo must look like 'org/name', got {hf_repo!r}")
    if not dirs:
        raise UploadError("dirs must map at least one path_in_repo to a local directory")
    for name, path in dirs.items():
        if "\\" in name or name.startswith("/") or ".." in name.split("/"):
            raise UploadError(f"illegal path_in_repo: {name!r}")
        if not Path(path).is_dir():
            raise UploadError(f"release directory does not exist: {path} (for {name!r})")

    visibility = "private" if private else "public"
    ops = [PlannedOp("create_repo", hf_repo, source=f"if absent ({visibility})")]
    ops.extend(
        PlannedOp("upload_folder", name or ".", source=str(dirs[name])) for name in sorted(dirs)
    )
    ops.append(PlannedOp("upload_file", README_FILENAME, source="<rendered model card>"))
    return ops


def upload_release(
    hf_repo: str,
    dirs: dict[str, Path],
    *,
    card_template: str = MODEL_CARD_TEMPLATE,
    model_name: str | None = None,
    benchmarks_table: str = "(pending)",
    provenance_root_hash: str = "(pending)",
    manifest_hash: str = "(pending)",
    replay_instructions: str = DEFAULT_REPLAY_INSTRUCTIONS,
    private: bool = True,
    token: str | None = None,
    commit_message: str = DEFAULT_COMMIT_MESSAGE,
    dry_run: bool = False,
) -> list[PlannedOp]:
    """Upload the release to `hf_repo`: create the repo if absent, push every
    directory in `dirs` (key = path_in_repo, '' = repo root), and write the
    rendered model card as README.md.

    dry_run=True validates everything, renders the card, and returns the
    operation plan without importing huggingface_hub or touching the network.
    Returns the executed (or planned) operations in order.
    """
    ops = plan_release(hf_repo, dirs, private=private)
    card = render_model_card(
        model_name=model_name or hf_repo.split("/", 1)[1],
        benchmarks_table=benchmarks_table,
        provenance_root_hash=provenance_root_hash,
        manifest_hash=manifest_hash,
        replay_instructions=replay_instructions,
        card_template=card_template,
    )
    if dry_run:
        return ops

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)
    if not api.repo_exists(repo_id=hf_repo, repo_type="model"):
        api.create_repo(repo_id=hf_repo, repo_type="model", private=private)
    for name in sorted(dirs):
        api.upload_folder(
            repo_id=hf_repo,
            repo_type="model",
            folder_path=str(dirs[name]),
            path_in_repo=name or ".",
            commit_message=f"{commit_message}: {name or 'root'}",
        )
    api.upload_file(
        repo_id=hf_repo,
        repo_type="model",
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo=README_FILENAME,
        commit_message=f"{commit_message}: model card",
    )
    return ops
