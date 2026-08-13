"""YAML config loading with overlay merging and ${ENV:default} interpolation.

A run is described by a base YAML plus zero or more step overlays
(C/configs/base.yaml + bulk.yaml, D/configs/anneal.yaml, ...). Overlays
deep-merge into the base; the merged dict validates into RunConfig.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .schemas import RunConfig

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            got = os.environ.get(var, default)
            if got is None:
                raise KeyError(f"environment variable {var!r} is required by config and unset")
            return got

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Overlay wins; dicts merge recursively; everything else replaces."""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    return data


def load_run_config(base: str | Path, *overlays: str | Path) -> RunConfig:
    merged = load_yaml(base)
    for overlay in overlays:
        merged = deep_merge(merged, load_yaml(overlay))
    return RunConfig.model_validate(_interpolate(merged))
