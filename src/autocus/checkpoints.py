from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from autocus.config import load_config
from autocus.models.factory import create_model


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    checkpoint: Path | None
    missing_keys: list[str]
    unexpected_keys: list[str]
    metadata: dict[str, Any]


def _extract_state_dict(blob: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(blob, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = blob.get(key)
            if isinstance(value, dict):
                metadata = blob.get("meta") or blob.get("metadata") or {}
                return value, dict(metadata) if isinstance(metadata, dict) else {}
        if blob and all(isinstance(key, str) for key in blob):
            tensor_values = [value for value in blob.values() if torch.is_tensor(value)]
            if tensor_values:
                return blob, {}
    raise ValueError("Checkpoint must be a state_dict or contain model_state_dict/state_dict/model")


def strip_state_dict_prefix(state_dict: dict[str, torch.Tensor], prefix: str = "module.") -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}


def load_checkpoint_state(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(checkpoint)
    blob = torch.load(path, map_location=map_location, weights_only=False)
    state_dict, metadata = _extract_state_dict(blob)
    return strip_state_dict_prefix(state_dict), metadata


def load_model_from_config(
    config: str | Path,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> LoadedModel:
    cfg = load_config(config)
    model_cfg = dict(cfg.get("model", {}))
    if "name" not in model_cfg:
        raise ValueError("Config requires model.name")
    name = model_cfg.pop("name")
    model = create_model(name, **model_cfg)
    model.to(device)
    missing: list[str] = []
    unexpected: list[str] = []
    metadata: dict[str, Any] = {}
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    if checkpoint_path is not None:
        state_dict, metadata = load_checkpoint_state(checkpoint_path, map_location=device)
        incompatible = model.load_state_dict(state_dict, strict=strict)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
    model.eval()
    return LoadedModel(
        model=model,
        checkpoint=checkpoint_path,
        missing_keys=missing,
        unexpected_keys=unexpected,
        metadata=metadata,
    )
