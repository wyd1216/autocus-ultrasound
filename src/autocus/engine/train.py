from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from autocus.config import load_config
from autocus.models.factory import create_model


def build_training_components(config_path: str | Path) -> dict[str, Any]:
    """Build model/config objects used by public training templates."""
    cfg = load_config(config_path)
    model_cfg = dict(cfg.get("model", {}))
    name = model_cfg.pop("name")
    model = create_model(name, **model_cfg)
    optimizer_cfg = cfg.get("optimizer", {"type": "adamw", "lr": 1e-4})
    lr = float(optimizer_cfg.get("lr", 1e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return {"config": cfg, "model": model, "optimizer": optimizer}
