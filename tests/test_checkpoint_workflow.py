from __future__ import annotations

import hashlib
import json
import torch
from typer.testing import CliRunner

from autocus.checkpoints import load_model_from_config
from autocus.models.factory import create_model
from autocus.weights import verify_registered_weights
from autocus.cli import app


def test_load_model_from_config_accepts_common_checkpoint_shapes(tmp_path):
    cfg = tmp_path / "focusnet.yaml"
    cfg.write_text(
        "model:\n"
        "  name: focusnet\n"
        "  in_channels: 1\n"
        "  num_classes: 1\n"
        "  fusion_channels: 16\n"
        "  pretrained_backbone: false\n",
        encoding="utf-8",
    )
    reference = create_model("focusnet", in_channels=1, num_classes=1, fusion_channels=16)
    checkpoint = tmp_path / "focusnet.pth"
    torch.save({"model_state_dict": reference.state_dict(), "meta": {"module": "focusnet"}}, checkpoint)

    loaded = load_model_from_config(cfg, checkpoint=checkpoint, device="cpu", strict=True)

    assert loaded.model.__class__.__name__ == "FocusNet"
    assert loaded.model.training is False
    assert loaded.missing_keys == []
    assert loaded.unexpected_keys == []
    assert loaded.metadata["module"] == "focusnet"


def test_load_model_from_config_strips_module_prefix(tmp_path):
    cfg = tmp_path / "focusnet.yaml"
    cfg.write_text(
        "model:\n"
        "  name: focusnet\n"
        "  in_channels: 1\n"
        "  num_classes: 1\n"
        "  fusion_channels: 16\n",
        encoding="utf-8",
    )
    reference = create_model("focusnet", in_channels=1, num_classes=1, fusion_channels=16)
    prefixed = {f"module.{key}": value for key, value in reference.state_dict().items()}
    checkpoint = tmp_path / "focusnet_module_prefix.pth"
    torch.save({"state_dict": prefixed}, checkpoint)

    loaded = load_model_from_config(cfg, checkpoint=checkpoint, device="cpu", strict=True)

    assert loaded.missing_keys == []
    assert loaded.unexpected_keys == []


def test_weight_registry_reports_checksum_status(tmp_path):
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    blob = weight_dir / "demo.bin"
    blob.write_bytes(b"autocus")
    digest = hashlib.sha256(b"autocus").hexdigest()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"weights": [{"module": "demo", "filename": "demo.bin", "sha256": digest}]}),
        encoding="utf-8",
    )

    assert verify_registered_weights(registry, weight_dir) == {"demo.bin": "ok"}
    blob.write_bytes(b"changed")
    assert verify_registered_weights(registry, weight_dir) == {"demo.bin": "sha256-mismatch"}


def test_default_weight_registry_treats_unreleased_weights_as_optional():
    statuses = verify_registered_weights("weights/registry.json", "weights")

    assert statuses
    assert set(statuses.values()) == {"not-released"}


def test_cli_model_smoke_loads_checkpoint(tmp_path):
    cfg = tmp_path / "focusnet.yaml"
    cfg.write_text(
        "model:\n"
        "  name: focusnet\n"
        "  in_channels: 1\n"
        "  num_classes: 1\n"
        "  fusion_channels: 16\n",
        encoding="utf-8",
    )
    model = create_model("focusnet", in_channels=1, num_classes=1, fusion_channels=16)
    checkpoint = tmp_path / "focusnet.pth"
    torch.save(model.state_dict(), checkpoint)

    result = CliRunner().invoke(
        app,
        ["model-smoke", "--config", str(cfg), "--checkpoint", str(checkpoint), "--device", "cpu"],
    )

    assert result.exit_code == 0, result.output
    assert "FocusNet" in result.output
