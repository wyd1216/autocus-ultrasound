from __future__ import annotations

import json
from pathlib import Path

import torch
from typer.testing import CliRunner

from autocus.cli import app
from autocus.config import load_config
from autocus.data.manifest import load_manifest
from autocus.models.factory import create_model
from autocus.pipelines.autocus import run_pipeline
from scripts.audit_release import audit_paths


def test_paper_configs_have_no_private_paths():
    offenders = audit_paths(Path.cwd())
    assert offenders == []


def test_manifest_schema_accepts_relative_entries():
    manifest = load_manifest(Path('examples/toy_manifest.json'))
    assert 'test' in manifest.splits
    assert manifest.splits['test'][0].image == 'sample_input/demo_ultrasound.png'


def test_factory_creates_paper_models_with_small_tensors():
    cases = [
        ('focusnet', {'pretrained_backbone': False}, torch.rand(1, 1, 64, 64)),
        ('aarformer', {'embed_dims': [8, 16, 32, 64], 'depths': [1, 1, 1, 1], 'decoder_depths': [1, 1, 1], 'bottleneck_depth': 1, 'num_heads': 4}, torch.rand(1, 1, 32, 32)),
        ('cuhat', {'embed_dim': 12, 'num_groups': 1, 'num_blocks_per_group': 1, 'num_heads': 3, 'use_ocab': False}, torch.rand(1, 1, 32, 32)),
        ('larsnet_v5', {'pretrained': False}, torch.rand(1, 1, 64, 64)),
        ('plaque_net_v1', {'pretrained': False}, torch.rand(1, 1, 64, 64)),
        ('plaque_senet', {'pretrained': False, 'use_msf': True, 'use_cbam': True, 'use_dsv': True}, torch.rand(1, 1, 96, 96)),
    ]
    for name, kwargs, x in cases:
        model = create_model(name, **kwargs)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output is not None


def test_cli_help_and_demo_pipeline(tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(app, ['--help'])
    assert help_result.exit_code == 0

    out_dir = tmp_path / 'demo'
    result = runner.invoke(
        app,
        [
            'infer',
            '--config',
            'configs/paper/autocus_pipeline.yaml',
            '--input',
            'examples/sample_input',
            '--output',
            str(out_dir),
            '--device',
            'cpu',
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out_dir / 'pipeline_result.json').read_text())
    assert payload['images'][0]['plaque_presence_score'] >= 0.0
    assert payload['images'][0]['iqe_outputs']['norm_method'] == 'iqe_norm_traditional'
    assert (out_dir / 'predictions.csv').exists()


def test_cli_demo_command_writes_expected_artifacts(tmp_path):
    out_dir = tmp_path / 'demo'
    result = CliRunner().invoke(app, ['demo', '--output', str(out_dir), '--device', 'cpu'])

    assert result.exit_code == 0, result.output
    assert (out_dir / 'pipeline_result.json').exists()
    assert (out_dir / 'predictions.csv').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'stage_outputs' / 'demo_ultrasound_iqe_norm.png').exists()
    assert (out_dir / 'stage_outputs' / 'demo_ultrasound_artery_mask.png').exists()
    assert (out_dir / 'stage_outputs' / 'demo_ultrasound_plaque_mask.png').exists()


def test_config_loader_and_programmatic_pipeline(tmp_path):
    cfg = load_config(Path('configs/paper/autocus_pipeline.yaml'))
    result = run_pipeline(cfg, Path('examples/sample_input'), tmp_path, device='cpu')
    assert result['images']
    assert Path(result['images'][0]['input']).name == 'demo_ultrasound.png'
