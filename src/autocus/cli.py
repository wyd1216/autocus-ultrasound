from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from autocus.checkpoints import load_model_from_config
from autocus.config import load_config
from autocus.data.manifest import write_manifest
from autocus.engine.evaluate import evaluate_from_manifest
from autocus.engine.train import build_training_components
from autocus.paper.ablation import write_iqe_ablation_template
from autocus.pipelines.autocus import run_pipeline
from autocus.weights import download_registered_weights, verify_registered_weights

app = typer.Typer(help="AutoCUS public research framework")
data_app = typer.Typer(help="Data recipe and manifest utilities")
weights_app = typer.Typer(help="External weight utilities")
paper_app = typer.Typer(help="Paper reproduction utilities")
app.add_typer(data_app, name="data")
app.add_typer(weights_app, name="weights")
app.add_typer(paper_app, name="paper")
console = Console()


@app.command("model-smoke")
def model_smoke(
    config: Path = typer.Option(..., exists=True),
    checkpoint: Path | None = typer.Option(None, exists=True),
    device: str = typer.Option("cpu"),
    strict: bool = typer.Option(True),
) -> None:
    """Instantiate a model config and optionally load a checkpoint."""
    loaded = load_model_from_config(config, checkpoint=checkpoint, device=device, strict=strict)
    console.print({
        "model": loaded.model.__class__.__name__,
        "checkpoint": str(loaded.checkpoint) if loaded.checkpoint else None,
        "missing_keys": loaded.missing_keys,
        "unexpected_keys": loaded.unexpected_keys,
        "metadata": loaded.metadata,
    })


@app.command()
def train(config: Path = typer.Option(..., exists=True)) -> None:
    """Build training components for a sanitized paper config."""
    comps = build_training_components(config)
    console.print({"model": comps["model"].__class__.__name__, "config": str(config)})


@app.command()
def evaluate(
    config: Path = typer.Option(..., exists=True),
    checkpoint: Path | None = typer.Option(None),
    manifest: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("outputs/eval")),
) -> None:
    """Evaluate a manifest and write a metrics JSON skeleton."""
    _ = load_config(config)
    _ = checkpoint
    metrics = evaluate_from_manifest(manifest, output)
    console.print(metrics)


@app.command()
def infer(
    config: Path = typer.Option(..., exists=True),
    input: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("outputs/demo")),
    device: str = typer.Option("cpu"),
) -> None:
    """Run the public AutoCUS demo inference pipeline."""
    cfg = load_config(config)
    payload = run_pipeline(cfg, input, output, device=device)
    console.print({"output": str(output), "num_images": len(payload["images"])})


@app.command()
def demo(
    output: Path = typer.Option(Path("outputs/demo")),
    device: str = typer.Option("cpu"),
) -> None:
    """Run the bundled CPU demo with the paper pipeline config and sample input."""
    cfg = load_config(Path("configs/paper/autocus_pipeline.yaml"))
    payload = run_pipeline(cfg, Path("examples/sample_input"), output, device=device)
    console.print({"output": str(output), "num_images": len(payload["images"])})


@data_app.command("build-manifest")
def build_manifest(
    recipe: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("examples/toy_manifest.json")),
) -> None:
    """Build a minimal manifest from a JSON recipe."""
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    write_manifest(output, payload)
    console.print({"manifest": str(output)})


@weights_app.command("download")
def weights_download(registry: Path = typer.Option(Path("weights/registry.json"))) -> None:
    files = download_registered_weights(registry)
    console.print({"downloaded": [str(p) for p in files]})


@weights_app.command("verify")
def weights_verify(registry: Path = typer.Option(Path("weights/registry.json"))) -> None:
    console.print(verify_registered_weights(registry))


@paper_app.command("ablate-iqe")
def ablate_iqe(
    manifest: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("outputs/iqe_ablation")),
) -> None:
    console.print(write_iqe_ablation_template(manifest, output))


if __name__ == "__main__":
    app()
