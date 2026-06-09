# Reproducibility Guide

The public workflow has three levels.

1. Smoke reproducibility: run the toy demo with CPU-only inference.
2. Structural reproducibility: instantiate each paper model and run forward-shape tests.
3. Experimental reproducibility: use public data recipes or approved private data with the sanitized configs in `configs/paper/`.

Paper-level numerical reproduction requires the frozen model weights and the same non-public evaluation cohorts. These materials are not bundled with the public repository. The registry in `weights/registry.json` records the expected checkpoint names and marks the default paper entries as `not_released`.

The intended public reproducibility claim is therefore code-level and workflow-level: readers can inspect the implemented model families, instantiate every module, run toy inference, validate manifests, and re-train with public or approved local data.
