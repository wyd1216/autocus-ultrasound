# Reproducibility Guide

The public workflow supports three reproducibility levels.

1. Execution reproducibility: run the bundled CPU demo and inspect the generated JSON, CSV, metrics, and stage images.
2. Structural reproducibility: instantiate each paper-associated model and run forward-shape tests.
3. Retraining reproducibility: use public data recipes or approved local data with the sanitized configs in `configs/paper/`.

Paper-level numerical reproduction requires the frozen model weights and the same non-public evaluation cohorts. These materials are not bundled with the public repository. The registry in `weights/registry.json` records the expected checkpoint names and marks the default paper entries as `not_released`.

The public reproducibility claim is code-level and workflow-level: readers can inspect the implemented model families, instantiate each module, run example inference, validate manifests, and re-train with public or approved local data.
