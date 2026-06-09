# Reproducibility Guide

The public workflow has three levels.

1. Smoke reproducibility: run the toy demo with CPU-only inference.
2. Structural reproducibility: instantiate each paper model and run forward-shape tests.
3. Experimental reproducibility: use public data recipes or approved private data with the sanitized configs in `configs/paper/`.

Paper-level numerical reproduction requires the frozen model weights and the same non-public evaluation cohorts. The repository records the expected weight filenames and checksum slots in `weights/registry.json`.
