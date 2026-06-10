# PlaqueNetV1 Model Card

## Intended Use

Research reproduction for the AutoCUS carotid-ultrasound workflow.

## Inputs

Grayscale B-mode ultrasound images normalized to the module-specific config.

## Outputs

Module-specific tensors or structured pipeline outputs documented in the README.

## Weights

Paper checkpoints are tracked through `weights/registry.json` and are not released by default.

## Limitations

Not intended for clinical use. Performance depends on acquisition domain, preprocessing, and validation cohort.
