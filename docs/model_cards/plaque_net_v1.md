# plaque_net_v1 Model Card

## Intended Use

Research reproduction for the AutoCUS carotid-ultrasound paper workflow.

## Inputs

Grayscale B-mode ultrasound images normalized to the module-specific config.

## Outputs

Module-specific tensors or structured pipeline outputs documented in the README.

## Weights

Frozen paper weights are tracked through `weights/registry.json` and hosted outside Git.

## Limitations

Not intended for clinical diagnosis. Performance depends on acquisition domain, preprocessing, and validation cohort.
