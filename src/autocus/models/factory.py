from __future__ import annotations

from typing import Any


def create_model(name: str, **kwargs: Any):
    """Create a paper-relevant AutoCUS model by public name."""
    normalized = name.lower().replace("-", "_")
    if normalized == "focusnet":
        from autocus.models.roi.focusnet import FocusNet

        return FocusNet(**kwargs)
    if normalized == "aarformer":
        from autocus.models.iqe.aarformer import AARFormer

        return AARFormer(**kwargs)
    if normalized == "cuhat":
        from autocus.models.iqe.cuhat import CUHAT

        return CUHAT(**kwargs)
    if normalized == "larsnet_v5":
        from autocus.models.segmentation.larsnet import LARSNetV5

        if "out_channels" in kwargs and "num_classes" not in kwargs:
            kwargs["num_classes"] = kwargs.pop("out_channels")
        return LARSNetV5(**kwargs)
    if normalized == "plaque_net_v1":
        from autocus.models.segmentation.plaque_net import PlaqueNetV1

        if "out_channels" in kwargs and "num_classes" not in kwargs:
            kwargs["num_classes"] = kwargs.pop("out_channels")
        return PlaqueNetV1(**kwargs)
    if normalized == "plaque_senet":
        from autocus.models.classification.plaque_senet import PlaqueSENet

        return PlaqueSENet(**kwargs)
    raise ValueError(f"Unknown AutoCUS model: {name}")


MODEL_NAMES = ["focusnet", "aarformer", "cuhat", "larsnet_v5", "plaque_net_v1", "plaque_senet"]
