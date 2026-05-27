"""Torch Hub entrypoints for FoL.

Usage examples:
    # Default FoL model: ViT-L + pretrained FoL_large.pth
    model = torch.hub.load("chenshunpeng/FoL", "FoL", pretrained=True, trust_repo=True)

    # Explicit ViT-L entrypoint
    model = torch.hub.load("chenshunpeng/FoL", "fol_vitl14", pretrained=True, trust_repo=True)

    # ViT-B entrypoint
    model = torch.hub.load("chenshunpeng/FoL", "fol_vitb14", pretrained=True, trust_repo=True)
"""

from typing import Dict

import torch

from network_FoL import FoLNet


dependencies = ["torch"]


_WEIGHT_URLS: Dict[str, str] = {
    "vitl": "https://huggingface.co/shunpeng/FoL/resolve/main/FoL_large.pth",
    "vitb": "https://huggingface.co/shunpeng/FoL/resolve/main/FoL_base.pth",
}

_MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    # num_channels 同时传给 FoL 聚合层和 network_FoL.py 第 18 行的 upconv 输入通道。
    "vitl": {"model_name": "dinov2_vitl14", "num_channels": 1024},
    "vitb": {"model_name": "dinov2_vitb14", "num_channels": 768},
}



def _build_fol(
    arch: str,
    pretrained: bool = True,
    progress: bool = True,
    map_location="cpu",
    strict: bool = False,
    num_trainable_blocks: int = 4,
):
    if arch not in _MODEL_CONFIGS:
        raise ValueError(f"Unsupported FoL architecture: {arch}. Choose from {sorted(_MODEL_CONFIGS)}.")

    config = _MODEL_CONFIGS[arch]


    model = FoLNet(
            num_channels=config["num_channels"],
            model_name=config["model_name"],
            num_trainable_blocks=num_trainable_blocks,
       )  

    if pretrained:
        # 下载对应架构的官方权重，并按 strict 参数决定是否严格校验所有键。
        checkpoint = torch.hub.load_state_dict_from_url(
            _WEIGHT_URLS[arch], map_location=map_location, progress=progress
        )
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=strict)
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}")

    return model


def FoL(
    pretrained: bool = True,
    backbone: str = "vitl",
    progress: bool = True,
    map_location="cpu",
    **kwargs,
):
    """Load FoL from Torch Hub.

    Args:
        pretrained: Download and load the official pretrained weights.
        backbone: 'vitl' for ViT-L (default) or 'vitb' for ViT-B.
        progress: Show download progress bars.
        map_location: Device mapping used while loading the checkpoint.
        **kwargs: Advanced options passed to the internal builder, such as
            strict, num_trainable_blocks, dinov2_repo, and dinov2_source.
    """
    return _build_fol(backbone, pretrained=pretrained, progress=progress, map_location=map_location, **kwargs)
