"""
DinoV2 vision encoder for patch-level perceptual tokens.
Outputs patch tokens (B, N_patches, D) suitable for BottleneckSE and obs conditioning.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def _get_dinov2_patch_tokens(
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
    pretrained: bool = True,
):
    """Load DinoV2 and return model. Use forward_features and slice CLS for patch tokens."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for DinoV2. Install with: pip install timm")
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, global_pool=""
    )
    return model


class DinoV2Encoder(ModuleAttrMixin):
    """
    DinoV2 wrapper that returns patch tokens (B, N, D).
    Input: (B, C, H, W) in [0, 1] or ImageNet-normalized.
    Output: (B, N_patches, D_vit) patch tokens.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        img_size: Tuple[int, int] = (224, 224),
        imagenet_norm: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.imagenet_norm = imagenet_norm
        self.backbone = _get_dinov2_patch_tokens(model_name, pretrained)
        if imagenet_norm:
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )
        self._embed_dim = self.backbone.embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W), C=3, values in [0, 1] typically
        Returns: (B, N_patches, D) patch tokens
        """
        B, C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            x = torch.nn.functional.interpolate(
                x, size=self.img_size, mode="bilinear", align_corners=False
            )
        if self.imagenet_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        out = self.backbone(x)
        if out.shape[1] > 1:
            patch_tokens = out[:, 1:, :]
        else:
            patch_tokens = out
        return patch_tokens


def create_dinov2_encoder(
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
    pretrained: bool = True,
    img_size: Tuple[int, int] = (224, 224),
    imagenet_norm: bool = True,
) -> DinoV2Encoder:
    """Factory for DinoV2Encoder."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for DinoV2. Install with: pip install timm")

    return DinoV2Encoder(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
        imagenet_norm=imagenet_norm,
    )
