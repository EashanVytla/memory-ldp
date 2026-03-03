"""
Diffusion Transformer with Perception Attention.
Adds cross-attention where Q = decoder output, K/V = per_tokens (tilde_p).
Perception attention runs after decoder, before final LN and head.
"""

from typing import Union, Optional, Tuple

import logging
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class TransformerForDiffusionWithPerceptionAttention(TransformerForDiffusion):
    """
    Extends TransformerForDiffusion with Perception Attention.
    per_tokens (tilde_p): (B, N_per, per_dim) used as K, V in cross-attention.
    Runs after decoder, before ln_f and head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        per_token_dim: int = 256,
        n_per_tokens: int = 256,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )
        self.per_token_dim = per_token_dim
        self.n_per_tokens = n_per_tokens
        self.per_proj = nn.Linear(per_token_dim, n_emb)
        self.perception_attn = nn.MultiheadAttention(
            embed_dim=n_emb,
            num_heads=n_head,
            dropout=p_drop_attn,
            batch_first=True,
        )
        self.perception_attn_norm = nn.LayerNorm(n_emb)
        for m in [self.per_proj, self.perception_attn]:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        torch.nn.init.ones_(self.perception_attn_norm.weight)
        torch.nn.init.zeros_(self.perception_attn_norm.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        per_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        per_tokens: (B, N_per, per_dim) memory-augmented perceptual tokens (tilde_p).
        """
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            cond_embeddings = time_emb
            if self.obs_as_cond and cond is not None:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(
                tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask
            )
            if per_tokens is not None:
                per_emb = self.per_proj(per_tokens)
                attn_out, _ = self.perception_attn(
                    query=x, key=per_emb, value=per_emb, need_weights=False
                )
                x = self.perception_attn_norm(x + attn_out)

        x = self.ln_f(x)
        x = self.head(x)
        return x
