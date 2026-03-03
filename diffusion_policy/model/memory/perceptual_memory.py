"""
Perceptual memory bank components ported from memory_vla.
Pipeline: p (current perceptual tokens) -> retrieve from bank -> GateFusion(p, retrieved) -> tilde_p
Bank stores tilde_p (update_fused=True). When full, merge (FIFO or ToMe).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(next(self.mlp.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            next(self.mlp.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossTransformerBlock(nn.Module):
    """Cross-attention block for memory retrieval: query attends to k, v."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        query: torch.Tensor,  # (B, N, D)
        k: torch.Tensor,  # (B, M, D)
        v: torch.Tensor,  # (B, M, D)
    ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = self.attn_norm(query + attn_out)
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class BottleneckSE(nn.Module):
    """Perceptual compression: patch tokens (B, N, C_in) -> (B, N, C_out) with spatial SE."""

    def __init__(self, C_in: int, C_mid: int, C_out: int):
        super().__init__()
        self.C_in = C_in
        self.C_mid = C_mid
        self.C_out = C_out
        excite_mid = max(C_mid // 16, 1)
        self.reduce = nn.Conv2d(C_in, C_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, excite_mid, 1),
            nn.ReLU(),
            nn.Conv2d(excite_mid, C_mid, 1),
            nn.Sigmoid(),
        )
        self.expand = nn.Conv2d(C_mid, C_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _n, _c = x.shape
        _h = _w = int(math.sqrt(_n))
        assert _h * _w == _n, "Input feature has no spatial structure"
        x = x.reshape(_b, _h, _w, _c).permute(0, 3, 1, 2)  # (B, C_in, H, W)
        z = self.act(self.reduce(x))
        w = self.excite(z)
        final = self.expand(z * w)
        final = final.reshape(_b, self.C_out, _n).permute(0, 2, 1)
        return final


class GateFusion(nn.Module):
    """Gating: scale*x1 + (1-scale)*x2 where scale = sigmoid(proj(concat(x1,x2)))."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.proj(torch.cat([x1, x2], dim=-1)))
        return scale * x1 + (1 - scale) * x2


class PerMemBank(nn.Module):
    """
    Perceptual memory bank: retrieve, fuse (GateFusion), store tilde_p when
    update_fused=True. When bank is full, consolidate via FIFO or ToMe.
    """

    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = True,
    ):
        super().__init__()
        assert dataloader_type in ("stream", "group")
        assert fusion_type in ("gate", "add")
        assert consolidate_type in ("fifo", "tome")

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.retrieval_blocks = nn.ModuleList(
            [CrossTransformerBlock(self.token_size) for _ in range(self.retrieval_layers)]
        )
        if self.fusion_type == "gate":
            self.gate_fusion_blocks = GateFusion(self.token_size)
        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                self.token_size, frequency_embedding_size=max(self.token_size // 4, 1)
            )
        else:
            self.timestep_encoder = None

        self.reset()

    def reset(self):
        self.bank: dict = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id: int):
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return
        feats = [feat for (_, feat) in bank]
        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
            f2 = feats[i + 1].flatten(1) if feats[i + 1].dim() > 1 else feats[i + 1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())
        idx_max = int(torch.tensor(sims).argmax().item())
        timestep_i, feat_i = bank[idx_max]
        _, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)
        bank[idx_max] = (timestep_i, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(
        self,
        episode_id: int,
        feat: torch.Tensor,
        timestep: Optional[torch.Tensor],
    ):
        if episode_id not in self.bank:
            self.bank[episode_id] = []
        self.bank[episode_id].append((timestep, feat.detach().clone()))
        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length :]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError

    def process_batch(
        self,
        tokens: torch.Tensor,  # [B, N, D]
        episode_ids: np.ndarray,
        timesteps: np.ndarray,
    ) -> torch.Tensor:
        """p -> retrieve, fuse -> tilde_p; store tilde_p in bank if update_fused."""
        assert episode_ids is not None, "episode_ids must be provided"
        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided when use_timestep_pe"

        B, N, D = tokens.shape
        outputs = []

        if self.training:
            if self.dataloader_type == "group":
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == "stream":
                first_eid = int(episode_ids[0])
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(B):
            eid = int(episode_ids[i])
            if self.training:
                if self.dataloader_type == "group":
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = int(episode_ids[i - self.group_size])
                        self.clear_episode(prev_group_eid)
                if self.dataloader_type == "stream":
                    if i > 0 and int(episode_ids[i]) != int(episode_ids[i - 1]):
                        self.clear_episode(int(episode_ids[i - 1]))
                        self.eid_stream = int(episode_ids[i])

            working_mem = tokens[i].unsqueeze(0)
            hist = self.bank.get(eid, [])
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)
                if self.use_timestep_pe:
                    hist_timesteps = [t for t, _ in hist]
                    hist_ts = torch.tensor(hist_timesteps, dtype=torch.float32, device=working_mem.device)
                    pe = self.timestep_encoder(hist_ts).unsqueeze(0)
                    pe = pe.repeat_interleave(N, dim=1)
                else:
                    pe = torch.zeros_like(episode_mem)
                query = working_mem
                for block in self.retrieval_blocks:
                    query = block(query, episode_mem + pe, episode_mem)
                retrieved_episode_mem = query
            else:
                retrieved_episode_mem = working_mem

            if self.fusion_type == "add":
                fused_feats = (working_mem + retrieved_episode_mem) * 0.5
            else:
                fused_feats = self.gate_fusion_blocks(working_mem, retrieved_episode_mem)
            outputs.append(fused_feats)

            timestep_i = float(timesteps[i]) if (self.use_timestep_pe and timesteps is not None) else None
            if self.update_fused:
                self._memory_consolidate(eid, fused_feats.squeeze(0), timestep_i)
            else:
                self._memory_consolidate(eid, tokens[i], timestep_i)

        return torch.cat(outputs, dim=0)


