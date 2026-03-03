"""
Diffusion Transformer Hybrid Image Policy with Perceptual Memory.
- DinoV2 replaces robomimic ResNet for vision
- Perceptual path: Image -> DinoV2 -> BottleneckSE -> p -> PerMemBank -> tilde_p
- tilde_p conditions diffusion via Perception Attention (K, V)
- obs cond: DinoV2 patch tokens pooled per image -> cond
"""

from typing import Dict, Tuple, Optional

import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion_with_perception import (
    TransformerForDiffusionWithPerceptionAttention,
)
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.dinov2_encoder import DinoV2Encoder
from diffusion_policy.model.memory import BottleneckSE, PerMemBank
from diffusion_policy.common.pytorch_util import dict_apply
import numpy as np


class DiffusionTransformerHybridImagePolicyWithMemory(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        crop_shape: Tuple[int, int] = (76, 76),
        n_layer: int = 8,
        n_cond_layers: int = 0,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        obs_as_cond: bool = True,
        pred_action_steps_only: bool = False,
        past_action_pred: bool = False,
        past_steps_reg: int = -1,
        dinov2_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        dinov2_pretrained: bool = True,
        dinov2_img_size: Tuple[int, int] = (224, 224),
        per_token_size: int = 256,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        dataloader_type: str = "group",
        group_size: int = 16,
        consolidate_type: str = "fifo",
        update_fused: bool = True,
        obs_encoder_freeze: bool = False,
        obs_encoder_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.past_action_pred = past_action_pred
        self.past_steps_reg = past_steps_reg

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        rgb_keys = [k for k, v in obs_shape_meta.items() if v.get("type") == "rgb"]
        lowdim_keys = [k for k, v in obs_shape_meta.items() if v.get("type", "low_dim") == "low_dim"]
        self.rgb_keys = sorted(rgb_keys)
        self.lowdim_keys = sorted(lowdim_keys)
        self.num_rgb = len(self.rgb_keys)

        self.vision_encoder = DinoV2Encoder(
            model_name=dinov2_model_name,
            pretrained=dinov2_pretrained,
            img_size=dinov2_img_size,
            imagenet_norm=True,
        )
        vision_dim = self.vision_encoder.embed_dim
        obs_feature_dim = self.num_rgb * vision_dim
        if self.lowdim_keys:
            lowdim_dim = sum(
                int(np.prod(obs_shape_meta[k]["shape"])) for k in self.lowdim_keys
            )
            obs_feature_dim += lowdim_dim

        num_patches = self.vision_encoder.backbone.patch_embed.num_patches
        self.per_compr = BottleneckSE(
            C_in=self.num_rgb * vision_dim,
            C_mid=per_token_size * 2,
            C_out=per_token_size,
        )
        self.per_mem_bank = PerMemBank(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=per_token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=True,
            fusion_type="gate",
            consolidate_type=consolidate_type,
            update_fused=update_fused,
        )

        cond_dim = obs_feature_dim if obs_as_cond else 0
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim

        self.model = TransformerForDiffusionWithPerceptionAttention(
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
            per_token_dim=per_token_size,
            n_per_tokens=num_patches,
        )
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.per_token_size = per_token_size
        self.kwargs = kwargs
        self.dinov2_img_size = dinov2_img_size

        if obs_encoder_dir:
            self._load_encoder(obs_encoder_dir, **kwargs)
        if obs_encoder_freeze:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.num_inference_steps = (
            num_inference_steps or noise_scheduler.config.num_train_timesteps
        )

    def _load_encoder(self, path: str, **kwargs):
        import dill
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        if "state_dicts" in payload and "obs_encoder" in payload["state_dicts"]:
            self.vision_encoder.load_state_dict(
                payload["state_dicts"]["obs_encoder"], strict=False, **kwargs
            )
        elif "state_dicts" in payload and "model" in payload["state_dicts"]:
            sd = {k.replace("obs_encoder.", ""): v for k, v in payload["state_dicts"]["model"].items() if k.startswith("obs_encoder")}
            if sd:
                self.vision_encoder.load_state_dict(sd, strict=False, **kwargs)

    def _encode_all_images(
        self, nobs: Dict[str, torch.Tensor], B: int, To: int
    ):
        """Single DINOv2 pass for all cameras/timesteps.

        Returns:
            obs_cond: (B, To, obs_feature_dim) for diffusion conditioning
            last_patches: (B, N, num_rgb * vision_dim) last-frame patches from
                          all cameras concatenated along the feature axis, for
                          the memory compression path.
        """
        pooled_list = []
        last_patches_list = []
        for key in self.rgb_keys:
            img = nobs[key]
            flat = img[:, :To].reshape(-1, *img.shape[2:])  # (B*To, 3, H, W)
            patch_tokens = self.vision_encoder(flat)         # (B*To, N, D)
            N = patch_tokens.shape[1]
            pooled = reduce(patch_tokens, "b n d -> b d", "mean")  # (B*To, D)
            pooled_list.append(pooled.reshape(B, To, -1))          # (B, To, D)
            last = patch_tokens.reshape(B, To, N, -1)[:, -1]       # (B, N, D)
            last_patches_list.append(last)
        cond_parts = pooled_list
        if self.lowdim_keys:
            ld = torch.cat([nobs[k][:, :To] for k in self.lowdim_keys], dim=-1)
            cond_parts.append(ld)
        obs_cond = torch.cat(cond_parts, dim=-1)              # (B, To, obs_feature_dim)
        last_patches = torch.cat(last_patches_list, dim=-1)   # (B, N, num_rgb*D)
        return obs_cond, last_patches

    def _get_perceptual_tokens(
        self,
        last_patches: torch.Tensor,
        episode_ids: np.ndarray,
        timesteps: np.ndarray,
    ) -> torch.Tensor:
        p = self.per_compr(last_patches)
        tilde_p = self.per_mem_bank.process_batch(p, episode_ids, timesteps)
        return tilde_p

    def reset(self):
        self.per_mem_bank.reset()

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        per_tokens: Optional[torch.Tensor] = None,
        generator=None,
        act: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            if act is not None:
                trajectory[:, : act.shape[1]] = act
            model_output = self.model(trajectory, t, cond=cond, per_tokens=per_tokens)
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        act_cond: Optional[torch.Tensor] = None,
        episode_ids: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        obs = obs_dict["obs"]
        nobs = self.normalizer.normalize(obs)
        B = nobs[self.rgb_keys[0]].shape[0]
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        cond, last_patches = self._encode_all_images(nobs, B, To)
        if episode_ids is None:
            episode_ids = np.zeros(B, dtype=np.int64)
        if timesteps is None:
            timesteps = np.zeros(B, dtype=np.float32)
        per_tokens = self._get_perceptual_tokens(last_patches, episode_ids, timesteps)

        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if act_cond is not None:
            act_cond = self.normalizer["action"].normalize(act_cond)

        nsample = self.conditional_sample(
            cond_data, cond_mask, cond=cond, per_tokens=per_tokens, act=act_cond, **self.kwargs
        )
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.vision_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay,
        })
        optim_groups.append({
            "params": list(self.per_compr.parameters()) + list(self.per_mem_bank.parameters()),
            "weight_decay": 0.0,
        })
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch: Dict, debug: bool = False) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        To = self.n_obs_steps
        episode_ids = batch["episode_ids"]
        timesteps = batch["timesteps"]
        if hasattr(episode_ids, "cpu"):
            episode_ids = episode_ids.cpu().numpy()
        if hasattr(timesteps, "cpu"):
            timesteps = timesteps.cpu().numpy()

        cond, last_patches = self._encode_all_images(nobs, batch_size, To)
        per_tokens = self._get_perceptual_tokens(last_patches, episode_ids, timesteps)

        trajectory = nactions
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps_rand = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps_rand)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy_trajectory, timesteps_rand, cond=cond, per_tokens=per_tokens)
        pred_type = self.noise_scheduler.config.prediction_type
        target = noise if pred_type == "epsilon" else trajectory
        if pred_type != "epsilon" and pred_type != "sample":
            raise ValueError(f"Unsupported prediction type {pred_type}")

        if not self.past_action_pred:
            pred = pred[:, self.n_obs_steps - 1 :]
            target = target[:, self.n_obs_steps - 1 :]
            loss_mask = loss_mask[:, self.n_obs_steps - 1 :]
        if self.past_steps_reg != -1:
            pred = pred[:, self.n_obs_steps - self.past_steps_reg - 1 :]
            target = target[:, self.n_obs_steps - self.past_steps_reg - 1 :]
            loss_mask = loss_mask[:, self.n_obs_steps - self.past_steps_reg - 1 :]

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()
        return loss
