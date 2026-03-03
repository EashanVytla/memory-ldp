# LDP + Perceptual Memory

## Project Goal

Testing whether MemoryVLA's perceptual-token memory strategy solves spatial recall in robot manipulation. We strip MemoryVLA down to just its perceptual memory module and plug it into a diffusion policy (LDP), then evaluate on:
1. **Spatial Recall:** pick up object → move to corner → return to exact spawn location
2. **Object Permanence:** remember which shelf an object is in after closing both shelves

## Paper Reference: MemoryVLA (arXiv:2508.19236, ICLR 2026)

Key architecture: at each timestep, DINOv2 patch tokens (256 × d) are stored in a fixed-capacity memory bank (L=16). Retrieved via dot-product attention with sinusoidal timestep positional encoding, fused back via a learned gate `g = sigmoid(MLP(concat[x, H])); x̃ = g⊙H + (1-g)⊙x`. When bank is full, token-merge consolidation finds the most similar consecutive pair and replaces it with their mean (beats FIFO by ~5pts). Memory-augmented tokens condition the diffusion denoiser via cross-attention.

Our simplification: no LLM, no cognitive token — just the perceptual path.

## What's Been Built

| File | Description |
|------|-------------|
| [diffusion_policy/model/memory/perceptual_memory.py](diffusion_policy/model/memory/perceptual_memory.py) | `PerMemBank`, `GateFusion`, `CrossTransformerBlock`, `BottleneckSE`, `TimestepEmbedder` |
| [diffusion_policy/model/vision/dinov2_encoder.py](diffusion_policy/model/vision/dinov2_encoder.py) | DinoV2 wrapper → (B, 256, 768) patch tokens via timm |
| [diffusion_policy/policy/diffusion_transformer_hybrid_image_policy_with_memory.py](diffusion_policy/policy/diffusion_transformer_hybrid_image_policy_with_memory.py) | Main policy: DinoV2 → BottleneckSE → PerMemBank → diffusion |
| [diffusion_policy/model/diffusion/transformer_for_diffusion_with_perception.py](diffusion_policy/model/diffusion/transformer_for_diffusion_with_perception.py) | Extends `TransformerForDiffusion` with perception cross-attention after decoder |
| [experiment_configs/aloha/transformer_aloha_perceptual_memory.yaml](experiment_configs/aloha/transformer_aloha_perceptual_memory.yaml) | Training config: ALOHA pickandplace twomodes, 3500 epochs, batch 64 |

## Codebase Conventions

- **Entry point:** `train.py` with Hydra config
- **Dataset:** `RobomimicReplayImageDataset` from `data/aloha_twomodes_single/demos.hdf5`
- **Obs keys:** `top` (3,84,84), `right_wrist` (3,84,84), `qpos` (7,); actions (7,)
- **Eval:** `AlohaImageRunner` runs vectorized sim rollouts; `policy.reset()` must be called between episodes
- **Configs:** `experiment_configs/aloha/<name>.yaml`, WandB project `diffusion_aloha`
- DinoV2 expects 224×224 — resizing from 84×84 is handled inside the encoder
- `group_size` in memory config should align with `n_obs_steps` (both 16)
- `consolidate_type: fifo` is current; `tome` (token-merge) is implemented but untested

## Next Steps

1. Train memory policy on ALOHA long-horizon task, compare vs baseline LDP
2. Modify task success criterion to use L2 distance from spawn position (spatial recall)
3. Build object permanence task (two shelves, one hidden object)
4. Ablate: ToMe vs FIFO, memory length L, timestep PE, gate vs additive fusion
