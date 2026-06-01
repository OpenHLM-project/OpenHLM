import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.rtc import InferenceTimeRTCConfig, InferenceTimeRTCProcessor


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim not in (1, 2):
        raise ValueError("The time tensor is expected to be of shape `(batch_size,)` or `(batch_size, action_horizon)`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    
    if time.ndim == 1:
        # Original case: (batch_size,) -> (batch_size, dimension)
        sin_input = scaling_factor[None, :] * time[:, None]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    else:
        # RTC case: (batch_size, action_horizon) -> (batch_size, action_horizon, dimension)
        # scaling_factor: (dimension//2,) -> (1, 1, dimension//2)
        # time: (batch_size, action_horizon) -> (batch_size, action_horizon, 1)
        sin_input = scaling_factor[None, None, :] * time[:, :, None]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=2)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def _cdist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Pairwise L2 distance: [B, N, D] x [B, M, D] -> [B, N, M].

    Uses dot-product formula for numerical stability (matches drifting-policy JAX impl).
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(
    gen: Tensor,
    fixed_pos: Tensor,
    fixed_neg: Tensor | None = None,
    weight_gen: Tensor | None = None,
    weight_pos: Tensor | None = None,
    weight_neg: Tensor | None = None,
    R_list: tuple[float, ...] = (0.02, 0.05, 0.2),  # noqa: N803
) -> tuple[Tensor, dict]:
    """Drifting loss (faithful PyTorch port of drifting-policy JAX implementation).

    Args:
        gen: [B, C_g, S] generated samples
        fixed_pos: [B, C_p, S] positive (real) samples
        fixed_neg: [B, C_n, S] negative samples (optional, None = no explicit negatives)
        weight_gen: [B, C_g] (optional, default 1)
        weight_pos: [B, C_p] (optional, default 1)
        weight_neg: [B, C_n] (optional, default 1)
        R_list: tuple of temperature values
    Returns:
        loss: [B]
        info: dict with 'scale' and 'loss_{R}' entries
    """
    B, C_g, S = gen.shape  # noqa: N806
    C_p = fixed_pos.shape[1]  # noqa: N806

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]  # noqa: N806

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        info = {}
        dist = _cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale

        scale_inputs = torch.clamp(scale / (S**0.5), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs

        dist_normed = dist / torch.clamp(scale, min=1e-3)

        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))
        block_mask = block_mask.unsqueeze(0)
        dist_normed = dist_normed + block_mask * mask_val

        force_across_R = torch.zeros_like(old_gen_scaled)  # noqa: N806

        for R in R_list:  # noqa: N806
            logits = -dist_normed / R

            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))

            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)  # noqa: N806

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)  # noqa: N806

            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs.unsqueeze(-1) * old_gen_scaled  # noqa: N806

            f_norm_val = (total_force_R**2).mean()
            info[f"loss_{R}"] = f_norm_val

            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale  # noqa: N806

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs.detach()
    diff = gen_scaled - goal_scaled.detach()
    loss = (diff**2).mean(dim=(-1, -2))

    info = {k: v.mean() for k, v in info.items()}

    return loss, info


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.training_compile_enabled = False
        self.inference_compile_enabled = False

        # Inference-time RTC processor will be initialized via init_inference_time_rtc_processor
        self.inference_time_rtc_config: InferenceTimeRTCConfig | None = None
        self.inference_time_rtc_processor: InferenceTimeRTCProcessor | None = None

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # Use config.action_dim if > 32, otherwise use default 32 (pretrained model limit)
        action_proj_dim = config.action_dim if config.action_dim > 32 else 32
        self.action_in_proj = nn.Linear(action_proj_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, action_proj_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.compile_for_inference()

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def compile_for_inference(self) -> None:
        """Compile the inference sampling path for faster action generation."""
        if self.inference_compile_enabled:
            return

        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        self.inference_compile_enabled = True

    def compile_for_training(self) -> None:
        """Compile the training forward path for faster training steps."""
        if self.training_compile_enabled:
            return

        self.forward = torch.compile(self.forward, mode="default")
        self.training_compile_enabled = True
        logging.info("Enabled torch.compile for the training forward path")

    def init_inference_time_rtc_processor(self, rtc_config: InferenceTimeRTCConfig) -> None:
        """Initialize inference-time RTC processor with the given config. 
        This method should be called after model instantiation to enable inference-time RTC.
        """
        self.inference_time_rtc_config = rtc_config
        self.inference_time_rtc_processor = InferenceTimeRTCProcessor(rtc_config)
        logging.info(f"Initialized inference-time RTC processor with config: enabled={rtc_config.enabled}, "
                     f"execution_horizon={rtc_config.execution_horizon}")

    def _inference_time_rtc_enabled(self) -> bool:
        """Check if inference-time RTC is enabled."""
        return self.inference_time_rtc_config is not None and self.inference_time_rtc_config.enabled and self.inference_time_rtc_processor is not None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        if self.config.timestep_distribution == "uniform":
            # Sample from uniform distribution [0, 1]
            time = torch.rand(bsize, dtype=torch.float32, device=device)
        elif self.config.timestep_distribution == "shifted_beta":
            # Sample from shifted beta distribution (default)
            time_beta = sample_beta(1.5, 1.0, bsize, device)
            time = time_beta * 0.999 + 0.001
        else:
            raise ValueError(
                f"Invalid timestep_distribution: {self.config.timestep_distribution}. "
                f"Must be 'shifted_beta' or 'uniform'."
            )
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # timestep shape: (batch_size,) or (batch_size, action_horizon) for training-time RTC
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            assert time_emb.ndim == 2
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb  # (bs, dim) or (bs, ah, dim) for RTC

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward_drifting(self, observation, actions, noise=None) -> Tensor:
        """Training forward pass for the drifting model.

        For each observation, generates G (gen_per_label) samples by repeating
        the observation context and sampling independent noise vectors. The
        drifting loss is then computed between generated and real actions using
        multi-temperature drift fields.

        Returns a scalar loss tensor.
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        bsize, action_horizon, action_dim = actions.shape
        device = actions.device
        G = self.config.drifting_gen_per_label  # noqa: N806

        # Embed prefix once with original batch, then repeat G times
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs = prefix_embs.repeat_interleave(G, dim=0)
        prefix_pad_masks = prefix_pad_masks.repeat_interleave(G, dim=0)
        prefix_att_masks = prefix_att_masks.repeat_interleave(G, dim=0)
        state_rep = state.repeat_interleave(G, dim=0)

        # Generate G independent noise samples per observation
        noise = self.sample_noise((bsize * G, action_horizon, action_dim), device)
        time = torch.ones(bsize * G, dtype=torch.float32, device=device)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state_rep, noise, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        gen = self._apply_checkpoint(action_out_proj_func, suffix_out)  # [B*G, T, D]
        gen = gen.reshape(bsize, G, action_horizon, action_dim)  # [B, G, T, D]

        R_list = tuple(self.config.drifting_temperatures)  # noqa: N806

        total_loss = 0.0
        for t in range(action_horizon):
            gen_t = gen[:, :, t, :]  # [B, G, D]
            pos_t = actions[:, t, :].unsqueeze(1)  # [B, 1, D]
            loss_t, _ = drift_loss(gen_t, pos_t, R_list=R_list)
            total_loss = total_loss + loss_t.mean()
        return total_loss / action_horizon

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if self.config.use_drifting:
            return self.forward_drifting(observation, actions, noise=noise)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        bsize, action_horizon, action_dim = actions.shape
        device = actions.device

        if noise is None:
            # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
            noise = self.sample_noise(actions.shape, device)

        if time is None:
            time = self.sample_time(bsize, device)

        # Check if training-time RTC is enabled (only for pi05)
        use_training_time_rtc = self.config.use_training_time_rtc and self.pi05
        
        if use_training_time_rtc:
            # Training-time RTC: sample delay and create prefix mask
            max_delay = self.config.rtc_max_delay
            delay = torch.randint(0, max_delay + 1, (bsize,), device=device)
            # Create prefix_mask: True for indices < delay
            # Shape: (batch_size, action_horizon)
            indices = torch.arange(action_horizon, device=device)[None, :]  # (1, ah)
            prefix_mask = indices < delay[:, None]  # (bs, ah)
            # Set time to 0 (action-prefix) for prefix positions, keep original time for postfix
            # time: (bs,) -> time_per_position: (bs, ah)
            time_per_position = time[:, None].expand(bsize, action_horizon).clone()
            time_per_position = torch.where(prefix_mask, torch.zeros_like(time_per_position), time_per_position)
            # For x_t computation: time shape (bs, ah, 1)
            time_for_xt = time_per_position[:, :, None]
            x_t = time_for_xt * noise + (1 - time_for_xt) * actions
            # Compute u_t based on prediction mode
            if self.config.x_pred:
                # For x prediction: u_t = (x_t - actions) / t
                u_t = (x_t - actions) / time_for_xt.clamp_min(0.05)
            else:
                # For v prediction: u_t = noise - actions
                u_t = noise - actions
            embed_time = time_per_position
        else:
            # Standard flow matching
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            # Compute u_t based on prediction mode
            if self.config.x_pred:
                # For x prediction: u_t = (x_t - actions) / t
                u_t = (x_t - actions) / time_expanded.clamp_min(0.05)
            else:
                # For v prediction: u_t = noise - actions
                u_t = noise - actions
            prefix_mask = None
            embed_time = time

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, embed_time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        network_output = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # Handle x prediction mode: convert x_pred to v_pred
        # In this codebase, t=1 is noise and t=0 is target, so v_pred = (x_t - x_pred) / t
        if self.config.x_pred:
            if use_training_time_rtc:
                # time_for_xt has shape (bs, ah, 1)
                v_t = (x_t - network_output) / time_for_xt.clamp_min(0.05)
            else:
                # time_expanded has shape (bs, 1, 1)
                v_t = (x_t - network_output) / time_expanded.clamp_min(0.05)
        else:
            v_t = network_output

        if use_training_time_rtc:
            if self.config.rtc_mask_loss:
                # Compute loss only on postfix positions (indices >= delay)
                postfix_mask = ~prefix_mask  # (bs, ah), True for postfix positions
                loss = (u_t - v_t) ** 2  # (bs, ah, ad)
                # Mask out prefix positions: (bs, ah, ad) * (bs, ah, 1)
                loss = loss * postfix_mask[:, :, None]
                # Normalize by number of postfix elements (sum over all dimensions, divide by postfix count)
                loss = torch.sum(loss) / (torch.sum(postfix_mask) * action_dim + 1e-8)
                return loss
            else:
                # Compute loss on all positions (both prefix and postfix)
                return F.mse_loss(u_t, v_t, reduction="none")
        else:
            return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions_drifting(self, device, observation, noise=None, **kwargs) -> Tensor:
        """One-step action generation for the drifting model.

        Passes noise through the network once to produce actions directly,
        without iterative denoising.
        """
        bsize = observation.state.shape[0]
        action_horizon = self.config.action_horizon
        action_dim = self.config.action_dim

        if noise is None:
            actions_shape = (bsize, action_horizon, action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        time = torch.ones(bsize, dtype=torch.float32, device=device)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, noise, time)

        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, **kwargs) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)."""
        if self.config.use_drifting:
            return self.sample_actions_drifting(device, observation, noise=noise)

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        bsize = observation.state.shape[0]
        action_horizon = self.config.action_horizon
        action_dim = self.config.action_dim

        if noise is None:
            actions_shape = (bsize, action_horizon, action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Check if training-time RTC is enabled (only for pi05)
        use_training_time_rtc = self.config.use_training_time_rtc and self.pi05

        if use_training_time_rtc:
            # action_prefix: supports multiple input formats:
            #   - None: will be initialized to zeros
            #   - (action_horizon, action_dim) or smaller: will add batch dim and pad
            #   - (batch_size, action_horizon, action_dim) or smaller in last 2 dims: will pad
            # inference_delay: (batch_size,) or scalar, number of valid prefix actions
            action_prefix = kwargs.get("action_prefix")
            inference_delay = kwargs.get("inference_delay")

            if action_prefix is None:
                action_prefix = torch.zeros(bsize, action_horizon, action_dim, device=device, dtype=noise.dtype)
            else:
                # Add batch dimension if needed
                if action_prefix.ndim == 2:
                    action_prefix = action_prefix.unsqueeze(0)
                # Pad to (bsize, action_horizon, action_dim) if needed
                if action_prefix.shape[1] < action_horizon or action_prefix.shape[2] < action_dim:
                    padded = torch.zeros(bsize, action_horizon, action_dim, device=device, dtype=action_prefix.dtype)
                    padded[:, :action_prefix.shape[1], :action_prefix.shape[2]] = action_prefix
                    action_prefix = padded
            assert action_prefix.shape == (bsize, action_horizon, action_dim), f"Action prefix shape: {action_prefix.shape}, expected: ({bsize}, {action_horizon}, {action_dim})"
            if inference_delay is None:
                inference_delay = torch.zeros(bsize, dtype=torch.long, device=device)
            elif not isinstance(inference_delay, torch.Tensor):
                inference_delay = torch.tensor(inference_delay, dtype=torch.long, device=device)
            if inference_delay.ndim == 0:
                inference_delay = inference_delay.expand(bsize)

            # Create prefix_mask: True for indices < inference_delay
            indices = torch.arange(action_horizon, device=device)[None, :]  # (1, ah)
            action_prefix_mask = indices < inference_delay[:, None]  # (bs, ah)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            if use_training_time_rtc:
                # Replace x_t with action_prefix where prefix_mask is True
                x_t = torch.where(action_prefix_mask[:, :, None], action_prefix, x_t)
                # Set time to 0.0 for prefix positions (action-prefix), keep current time for postfix
                time_per_position = torch.where(
                    action_prefix_mask,
                    torch.zeros(bsize, action_horizon, device=device, dtype=torch.float32),
                    time.expand(bsize, action_horizon),
                )
                v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time_per_position)
            else:
                expanded_time = time.expand(bsize)

                # Create partial function for denoising step
                def denoise_step_partial(input_x_t, current_timestep=expanded_time):
                    return self.denoise_step(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        input_x_t,
                        current_timestep,
                    )

                # Apply inference-time RTC guidance if enabled
                if self._inference_time_rtc_enabled():
                    inference_delay = kwargs.get("inference_delay")
                    action_prefix = kwargs.get("action_prefix")
                    execution_horizon = kwargs.get("execution_horizon")

                    v_t = self.inference_time_rtc_processor.denoise_step(
                        x_t=x_t,
                        action_prefix=action_prefix,
                        inference_delay=inference_delay,
                        time=time,
                        original_denoise_step_partial=denoise_step_partial,
                        execution_horizon=execution_horizon,
                    )
                else:
                    v_t = denoise_step_partial(x_t)

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.
        
        Returns the velocity v_t for the Euler step. When x_pred mode is enabled,
        the network predicts x (clean target) and we convert to velocity using
        v_pred = (x_t - x_pred) / t.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        network_output = self.action_out_proj(suffix_out)

        # Handle x prediction mode: convert x_pred to v_pred
        # In this codebase, t=1 is noise and t=0 is target, so v_pred = (x_t - x_pred) / t
        if self.config.x_pred:
            # timestep can be (batch_size,) or (batch_size, action_horizon) for RTC
            if timestep.ndim == 1:
                # Shape: (batch_size,) -> (batch_size, 1, 1) for broadcasting
                v_t = (x_t - network_output) / timestep[:, None, None].clamp_min(1e-4)
            else:
                # Shape: (batch_size, action_horizon) -> (batch_size, action_horizon, 1)
                v_t = (x_t - network_output) / timestep[:, :, None].clamp_min(1e-4)
            return v_t
        else:
            return network_output
