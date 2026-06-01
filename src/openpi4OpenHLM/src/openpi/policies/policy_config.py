import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

try:
    import torch
    from safetensors.torch import load_file, save_file
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def create_merged_policy(
    train_config: _config.TrainConfig,
    base_checkpoint_dir: pathlib.Path | str,
    finetuned_checkpoint_dir: pathlib.Path | str,
    merge_weight: float,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from two merged checkpoints using linear interpolation.
    
    The final model weights are computed as:
        θ̃ = (1 - α) · θ_base + α · θ_finetuned
    
    where α is the merge_weight parameter.
    
    Args:
        train_config: The training config to use to create the model.
        base_checkpoint_dir: The directory to load the base model from.
        finetuned_checkpoint_dir: The directory to load the finetuned model from.
        merge_weight: Weight for linear interpolation (α). Range: [0, 1].
                     0 = use only base model, 1 = use only finetuned model.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method.
        default_prompt: The default prompt to use for the policy.
        norm_stats: The norm stats to use for the policy. If not provided, will be loaded from
                   the finetuned checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
    
    Returns:
        A policy with merged model weights.
    
    Note:
        Currently only supports PyTorch models. JAX models are not supported.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and safetensors are required for model merging. Please install them.")
    
    if not (0 <= merge_weight <= 1):
        raise ValueError(f"merge_weight must be in range [0, 1], got {merge_weight}")
    
    repack_transforms = repack_transforms or transforms.Group()
    base_checkpoint_dir = download.maybe_download(str(base_checkpoint_dir))
    finetuned_checkpoint_dir = download.maybe_download(str(finetuned_checkpoint_dir))
    
    # Check if both are PyTorch models
    base_weight_path = os.path.join(base_checkpoint_dir, "model.safetensors")
    finetuned_weight_path = os.path.join(finetuned_checkpoint_dir, "model.safetensors")
    
    if not os.path.exists(base_weight_path):
        raise ValueError(f"Base checkpoint is not a PyTorch model (model.safetensors not found): {base_checkpoint_dir}")
    if not os.path.exists(finetuned_weight_path):
        raise ValueError(f"Finetuned checkpoint is not a PyTorch model (model.safetensors not found): {finetuned_checkpoint_dir}")
    
    logging.info("Loading base model weights from: %s", base_weight_path)
    base_state_dict = load_file(base_weight_path)
    
    logging.info("Loading finetuned model weights from: %s", finetuned_weight_path)
    finetuned_state_dict = load_file(finetuned_weight_path)
    
    # Verify that both models have the same architecture
    if set(base_state_dict.keys()) != set(finetuned_state_dict.keys()):
        raise ValueError("Base and finetuned models have different architectures (different parameter names)")
    
    # Perform linear interpolation: θ̃ = (1 - α) · θ_base + α · θ_finetuned
    logging.info("Merging model weights with merge_weight=%.4f", merge_weight)
    merged_state_dict = {}
    for key in base_state_dict.keys():
        base_param = base_state_dict[key]
        finetuned_param = finetuned_state_dict[key]
        
        if base_param.shape != finetuned_param.shape:
            raise ValueError(f"Parameter {key} has different shapes in base and finetuned models: "
                           f"{base_param.shape} vs {finetuned_param.shape}")
        
        # Linear interpolation
        merged_state_dict[key] = (1 - merge_weight) * base_param + merge_weight * finetuned_param
    
    logging.info("Successfully merged %d parameters", len(merged_state_dict))
    
    # Create model structure and load merged weights
    logging.info("Creating model with merged weights...")
    from openpi.models_pytorch import pi0_pytorch
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    
    # Load the merged weights into the model
    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)
    if missing_keys:
        logging.warning(f"Missing keys when loading merged weights: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys when loading merged weights: {unexpected_keys}")
    
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    
    # Load data config and norm stats from finetuned checkpoint
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(finetuned_checkpoint_dir / "assets", data_config.asset_id)
    
    # Determine the device to use for PyTorch models
    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=pytorch_device,
    )
