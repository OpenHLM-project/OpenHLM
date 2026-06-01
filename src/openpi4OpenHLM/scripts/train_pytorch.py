"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.optimizer as _optimizer


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


# Keys for action projection layers that need special handling when action_dim > 32
# Pretrained models only support action_dim <= 32
ACTION_PROJ_KEYS = [
    "action_in_proj.weight",
    "action_in_proj.bias",
    "action_out_proj.weight",
    "action_out_proj.bias",
]

# Pretrained action dimension limit
PRETRAINED_ACTION_DIM = 32


def load_model_with_action_proj_exclusion(
    model: torch.nn.Module,
    filename: str,
    device: str = "cpu",
) -> tuple[list[str], list[str]]:
    """
    Load model weights from a safetensors file, excluding action projection layers.
    
    This function is used when model_cfg.action_dim > 32 and weight surgery is disabled.
    The action projection layers will be randomly initialized.
    
    Based on safetensors.torch.load_model but with key exclusion logic and
    proper handling of tensor sharing (duplicate names).
    """
    # Load state dict from safetensors file
    state_dict = safetensors.torch.load_file(filename, device=device)
    
    # Remove action projection keys from loaded state dict
    excluded_keys = []
    for key in ACTION_PROJ_KEYS:
        if key in state_dict:
            del state_dict[key]
            excluded_keys.append(key)
    
    if excluded_keys:
        logging.info(f"Excluded action projection keys from checkpoint: {excluded_keys}")
    
    # Handle tensor sharing issues (duplicate names) similar to safetensors.torch.load_model
    # This is needed because safetensors doesn't allow tensor sharing
    model_state_dict = model.state_dict()
    to_removes = safetensors.torch._remove_duplicate_names(model_state_dict, preferred_names=state_dict.keys())
    
    # Load the filtered state dict with strict=False to allow missing keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Process duplicate names to adjust missing and unexpected keys
    missing = set(missing)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)
    
    # Convert missing back to list for consistent return type
    missing = list(missing)
    
    # Log missing and unexpected keys (excluding the intentionally excluded ones)
    actual_missing = [k for k in missing if k not in ACTION_PROJ_KEYS]
    if actual_missing:
        logging.warning(f"Missing keys in checkpoint (excluding action proj): {actual_missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint: {unexpected}")
    
    return missing, unexpected


def load_model_with_action_expert_exclusion(
    model: torch.nn.Module,
    filename: str,
    device: str = "cpu",
) -> tuple[list[str], list[str]]:
    """
    Load model weights from a safetensors file, excluding action expert parameters.
    
    This function excludes parameters with keys containing:
      - "paligemma_with_expert.gemma_expert"
      - "action_in_proj"
      - "action_out_proj"
    
    These excluded parameters will be randomly initialized, allowing training
    of the action expert from scratch while keeping other weights pretrained.
    """
    # Load state dict from safetensors file
    state_dict = safetensors.torch.load_file(filename, device=device)
    
    # Remove action expert and action projection keys from loaded state dict
    excluded_keys = []
    keys_to_check = list(state_dict.keys())  # Make a copy since we'll modify the dict
    for key in keys_to_check:
        # Check if key contains any of the patterns to exclude
        if ("paligemma_with_expert.gemma_expert" in key or 
            "action_in_proj" in key or 
            "action_out_proj" in key):
            del state_dict[key]
            excluded_keys.append(key)
    
    if excluded_keys:
        logging.info(f"Excluded action expert keys from checkpoint ({len(excluded_keys)} keys): {excluded_keys[:5]}...")
    
    # Handle tensor sharing issues (duplicate names) similar to safetensors.torch.load_model
    # This is needed because safetensors doesn't allow tensor sharing
    model_state_dict = model.state_dict()
    to_removes = safetensors.torch._remove_duplicate_names(model_state_dict, preferred_names=state_dict.keys())
    
    # Load the filtered state dict with strict=False to allow missing keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Process duplicate names to adjust missing and unexpected keys
    missing = set(missing)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)
    
    # Convert missing back to list for consistent return type
    missing = list(missing)
    
    # Log missing and unexpected keys (excluding the intentionally excluded ones)
    actual_missing = [k for k in missing if k not in excluded_keys]
    if actual_missing:
        logging.warning(f"Missing keys in checkpoint (excluding action expert): {actual_missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint: {unexpected}")
    
    return missing, unexpected


def load_model_with_action_proj_expansion(
    model: torch.nn.Module,
    filename: str,
    new_action_dim: int,
    device: str = "cpu",
) -> tuple[list[str], list[str]]:
    """
    Load model weights with action projection layer expansion (weight surgery).
    
    This function preserves pretrained weights for the first 32 dimensions and
    initializes the extra dimensions with Xavier/Glorot initialization.
    This approach is better than random initialization because it maintains the
    learned representations for the original action dimensions.
    """
    # Load state dict from safetensors file
    state_dict = safetensors.torch.load_file(filename, device=device)
    model_state_dict = model.state_dict()
    
    logging.info(
        f"Performing weight surgery: expanding action projection layers from "
        f"{PRETRAINED_ACTION_DIM} to {new_action_dim} dimensions"
    )
    
    # Handle action_in_proj.weight: (hidden_dim, action_dim)
    # Pretrained shape: (hidden_dim, 32), New shape: (hidden_dim, new_action_dim)
    if "action_in_proj.weight" in state_dict:
        pretrained_weight = state_dict["action_in_proj.weight"]  # (hidden_dim, 32)
        new_weight = model_state_dict["action_in_proj.weight"].clone()  # (hidden_dim, new_action_dim)
        hidden_dim = pretrained_weight.shape[0]
        
        # Copy pretrained weights for first 32 dimensions
        new_weight[:, :PRETRAINED_ACTION_DIM] = pretrained_weight
        
        # Initialize remaining dimensions with Xavier/Glorot uniform
        # Calculate the gain for Xavier initialization
        extra_dims = new_action_dim - PRETRAINED_ACTION_DIM
        if extra_dims > 0:
            # Use Xavier uniform initialization for the extra dimensions
            # fan_in = extra_dims, fan_out = hidden_dim
            std = np.sqrt(2.0 / (extra_dims + hidden_dim))
            bound = np.sqrt(3.0) * std
            new_weight[:, PRETRAINED_ACTION_DIM:] = torch.empty(
                hidden_dim, extra_dims, device=device, dtype=new_weight.dtype
            ).uniform_(-bound, bound)
            logging.info(
                f"action_in_proj.weight: copied {PRETRAINED_ACTION_DIM} dims from pretrained, "
                f"initialized {extra_dims} extra dims with Xavier uniform (bound={bound:.4f})"
            )
        
        state_dict["action_in_proj.weight"] = new_weight
    
    # Handle action_in_proj.bias: (hidden_dim,)
    # Bias shape doesn't change with action_dim, so we can copy directly
    # (The bias is applied after the linear transformation, so it has shape hidden_dim)
    if "action_in_proj.bias" in state_dict:
        # Bias shape is (hidden_dim,), no expansion needed
        logging.info("action_in_proj.bias: copied directly (no expansion needed)")
    
    # Handle action_out_proj.weight: (action_dim, hidden_dim)
    # Pretrained shape: (32, hidden_dim), New shape: (new_action_dim, hidden_dim)
    if "action_out_proj.weight" in state_dict:
        pretrained_weight = state_dict["action_out_proj.weight"]  # (32, hidden_dim)
        new_weight = model_state_dict["action_out_proj.weight"].clone()  # (new_action_dim, hidden_dim)
        hidden_dim = pretrained_weight.shape[1]
        
        # Copy pretrained weights for first 32 dimensions
        new_weight[:PRETRAINED_ACTION_DIM, :] = pretrained_weight
        
        # Initialize remaining dimensions with Xavier/Glorot uniform
        extra_dims = new_action_dim - PRETRAINED_ACTION_DIM
        if extra_dims > 0:
            # Use Xavier uniform initialization for the extra dimensions
            # fan_in = hidden_dim, fan_out = extra_dims
            std = np.sqrt(2.0 / (hidden_dim + extra_dims))
            bound = np.sqrt(3.0) * std
            new_weight[PRETRAINED_ACTION_DIM:, :] = torch.empty(
                extra_dims, hidden_dim, device=device, dtype=new_weight.dtype
            ).uniform_(-bound, bound)
            logging.info(
                f"action_out_proj.weight: copied {PRETRAINED_ACTION_DIM} dims from pretrained, "
                f"initialized {extra_dims} extra dims with Xavier uniform (bound={bound:.4f})"
            )
        
        state_dict["action_out_proj.weight"] = new_weight
    
    # Handle action_out_proj.bias: (action_dim,)
    # Pretrained shape: (32,), New shape: (new_action_dim,)
    if "action_out_proj.bias" in state_dict:
        pretrained_bias = state_dict["action_out_proj.bias"]  # (32,)
        new_bias = model_state_dict["action_out_proj.bias"].clone()  # (new_action_dim,)
        
        # Copy pretrained bias for first 32 dimensions
        new_bias[:PRETRAINED_ACTION_DIM] = pretrained_bias
        
        # Initialize extra dimensions to zero (standard practice for bias)
        extra_dims = new_action_dim - PRETRAINED_ACTION_DIM
        if extra_dims > 0:
            new_bias[PRETRAINED_ACTION_DIM:] = 0.0
            logging.info(
                f"action_out_proj.bias: copied {PRETRAINED_ACTION_DIM} dims from pretrained, "
                f"initialized {extra_dims} extra dims to zero"
            )
        
        state_dict["action_out_proj.bias"] = new_bias
    
    # Handle tensor sharing issues (duplicate names) similar to safetensors.torch.load_model
    to_removes = safetensors.torch._remove_duplicate_names(model_state_dict, preferred_names=state_dict.keys())
    
    # Load the modified state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Process duplicate names to adjust missing and unexpected keys
    missing = set(missing)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)
    
    missing = list(missing)
    
    if missing:
        logging.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint: {unexpected}")
    
    logging.info("Weight surgery completed successfully")
    return missing, unexpected


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def create_muon_optimizer(model, config: _config.TrainConfig, use_ddp: bool):
    """Create MuonWithAuxAdam optimizer with proper parameter grouping.
    
    Muon is designed for hidden weight matrices (2D+ params). Non-hidden parameters
    (embeddings, projection layers, biases, layer norms) are optimized with AdamW.
    
    In DDP mode, parameters must be grouped by dtype to avoid all_gather errors.
    
    Args:
        model: The model to optimize (may be wrapped in DDP)
        config: Training configuration containing MuonWithAuxAdam optimizer config
        use_ddp: Whether distributed training is enabled
        
    Returns:
        Tuple of (optimizer, initial_lrs) where initial_lrs is a list of initial learning rates
        for each parameter group
    """
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    from collections import defaultdict
    
    muon_cfg = config.optimizer
    model_to_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    # Classify parameters into hidden (Muon) and non-hidden (AdamW) groups
    # In DDP mode, further group by dtype to avoid all_gather dtype mismatch
    hidden_by_dtype = defaultdict(list)  # dtype -> [params]
    nonhidden_by_dtype = defaultdict(list)  # dtype -> [params]
    
    hidden_count = 0
    nonhidden_count = 0
    
    for name, param in model_to_inspect.named_parameters():
        if not param.requires_grad:
            continue
        
        # A parameter is non-hidden if:
        # 1. It's 1D (biases, layer norms, etc.)
        # 2. Its name contains any of the nonhidden patterns (embeddings, projections)
        is_nonhidden = (
            param.ndim < 2 or
            any(pattern in name for pattern in muon_cfg.nonhidden_patterns)
        )
        
        if is_nonhidden:
            nonhidden_by_dtype[param.dtype].append(param)
            nonhidden_count += 1
        else:
            hidden_by_dtype[param.dtype].append(param)
            hidden_count += 1
    
    logging.info(
        f"MuonWithAuxAdam parameter grouping: "
        f"{hidden_count} hidden params (Muon), {nonhidden_count} non-hidden params (AdamW)"
    )
    
    # Log dtype distribution
    for dtype, params in hidden_by_dtype.items():
        logging.info(f"  Hidden params with dtype {dtype}: {len(params)}")
    for dtype, params in nonhidden_by_dtype.items():
        logging.info(f"  Non-hidden params with dtype {dtype}: {len(params)}")
    
    # Build parameter groups as expected by MuonWithAuxAdam
    # In DDP mode, create separate groups for each dtype to avoid all_gather errors
    # Note: MuonWithAuxAdam validates exact keys, so we cannot add initial_lr here.
    # Initial LRs are returned separately for LR scheduling.
    param_groups = []
    initial_lrs = []
    
    # Add hidden weight groups (Muon), one per dtype
    for dtype, params in sorted(hidden_by_dtype.items(), key=lambda x: str(x[0])):
        param_groups.append(dict(
            params=params,
            use_muon=True,
            lr=muon_cfg.muon_lr,
            momentum=muon_cfg.muon_momentum,
            weight_decay=muon_cfg.muon_weight_decay,
        ))
        initial_lrs.append(muon_cfg.muon_lr)
    
    # Add non-hidden param groups (AdamW), one per dtype
    for dtype, params in sorted(nonhidden_by_dtype.items(), key=lambda x: str(x[0])):
        param_groups.append(dict(
            params=params,
            use_muon=False,
            lr=muon_cfg.adam_lr,
            betas=(muon_cfg.adam_b1, muon_cfg.adam_b2),
            eps=muon_cfg.adam_eps,
            weight_decay=muon_cfg.adam_weight_decay,
        ))
        initial_lrs.append(muon_cfg.adam_lr)
    
    # Use distributed or single-device variant based on DDP status
    OptimizerClass = MuonWithAuxAdam if use_ddp else SingleDeviceMuonWithAuxAdam
    logging.info(
        f"Using {'distributed' if use_ddp else 'single-device'} Muon optimizer "
        f"with {len(param_groups)} parameter groups"
    )
    
    optimizer = OptimizerClass(param_groups)
    return optimizer, initial_lrs


def init_ema_parameters(model):
    """Initialize EMA parameters as a deep copy of model parameters."""
    model_to_copy = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    ema_state_dict = {}
    for name, param in model_to_copy.named_parameters():
        ema_state_dict[name] = param.data.clone().detach()
    return ema_state_dict


def update_ema_parameters(ema_state_dict, model, ema_decay):
    """Update EMA parameters using exponential moving average.
    
    EMA formula: ema_param = ema_decay * ema_param + (1 - ema_decay) * param
    
    Args:
        ema_state_dict: Dictionary containing EMA parameters
        model: Current model (may be wrapped in DDP)
        ema_decay: EMA decay rate (typically 0.99 or 0.999)
    """
    model_to_update = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    with torch.no_grad():
        for name, param in model_to_update.named_parameters():
            if name in ema_state_dict:
                ema_state_dict[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, ema_state_dict=None, save_optimizer=False):
    """Save a checkpoint with model state, optimizer state, and metadata.
    
    If EMA is enabled (ema_state_dict is not None), saves the EMA parameters as the main model weights
    for inference, following the same pattern as the JAX training script.
    """
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        # If EMA is enabled, save EMA parameters instead of regular parameters
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        if ema_state_dict is not None:
            # Save EMA parameters for inference (matching JAX behavior)
            # Temporarily load EMA parameters into model for saving
            original_state_dict = {name: param.data.clone() for name, param in model_to_save.named_parameters()}
            try:
                for name, param in model_to_save.named_parameters():
                    if name in ema_state_dict:
                        param.data.copy_(ema_state_dict[name])
                safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
                logging.info(f"Saved EMA parameters to checkpoint at step {global_step}")
            finally:
                # Restore original parameters
                for name, param in model_to_save.named_parameters():
                    if name in original_state_dict:
                        param.data.copy_(original_state_dict[name])
        else:
            # Save regular parameters
            safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        if save_optimizer:
            torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
            "has_ema": ema_state_dict is not None,
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, ema_state_dict=None):
    """Load the latest checkpoint and return the global step and EMA state dict.
    
    When EMA was used during training, the checkpoint contains EMA parameters.
    We load them into the model (for continued training) and also return them
    as the EMA state dict.
    
    Returns:
        tuple: (global_step, ema_state_dict) where ema_state_dict is None if EMA was not used
    """
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load metadata first to check if EMA was used
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        has_ema = metadata.get("has_ema", False)
        
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            
            if has_ema:
                logging.info("Loaded EMA parameters from checkpoint (saved as model weights)")
                # The loaded parameters are EMA parameters, so we need to copy them to ema_state_dict
                if ema_state_dict is not None:
                    for name, param in model_to_load.named_parameters():
                        if name in ema_state_dict:
                            ema_state_dict[name].copy_(param.data)
                else:
                    # Initialize EMA state dict from loaded parameters
                    ema_state_dict = {}
                    for name, param in model_to_load.named_parameters():
                        ema_state_dict[name] = param.data.clone().detach()
            else:
                logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict
        else:
            logging.warning(f"No optimizer checkpoint found at {ckpt_dir}, optimizer state will be reset")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step, ema_state_dict if has_ema else None

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def evaluate_sample_actions(model, observation, actions, device, num_denoising_steps=10, action_prefix=None, inference_delay=None):
    """Evaluate model by sampling actions and computing MSE with ground truth.
    
    Returns:
        dict: Dictionary containing MSE metrics:
            - 'mse': Overall MSE between sampled and ground truth actions
            - 'prefix_mse': (Optional) MSE for the first inference_delay horizon actions
            - 'remaining_mse': (Optional) MSE for actions after inference_delay horizon
    """
    model_to_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    # Save training state
    was_training = model_to_eval.training
    model_to_eval.eval()
    
    try:
        with torch.no_grad():
            # Sample actions using the model's inference path
            sample_kwargs = {}
            if action_prefix is not None:
                sample_kwargs["action_prefix"] = action_prefix
            if inference_delay is not None:
                sample_kwargs["inference_delay"] = inference_delay
                
            sampled_actions = model_to_eval.sample_actions(
                device=device,
                observation=observation,
                num_steps=num_denoising_steps,
                **sample_kwargs,
            )
            
            # Ensure both are float32 for comparison
            sampled_actions = sampled_actions.to(torch.float32)
            gt_actions = actions.to(torch.float32)
            
            # Handle potential dimension mismatch by truncating to min dimension
            # This can happen if action_dim differs between model and data
            min_action_dim = min(sampled_actions.shape[-1], gt_actions.shape[-1])
            min_horizon = min(sampled_actions.shape[-2], gt_actions.shape[-2])
            
            sampled_actions = sampled_actions[..., :min_horizon, :min_action_dim]
            gt_actions = gt_actions[..., :min_horizon, :min_action_dim]
            
            # Compute overall MSE
            mse = F.mse_loss(sampled_actions, gt_actions)
            
            result = {'mse': mse.item()}
            
            # Compute additional MSE metrics when action_prefix is provided
            if action_prefix is not None and inference_delay is not None:
                # MSE for the first inference_delay horizon actions (prefix region)
                prefix_sampled = sampled_actions[..., :inference_delay, :]
                prefix_gt = gt_actions[..., :inference_delay, :]
                prefix_mse = F.mse_loss(prefix_sampled, prefix_gt)
                result['prefix_mse'] = prefix_mse.item()
                
                # MSE for the remaining actions (non-prefix region)
                if min_horizon > inference_delay:
                    remaining_sampled = sampled_actions[..., inference_delay:, :]
                    remaining_gt = gt_actions[..., inference_delay:, :]
                    remaining_mse = F.mse_loss(remaining_sampled, remaining_gt)
                    result['remaining_mse'] = remaining_mse.item()
            
            return result
    finally:
        # Restore training state
        if was_training:
            model_to_eval.train()


def train_loop(config: _config.TrainConfig):
    if config.debug:
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print("\033[91mDebug server started on 127.0.0.1:5678\033[0m")
        print("\033[91mWaiting for debugger to attach...\033[0m")
        debugpy.wait_for_client()
        print("Debugger attached, continuing execution...")
    
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if config.enable_training_compile:
        if hasattr(model, "compile_for_training"):
            model.compile_for_training()
        else:
            logging.info("Training compile is not supported for this model")

    # Enable gradient checkpointing based on config and model support
    enable_gradient_checkpointing = False
    if config.enable_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            enable_gradient_checkpointing = True
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing for memory optimization")
        else:
            logging.info("Gradient checkpointing is not supported for this model")
    else:
        logging.info("Gradient checkpointing is disabled by config")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        # Check if we should randomly initialize action expert parameters
        if config.random_init_action_expert:
            # Exclude action expert and action projection parameters from loading
            logging.info(
                "random_init_action_expert=True, excluding action expert and action projection "
                "parameters from checkpoint (will be randomly initialized)"
            )
            load_model_with_action_expert_exclusion(model_to_load, model_path, device=str(device))
        elif model_cfg.action_dim > PRETRAINED_ACTION_DIM:
            # Pretrained models only support action_dim <= 32
            if config.action_proj_init_mode == "weight_surgery":
                # Use weight surgery: preserve pretrained weights for first 32 dims,
                # initialize extra dims with Xavier/Glorot initialization
                logging.info(
                    f"action_dim={model_cfg.action_dim} > {PRETRAINED_ACTION_DIM}, "
                    f"using weight surgery to expand action projection layers"
                )
                load_model_with_action_proj_expansion(
                    model_to_load, model_path, model_cfg.action_dim, device=str(device)
                )
            elif config.action_proj_init_mode == "random":
                # Random initialization: exclude action projection layers entirely
                logging.info(
                    f"action_dim={model_cfg.action_dim} > {PRETRAINED_ACTION_DIM}, "
                    f"excluding action projection layers (random initialization)"
                )
                load_model_with_action_proj_exclusion(model_to_load, model_path, device=str(device))
            else:
                raise ValueError(f"Unknown action_proj_init_mode: {config.action_proj_init_mode}")
        else:
            safetensors.torch.load_model(model_to_load, model_path)
        
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = None
    is_muon_optimizer = isinstance(config.optimizer, _optimizer.MuonWithAuxAdam)
    muon_initial_lrs = None  # Store initial LRs for Muon param groups
    
    if is_muon_optimizer:
        logging.info("Using MuonWithAuxAdam optimizer")
        optim, muon_initial_lrs = create_muon_optimizer(model, config, use_ddp)
    elif config.use_8bit_adam:
        logging.info("Using 8-bit Adam optimizer")
        import bitsandbytes as bnb
        
        optim = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        # move all optimizer state tensors to CPU
        # increse step execution time but save GPU memory
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
    else:
        logging.info("Using regular AdamW optimizer")
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

    # Initialize EMA parameters if enabled
    ema_state_dict = None
    if config.ema_decay is not None:
        ema_state_dict = init_ema_parameters(model)
        logging.info(f"Initialized EMA parameters with decay={config.ema_decay}")
    
    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step, loaded_ema = load_checkpoint(model, optim, config.checkpoint_dir, device, ema_state_dict)
        if loaded_ema is not None:
            ema_state_dict = loaded_ema
            logging.info(f"Resumed training from step {global_step} with EMA parameters")
        else:
            logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int, base_lr: float = peak_lr):
        """
        Compute learning rate for a given step.
        
        Args:
            step: Current training step
            base_lr: Base learning rate for this parameter group (default: peak_lr)
        """
        # Compute the LR multiplier relative to peak_lr
        lr_multiplier = base_lr / peak_lr if peak_lr > 0 else 1.0
        
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            scheduled_lr = init_lr + (peak_lr - init_lr) * step / warmup_steps
        else:
            # cosine decay
            progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
            cos = 0.5 * (1 + np.cos(np.pi * progress))
            scheduled_lr = end_lr + (peak_lr - end_lr) * cos
        
        # Apply the multiplier to maintain relative LR between parameter groups
        return scheduled_lr * lr_multiplier

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(
            f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}, training_compile={config.enable_training_compile}"
        )
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        if is_muon_optimizer:
            logging.info(
                f"Optimizer: {type(config.optimizer).__name__}, "
                f"muon_lr={config.optimizer.muon_lr:.2e}, muon_wd={config.optimizer.muon_weight_decay}, "
                f"adam_lr={config.optimizer.adam_lr:.2e}, adam_wd={config.optimizer.adam_weight_decay}, "
                f"clip_norm={config.optimizer.clip_gradient_norm}"
            )
        else:
            logging.info(
                f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
            )
        if config.ema_decay is not None:
            logging.info(f"EMA enabled with decay={config.ema_decay}")
        else:
            logging.info("EMA is disabled")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR for each parameter group based on its initial LR
            for i, pg in enumerate(optim.param_groups):
                # For Muon optimizer, use stored initial LRs since param groups have strict key validation
                if is_muon_optimizer and muon_initial_lrs is not None:
                    initial_lr = muon_initial_lrs[i]
                else:
                    # For other optimizers, use stored 'initial_lr' or fall back to peak_lr
                    initial_lr = pg.get("initial_lr", peak_lr)
                pg["lr"] = lr_schedule(global_step, initial_lr)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Update EMA parameters after optimizer step
            if ema_state_dict is not None:
                update_ema_parameters(ema_state_dict, model, config.ema_decay)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                info_dict = {
                    "loss": loss.item(),
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }
                
                # For Muon optimizer, log both learning rates (Muon and AdamW)
                # Find the first Muon group and first AdamW group
                if is_muon_optimizer:
                    muon_lr = None
                    adam_lr = None
                    for pg in optim.param_groups:
                        if pg.get("use_muon", False) and muon_lr is None:
                            muon_lr = pg["lr"]
                        elif not pg.get("use_muon", False) and adam_lr is None:
                            adam_lr = pg["lr"]
                    
                    if muon_lr is not None:
                        info_dict["learning_rate_muon"] = muon_lr
                    if adam_lr is not None:
                        info_dict["learning_rate_adam"] = adam_lr
                else:
                    info_dict["learning_rate"] = optim.param_groups[0]["lr"]
                
                infos.append(info_dict)

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                
                # Handle learning rate logging based on optimizer type
                avg_lr_muon = None
                avg_lr_adam = None
                avg_lr = None
                
                if is_muon_optimizer:
                    # Average Muon and AdamW learning rates if present
                    if any("learning_rate_muon" in info for info in infos):
                        avg_lr_muon = sum(info.get("learning_rate_muon", 0) for info in infos) / len(infos)
                    if any("learning_rate_adam" in info for info in infos):
                        avg_lr_adam = sum(info.get("learning_rate_adam", 0) for info in infos) / len(infos)
                else:
                    avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                
                # Build log message based on optimizer type
                if is_muon_optimizer:
                    log_msg = f"step={global_step} loss={avg_loss:.4f}"
                    if avg_lr_muon is not None:
                        log_msg += f" lr_muon={avg_lr_muon:.2e}"
                    if avg_lr_adam is not None:
                        log_msg += f" lr_adam={avg_lr_adam:.2e}"
                    if avg_grad_norm is not None:
                        log_msg += f" grad_norm={avg_grad_norm:.2f}"
                    log_msg += f" time={elapsed:.1f}s"
                else:
                    log_msg = f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e}"
                    if avg_grad_norm is not None:
                        log_msg += f" grad_norm={avg_grad_norm:.2f}"
                    log_msg += f" time={elapsed:.1f}s"
                
                logging.info(log_msg)

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    
                    # Add learning rate(s) to wandb payload
                    if is_muon_optimizer:
                        if avg_lr_muon is not None:
                            log_payload["learning_rate_muon"] = avg_lr_muon
                        if avg_lr_adam is not None:
                            log_payload["learning_rate_adam"] = avg_lr_adam
                    else:
                        log_payload["learning_rate"] = avg_lr
                    
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            # Run evaluation: sample actions and compute MSE with ground truth
            if is_main and config.eval_interval > 0 and (global_step % config.eval_interval == 0):
                eval_start_time = time.time()
                
                try:
                    # Evaluation with num_denoising_steps=10
                    eval_result_10 = evaluate_sample_actions(
                        model=model,
                        observation=observation,
                        actions=actions,
                        device=device,
                        num_denoising_steps=10,
                    )
                    eval_mse_10 = eval_result_10['mse']

                    # Evaluation with num_denoising_steps=1
                    eval_result_1 = evaluate_sample_actions(
                        model=model,
                        observation=observation,
                        actions=actions,
                        device=device,
                        num_denoising_steps=1,
                    )
                    eval_mse_1 = eval_result_1['mse']
                    
                    # Evaluation with training-time RTC if enabled
                    eval_mse_rtc = None
                    eval_mse_rtc_prefix = None
                    eval_mse_rtc_remaining = None
                    if model_cfg.use_training_time_rtc:
                        # Use first 6 actions as prefix (assuming inference_delay=6)
                        inference_delay = 6
                        action_prefix = actions[:, :inference_delay, :].clone()  # (batch_size, 6, action_dim)
                        
                        eval_result_rtc = evaluate_sample_actions(
                            model=model,
                            observation=observation,
                            actions=actions,
                            device=device,
                            num_denoising_steps=10,
                            action_prefix=action_prefix,
                            inference_delay=inference_delay,
                        )
                        eval_mse_rtc = eval_result_rtc['mse']
                        eval_mse_rtc_prefix = eval_result_rtc.get('prefix_mse')
                        eval_mse_rtc_remaining = eval_result_rtc.get('remaining_mse')
                    
                    # Evaluate EMA model if EMA is enabled
                    eval_ema_mse = None
                    if ema_state_dict is not None:
                        # Temporarily swap in EMA parameters for evaluation
                        model_to_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        original_state_dict = {name: param.data.clone() for name, param in model_to_eval.named_parameters()}
                        
                        try:
                            # Load EMA parameters into model
                            for name, param in model_to_eval.named_parameters():
                                if name in ema_state_dict:
                                    param.data.copy_(ema_state_dict[name])
                            
                            # Evaluate with EMA parameters
                            eval_result_ema = evaluate_sample_actions(
                                model=model,
                                observation=observation,
                                actions=actions,
                                device=device,
                                num_denoising_steps=10,
                            )
                            eval_ema_mse = eval_result_ema['mse']
                        finally:
                            # Restore original parameters
                            for name, param in model_to_eval.named_parameters():
                                if name in original_state_dict:
                                    param.data.copy_(original_state_dict[name])
                    
                    eval_elapsed = time.time() - eval_start_time
                    
                    # Build log message
                    log_msg = (
                        f"Evaluation at step {global_step}: "
                        f"sample_mse_10steps={eval_mse_10:.6f} "
                        f"sample_mse_1step={eval_mse_1:.6f} "
                    )
                    if eval_mse_rtc is not None:
                        log_msg += f"sample_mse_rtc={eval_mse_rtc:.6f} "
                        if eval_mse_rtc_prefix is not None:
                            log_msg += f"sample_mse_rtc_prefix={eval_mse_rtc_prefix:.6f} "
                        if eval_mse_rtc_remaining is not None:
                            log_msg += f"sample_mse_rtc_remaining={eval_mse_rtc_remaining:.6f} "
                    if eval_ema_mse is not None:
                        log_msg += f"ema_mse_10steps={eval_ema_mse:.6f} "
                    log_msg += f"eval_time={eval_elapsed:.2f}s"
                    logging.info(log_msg)
                    
                    # Log evaluation MSE to wandb
                    if config.wandb_enabled:
                        log_dict = {
                            "eval_sample_mse_10steps": eval_mse_10,
                            "eval_sample_mse_1step": eval_mse_1,
                        }
                        if eval_mse_rtc is not None:
                            log_dict["eval_sample_mse_rtc"] = eval_mse_rtc
                            if eval_mse_rtc_prefix is not None:
                                log_dict["eval_sample_mse_rtc_prefix"] = eval_mse_rtc_prefix
                            if eval_mse_rtc_remaining is not None:
                                log_dict["eval_sample_mse_rtc_remaining"] = eval_mse_rtc_remaining
                        if eval_ema_mse is not None:
                            log_dict["eval_ema_mse_10steps"] = eval_ema_mse
                        wandb.log(log_dict, step=global_step)
                        
                except Exception as e:
                    logging.warning(f"Evaluation failed at step {global_step}: {e}")

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config, ema_state_dict, save_optimizer=False)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()