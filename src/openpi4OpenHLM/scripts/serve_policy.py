import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.rtc import InferenceTimeRTCAttentionSchedule, InferenceTimeRTCConfig
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    TWIST2G1 = "twist2_g1"
    SONICG1 = "sonic_g1"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class InferenceTimeRTCArgs:
    """Arguments for inference-time Real-Time Chunking (RTC) configuration.
    
    Inference-time RTC applies action corrections during inference without
    requiring special training. This is different from training-time RTC
    which requires the model to be trained with use_training_time_rtc=True.
    """

    # Whether inference-time RTC is enabled.
    enabled: bool = False
    # Execution horizon for inference-time RTC.
    execution_horizon: int = 20
    # Maximum guidance weight for inference-time RTC correction.
    max_guidance_weight: float = 10.0
    # Prefix attention schedule. Options: zeros, ones, linear, exp.
    prefix_attention_schedule: str = "exp"


@dataclasses.dataclass
class TrainingTimeRTCArgs:
    """Arguments for training-time Real-Time Chunking (RTC) configuration.
    
    Training-time RTC requires the model to be trained with use_training_time_rtc=True.
    This argument allows enabling it if it wasn't already enabled in the checkpoint.
    """

    # Whether training-time RTC is enabled.
    enabled: bool = False


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Number of sampling steps for action generation.
    num_steps: int = 10

    # Inference-time RTC configuration.
    # Note: For training-time RTC, the model config must have use_training_time_rtc=True.
    # Training-time RTC does not require additional server-side configuration.
    inference_time_rtc: InferenceTimeRTCArgs = dataclasses.field(default_factory=InferenceTimeRTCArgs)

    # Training-time RTC configuration.
    training_time_rtc: TrainingTimeRTCArgs = dataclasses.field(default_factory=TrainingTimeRTCArgs)
    # debug mode
    debug: bool = False


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.TWIST2G1: Checkpoint(
        config="pi05_g1",
        dir="checkpoints/pi05_g1",
    ),
    EnvMode.SONICG1: Checkpoint(
        config="pi05_g1_pick_sprite_turnback",
        dir="checkpoints/pi05_g1_pick_sprite/g1_pick_sprite_bs64/20000",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None, num_steps: int = 10) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
             _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt, sample_kwargs={"num_steps": num_steps}
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt, sample_kwargs={"num_steps": args.num_steps}
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, num_steps=args.num_steps)


def main(args: Args) -> None:
    if args.debug:
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print("\033[91mDebug server started on 127.0.0.1:5678\033[0m")
        print("\033[91mWaiting for debugger to attach...\033[0m")
        debugpy.wait_for_client()
        print("Debugger attached, continuing execution...")
    
    logging.info(f"Using num_steps: {args.num_steps} for action generation")
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Check if model uses training-time RTC (check original config first)
    training_time_rtc_enabled = False
    if hasattr(policy, "_model") and hasattr(policy._model, "config"):
        if hasattr(policy._model.config, "use_training_time_rtc"):
            training_time_rtc_enabled = policy._model.config.use_training_time_rtc
            if training_time_rtc_enabled:
                logging.info("Model is configured with training-time RTC (use_training_time_rtc=True)")
    
    # Enable training-time RTC via command line argument if requested
    if args.training_time_rtc.enabled:
        if hasattr(policy, "_model") and hasattr(policy._model, "config"):
            policy._model.config = dataclasses.replace(
                policy._model.config,
                use_training_time_rtc=True
            )
            training_time_rtc_enabled = True
            logging.info("Training-time RTC enabled via arguments")

    # Initialize inference-time RTC processor if enabled
    # Note: Inference-time RTC and training-time RTC are mutually exclusive
    if args.inference_time_rtc.enabled:
        if training_time_rtc_enabled:
            logging.warning(
                "Both inference-time RTC and training-time RTC are enabled. "
                "This is unusual - typically only one should be used."
            )
            raise ValueError("Both inference-time RTC and training-time RTC are enabled. This is unusual - typically only one should be used.")
        inference_time_rtc_config = InferenceTimeRTCConfig(
            enabled=args.inference_time_rtc.enabled,
            execution_horizon=args.inference_time_rtc.execution_horizon,
            max_guidance_weight=args.inference_time_rtc.max_guidance_weight,
            prefix_attention_schedule=InferenceTimeRTCAttentionSchedule(args.inference_time_rtc.prefix_attention_schedule),
        )
        if hasattr(policy, "_model") and hasattr(policy._model, "init_inference_time_rtc_processor"):
            policy._model.init_inference_time_rtc_processor(inference_time_rtc_config)
            logging.info("Inference-time RTC processor initialized on model with config: %s", inference_time_rtc_config)
        else:
            logging.warning("Model does not support init_inference_time_rtc_processor, inference-time RTC may not work correctly")

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
