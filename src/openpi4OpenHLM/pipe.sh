uv sync
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

### checkpoint
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --config_name pi05_aloha \
    --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch

### data processing
uv run examples/unitree_g1/convert_g1_data_to_lerobot.py \
    --data_dir ~/codebase/OpenHLM/src/openpi-humanoid/sonic_demonstration/20260521_1102_test \
    --repo_name test_video

uv run examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
    --parent_dir ~/codebase/OpenHLM/src/openpi-humanoid/sonic_demonstration \
    --dataset_folders 20260521_1102_test 20260521_1102_test_2 \
    --repo_name test_video2

CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name openhlm_example \
    --max-frames 500000

### training
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py \
    openhlm_example \
    --save_interval 5000 \
    --batch_size 32 \
    --no-enable-gradient-checkpointing \
    --enable-training-compile \
    --exp_name g1_test_video \
    --pytorch-weight-path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch

### evaluation
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
--env SONICG1 \
--num-steps 10 \
policy:checkpoint \
--policy.config=openhlm_example \
--policy.dir=~/codebase/OpenHLM/src/openpi-humanoid/checkpoints/openhlm_example/g1_test_video_freeze_paligemma/30000

python openpi-eval/main.py \
--control_hz 30 \
--max_steps 10000 \
--save_video \
--instruction "test" \
--exp_name g1_test_video

uv pip install --python .venv_data_collection/bin/python -e openpi-client