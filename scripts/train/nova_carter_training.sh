#!/bin/bash
set -euo pipefail
export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "${DREAMZERO_ROOT:-}" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    : # keep caller-provided repo root
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/workspace/dreamzero-vln}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the repo root mounted inside Docker."
    exit 1
fi

# torchrun launches experiment.py as a file, so make the repo importable inside
# Docker where the package is usually not pip-installed.
export PYTHONPATH="$DREAMZERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$DREAMZERO_ROOT"

# ============ CONFIGURATION ============
# DATA_ROOT is the GEAR-converted dataset root as seen from the runtime
# environment. When training inside Docker, use the container-mounted path.
DATA_ROOT=${DATA_ROOT:-"/data/datasets/mas-vln-lerobot/nova_carter_lerobot_train"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_nova_carter_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NUM_GPUS=0
    fi
fi
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: NUM_GPUS must be a positive integer. Set NUM_GPUS explicitly inside Docker if GPU detection is unavailable."
    exit 1
fi

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
DREAMZERO_CKPT_DIR=${DREAMZERO_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/DreamZero-AgiBot"}
# =======================================

# Auto-download weights if missing
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    if ! command -v hf >/dev/null 2>&1; then
        echo "ERROR: hf CLI not found. Install huggingface_hub[cli] or pre-populate WAN_CKPT_DIR."
        exit 1
    fi
    hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    if ! command -v hf >/dev/null 2>&1; then
        echo "ERROR: hf CLI not found. Install huggingface_hub[cli] or pre-populate TOKENIZER_DIR."
        exit 1
    fi
    hf download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first"
    exit 1
fi
if [ ! -d "$DREAMZERO_CKPT_DIR" ] || [ -z "$(ls -A "$DREAMZERO_CKPT_DIR" 2>/dev/null)" ]; then
    echo "ERROR: DreamZero checkpoint not found at $DREAMZERO_CKPT_DIR"
    echo "Download with: hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir $DREAMZERO_CKPT_DIR"
    exit 1
fi

# Nova Carter packs three 224x224 views into a 448x448 2x2 canvas; Wan2.1 VAE
# plus patch embedding gives 28*28 = 784 tokens per frame.
EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
torchrun --nproc_per_node $NUM_GPUS --standalone \
    "$EXPERIMENT_PY" \
    report_to=wandb \
    data=dreamzero/nova_carter \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=4 \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=224 \
    image_resolution_height=224 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=784 \
    save_strategy=steps \
    nova_carter_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$DREAMZERO_CKPT_DIR \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    "$@"
