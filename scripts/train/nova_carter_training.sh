#!/bin/bash
export HYDRA_FULL_ERROR=1

# ============ CONFIGURATION ============
# DATA_ROOT is the GEAR-converted dataset root as seen from the runtime
# environment. When training inside Docker, use the container-mounted path.
DATA_ROOT=${DATA_ROOT:-"/data/datasets/mas-vln-lerobot/nova_carter_lerobot_train"}
OUTPUT_DIR=${OUTPUT_DIR:-"/data/checkpoints/dreamzero_nova_carter_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/data/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/data/checkpoints/umt5-xxl"}
# =======================================

# Auto-download weights if missing
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first"
    exit 1
fi
if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null || [ ! -w "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory is not writable: $OUTPUT_DIR"
    echo "Fix the directory ownership on the host, or rerun with OUTPUT_DIR=/data/checkpoints/<new-run-name>."
    exit 1
fi
if [ "${HF_HOME:-}" = "/data/cache/huggingface" ]; then
    HF_HOME="/data/cache/hf_home"
fi
HF_HOME=${HF_HOME:-"/data/cache/hf_home"}
HF_HUB_CACHE=${HF_HUB_CACHE:-"/data/cache/hf_hub"}
HF_XET_CACHE=${HF_XET_CACHE:-"/data/cache/hf_xet"}
export HF_HOME HF_HUB_CACHE HF_XET_CACHE
for CACHE_DIR in "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"; do
    if ! mkdir -p "$CACHE_DIR" 2>/dev/null || [ ! -w "$CACHE_DIR" ]; then
        echo "ERROR: Cache directory is not writable: $CACHE_DIR"
        echo "Relaunch Docker with docker/dreamzero/run_interactive.sh, or choose a fresh writable cache path."
        exit 1
    fi
done

# Nova Carter packs three 224x224 views into a 448x448 2x2 canvas; Wan2.1 VAE
# plus patch embedding gives 28*28 = 784 tokens per frame.
torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
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
    pretrained_model_path=/data/checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    "$@"
