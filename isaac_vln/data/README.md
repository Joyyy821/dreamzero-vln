# Isaac VLN Data Conversion

This folder contains the MAS-VLN Isaac Sim to LeRobot/GEAR conversion tools used
for DreamZero fine-tuning. V1 is RGB-only and one controlled ego embodiment per
output dataset.

## Expected Input

Use the packaged MAS-VLN HF-style release as input:

```text
/home/yjiao/Datasets/ma_vln_isaac_sim_hf/
  metadata/
    rollouts.parquet
    frames.parquet
  rollouts/scene_XXX/
    rollout_YYY.tar
  scenes/scene_XXX/
    ...
```

`rollouts.parquet` and `frames.parquet` drive selection and frame indexing.
`run_config.yaml` inside each rollout tar is used for the robot list, goal
poses, and language instruction. The converter does not modify this source
dataset.

## Isaac To LeRobot

Convert all successful packaged `nova_carter` ego episodes:

```bash
python isaac_vln/data/convert_manifest_to_lerobot.py \
  --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf \
  --out-root ./datasets/nova_carter_lerobot_train \
  --embodiment nova_carter \
  --scene-ids 1 2 3 6 7 8 11 12 13 16 17 18 21 22 23 \
  --split train \
  --overwrite
```

The converter shows a `Converting episodes` progress bar. Omit `--max-episodes`
and `--max-steps` for full conversion. Use those flags only for smoke tests.

For a tiny smoke test:

```bash
python isaac_vln/data/convert_manifest_to_lerobot.py \
  --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf \
  --out-root /tmp/isaac_debug/nova_carter_smoke \
  --embodiment nova_carter \
  --scene-ids 1 2 \
  --max-episodes 2 \
  --max-steps 20 \
  --overwrite
```

Expected LeRobot output:

```text
nova_carter_lerobot_train/
  data/chunk-000/
    episode_000000.parquet
    ...
  videos/chunk-000/
    observation.images.ego_front/
      episode_000000.mp4
    observation.images.third_view_0/
      episode_000000.mp4
    observation.images.third_view_1/
      episode_000000.mp4
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    isaac_vln_debug.json
```

## State And Action

Each parquet row stores one packed `observation.state` and one packed `action`.
The source global `map` poses are used internally, but global `x/y/yaw` are not
stored directly as the training state.

```text
state_v1:
  ego:
    v                  # recorded vx
    omega              # recorded wz
  goal:
    dx_goal_body       # goal relative to ego, ego body frame
    dy_goal_body
    dtheta_goal
  team:
    repeated team slots:
      valid_mask       # 1 when this slot contains a real teammate, else 0
      dx_body          # teammate relative to ego, ego body frame
      dy_body
      dtheta
      v                # teammate recorded vx
      omega            # teammate recorded wz
      agent_type_id

action_v1:
  cmd_vx
  cmd_wz
```

For the stratified `nova_carter` split below, all train/val/test roots include
up to four total robots, so the state/action GEAR slices are:

```bash
--state-keys '{"ego": [0, 2], "goal": [2, 5], "team": [5, 26]}'
--action-keys '{"cmd_vel": [0, 2]}'
```

Always check `<lerobot_root>/meta/isaac_vln_debug.json` for the exact
`gear_command_hint` before running the GEAR converter.

## Train/Val/Test Split

`convert_lerobot_to_gear.py` does not split datasets. It converts every episode
under one LeRobot root into GEAR metadata. Create separate LeRobot roots for
train/val/test first, then run the GEAR converter separately on each root.

The current scenes are organized as five scene templates with five randomized
scenes per template:

```text
template_0: scenes 1-5
template_1: scenes 6-10
template_2: scenes 11-15
template_3: scenes 16-20
template_4: scenes 21-25
```

For the first single-agent `nova_carter` policy, use a stratified split so every
template, including the unique-object final template, appears in train/val/test:

```text
train: scenes 1 2 3 6 7 8 11 12 13 16 17 18 21 22 23
val:   scenes 4 9 14 19 24
test:  scenes 5 10 15 20 25
```

Commands:

```bash
# Train
python isaac_vln/data/convert_manifest_to_lerobot.py \
  --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf \
  --out-root ./datasets/nova_carter_lerobot_train \
  --embodiment nova_carter \
  --scene-ids 1 2 3 6 7 8 11 12 13 16 17 18 21 22 23 \
  --split train \
  --overwrite

# Val
python isaac_vln/data/convert_manifest_to_lerobot.py \
  --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf \
  --out-root ./datasets/nova_carter_lerobot_val \
  --embodiment nova_carter \
  --scene-ids 4 9 14 19 24 \
  --split val \
  --overwrite

# Test
python isaac_vln/data/convert_manifest_to_lerobot.py \
  --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf \
  --out-root ./datasets/nova_carter_lerobot_test \
  --embodiment nova_carter \
  --scene-ids 5 10 15 20 25 \
  --split test \
  --overwrite

# For the very first version of stablizing training loop and 
# no testing involves
# Train
python isaac_vln/data/convert_manifest_to_lerobot.py \
  --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf \
  --out-root ./datasets/nova_carter_lerobot_train \
  --embodiment nova_carter \
  --scene-ids 1 2 3 4 6 7 8 9 11 12 13 14 16 17 18 19 21 22 23 24 \
  --split train \
  --overwrite

# Val
python isaac_vln/data/convert_manifest_to_lerobot.py \
  --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf \
  --out-root ./datasets/nova_carter_lerobot_val \
  --embodiment nova_carter \
  --scene-ids 5 10 15 20 25 \
  --split val \
  --overwrite
```

## LeRobot To GEAR

Run GEAR conversion separately per split root:

```bash
python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path ./datasets/nova_carter_lerobot_train \
  --embodiment-tag nova_carter \
  --state-keys '{"ego": [0, 2], "goal": [2, 5], "team": [5, 26]}' \
  --action-keys '{"cmd_vel": [0, 2]}' \
  --task-key annotation.task \
  --force
```

Repeat for `nova_carter_lerobot_val` and `nova_carter_lerobot_test` if you want
GEAR metadata for evaluation roots. Do not pass `--relative-action-keys` for V1;
the action is absolute commanded velocity `[cmd_vx, cmd_wz]`.

## DreamZero Model-Side Setup

The LeRobot and GEAR converters only prepare data files and metadata. A new
DreamZero embodiment also needs model-side registration before training.

For `nova_carter`, check that these pieces exist:

- `groot/vla/data/schema/embodiment_tags.py` has `NOVA_CARTER = "nova_carter"`.
- `groot/vla/configs/model/dreamzero/transform/base.yaml` maps
  `nova_carter` to a projector id.
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` handles the
  `NOVA_CARTER` language prompt and view layout.
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
  defines `modality_config_nova_carter`, `transform_nova_carter`, and
  `fps.nova_carter: 10`.
- `groot/vla/configs/data/dreamzero/nova_carter.yaml` uses
  `relative_action: false`, `relative_action_keys: []`, and points
  `dataset_path.nova_carter` at `${nova_carter_data_root}`.

The expected Nova Carter view order is:

```text
modality_config_nova_carter.video.modality_keys:
  - video.ego_front
  - video.third_view_0
  - video.third_view_1

DreamZero packed canvas:
  [ego_front    | third_view_1]
  [third_view_0 | black]
```

The canvas packing happens during training, not during conversion. The MP4 files
remain normal per-camera videos.

## Training Resolution And `frame_seqlen`

`image_resolution_width` and `image_resolution_height` are per-view resize
settings in the DreamZero data transform. For Nova Carter, the transform packs
three views into a 2x2 canvas, so the model sees a canvas with doubled width and
height.

For the Wan2.1 14B action head, `frame_seqlen` is the number of latent patch
tokens per frame:

```text
canvas_width = 2 * image_resolution_width
canvas_height = 2 * image_resolution_height
frame_seqlen = (canvas_width / 16) * (canvas_height / 16)
```

Common settings:

```text
320x176 per view -> 640x352 canvas -> frame_seqlen = 40 * 22 = 880
224x224 per view -> 448x448 canvas -> frame_seqlen = 28 * 28 = 784
```

If you keep the native Isaac camera resolution at `224x224`, use
`frame_seqlen=784` in the training script. If you switch to the existing
DreamZero rectangular setting `320x176`, use `frame_seqlen=880`.

## Training Launch

Run training against the GEAR-converted train root. The GEAR converter writes
metadata into the same dataset root, so the directory name may still include
`lerobot`, but it must contain files such as:

```text
meta/embodiment.json
meta/modality.json
meta/stats.json
meta/tasks.jsonl
meta/episodes.jsonl
```

If running inside Docker, `DATA_ROOT` must be the path visible inside the
container. Set `NUM_GPUS` explicitly if `nvidia-smi` is not visible from the
container shell or if you want to pin the number of ranks:

```bash
DATA_ROOT=/data/datasets/mas-vln-lerobot/nova_carter_lerobot_train \
NUM_GPUS=8 \
  bash scripts/train/nova_carter_training.sh
```

The training script auto-detects the repo root from its own path, exports that
path on `PYTHONPATH`, and runs from the repo root so `groot` imports work inside
Docker even when the package is not pip-installed. If the repo is mounted at a
nonstandard path, set `DREAMZERO_ROOT` to the container-visible repo root:

```bash
DREAMZERO_ROOT=/workspace/dreamzero-vln \
DATA_ROOT=/data/datasets/mas-vln-lerobot/nova_carter_lerobot_train \
  bash scripts/train/nova_carter_training.sh
```

Extra Hydra overrides can be passed after the script name:

```bash
DATA_ROOT=/data/datasets/mas-vln-lerobot/nova_carter_lerobot_train \
  bash scripts/train/nova_carter_training.sh max_steps=1000 save_steps=500
```

The first V1 run should use the available DreamZero checkpoint:

```bash
hf download GEAR-Dreams/DreamZero-AgiBot \
  --repo-type model \
  --local-dir ./checkpoints/DreamZero-AgiBot
```

This checkpoint is used as the LoRA fine-tuning initialization. It is not
Nova-Carter-specific, but it is the current DreamZero pretrained policy
checkpoint expected by `scripts/train/nova_carter_training.sh`.
