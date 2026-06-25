# Isaac VLN Raw Rollout Schema

This document defines the v1 contract between MAS-VLN Isaac Sim rollouts and the
DreamZero Isaac VLN data converter. It is intentionally grounded in the current
packaged release at [`mas-vln-isaac-rgbd`](https://huggingface.co/datasets/yang-jiao/mas-vln-isaac-rgbd)
and the raw randomized-warehouse rollout layout produced by [`mas-vln`](https://github.com/Joyyy821/mas-vln).

<!-- First version uses dataset dir on my local machine: /home/yjiao/Datasets/ma_vln_isaac_sim_hf -->

## Source Layouts

### Raw MAS-VLN rollout tree

Raw randomized-warehouse data is organized by scene and numeric rollout id:

```text
experiments/randomized_warehouse/
  collection_metadata.yaml
  scene_<id>/
    mapf_map.png
    mapf_map.yaml
    nav2_map.png
    nav2_map.yaml
    scene.usd
    scene_manifest.yaml
    team_config.yaml
    rollouts/
      <rollout_id>/
        run_config.yaml
        render_manifest.csv
        <agent_name>_velocity.csv
        rgb/<camera_folder>/frame_<timestamp_ns>.png
        depth/<camera_folder>/frame_<timestamp_ns>.png
```

`run_config.yaml` is the per-rollout snapshot used for conversion. It contains
the language instruction, record settings, and the rollout-specific robot list
with initial and goal poses. `team_config.yaml` contains all rollout pose sets
for the scene and is useful for broader dataset indexing, but the converter must
prefer the matching per-rollout snapshot when both exist.

### HF packaged release

The released dataset mirrors scene metadata and stores rollout contents in one
plain tar per rollout:

```text
ma_vln_isaac_sim_hf/
  README.md
  collection_metadata.yaml
  metadata/
    dataset_release.json
    schema.md
    scenes.parquet
    rollouts.parquet
    frames.parquet
  scenes/scene_XXX/
    mapf_map.png
    mapf_map.yaml
    nav2_map.png
    nav2_map.yaml
    scene_manifest.yaml
    team_config.yaml
  rollouts/scene_XXX/
    rollout_YYY.tar
```

`metadata/rollouts.parquet` is one row per packaged rollout. `tar_path` points
to the rollout tar. `metadata/frames.parquet` is one row per rendered camera
frame and includes `rgb_path_in_tar`, `depth_path_in_tar`, timestamp, camera
metadata, and nearest recorded robot pose and command velocity for robot camera
rows.

The DreamZero debug converter treats this packaged release as the preferred
manifest source. It must not modify the packaged release.

## Camera Names

Camera keys are sourced from `render_manifest.csv` and `frames.parquet`.

- Robot cameras use the agent namespace as `camera_name`, for example
  `nova_carter`, `carter_v1`, `jackal`, or `limo`.
- Third-view cameras use `camera_type == "bev"` in metadata and are stored in
  tar folders with a `bev_` prefix, for example `rgb/bev_half_north/...`.
- LeRobot output uses stable DreamZero-facing names:
  - `observation.images.ego_front`
  - `observation.images.third_view_0`
  - `observation.images.third_view_1`

By default, third-view cameras are all BEV cameras in stable name order. The
current release normally provides `half_north` and `half_south`.

## Timestamp Convention

Raw and packaged timestamps are integer nanoseconds from ROS or Isaac Sim
capture time. The LeRobot output writes fixed-rate RGB MP4 videos, so
`timestamp` is the video playback timestamp:

```text
timestamp = frame_index / output_fps
```

The original capture timestamp is preserved as `source_timestamp_ns`, and
`source_elapsed_s` stores elapsed source time relative to the first selected
source frame.

The converter only emits rows whose selected ego camera and selected third-view
cameras exist at the same source timestamp. For v1 debug conversion, optional
downsampling selects a monotonic subsequence near the requested output FPS.

## Pose Frame Convention

Robot poses in `<agent>_velocity.csv` and `frames.parquet` are planar poses in
the ROS `map` frame:

```text
x, y: meters in map
yaw: radians in map, wrapped to [-pi, pi]
```

Goal poses in `run_config.yaml` are also in the same `map` frame. State vectors
express relative goal and team geometry in the ego robot body frame at the
current timestep:

```text
dx_map = target_x - ego_x
dy_map = target_y - ego_y
dx_body =  cos(ego_yaw) * dx_map + sin(ego_yaw) * dy_map
dy_body = -sin(ego_yaw) * dx_map + cos(ego_yaw) * dy_map
dtheta = wrap_to_pi(target_yaw - ego_yaw)
```

## Velocity Convention

Recorded odometry velocity fields are:

- `vx`: robot forward linear velocity, meters per second
- `vy`: lateral velocity, meters per second, recorded for diagnostics but not
  used by differential-drive v1
- `wz`: yaw angular velocity, radians per second

Commanded action fields are:

- `cmd_vx`: commanded forward linear velocity, meters per second
- `cmd_vy`: commanded lateral velocity, ignored by v1
- `cmd_wz`: commanded yaw angular velocity, radians per second

For state v1, `v = vx` and `omega = wz`. For action v1,
`action = [cmd_vx, cmd_wz]`.

## Instruction Format

Language instruction text is read from `run_config.yaml` field
`language_instruction`. Packaged metadata duplicates this in
`rollouts.parquet["instruction"]`. The LeRobot output stores task ids in
`task_index` and `annotation.task`, with text resolved through
`meta/tasks.jsonl`.

Instructions must be non-empty for converted episodes.

## Agent Naming Convention

Agent names are ROS namespaces and are stable within a rollout:

```text
nova_carter
carter_v1
jackal
limo
```

The converter preserves the robot order from `run_config.yaml`. For per-agent
episodes, `ego_agent` selects the controlled robot. Other robots are packed into
team slots in that same stable rollout order, excluding the ego.

If a non-ego teammate stops emitting robot-camera rows before the ego episode
ends, the v1 converter keeps that teammate's slot and holds its last known pose
with zero velocity. This avoids shortening the ego imitation episode just
because another robot reached its goal earlier. A slot uses `valid_mask = 0`
only when no teammate sample is available yet or when the slot is empty.

`agent_type_id` is a deterministic integer assigned from sorted robot model
names observed in the selected conversion set. The mapping is written to
`meta/isaac_vln_debug.json`.

## Split Convention

The packaged release currently contains `split == "train"` rows. The converter
preserves an explicit split in manifest JSONL or packaged metadata. If no split
is present, it writes `train`. LeRobot `meta/info.json` uses split ranges over
converted episode indices.

## State and Action V1

All v1 LeRobot rows use one packed `observation.state` vector and one packed
`action` vector. `meta/modality.json`, generated later by the LeRobot-to-GEAR
converter, should expose named slices.

For `max_agents = N`, `max_team_slots = max(0, N - 1)`:

```text
state_v1:
  ego:
    v
    omega
  goal:
    dx_goal_body
    dy_goal_body
    dtheta_goal
  team:
    repeated max_team_slots times:
      valid_mask
      dx_body
      dy_body
      dtheta
      v
      omega
      agent_type_id
```

Thus:

```text
state_dim = 2 + 3 + 7 * max_team_slots
action_dim = 2
```

For the two-agent debug default, `state_dim = 12`.

The v1 action is:

```text
action_v1:
  cmd_vx
  cmd_wz
```

Future action chunks are produced by DreamZero `action.delta_indices` during
training, not packed into each parquet row.

## LeRobot Debug Output

The debug converter writes a LeRobot-style folder:

```text
isaac_vln_lerobot/
  data/chunk-000/
    episode_000000.parquet
  videos/chunk-000/
    observation.images.ego_front/
      episode_000000.mp4
    observation.images.third_view_0/
      episode_000000.mp4
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    isaac_vln_debug.json
```

Required parquet columns are:

- `observation.state`: list of float64, shape `[state_dim]`
- `action`: list of float64, shape `[2]`
- `timestamp`: float64 seconds
- `episode_index`: int64
- `frame_index`: int64, zero-based within the converted episode
- `task_index`: int64
- `annotation.task`: string language instruction text
- debug columns such as `scene_id`, `rollout_id`, `ego_agent`, and
  `source_timestamp_ns`

The converter writes RGB MP4 videos only. Depth is intentionally out of scope for
debug v1.
