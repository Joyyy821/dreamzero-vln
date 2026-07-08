#!/usr/bin/env python3
"""Convert MAS-VLN Isaac Sim rollout metadata to LeRobot datasets."""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
import sys
import tarfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from PIL import Image


DEFAULT_DATASET_ROOT = Path("/home/yjiao/Datasets/ma_vln_isaac_sim_hf")
DEFAULT_SCENE_ID = 1
DEFAULT_ROLLOUT_ID = 3
DEFAULT_EGO_AGENTS = ("nova_carter",)
DEFAULT_DEBUG_MAX_EPISODES = 2
DEFAULT_DEBUG_MAX_STEPS = 300

STATE_EGO_KEYS = ("v", "omega")
STATE_GOAL_KEYS = ("dx_goal_body", "dy_goal_body", "dtheta_goal")
STATE_TEAM_KEYS = (
    "valid_mask",
    "dx_body",
    "dy_body",
    "dtheta",
    "v",
    "omega",
    "agent_type_id",
)
ACTION_KEYS = ("cmd_vx", "cmd_wz")
TEAM_SLOT_WIDTH = len(STATE_TEAM_KEYS)
CHUNK_SIZE = 1000


@dataclass(frozen=True)
class ManifestEntry:
    dataset_root: Path
    scene_id: int
    rollout_id: int
    ego_agent: str
    third_view_cameras: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    scene_id: int
    rollout_id: int
    ego_agent: str
    embodiment_tag: str
    split: str
    instruction: str
    length: int
    timestamps_ns: tuple[int, ...]
    video_keys: tuple[str, ...]
    source_cameras: dict[str, str]
    rows: list[dict[str, Any]]
    state_dim: int
    action_dim: int
    fps: float


@dataclass(frozen=True)
class RobotTimeline:
    timestamps_ns: tuple[int, ...]
    rows: tuple[dict[str, Any], ...]
    by_timestamp_ns: dict[int, dict[str, Any]]


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def relative_pose_body(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    target_x: float,
    target_y: float,
    target_yaw: float,
) -> tuple[float, float, float]:
    """Return target pose relative to ego, expressed in ego body frame."""
    dx_map = float(target_x) - float(ego_x)
    dy_map = float(target_y) - float(ego_y)
    cos_yaw = math.cos(float(ego_yaw))
    sin_yaw = math.sin(float(ego_yaw))
    dx_body = cos_yaw * dx_map + sin_yaw * dy_map
    dy_body = -sin_yaw * dx_map + cos_yaw * dy_map
    dtheta = wrap_to_pi(float(target_yaw) - float(ego_yaw))
    return dx_body, dy_body, dtheta


def stable_team_names(robot_order: Sequence[str], ego_agent: str) -> tuple[str, ...]:
    """Return non-ego agents in stable rollout order."""
    return tuple(name for name in robot_order if name != ego_agent)


def pack_state_v1(
    *,
    ego_sample: Mapping[str, float],
    goal_pose: Mapping[str, float],
    team_samples: Sequence[tuple[str, Mapping[str, float] | None, int]],
    max_team_slots: int,
) -> list[float]:
    """Pack the differential-drive v1 state vector."""
    ego_x = _required_float(ego_sample, "x")
    ego_y = _required_float(ego_sample, "y")
    ego_yaw = _required_float(ego_sample, "yaw")

    dx_goal, dy_goal, dtheta_goal = relative_pose_body(
        ego_x,
        ego_y,
        ego_yaw,
        _required_float(goal_pose, "x"),
        _required_float(goal_pose, "y"),
        _required_float(goal_pose, "yaw"),
    )

    state = [
        _required_float(ego_sample, "vx"),
        _required_float(ego_sample, "wz"),
        dx_goal,
        dy_goal,
        dtheta_goal,
    ]

    for _, team_sample, agent_type_id in team_samples[:max_team_slots]:
        if team_sample is None:
            state.extend([0.0] * TEAM_SLOT_WIDTH)
            continue
        dx_body, dy_body, dtheta = relative_pose_body(
            ego_x,
            ego_y,
            ego_yaw,
            _required_float(team_sample, "x"),
            _required_float(team_sample, "y"),
            _required_float(team_sample, "yaw"),
        )
        state.extend(
            [
                1.0,
                dx_body,
                dy_body,
                dtheta,
                _required_float(team_sample, "vx"),
                _required_float(team_sample, "wz"),
                float(agent_type_id),
            ]
        )

    for _ in range(max(0, max_team_slots - len(team_samples))):
        state.extend([0.0] * TEAM_SLOT_WIDTH)

    return [float(value) for value in state]


def pack_action_v1(ego_sample: Mapping[str, float]) -> list[float]:
    return [
        _required_float(ego_sample, "cmd_vx"),
        _required_float(ego_sample, "cmd_wz"),
    ]


def state_key_slices(max_team_slots: int) -> dict[str, list[int]]:
    start = 0
    mapping = {"ego": [start, start + len(STATE_EGO_KEYS)]}
    start += len(STATE_EGO_KEYS)
    mapping["goal"] = [start, start + len(STATE_GOAL_KEYS)]
    start += len(STATE_GOAL_KEYS)
    if max_team_slots:
        mapping["team"] = [start, start + max_team_slots * TEAM_SLOT_WIDTH]
    return mapping


def action_key_slices() -> dict[str, list[int]]:
    return {"cmd_vel": [0, len(ACTION_KEYS)]}


def build_default_manifest_entries(
    *,
    dataset_root: Path,
    scene_id: int,
    rollout_id: int,
    ego_agents: Sequence[str],
    third_view_cameras: Sequence[str] | None,
    split: str,
) -> list[ManifestEntry]:
    return [
        ManifestEntry(
            dataset_root=dataset_root,
            scene_id=int(scene_id),
            rollout_id=int(rollout_id),
            ego_agent=str(ego_agent),
            third_view_cameras=tuple(third_view_cameras or ()),
            split=split,
        )
        for ego_agent in ego_agents
    ]


def read_manifest_jsonl(path: Path, fallback_dataset_root: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            dataset_root = Path(payload.get("dataset_root") or fallback_dataset_root).expanduser()
            ego_agents = payload.get("ego_agents")
            if ego_agents is None:
                ego_agents = [payload["ego_agent"]]
            third_views = tuple(payload.get("third_view_cameras") or ())
            for ego_agent in ego_agents:
                try:
                    entries.append(
                        ManifestEntry(
                            dataset_root=dataset_root,
                            scene_id=int(payload["scene_id"]),
                            rollout_id=int(payload["rollout_id"]),
                            ego_agent=str(ego_agent),
                            third_view_cameras=third_views,
                            split=str(payload.get("split") or "train"),
                        )
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"{path}:{line_number} is missing required field {exc.args[0]!r}"
                    ) from exc
    if not entries:
        raise ValueError(f"No manifest entries found in {path}")
    return entries


def build_embodiment_manifest_entries(
    *,
    dataset_root: Path,
    embodiment: str,
    rollouts: pd.DataFrame,
    third_view_cameras: Sequence[str] | None,
    split_override: str | None,
    scene_id: int | None = None,
    rollout_id: int | None = None,
) -> list[ManifestEntry]:
    """Build one per-ego entry for every packaged rollout containing an embodiment."""
    embodiment = str(embodiment).strip()
    if not embodiment:
        raise ValueError("--embodiment must be non-empty")

    rows = rollouts.copy()
    if "success" in rows.columns:
        rows = rows[rows["success"].astype(bool)]
    if "package_status" in rows.columns:
        rows = rows[rows["package_status"].astype(str) == "packaged"]
    if scene_id is not None:
        rows = rows[rows["scene_id"] == int(scene_id)]
    if rollout_id is not None:
        rows = rows[rows["rollout_id"] == int(rollout_id)]
    rows = rows.sort_values(["scene_id", "rollout_id"], kind="stable")

    entries: list[ManifestEntry] = []
    for row in rows.to_dict(orient="records"):
        robot_names = _json_list(row.get("robot_names"))
        robot_models = _json_list(row.get("robot_models"))
        if len(robot_models) < len(robot_names):
            robot_models.extend(robot_names[len(robot_models) :])
        for robot_name, robot_model in zip(robot_names, robot_models):
            if embodiment not in {str(robot_name), str(robot_model)}:
                continue
            entries.append(
                ManifestEntry(
                    dataset_root=dataset_root,
                    scene_id=int(row["scene_id"]),
                    rollout_id=int(row["rollout_id"]),
                    ego_agent=str(robot_name),
                    third_view_cameras=tuple(third_view_cameras or ()),
                    split=str(split_override or row.get("split") or "train"),
                )
            )

    if not entries:
        available = ", ".join(available_embodiments_from_rollouts(rollouts)) or "none"
        raise ValueError(
            f"No packaged rollout entries found for embodiment {embodiment!r}. "
            f"Available embodiments: {available}"
        )
    return entries


def available_embodiments_from_rollouts(rollouts: pd.DataFrame) -> list[str]:
    embodiments: set[str] = set()
    for row in rollouts.to_dict(orient="records"):
        for value in _json_list(row.get("robot_models")):
            if str(value):
                embodiments.add(str(value))
        for value in _json_list(row.get("robot_names")):
            if str(value):
                embodiments.add(str(value))
    return sorted(embodiments)


def convert_dataset(
    entries: Sequence[ManifestEntry],
    out_root: Path,
    *,
    max_episodes: int | None,
    max_steps: int | None,
    requested_fps: float | None,
    image_width: int | None,
    image_height: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    entries = list(entries)
    if max_episodes is not None:
        entries = entries[:max_episodes]
    if not entries:
        raise ValueError("No conversion entries selected")

    roots = {entry.dataset_root.expanduser().resolve() for entry in entries}
    metadata_cache = {root: _load_release_metadata(root) for root in roots}
    run_config_cache: dict[tuple[Path, int, int], dict[str, Any]] = {}

    max_team_slots = _infer_max_team_slots(entries, metadata_cache, run_config_cache)
    agent_type_ids = _infer_agent_type_ids(entries, metadata_cache, run_config_cache)
    embodiment_tag = _single_embodiment_tag(
        _infer_ego_embodiment_tags(entries, metadata_cache, run_config_cache)
    )

    out_root = out_root.expanduser()
    if out_root.exists():
        if overwrite:
            shutil.rmtree(out_root)
        elif any(out_root.iterdir()):
            raise FileExistsError(f"Output root already exists and is not empty: {out_root}")
    (out_root / "meta").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    all_results: list[EpisodeResult] = []
    global_row_index = 0

    for episode_index, entry in enumerate(entries):
        result = _build_episode(
            entry,
            episode_index=episode_index,
            task_index_lookup=tasks,
            out_root=out_root,
            metadata_cache=metadata_cache,
            run_config_cache=run_config_cache,
            max_team_slots=max_team_slots,
            agent_type_ids=agent_type_ids,
            max_steps=max_steps,
            requested_fps=requested_fps,
            image_width=image_width,
            image_height=image_height,
            global_row_index_start=global_row_index,
        )
        global_row_index += result.length
        all_results.append(result)

    _write_meta_files(
        out_root,
        all_results,
        tasks,
        max_team_slots,
        agent_type_ids,
        embodiment_tag=embodiment_tag,
    )

    summary = {
        "out_root": str(out_root),
        "episodes": len(all_results),
        "total_frames": sum(result.length for result in all_results),
        "state_dim": all_results[0].state_dim,
        "action_dim": all_results[0].action_dim,
        "video_keys": list(all_results[0].video_keys),
        "embodiment_tag": embodiment_tag,
        "tasks": tasks,
    }
    return summary


def _build_episode(
    entry: ManifestEntry,
    *,
    episode_index: int,
    task_index_lookup: dict[str, int],
    out_root: Path,
    metadata_cache: Mapping[Path, dict[str, pd.DataFrame]],
    run_config_cache: dict[tuple[Path, int, int], dict[str, Any]],
    max_team_slots: int,
    agent_type_ids: Mapping[str, int],
    max_steps: int | None,
    requested_fps: float | None,
    image_width: int | None,
    image_height: int | None,
    global_row_index_start: int,
) -> EpisodeResult:
    dataset_root = entry.dataset_root.expanduser().resolve()
    metadata = metadata_cache[dataset_root]
    rollout_row = _select_rollout_row(metadata["rollouts"], entry.scene_id, entry.rollout_id)
    tar_path = dataset_root / str(rollout_row["tar_path"])
    frames = _select_frame_rows(metadata["frames"], entry.scene_id, entry.rollout_id)
    run_config = _load_run_config(dataset_root, entry.scene_id, entry.rollout_id, tar_path, run_config_cache)

    robot_entries = list(((run_config.get("team_config") or {}).get("robots") or []))
    if not robot_entries:
        raise ValueError(f"Rollout {entry.scene_id}/{entry.rollout_id} has no robots")
    robot_order = [str(robot["name"]) for robot in robot_entries]
    if entry.ego_agent not in robot_order:
        raise ValueError(
            f"Ego agent {entry.ego_agent!r} is not in rollout robots {robot_order}"
        )
    robot_by_name = {str(robot["name"]): robot for robot in robot_entries}
    model_by_name = {
        name: str(robot_by_name[name].get("model") or name)
        for name in robot_order
    }
    embodiment_tag = _robot_embodiment_tag(robot_by_name[entry.ego_agent], entry.ego_agent)

    third_view_cameras = entry.third_view_cameras or _default_third_view_cameras(frames)
    if not third_view_cameras:
        raise ValueError(f"No third-view BEV cameras found for scene {entry.scene_id} rollout {entry.rollout_id}")

    selected_camera_names = (entry.ego_agent, *third_view_cameras)
    required_source_names = tuple(dict.fromkeys(selected_camera_names))
    timestamps = _common_timestamps(frames, required_source_names)
    output_fps = requested_fps or infer_native_fps(timestamps)
    timestamps = select_timestamps(timestamps, fps=requested_fps, max_steps=max_steps)
    if not timestamps:
        raise ValueError(
            f"No synchronized timestamps for scene {entry.scene_id} rollout {entry.rollout_id} ego {entry.ego_agent}"
        )

    view_keys = ("ego_front", *[f"third_view_{idx}" for idx in range(len(third_view_cameras))])
    source_cameras = dict(zip(view_keys, selected_camera_names))
    frame_lookup = _build_frame_lookup(frames)
    robot_timelines = _build_robot_timelines(frames, robot_order)

    instruction = str(run_config.get("language_instruction") or rollout_row.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"Empty instruction for scene {entry.scene_id} rollout {entry.rollout_id}")
    task_index = task_index_lookup.setdefault(instruction, len(task_index_lookup))

    rows: list[dict[str, Any]] = []
    team_order = stable_team_names(robot_order, entry.ego_agent)
    for local_frame_index, timestamp_ns in enumerate(timestamps):
        ego_sample = _sample_robot_timeline(
            robot_timelines[entry.ego_agent],
            timestamp_ns,
            hold_last=False,
            robot_name=entry.ego_agent,
        )
        goal_pose = robot_by_name[entry.ego_agent]["goal_pose"]
        team_samples = [
            (
                team_name,
                _sample_robot_timeline(
                    robot_timelines[team_name],
                    timestamp_ns,
                    hold_last=True,
                    robot_name=team_name,
                ),
                agent_type_ids[model_by_name[team_name]],
            )
            for team_name in team_order
        ]
        state = pack_state_v1(
            ego_sample=ego_sample,
            goal_pose=goal_pose,
            team_samples=team_samples,
            max_team_slots=max_team_slots,
        )
        action = pack_action_v1(ego_sample)
        rows.append(
            {
                "observation.state": state,
                "action": action,
                "timestamp": float(local_frame_index / output_fps),
                "episode_index": int(episode_index),
                "frame_index": int(local_frame_index),
                "task_index": int(task_index),
                "annotation.task": instruction,
                "index": int(global_row_index_start + local_frame_index),
                "scene_id": int(entry.scene_id),
                "rollout_id": int(entry.rollout_id),
                "ego_agent": entry.ego_agent,
                "source_timestamp_ns": int(timestamp_ns),
                "source_elapsed_s": float(
                    (int(timestamp_ns) - int(timestamps[0])) / 1_000_000_000.0
                ),
            }
        )

    chunk_idx = episode_index // CHUNK_SIZE
    data_dir = out_root / f"data/chunk-{chunk_idx:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / f"episode_{episode_index:06d}.parquet")

    _write_episode_videos(
        tar_path=tar_path,
        frame_lookup=frame_lookup,
        timestamps=timestamps,
        source_cameras=source_cameras,
        episode_index=episode_index,
        out_root=out_root,
        fps=output_fps,
        image_width=image_width,
        image_height=image_height,
    )

    return EpisodeResult(
        episode_index=episode_index,
        scene_id=entry.scene_id,
        rollout_id=entry.rollout_id,
        ego_agent=entry.ego_agent,
        embodiment_tag=embodiment_tag,
        split=entry.split or str(rollout_row.get("split") or "train"),
        instruction=instruction,
        length=len(rows),
        timestamps_ns=tuple(timestamps),
        video_keys=view_keys,
        source_cameras=source_cameras,
        rows=rows,
        state_dim=len(rows[0]["observation.state"]),
        action_dim=len(rows[0]["action"]),
        fps=output_fps,
    )


def _write_episode_videos(
    *,
    tar_path: Path,
    frame_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    timestamps: Sequence[int],
    source_cameras: Mapping[str, str],
    episode_index: int,
    out_root: Path,
    fps: float,
    image_width: int | None,
    image_height: int | None,
) -> None:
    chunk_idx = episode_index // CHUNK_SIZE
    with tarfile.open(tar_path, "r") as tar:
        for video_key, source_camera in source_cameras.items():
            frames: list[np.ndarray] = []
            for timestamp_ns in timestamps:
                row = frame_lookup[(source_camera, int(timestamp_ns))]
                frame = _read_rgb_frame(
                    tar,
                    str(row["rgb_path_in_tar"]),
                    image_width=image_width,
                    image_height=image_height,
                )
                frames.append(frame)
            video_dir = out_root / f"videos/chunk-{chunk_idx:03d}/observation.images.{video_key}"
            video_dir.mkdir(parents=True, exist_ok=True)
            _write_mp4_ffmpeg(
                np.stack(frames, axis=0),
                video_dir / f"episode_{episode_index:06d}.mp4",
                fps=fps,
            )


def _read_rgb_frame(
    tar: tarfile.TarFile,
    member_name: str,
    *,
    image_width: int | None,
    image_height: int | None,
) -> np.ndarray:
    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract {member_name}")
    with Image.open(io.BytesIO(extracted.read())) as image:
        image = image.convert("RGB")
        if image_width is not None or image_height is not None:
            if image_width is None or image_height is None:
                raise ValueError("--image-width and --image-height must be provided together")
            image = image.resize((int(image_width), int(image_height)), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _write_mp4_ffmpeg(frames: np.ndarray, output_path: Path, *, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames, got {frames.dtype}")
    height, width = frames.shape[1:3]
    if width % 2 or height % 2:
        raise ValueError(f"H.264 yuv420p output requires even resolution, got {width}x{height}")
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(output_path),
    ]
    try:
        process = subprocess.run(
            command,
            input=frames.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to write debug MP4 videos") from exc
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed writing {output_path}:\n{stderr[-4000:]}")


def _write_meta_files(
    out_root: Path,
    results: Sequence[EpisodeResult],
    tasks: Mapping[str, int],
    max_team_slots: int,
    agent_type_ids: Mapping[str, int],
    *,
    embodiment_tag: str,
) -> None:
    result_tags = {result.embodiment_tag for result in results}
    if result_tags != {embodiment_tag}:
        raise ValueError(
            f"Converted episodes have mixed embodiment tags {sorted(result_tags)}; "
            "write one ego embodiment per LeRobot/GEAR dataset for v1."
        )

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for name in ("tasks.jsonl", "episodes.jsonl"):
        path = meta_dir / name
        if path.exists():
            path.unlink()

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task, task_index in sorted(tasks.items(), key=lambda item: item[1]):
            stream.write(json.dumps({"task_index": int(task_index), "task": task}) + "\n")

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(
                json.dumps(
                    {
                        "episode_index": result.episode_index,
                        "tasks": [result.instruction],
                        "length": result.length,
                        "scene_id": result.scene_id,
                        "rollout_id": result.rollout_id,
                        "ego_agent": result.ego_agent,
                        "embodiment_tag": result.embodiment_tag,
                        "split": result.split,
                    }
                )
                + "\n"
            )

    state_dim = results[0].state_dim
    action_dim = results[0].action_dim
    height, width = _first_video_resolution(out_root, results[0].video_keys[0])
    video_features = {
        f"observation.images.{video_key}": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": results[0].fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        for video_key in results[0].video_keys
    }
    total_frames = sum(result.length for result in results)
    split_ranges = _build_split_ranges(results)
    info = {
        "codebase_version": "v2.0",
        "robot_type": embodiment_tag,
        "total_episodes": len(results),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(results[0].video_keys),
        "total_chunks": (len(results) // CHUNK_SIZE) + (1 if len(results) % CHUNK_SIZE else 0),
        "chunks_size": CHUNK_SIZE,
        "fps": results[0].fps,
        "splits": split_ranges,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **video_features,
            "observation.state": {
                "dtype": "float64",
                "shape": [state_dim],
                "names": _state_names(max_team_slots),
            },
            "action": {
                "dtype": "float64",
                "shape": [action_dim],
                "names": list(ACTION_KEYS),
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.task": {"dtype": "string", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "scene_id": {"dtype": "int64", "shape": [1]},
            "rollout_id": {"dtype": "int64", "shape": [1]},
            "ego_agent": {"dtype": "string", "shape": [1]},
            "source_timestamp_ns": {"dtype": "int64", "shape": [1]},
            "source_elapsed_s": {"dtype": "float64", "shape": [1]},
        },
    }
    _write_json(meta_dir / "info.json", info)

    debug = {
        "schema_version": "isaac_vln_lerobot_debug_v1",
        "state_version": "v1",
        "action_type": "cmd_vel",
        "state_keys": state_key_slices(max_team_slots),
        "action_keys": action_key_slices(),
        "state_names": _state_names(max_team_slots),
        "action_names": list(ACTION_KEYS),
        "embodiment_tag": embodiment_tag,
        "max_team_slots": max_team_slots,
        "agent_type_ids": dict(sorted(agent_type_ids.items(), key=lambda item: item[1])),
        "gear_command_hint": {
            "state_keys": json.dumps(state_key_slices(max_team_slots), sort_keys=True),
            "action_keys": json.dumps(action_key_slices(), sort_keys=True),
            "relative_action_keys": [],
            "task_key": "annotation.task",
            "embodiment_tag": embodiment_tag,
        },
        "episodes": [
            {
                "episode_index": result.episode_index,
                "scene_id": result.scene_id,
                "rollout_id": result.rollout_id,
                "ego_agent": result.ego_agent,
                "embodiment_tag": result.embodiment_tag,
                "split": result.split,
                "length": result.length,
                "fps": result.fps,
                "source_cameras": result.source_cameras,
                "first_timestamp_ns": result.timestamps_ns[0],
                "last_timestamp_ns": result.timestamps_ns[-1],
            }
            for result in results
        ],
    }
    _write_json(meta_dir / "isaac_vln_debug.json", debug)


def _first_video_resolution(out_root: Path, video_key: str) -> tuple[int, int]:
    path = out_root / f"videos/chunk-000/observation.images.{video_key}/episode_000000.mp4"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {probe.stderr.decode(errors='replace')}")
    payload = json.loads(probe.stdout.decode("utf-8"))
    stream = payload["streams"][0]
    return int(stream["height"]), int(stream["width"])


def _state_names(max_team_slots: int) -> list[str]:
    names = [f"ego.{name}" for name in STATE_EGO_KEYS]
    names.extend(f"goal.{name}" for name in STATE_GOAL_KEYS)
    for slot_idx in range(max_team_slots):
        names.extend(f"team_{slot_idx}.{name}" for name in STATE_TEAM_KEYS)
    return names


def _build_split_ranges(results: Sequence[EpisodeResult]) -> dict[str, str]:
    by_split: dict[str, list[int]] = {}
    for result in results:
        by_split.setdefault(result.split or "train", []).append(result.episode_index)
    ranges: dict[str, str] = {}
    for split, indices in sorted(by_split.items()):
        indices = sorted(indices)
        ranges[split] = f"{indices[0]}:{indices[-1] + 1}"
    return ranges


def _infer_max_team_slots(
    entries: Sequence[ManifestEntry],
    metadata_cache: Mapping[Path, dict[str, pd.DataFrame]],
    run_config_cache: dict[tuple[Path, int, int], dict[str, Any]],
) -> int:
    max_robots = 1
    for entry in entries:
        dataset_root = entry.dataset_root.expanduser().resolve()
        rollout_row = _select_rollout_row(
            metadata_cache[dataset_root]["rollouts"],
            entry.scene_id,
            entry.rollout_id,
        )
        run_config = _load_run_config(
            dataset_root,
            entry.scene_id,
            entry.rollout_id,
            dataset_root / str(rollout_row["tar_path"]),
            run_config_cache,
        )
        robots = list(((run_config.get("team_config") or {}).get("robots") or []))
        max_robots = max(max_robots, len(robots))
    return max(0, max_robots - 1)


def _infer_agent_type_ids(
    entries: Sequence[ManifestEntry],
    metadata_cache: Mapping[Path, dict[str, pd.DataFrame]],
    run_config_cache: dict[tuple[Path, int, int], dict[str, Any]],
) -> dict[str, int]:
    models: set[str] = set()
    for entry in entries:
        dataset_root = entry.dataset_root.expanduser().resolve()
        rollout_row = _select_rollout_row(
            metadata_cache[dataset_root]["rollouts"],
            entry.scene_id,
            entry.rollout_id,
        )
        run_config = _load_run_config(
            dataset_root,
            entry.scene_id,
            entry.rollout_id,
            dataset_root / str(rollout_row["tar_path"]),
            run_config_cache,
        )
        for robot in list(((run_config.get("team_config") or {}).get("robots") or [])):
            name = str(robot.get("name") or "")
            models.add(str(robot.get("model") or name))
    return {model: idx for idx, model in enumerate(sorted(models))}


def _infer_ego_embodiment_tags(
    entries: Sequence[ManifestEntry],
    metadata_cache: Mapping[Path, dict[str, pd.DataFrame]],
    run_config_cache: dict[tuple[Path, int, int], dict[str, Any]],
) -> list[str]:
    tags: list[str] = []
    for entry in entries:
        dataset_root = entry.dataset_root.expanduser().resolve()
        rollout_row = _select_rollout_row(
            metadata_cache[dataset_root]["rollouts"],
            entry.scene_id,
            entry.rollout_id,
        )
        run_config = _load_run_config(
            dataset_root,
            entry.scene_id,
            entry.rollout_id,
            dataset_root / str(rollout_row["tar_path"]),
            run_config_cache,
        )
        robot_entries = list(((run_config.get("team_config") or {}).get("robots") or []))
        robot_by_name = {str(robot.get("name") or ""): robot for robot in robot_entries}
        if entry.ego_agent not in robot_by_name:
            raise ValueError(
                f"Ego agent {entry.ego_agent!r} is not in rollout robots {sorted(robot_by_name)}"
            )
        tags.append(_robot_embodiment_tag(robot_by_name[entry.ego_agent], entry.ego_agent))
    return tags


def _robot_embodiment_tag(robot_entry: Mapping[str, Any], fallback_name: str) -> str:
    tag = str(robot_entry.get("model") or fallback_name).strip()
    if not tag:
        raise ValueError(f"Could not infer embodiment tag for ego agent {fallback_name!r}")
    return tag


def _single_embodiment_tag(tags: Sequence[str]) -> str:
    unique = sorted({str(tag) for tag in tags})
    if not unique:
        raise ValueError("No ego embodiment tags inferred")
    if len(unique) > 1:
        raise ValueError(
            "V1 debug conversion requires one controlled ego embodiment per output dataset. "
            f"Selected ego agents resolve to multiple embodiment tags: {unique}. "
            "Run the converter once per ego robot type, for example with "
            "--ego-agents nova_carter, then --ego-agents carter_v1."
        )
    return unique[0]


def _load_release_metadata(dataset_root: Path) -> dict[str, pd.DataFrame]:
    metadata_dir = dataset_root / "metadata"
    required = {
        "rollouts": metadata_dir / "rollouts.parquet",
        "frames": metadata_dir / "frames.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing packaged dataset metadata files: {missing}")
    return {name: pd.read_parquet(path) for name, path in required.items()}


def _select_rollout_row(rollouts: pd.DataFrame, scene_id: int, rollout_id: int) -> pd.Series:
    rows = rollouts[(rollouts["scene_id"] == int(scene_id)) & (rollouts["rollout_id"] == int(rollout_id))]
    if rows.empty:
        raise ValueError(f"No rollout metadata row for scene {scene_id} rollout {rollout_id}")
    return rows.iloc[0]


def _select_frame_rows(frames: pd.DataFrame, scene_id: int, rollout_id: int) -> pd.DataFrame:
    rows = frames[(frames["scene_id"] == int(scene_id)) & (frames["rollout_id"] == int(rollout_id))]
    if rows.empty:
        raise ValueError(f"No frame rows for scene {scene_id} rollout {rollout_id}")
    return rows.copy()


def _load_run_config(
    dataset_root: Path,
    scene_id: int,
    rollout_id: int,
    tar_path: Path,
    cache: dict[tuple[Path, int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (dataset_root, int(scene_id), int(rollout_id))
    if key in cache:
        return cache[key]
    with tarfile.open(tar_path, "r") as tar:
        extracted = tar.extractfile("run_config.yaml")
        if extracted is None:
            raise FileNotFoundError(f"{tar_path} does not contain run_config.yaml")
        payload = yaml.safe_load(extracted.read().decode("utf-8")) or {}
    cache[key] = payload
    return payload


def _default_third_view_cameras(frames: pd.DataFrame) -> tuple[str, ...]:
    rows = frames[frames["camera_type"] == "bev"]
    return tuple(sorted(str(value) for value in rows["camera_name"].dropna().unique()))


def _common_timestamps(frames: pd.DataFrame, camera_names: Sequence[str]) -> list[int]:
    timestamp_sets: list[set[int]] = []
    for camera_name in camera_names:
        rows = frames[frames["camera_name"] == camera_name]
        if rows.empty:
            raise ValueError(f"Missing required camera/agent rows for {camera_name!r}")
        timestamp_sets.append(set(rows["timestamp_ns"].dropna().astype(np.int64).astype(int)))
    common = set.intersection(*timestamp_sets)
    return sorted(common)


def _build_frame_lookup(frames: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frames.to_dict(orient="records"):
        camera_name = str(row["camera_name"])
        timestamp_ns = int(row["timestamp_ns"])
        lookup[(camera_name, timestamp_ns)] = row
    return lookup


def _build_robot_timelines(frames: pd.DataFrame, robot_names: Sequence[str]) -> dict[str, RobotTimeline]:
    timelines: dict[str, RobotTimeline] = {}
    for robot_name in robot_names:
        rows = frames[frames["camera_name"] == robot_name].sort_values("timestamp_ns")
        if rows.empty:
            raise ValueError(f"Missing required robot rows for {robot_name!r}")
        by_timestamp: dict[int, dict[str, Any]] = {}
        for row in rows.to_dict(orient="records"):
            by_timestamp[int(row["timestamp_ns"])] = row
        timestamps = tuple(sorted(by_timestamp))
        ordered_rows = tuple(by_timestamp[timestamp] for timestamp in timestamps)
        timelines[robot_name] = RobotTimeline(
            timestamps_ns=timestamps,
            rows=ordered_rows,
            by_timestamp_ns=by_timestamp,
        )
    return timelines


def _sample_robot_timeline(
    timeline: RobotTimeline,
    timestamp_ns: int,
    *,
    hold_last: bool,
    robot_name: str,
) -> dict[str, Any] | None:
    timestamp_ns = int(timestamp_ns)
    exact = timeline.by_timestamp_ns.get(timestamp_ns)
    if exact is not None:
        return exact
    if not hold_last:
        raise KeyError(f"Missing exact timestamp {timestamp_ns} for required ego robot {robot_name!r}")

    idx = bisect_right(timeline.timestamps_ns, timestamp_ns) - 1
    if idx < 0:
        return None

    held = dict(timeline.rows[idx])
    held["timestamp_ns"] = timestamp_ns
    for key in ("vx", "vy", "wz", "cmd_vx", "cmd_vy", "cmd_wz"):
        if key in held:
            held[key] = 0.0
    return held


def infer_native_fps(timestamps_ns: Sequence[int]) -> float:
    if len(timestamps_ns) < 2:
        return 1.0
    diffs = np.diff(np.array(sorted(timestamps_ns), dtype=np.int64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    fps = 1_000_000_000.0 / float(np.median(diffs))
    return float(round(fps, 6))


def select_timestamps(
    timestamps_ns: Sequence[int],
    *,
    fps: float | None,
    max_steps: int | None,
) -> list[int]:
    timestamps = [int(value) for value in sorted(timestamps_ns)]
    if fps is not None:
        if fps <= 0:
            raise ValueError("--fps must be positive")
        target_dt_ns = int(round(1_000_000_000.0 / fps))
        selected: list[int] = []
        last_ts: int | None = None
        tolerance_ns = int(0.01 * 1_000_000_000)
        for timestamp in timestamps:
            if last_ts is None or timestamp - last_ts >= target_dt_ns - tolerance_ns:
                selected.append(timestamp)
                last_ts = timestamp
        timestamps = selected
    if max_steps is None:
        return timestamps
    return timestamps[:max_steps]


def _required_float(sample: Mapping[str, Any], key: str) -> float:
    value = sample.get(key)
    if value is None:
        raise ValueError(f"Missing required numeric field {key!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {key!r} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Field {key!r} is not finite: {value!r}")
    return number


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if isinstance(parsed, np.ndarray):
        return parsed.tolist()
    if isinstance(parsed, tuple):
        return list(parsed)
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"Expected JSON/list value, got {type(value).__name__}: {value!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MAS-VLN packaged Isaac Sim rollouts to a LeRobot dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--embodiment",
        default=None,
        help=(
            "Convert every packaged rollout containing this controlled ego robot model/name "
            "(for example nova_carter, carter_v1, jackal, or limo)."
        ),
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help=(
            "Optional scene filter. In debug mode, defaults to "
            f"{DEFAULT_SCENE_ID}; in --embodiment mode, omitted means all scenes."
        ),
    )
    parser.add_argument(
        "--rollout-id",
        type=int,
        default=None,
        help=(
            "Optional rollout filter. In debug mode, defaults to "
            f"{DEFAULT_ROLLOUT_ID}; in --embodiment mode, omitted means all rollouts."
        ),
    )
    parser.add_argument("--ego-agents", nargs="+", default=list(DEFAULT_EGO_AGENTS))
    parser.add_argument("--third-view-cameras", nargs="*", default=None)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Optional episode cap. Debug mode defaults to "
            f"{DEFAULT_DEBUG_MAX_EPISODES}; --embodiment and manifest modes default to no cap."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional per-episode timestep cap. Debug mode defaults to "
            f"{DEFAULT_DEBUG_MAX_STEPS}; --embodiment and manifest modes default to no cap."
        ),
    )
    parser.add_argument("--fps", type=float, default=None, help="Optional fixed output FPS/downsample rate.")
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--action-type", choices=["cmd_vel"], default="cmd_vel")
    parser.add_argument("--state-version", choices=["v1"], default="v1")
    parser.add_argument("--split", default=None, help="Optional split override; otherwise source split is preserved.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = args.dataset_root.expanduser()
    try:
        if args.manifest and args.embodiment:
            raise ValueError("--manifest and --embodiment are mutually exclusive")
        if args.manifest:
            entries = read_manifest_jsonl(args.manifest.expanduser(), dataset_root)
        elif args.embodiment:
            metadata = _load_release_metadata(dataset_root.resolve())
            entries = build_embodiment_manifest_entries(
                dataset_root=dataset_root,
                embodiment=args.embodiment,
                rollouts=metadata["rollouts"],
                third_view_cameras=args.third_view_cameras,
                split_override=args.split,
                scene_id=args.scene_id,
                rollout_id=args.rollout_id,
            )
        else:
            entries = build_default_manifest_entries(
                dataset_root=dataset_root,
                scene_id=args.scene_id if args.scene_id is not None else DEFAULT_SCENE_ID,
                rollout_id=args.rollout_id if args.rollout_id is not None else DEFAULT_ROLLOUT_ID,
                ego_agents=args.ego_agents,
                third_view_cameras=args.third_view_cameras,
                split=args.split or "train",
            )

        debug_default_mode = args.manifest is None and args.embodiment is None
        max_episodes = args.max_episodes
        max_steps = args.max_steps
        if debug_default_mode:
            if max_episodes is None:
                max_episodes = DEFAULT_DEBUG_MAX_EPISODES
            if max_steps is None:
                max_steps = DEFAULT_DEBUG_MAX_STEPS
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        summary = convert_dataset(
            entries,
            args.out_root,
            max_episodes=max_episodes,
            max_steps=max_steps,
            requested_fps=args.fps,
            image_width=args.image_width,
            image_height=args.image_height,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


# Command for v1 test on my local machine 
# (set max-steps to an arbitary large number for full episode conversion)
# python isaac_vln/data/convert_manifest_to_lerobot.py \
# --dataset-root /home/yjiao/Datasets/ma_vln_isaac_sim_hf/ \
# --out-root ./datasets/tmp_isaac_debug \
# --scene-id 1 --rollout-id 1 \
# --ego-agents nova_carter --max-steps 10000 --overwrite

if __name__ == "__main__":
    raise SystemExit(main())
