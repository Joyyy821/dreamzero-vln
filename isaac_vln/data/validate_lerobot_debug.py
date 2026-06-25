#!/usr/bin/env python3
"""Validate and visualize a tiny Isaac VLN LeRobot debug dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


DEFAULT_OUT_ROOT = Path("/tmp/isaac_debug/validation")


def validate_dataset(
    dataset_root: Path,
    out_root: Path,
    *,
    episode_index: int = 0,
    max_alignment_frames: int = 120,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    out_root = out_root.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    info = _load_json(dataset_root / "meta/info.json")
    debug_meta = _load_json(dataset_root / "meta/isaac_vln_debug.json")
    tasks = _load_jsonl(dataset_root / "meta/tasks.jsonl")
    episodes = _load_jsonl(dataset_root / "meta/episodes.jsonl")
    episode_meta = _episode_meta(episodes, episode_index)
    df = pd.read_parquet(_parquet_path(dataset_root, info, episode_index))

    video_keys = _video_keys(info)
    video_frames = {
        key: read_video_frames(_video_path(dataset_root, info, episode_index, f"observation.images.{key}"))
        for key in video_keys
    }

    errors: list[str] = []
    warnings: list[str] = []

    _check_lengths(df, video_frames, errors)
    state = _stack_column(df, "observation.state", errors)
    action = _stack_column(df, "action", errors)
    if state is not None:
        _check_state(state, debug_meta, errors, warnings)
    if action is not None:
        _check_action(action, errors, warnings)
    _check_timestamps(df, errors)
    _check_camera_order(video_keys, debug_meta, episode_index, errors)
    _check_instruction(tasks, episode_meta, errors)

    contact_sheet_path = out_root / f"episode_{episode_index:06d}_contact_sheet.png"
    action_plot_path = out_root / f"episode_{episode_index:06d}_action_plot.png"
    state_plot_path = out_root / f"episode_{episode_index:06d}_state_plot.png"
    alignment_path = out_root / f"episode_{episode_index:06d}_alignment.mp4"

    write_contact_sheet(video_frames, contact_sheet_path)
    if action is not None:
        write_action_plot(action, action_plot_path)
    if state is not None:
        write_state_plot(state, debug_meta, state_plot_path)
    write_alignment_video(
        df,
        video_frames,
        alignment_path,
        fps=float(info.get("fps") or 10.0),
        max_frames=max_alignment_frames,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "episode_index": episode_index,
        "rows": int(len(df)),
        "video_frames": {key: int(frames.shape[0]) for key, frames in video_frames.items()},
        "artifacts": {
            "contact_sheet": str(contact_sheet_path),
            "action_plot": str(action_plot_path),
            "state_plot": str(state_plot_path),
            "alignment": str(alignment_path),
        },
        "errors": errors,
        "warnings": warnings,
    }
    _write_json(out_root / f"episode_{episode_index:06d}_validation.json", summary)
    if errors:
        raise ValidationError(summary)
    return summary


class ValidationError(RuntimeError):
    def __init__(self, summary: Mapping[str, Any]) -> None:
        super().__init__("Isaac VLN LeRobot validation failed")
        self.summary = dict(summary)


def read_video_frames(path: Path) -> np.ndarray:
    width, height = probe_video_resolution(path)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for validation video reads") from exc
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed reading {path}: {process.stderr.decode('utf-8', errors='replace')}"
        )
    frame_size = width * height * 3
    if len(process.stdout) % frame_size:
        raise RuntimeError(f"Raw frame byte count is not divisible by frame size for {path}")
    num_frames = len(process.stdout) // frame_size
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(num_frames, height, width, 3)


def probe_video_resolution(path: Path) -> tuple[int, int]:
    command = [
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
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}: {process.stderr.decode('utf-8', errors='replace')}"
        )
    payload = json.loads(process.stdout.decode("utf-8"))
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"])


def write_contact_sheet(video_frames: Mapping[str, np.ndarray], path: Path) -> None:
    columns = min(6, min(frames.shape[0] for frames in video_frames.values()))
    sample_indices = np.linspace(0, min(frames.shape[0] for frames in video_frames.values()) - 1, columns)
    sample_indices = [int(round(value)) for value in sample_indices]
    thumb_w, thumb_h = 160, 160
    label_h = 18
    sheet = Image.new("RGB", (columns * thumb_w, len(video_frames) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, (key, frames) in enumerate(video_frames.items()):
        y0 = row_idx * (thumb_h + label_h)
        draw.text((4, y0 + 2), key, fill=(0, 0, 0))
        for col_idx, frame_idx in enumerate(sample_indices):
            image = _fit_image_to_panel(
                Image.fromarray(frames[frame_idx]),
                (thumb_w, thumb_h),
                fill=(0, 0, 0),
            )
            image_draw = ImageDraw.Draw(image)
            image_draw.rectangle((0, 0, 42, 16), fill=(0, 0, 0))
            image_draw.text((4, 3), str(frame_idx), fill=(255, 255, 255))
            sheet.paste(image, (col_idx * thumb_w, y0 + label_h))
    sheet.save(path)


def _fit_image_to_panel(
    image: Image.Image,
    size: tuple[int, int],
    *,
    fill: tuple[int, int, int],
) -> Image.Image:
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, fill)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def write_action_plot(action: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    labels = ["cmd_vx", "cmd_wz"]
    for idx, ax in enumerate(axes):
        ax.plot(action[:, idx])
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_state_plot(state: np.ndarray, debug_meta: Mapping[str, Any], path: Path) -> None:
    state_names = list(debug_meta.get("state_names") or [])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    if state.shape[1] >= 2:
        axes[0].plot(state[:, 0], label=state_names[0] if state_names else "ego.v")
        axes[0].plot(state[:, 1], label=state_names[1] if len(state_names) > 1 else "ego.omega")
    if state.shape[1] >= 5:
        axes[1].plot(state[:, 2], label="goal.dx")
        axes[1].plot(state[:, 3], label="goal.dy")
        axes[1].plot(state[:, 4], label="goal.dtheta")
    if state.shape[1] >= 12:
        team_dx = state[:, 6]
        team_dy = state[:, 7]
        axes[2].plot(np.sqrt(team_dx * team_dx + team_dy * team_dy), label="team_0.distance")
        axes[2].plot(state[:, 10], label="team_0.omega")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_alignment_video(
    df: pd.DataFrame,
    video_frames: Mapping[str, np.ndarray],
    path: Path,
    *,
    fps: float,
    max_frames: int,
) -> None:
    keys = list(video_frames)
    count = min(max_frames, len(df), *(frames.shape[0] for frames in video_frames.values()))
    frames: list[np.ndarray] = []
    for idx in range(count):
        panels = []
        for key in keys:
            image = _fit_image_to_panel(
                Image.fromarray(video_frames[key][idx]),
                (224, 224),
                fill=(0, 0, 0),
            )
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 224, 24), fill=(0, 0, 0))
            draw.text((4, 5), f"{key} f={idx}", fill=(255, 255, 255))
            panels.append(np.asarray(image))
        combined = np.concatenate(panels, axis=1)
        frames.append(combined)
    _write_mp4_ffmpeg(np.stack(frames, axis=0), path, fps=fps)


def _write_mp4_ffmpeg(frames: np.ndarray, output_path: Path, *, fps: float) -> None:
    height, width = frames.shape[1:3]
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
    process = subprocess.run(
        command,
        input=frames.astype(np.uint8, copy=False).tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed writing {output_path}: {process.stderr.decode('utf-8', errors='replace')}"
        )


def _check_lengths(
    df: pd.DataFrame,
    video_frames: Mapping[str, np.ndarray],
    errors: list[str],
) -> None:
    for key, frames in video_frames.items():
        if frames.shape[0] != len(df):
            errors.append(f"{key}: video frame count {frames.shape[0]} != parquet rows {len(df)}")


def _check_state(
    state: np.ndarray,
    debug_meta: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not np.all(np.isfinite(state)):
        errors.append("observation.state contains NaN or Inf")
    if state.shape[1] >= 5:
        goal = state[:, 2:5]
        goal_step = np.linalg.norm(np.diff(goal, axis=0), axis=1)
        if len(goal_step) and float(np.max(goal_step)) > 10.0:
            warnings.append(f"large goal vector jump detected: {float(np.max(goal_step)):.3f}")
    max_team_slots = int(debug_meta.get("max_team_slots") or 0)
    for slot_idx in range(max_team_slots):
        start = 5 + slot_idx * 7
        if state.shape[1] < start + 7:
            errors.append(f"state vector too short for team slot {slot_idx}")
            continue
        valid = state[:, start]
        if not np.all((np.isclose(valid, 0.0)) | (np.isclose(valid, 1.0))):
            errors.append(f"team slot {slot_idx} valid_mask is not binary")
        active = valid > 0.5
        if np.any(active):
            dist = np.linalg.norm(state[active, start + 1 : start + 3], axis=1)
            if float(np.max(dist)) > 100.0:
                warnings.append(f"team slot {slot_idx} relative distance exceeds 100m")
            dtheta = state[active, start + 3]
            if np.any(np.abs(dtheta) > math.pi + 1e-6):
                errors.append(f"team slot {slot_idx} dtheta outside [-pi, pi]")


def _check_action(action: np.ndarray, errors: list[str], warnings: list[str]) -> None:
    if not np.all(np.isfinite(action)):
        errors.append("action contains NaN or Inf")
    if action.shape[1] != 2:
        errors.append(f"expected action dimension 2, got {action.shape[1]}")
    if action.size:
        if float(np.max(np.abs(action[:, 0]))) > 2.0:
            warnings.append("cmd_vx magnitude exceeds 2 m/s")
        if float(np.max(np.abs(action[:, 1]))) > 8.0:
            warnings.append("cmd_wz magnitude exceeds 8 rad/s")


def _check_timestamps(df: pd.DataFrame, errors: list[str]) -> None:
    if "timestamp" not in df:
        errors.append("missing timestamp column")
        return
    timestamps = df["timestamp"].to_numpy(dtype=float)
    if not np.all(np.isfinite(timestamps)):
        errors.append("timestamp contains NaN or Inf")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        errors.append("timestamps are not strictly monotonic")


def _check_camera_order(
    video_keys: Sequence[str],
    debug_meta: Mapping[str, Any],
    episode_index: int,
    errors: list[str],
) -> None:
    expected = ["ego_front"]
    expected.extend(f"third_view_{idx}" for idx in range(len(video_keys) - 1))
    if list(video_keys) != expected:
        errors.append(f"camera order {list(video_keys)} != expected {expected}")
    episodes = debug_meta.get("episodes") or []
    if episode_index < len(episodes):
        source_cameras = episodes[episode_index].get("source_cameras") or {}
        if list(source_cameras) != list(video_keys):
            errors.append("debug source camera order does not match info.json video order")


def _check_instruction(
    tasks: Sequence[Mapping[str, Any]],
    episode_meta: Mapping[str, Any],
    errors: list[str],
) -> None:
    task_texts = [str(task.get("task") or "") for task in tasks]
    if not task_texts or not any(text.strip() for text in task_texts):
        errors.append("instruction/task text is empty")
    episode_tasks = [str(task) for task in episode_meta.get("tasks", [])]
    if not episode_tasks or not any(task.strip() for task in episode_tasks):
        errors.append("episode task list is empty")


def _stack_column(df: pd.DataFrame, column: str, errors: list[str]) -> np.ndarray | None:
    if column not in df:
        errors.append(f"missing {column} column")
        return None
    try:
        array = np.stack(df[column].to_numpy()).astype(np.float64)
    except Exception as exc:
        errors.append(f"could not stack {column}: {exc}")
        return None
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _episode_meta(episodes: Sequence[Mapping[str, Any]], episode_index: int) -> Mapping[str, Any]:
    for episode in episodes:
        if int(episode["episode_index"]) == episode_index:
            return episode
    raise ValueError(f"No episode metadata for episode {episode_index}")


def _video_keys(info: Mapping[str, Any]) -> list[str]:
    features = info.get("features") or {}
    keys = []
    for key, meta in features.items():
        if isinstance(meta, Mapping) and meta.get("dtype") == "video":
            keys.append(key.replace("observation.images.", ""))
    return keys


def _parquet_path(dataset_root: Path, info: Mapping[str, Any], episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size") or 1000)
    pattern = str(info["data_path"])
    return dataset_root / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def _video_path(
    dataset_root: Path,
    info: Mapping[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    chunk_size = int(info.get("chunks_size") or 1000)
    pattern = str(info["video_path"])
    return dataset_root / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an Isaac VLN LeRobot debug dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-alignment-frames", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = validate_dataset(
            args.dataset_root,
            args.out_root,
            episode_index=args.episode_index,
            max_alignment_frames=args.max_alignment_frames,
        )
    except ValidationError as exc:
        print(json.dumps(exc.summary, indent=2, sort_keys=True))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


# Command for v1 test on my local machine 
# python isaac_vln/data/validate_lerobot_debug.py   \
# --dataset-root ./datasets/tmp_isaac_debug   \
# --out-root ./datasets/tmp_isaac_debug/validation_rollout1_nova_full   \
# --episode-index 0   --max-alignment-frames 1025

if __name__ == "__main__":
    raise SystemExit(main())
