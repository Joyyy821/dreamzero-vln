import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from isaac_vln.data.convert_manifest_to_lerobot import (
    _build_robot_timelines,
    _common_timestamps,
    _sample_robot_timeline,
    build_default_manifest_entries,
    pack_action_v1,
    pack_state_v1,
    read_manifest_jsonl,
    relative_pose_body,
    stable_team_names,
    state_key_slices,
    wrap_to_pi,
)


class IsaacVlnConverterTest(unittest.TestCase):
    def test_relative_pose_body_uses_ego_frame(self):
        dx, dy, dtheta = relative_pose_body(
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=math.pi / 2.0,
            target_x=0.0,
            target_y=2.0,
            target_yaw=math.pi,
        )
        self.assertAlmostEqual(dx, 2.0)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dtheta, math.pi / 2.0)

    def test_wrap_to_pi_bounds(self):
        self.assertAlmostEqual(wrap_to_pi(0.0), 0.0)
        self.assertAlmostEqual(wrap_to_pi(3.0 * math.pi), -math.pi)
        self.assertGreaterEqual(wrap_to_pi(-10.0), -math.pi)
        self.assertLessEqual(wrap_to_pi(10.0), math.pi)

    def test_pack_state_and_action_v1(self):
        ego = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "vx": 1.0,
            "wz": 0.2,
            "cmd_vx": 0.5,
            "cmd_wz": -0.1,
        }
        goal = {"x": 2.0, "y": 1.0, "yaw": math.pi / 2.0}
        other = {
            "x": 0.0,
            "y": 2.0,
            "yaw": math.pi,
            "vx": 0.25,
            "wz": -0.3,
        }
        state = pack_state_v1(
            ego_sample=ego,
            goal_pose=goal,
            team_samples=[("nova_carter", other, 7)],
            max_team_slots=2,
        )
        self.assertEqual(len(state), 19)
        self.assertEqual(state[:2], [1.0, 0.2])
        self.assertAlmostEqual(state[2], 2.0)
        self.assertAlmostEqual(state[3], 1.0)
        self.assertAlmostEqual(state[4], math.pi / 2.0)
        self.assertEqual(state[5], 1.0)
        self.assertAlmostEqual(state[6], 0.0)
        self.assertAlmostEqual(state[7], 2.0)
        self.assertAlmostEqual(abs(state[8]), math.pi)
        self.assertEqual(state[11], 7.0)
        self.assertEqual(state[12:], [0.0] * 7)
        self.assertEqual(pack_action_v1(ego), [0.5, -0.1])

    def test_stable_team_names_preserves_rollout_order(self):
        order = ["nova_carter", "carter_v1", "jackal"]
        self.assertEqual(stable_team_names(order, "carter_v1"), ("nova_carter", "jackal"))

    def test_state_key_slices(self):
        self.assertEqual(
            state_key_slices(2),
            {"ego": [0, 2], "goal": [2, 5], "team": [5, 19]},
        )

    def test_episode_timestamps_do_not_require_team_rows(self):
        frames = pd.DataFrame(
            [
                {"camera_name": "ego", "camera_type": "robot", "timestamp_ns": 10, "x": 1, "y": 0, "yaw": 0, "vx": 0.1, "wz": 0.2},
                {"camera_name": "ego", "camera_type": "robot", "timestamp_ns": 20, "x": 2, "y": 0, "yaw": 0, "vx": 0.1, "wz": 0.2},
                {"camera_name": "ego", "camera_type": "robot", "timestamp_ns": 30, "x": 3, "y": 0, "yaw": 0, "vx": 0.1, "wz": 0.2},
                {"camera_name": "third", "camera_type": "bev", "timestamp_ns": 10},
                {"camera_name": "third", "camera_type": "bev", "timestamp_ns": 20},
                {"camera_name": "third", "camera_type": "bev", "timestamp_ns": 30},
                {"camera_name": "team", "camera_type": "robot", "timestamp_ns": 10, "x": 4, "y": 0, "yaw": 0, "vx": 0.3, "wz": 0.4},
                {"camera_name": "team", "camera_type": "robot", "timestamp_ns": 20, "x": 5, "y": 0, "yaw": 0, "vx": 0.3, "wz": 0.4},
            ]
        )
        self.assertEqual(_common_timestamps(frames, ["ego", "third"]), [10, 20, 30])
        timelines = _build_robot_timelines(frames, ["team"])
        held = _sample_robot_timeline(timelines["team"], 30, hold_last=True, robot_name="team")
        self.assertIsNotNone(held)
        self.assertEqual(held["x"], 5)
        self.assertEqual(held["vx"], 0.0)
        self.assertEqual(held["wz"], 0.0)

    def test_manifest_defaults_and_jsonl_expansion(self):
        root = Path("/tmp/dataset")
        defaults = build_default_manifest_entries(
            dataset_root=root,
            scene_id=1,
            rollout_id=3,
            ego_agents=["carter_v1", "nova_carter"],
            third_view_cameras=["half_north"],
            split="train",
        )
        self.assertEqual([entry.ego_agent for entry in defaults], ["carter_v1", "nova_carter"])
        self.assertEqual(defaults[0].third_view_cameras, ("half_north",))

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "scene_id": 2,
                        "rollout_id": 5,
                        "ego_agents": ["a", "b"],
                        "third_view_cameras": ["half_south"],
                        "split": "val",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            entries = read_manifest_jsonl(manifest, root)
        self.assertEqual([(entry.ego_agent, entry.split) for entry in entries], [("a", "val"), ("b", "val")])
        self.assertEqual(entries[0].dataset_root, root)


if __name__ == "__main__":
    unittest.main()
