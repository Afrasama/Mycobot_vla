"""Quick checks before a full PyBullet GUI run."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import utils.env_loader  # noqa: F401 — loads .env

from reflection.vla_recovery_agent import VLARecoveryAgent, _normalize_vla_backend


class PreRunSmokeTest(unittest.TestCase):
    def test_env_inject_disabled_for_motion(self):
        self.assertEqual(os.getenv("INJECT_PERCEPTION_FAILURE"), "0")
        self.assertEqual(os.getenv("INJECT_AFFECTS_MOTION"), "0")

    def test_vla_backend_heuristic(self):
        self.assertEqual(_normalize_vla_backend("simulation"), "heuristic")
        agent = VLARecoveryAgent(backend="heuristic")
        self.assertEqual(agent.backend, "heuristic")

    def test_vla_no_close_when_far(self):
        """Regression: must not close gripper at ~60 mm XY (user log failure)."""
        agent = VLARecoveryAgent(backend="heuristic")
        state = {
            "gripper_pos": [0.22, 0.27, 0.023],
            "cube_pos": [0.24, 0.33, 0.025],
            "contacts_count": 2,
            "pixel_error_x": 0.0,
            "pixel_error_y": 0.0,
        }
        action = agent.predict_action(rgb=None, text_instruction="grasp", relative_state=state)
        self.assertFalse(action["gripper_close"], action)
        self.assertGreater(abs(action["dx"]) + abs(action["dy"]), 0.01)

    def test_vla_close_when_aligned(self):
        agent = VLARecoveryAgent(backend="heuristic")
        state = {
            "gripper_pos": [0.20, 0.10, 0.021],
            "cube_pos": [0.205, 0.105, 0.02],
            "contacts_count": 0,
            "pixel_error_x": 0.0,
            "pixel_error_y": 0.0,
        }
        action = agent.predict_action(rgb=None, text_instruction="grasp", relative_state=state)
        self.assertTrue(action["gripper_close"], action)

    def test_vla_moves_toward_cube(self):
        agent = VLARecoveryAgent(backend="heuristic")
        state = {
            "gripper_pos": [0.0246, 0.3590, 0.023],
            "cube_pos": [0.0043, 0.4271, 0.025],
            "contacts_count": 0,
            "pixel_error_x": 10.0,
            "pixel_error_y": -15.0,
        }
        action = agent.predict_action(rgb=None, text_instruction="re-align", relative_state=state)
        self.assertLess(action["dx"], 0.0)
        self.assertGreater(action["dy"], 0.0)
        self.assertFalse(action["gripper_close"])


if __name__ == "__main__":
    unittest.main()
