import unittest

import numpy as np

from zson.plan_a.goal_selector import select_frontier_candidates, select_semantic_candidates
from zson.plan_a.reporting import classify_failure
from zson.plan_a.semantic_map import frontier_mask, gaussian_smooth


class PlanAComponentTests(unittest.TestCase):
    def test_gaussian_smooth_respects_mask(self):
        values = np.zeros((5, 5), dtype=np.float32)
        valid = np.zeros((5, 5), dtype=bool)
        values[2, 2] = 1.0
        valid[2, 2] = True
        smoothed = gaussian_smooth(values, valid, kernel_size=3)
        self.assertAlmostEqual(smoothed[2, 2], 1.0, places=4)
        self.assertEqual(float(smoothed[0, 0]), -1.0)

    def test_frontier_mask(self):
        free = np.ones((4, 4), dtype=bool)
        explored = np.zeros((4, 4), dtype=bool)
        explored[1, 1] = True
        frontier = frontier_mask(free, explored)
        self.assertTrue(frontier[1, 2])
        self.assertTrue(frontier[2, 1])
        self.assertFalse(frontier[1, 1])

    def test_semantic_candidate_selection(self):
        planning = np.full((4, 4), -1.0, dtype=np.float32)
        planning[1, 2] = 0.4
        planning[2, 1] = 0.3
        hit_count = np.zeros((4, 4), dtype=np.int32)
        hit_count[1, 2] = 1
        hit_count[2, 1] = 1
        free = np.ones((4, 4), dtype=bool)
        explored = np.ones((4, 4), dtype=bool)
        suppression = np.zeros((4, 4), dtype=np.int32)
        candidates = select_semantic_candidates(
            planning,
            hit_count,
            free,
            explored,
            suppression,
            threshold=0.2,
            top_k=2,
        )
        self.assertEqual(candidates[0].cell, (1, 2))

    def test_frontier_candidate_selection(self):
        free = np.ones((4, 4), dtype=bool)
        explored = np.zeros((4, 4), dtype=bool)
        explored[1, 1] = True
        suppression = np.zeros((4, 4), dtype=np.int32)
        candidates = select_frontier_candidates(
            free,
            explored,
            suppression,
            agent_cell=(1, 1),
            limit=4,
            unknown_radius=1,
            distance_penalty=0.1,
        )
        self.assertGreaterEqual(len(candidates), 1)

    def test_failure_classification(self):
        metrics = {
            "success": 0.0,
            "stop_called": False,
            "stuck_events": 0,
            "max_similarity_seen": 0.1,
            "failed_goal_count": 0,
            "timed_out": True,
        }
        self.assertEqual(classify_failure(metrics, semantic_threshold=0.2), "target_not_found")


if __name__ == "__main__":
    unittest.main()
