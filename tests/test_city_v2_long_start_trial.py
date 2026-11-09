"""The four City pilot launches must use coherent, 100–200 m start contracts."""

import hashlib
import json
import math
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PILOTS = ("C01", "C06", "C11", "C16")


class CityV2LongStartTrialTests(unittest.TestCase):
    def test_pilot_launches_have_long_consistent_starts_and_traceable_tasks(self):
        table = json.loads((ROOT / "configs/city_200m_v2_task_table.json").read_text(encoding="utf-8"))
        episodes = json.loads((ROOT / "configs/collection_episodes/city_v2.json").read_text(encoding="utf-8"))
        rows = {row["episode"]: row for row in table["tasks"]}
        launches = {row["episode"]: row for row in episodes["episodes"]}
        for episode in PILOTS:
            with self.subTest(episode=episode):
                row = rows[episode]
                launch = launches[episode]
                task_path = Path(row["file"])
                task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
                start = task["start_position"]
                target = task["target_position"]
                distance = math.dist(start[:2], target[:2])
                self.assertGreaterEqual(distance, 100.0)
                self.assertLessEqual(distance, 200.0)
                self.assertEqual(start, row["start"])
                self.assertEqual(start, launch["start_position_m"])
                self.assertEqual(row["yaw"], launch["initial_yaw_degrees"])
                self.assertEqual(row["yaw"], task["environment_evidence"]["initial_yaw_degrees"])
                self.assertEqual(row["file"], launch["task_file"])
                self.assertEqual(row["task_version"], launch["task_version"])
                self.assertEqual(row["task_version"], task["environment_evidence"]["city_collection"]["version"])
                self.assertEqual(row["start_distance_m"], round(distance, 2))
                self.assertIn("several blocks ahead", task["instruction"])
                self.assertNotIn("℃", task["instruction"])
                source = task["environment_evidence"]["city_collection"]
                source_path = Path(source["source_task_path"])
                self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), source["source_task_sha256"])


if __name__ == "__main__":
    unittest.main()
