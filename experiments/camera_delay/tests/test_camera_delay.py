from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.camera_delay import run_eval
from experiments.camera_delay.analyze import collect_records
from experiments.camera_delay.config import RunConfig
from experiments.camera_delay.perturbation import VisionDelay


ROOT = Path(__file__).resolve().parents[3]
RUN_EVAL = ROOT / "experiments" / "camera_delay" / "run_eval.py"


class CameraDelayDriverTests(unittest.TestCase):
    def test_analyzer_ignores_incomplete_summary_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="camera-delay-analysis-") as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "delay_frames": 0,
                                "complete": False,
                                "success_rate": 0.0,
                                "score": 0.0,
                                "eval_time": 0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(collect_records([tmp]), [])

    def test_run_config_requires_a_positive_numeric_eval_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            RunConfig(eval_num="native")

    def test_run_config_requires_at_least_one_delay(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            RunConfig(delays=[])

    def test_delay_changes_only_rgb_color_payload(self) -> None:
        delay = VisionDelay(delay_frames=1)

        delay.push_and_peek(
            0,
            {
                "cam_head": {
                    "color": np.array([0]),
                    "depth": np.array([100]),
                    "intrinsic_matrix": np.array([200]),
                }
            },
        )
        delayed = delay.push_and_peek(
            0,
            {
                "cam_head": {
                    "color": np.array([1]),
                    "depth": np.array([101]),
                    "intrinsic_matrix": np.array([201]),
                }
            },
        )

        self.assertEqual(delayed["cam_head"]["color"].tolist(), [0])
        self.assertEqual(delayed["cam_head"]["depth"].tolist(), [101])
        self.assertEqual(delayed["cam_head"]["intrinsic_matrix"].tolist(), [201])

    def test_zero_delay_dry_run_uses_stock_client_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="camera-delay-dry-run-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "task: stack_blocks",
                        "ckpt: test-checkpoint",
                        "delays: [0]",
                        f"results_dir: {tmp_path / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUN_EVAL),
                    "--config",
                    str(config_path),
                    "--sweep",
                    "",
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("stock client entry", completed.stdout)
            self.assertNotIn("ROBODOJO_CLIENT_ENTRY=", completed.stdout)

    def test_result_validation_requires_analysis_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="camera-delay-result-") as tmp:
            result_path = Path(tmp) / "_result.json"
            result_path.write_text(json.dumps({"eval_time": 1, "details": {}}), encoding="utf-8")

            _, error = run_eval.validate_eval_result(
                str(result_path), returncode=0, expected_eval_num="1"
            )

            self.assertIn("missing required metric", error)

    def test_result_validation_rejects_too_few_completed_episodes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="camera-delay-result-") as tmp:
            result_path = Path(tmp) / "_result.json"
            result_path.write_text(
                json.dumps({"success_rate": 0.0, "score": 0.0, "eval_time": 0, "details": {}}),
                encoding="utf-8",
            )

            metrics, error = run_eval.validate_eval_result(
                str(result_path), returncode=0, expected_eval_num="1"
            )

            self.assertEqual(metrics["eval_time"], 0)
            self.assertIn("expected at least 1", error)

    def test_driver_returns_nonzero_when_eval_command_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="camera-delay-test-") as tmp:
            tmp_path = Path(tmp)
            launcher = tmp_path / "fail.sh"
            launcher.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")

            results_dir = tmp_path / "results"
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "task: stack_blocks",
                        "ckpt: test-checkpoint",
                        "delays: [0]",
                        f"robodojo_sh: {launcher}",
                        f"results_dir: {results_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUN_EVAL),
                    "--config",
                    str(config_path),
                    "--sweep",
                    "",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summaries = list(results_dir.glob("*/summary.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["runs"][0]["returncode"], 7)


if __name__ == "__main__":
    unittest.main()
