#!/usr/bin/env python3
"""Camera-delay experiment: driver + sim-client entrypoint.

Two modes, selected by the ``ROBODOJO_CAMERA_DELAY_CLIENT`` environment variable.

Driver mode (default)::

    python experiments/camera_delay/run_eval.py \\
        --config experiments/camera_delay/configs/base.yaml \\
        [--sweep experiments/camera_delay/configs/delays.yaml] \\
        [--delays 0,1,2,4] [--task stack_blocks] [--ckpt NAME] \\
        [--policy-env ENV] [--eval-num N] [--dry-run]

    For each delay value it exports ``ROBODOJO_CLIENT_ENTRY`` /
    ``ROBODOJO_CAMERA_DELAY_*`` and launches the stock RoboDojo eval
    (``scripts/robodojo.sh eval``), then archives each run's ``_result.json``
    under ``experiments/camera_delay/results/<stamp>/delay_<d>/``.

Client mode (invoked automatically by ``scripts/eval_policy.sh`` because the
driver set ``ROBODOJO_CLIENT_ENTRY``)::

    ROBODOJO_CAMERA_DELAY_CLIENT=1 python experiments/camera_delay/run_eval.py <stock main.py args...>

    It imports the stock ``src.eval_client.main``, wraps ``create_eval_env`` so
    the freshly built ``EvalEnv`` gets the observation-delay perturbation
    installed, then runs ``main()`` unchanged.

All experiment logic lives in this directory; the only upstream edit is the
one-line entrypoint override in ``scripts/eval_policy.sh`` (see README).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_CONFIGS_DIR = os.path.join(_HERE, "configs")


def _load_local(module_name: str):
    """Load a sibling module by file path (avoids import-name collisions)."""
    path = os.path.join(_HERE, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"_camera_delay_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves cls.__module__ via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp_config = _load_local("config")
perturbation = _load_local("perturbation")


# ---------------------------------------------------------------------------
# Client mode
# ---------------------------------------------------------------------------
def run_client() -> None:
    """Run inside the Isaac sim-client process: patch, then delegate to main()."""
    cfg = exp_config.DelayConfig.from_env()
    print(f"[camera_delay] client mode: {cfg.to_dict()}")

    import src.eval_client.main as main_module  # launches AppLauncher on import

    original_create_eval_env = main_module.create_eval_env

    def create_eval_env_with_delay(*args, **kwargs):
        env = original_create_eval_env(*args, **kwargs)
        perturbation.install(env, cfg)
        return env

    main_module.create_eval_env = create_eval_env_with_delay
    main_module.main()


# ---------------------------------------------------------------------------
# Driver mode
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera-delay experiment driver.")
    parser.add_argument(
        "--config",
        default=os.path.join(_CONFIGS_DIR, "base.yaml"),
        help="Base run config YAML.",
    )
    parser.add_argument(
        "--sweep",
        default=os.path.join(_CONFIGS_DIR, "delays.yaml"),
        help="Sweep config YAML (delay list + perturbation options). Pass an empty string to skip.",
    )
    parser.add_argument("--delays", default=None, help="Comma-separated override, e.g. 0,1,2,4.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--policy-dir", default=None)
    parser.add_argument("--policy-env", default=None)
    parser.add_argument("--eval-env", default=None)
    parser.add_argument("--env-cfg", default=None)
    parser.add_argument("--action-type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-num", default=None)
    parser.add_argument(
        "--record-policy-input",
        action="store_true",
        default=None,
        help="Also record the delayed frames the policy receives.",
    )
    parser.add_argument("--dry-run", action="store_true", default=None)
    return parser.parse_args(argv)


def _build_run_config(args: argparse.Namespace):
    overrides: Dict[str, Any] = {
        "task": args.task,
        "ckpt": args.ckpt,
        "policy_dir": args.policy_dir,
        "policy_env": args.policy_env,
        "eval_env": args.eval_env,
        "env_cfg": args.env_cfg,
        "action_type": args.action_type,
        "seed": args.seed,
        "eval_num": args.eval_num,
    }
    if args.dry_run is not None:
        overrides["dry_run"] = args.dry_run
    if args.record_policy_input:
        overrides["record_policy_input"] = True
    sweep = args.sweep or None
    cfg = exp_config.load_run_config(args.config, sweep, overrides)
    if args.delays:
        cfg.delays = [int(x) for x in str(args.delays).split(",") if x.strip() != ""]
    cfg.__post_init__()
    return cfg


def _build_robodojo_cmd(cfg) -> List[str]:
    return [
        "bash",
        cfg.robodojo_sh,
        "eval",
        "--policy-dir",
        cfg.policy_dir,
        "--task",
        cfg.task,
        "--ckpt",
        cfg.ckpt,
        "--policy-env",
        cfg.policy_env,
        "--eval-env",
        cfg.eval_env,
        "--env-cfg",
        cfg.env_cfg,
        "--action-type",
        cfg.action_type,
        "--seed",
        str(cfg.seed),
        "--eval-num",
        str(cfg.eval_num),
        "--policy-gpu",
        str(cfg.policy_gpu),
        "--env-gpu",
        str(cfg.env_gpu),
    ]


def _client_env(delay_cfg, run_dir: str) -> Dict[str, str]:
    env = os.environ.copy()
    for key in (
        exp_config.ENV_CLIENT_ENTRY,
        exp_config.ENV_CLIENT_FLAG,
        exp_config.ENV_CONFIG_PATH,
        exp_config.ENV_FRAMES_OVERRIDE,
    ):
        env.pop(key, None)
    if not delay_cfg.active:
        return env

    env[exp_config.ENV_CLIENT_ENTRY] = exp_config.CLIENT_ENTRY_RELPATH
    env[exp_config.ENV_CLIENT_FLAG] = "1"
    config_path = os.path.abspath(os.path.join(run_dir, "client_config.json"))
    delay_cfg.to_json(config_path)
    env[exp_config.ENV_CONFIG_PATH] = config_path
    env[exp_config.ENV_FRAMES_OVERRIDE] = str(delay_cfg.delay_frames)
    return env


def _find_result(cfg, run_id: str) -> Optional[str]:
    policy_name = os.path.basename(os.path.normpath(cfg.policy_dir))
    pattern = os.path.join(
        _ROOT,
        "eval_result",
        cfg.dataset,
        cfg.task,
        policy_name,
        "*",
        "*",
        run_id,
        "_result.json",
    )
    matches = sorted(glob(pattern))
    return matches[-1] if matches else None


def _read_result(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def validate_eval_result(
    result_path: Optional[str], returncode: int, expected_eval_num: str
) -> tuple[Dict[str, Any], Optional[str]]:
    """Validate the observable completion contract of one RoboDojo eval.

    A run is complete only when the launcher exits successfully, writes a
    readable ``_result.json``, and records at least the requested number of
    evaluated episodes.
    """
    metrics = _read_result(result_path) if result_path else {}
    if returncode != 0:
        return metrics, f"eval command exited with returncode={returncode}"
    if not result_path:
        return {}, "no _result.json found"

    if not metrics:
        return {}, f"_result.json is empty or unreadable: {result_path}"

    try:
        eval_time = int(metrics.get("eval_time"))
    except (TypeError, ValueError):
        return metrics, "_result.json has no valid integer eval_time"

    for key in ("success_rate", "score"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return metrics, f"_result.json is missing required metric: {key}"

    expected = int(expected_eval_num)
    if eval_time < expected:
        return metrics, f"eval_time={eval_time}; expected at least {expected} completed episode(s)"
    return metrics, None


def run_driver(args: argparse.Namespace) -> int:
    cfg = _build_run_config(args)
    if not cfg.task:
        print("[camera_delay] --task (or task: in YAML) is required", file=sys.stderr)
        return 2
    if not cfg.ckpt:
        print("[camera_delay] --ckpt (or ckpt: in YAML) is required", file=sys.stderr)
        return 2

    if not os.path.isabs(cfg.results_dir):
        results_root = os.path.join(_ROOT, cfg.results_dir)
    else:
        results_root = cfg.results_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = os.path.join(results_root, stamp)
    if not cfg.dry_run:
        os.makedirs(sweep_root, exist_ok=True)

    cmd = _build_robodojo_cmd(cfg)
    print(f"[camera_delay] delays={cfg.delays} warmup={cfg.warmup_mode} "
          f"cameras={cfg.cameras or 'all'} record_policy_input={cfg.record_policy_input}")
    print(f"[camera_delay] archiving results under {sweep_root}")

    summary: List[Dict[str, Any]] = []
    all_runs_succeeded = True
    for delay in cfg.delays:
        run_dir = os.path.join(sweep_root, f"delay_{delay}")
        policy_input_dir = os.path.join(run_dir, "policy_input") if cfg.record_policy_input else None
        delay_cfg = cfg.delay_config(delay, policy_input_dir)
        run_id = f"camdelay_d{delay}_{stamp}"

        if cfg.dry_run:
            if delay_cfg.active:
                config_path = os.path.abspath(os.path.join(run_dir, "client_config.json"))
                print(
                    f"[camera_delay][dry-run] delay={delay} "
                    f"{exp_config.ENV_CONFIG_PATH}={config_path} "
                    f"{exp_config.ENV_CLIENT_ENTRY}={exp_config.CLIENT_ENTRY_RELPATH}"
                )
            else:
                print(f"[camera_delay][dry-run] delay={delay} stock client entry (unmodified baseline)")
            print(f"[camera_delay][dry-run] ROBODOJO_RUN_ID={run_id} {' '.join(cmd)}")
            continue

        os.makedirs(run_dir, exist_ok=True)
        env = _client_env(delay_cfg, run_dir)
        env["ROBODOJO_RUN_ID"] = run_id

        print(f"\n[camera_delay] ===== delay_frames={delay} (run_id={run_id}) =====")
        rc = subprocess.run(cmd, cwd=_ROOT, env=env).returncode

        result_src = _find_result(cfg, run_id)
        metrics, validation_error = validate_eval_result(result_src, rc, cfg.eval_num)
        record: Dict[str, Any] = {
            "delay_frames": delay,
            "returncode": rc,
            "run_id": run_id,
            "complete": validation_error is None,
        }
        if validation_error is not None:
            record["error"] = validation_error
            all_runs_succeeded = False
        if result_src:
            archive_name = "_result.json" if validation_error is None else "_result.partial.json"
            shutil.copy2(result_src, os.path.join(run_dir, archive_name))
            record.update(
                {
                    "success_rate": metrics.get("success_rate"),
                    "score": metrics.get("score"),
                    "eval_time": metrics.get("eval_time"),
                }
            )
            print(
                f"[camera_delay] delay={delay} rc={rc} complete={record['complete']} "
                f"success_rate={record['success_rate']} score={record['score']} "
                f"eval_time={record['eval_time']}"
            )
        else:
            print(f"[camera_delay] delay={delay} rc={rc}: no _result.json found for run_id={run_id}")
        if validation_error is not None:
            print(f"[camera_delay] delay={delay} FAILED: {validation_error}", file=sys.stderr)
        summary.append(record)

    if not cfg.dry_run:
        summary_path = os.path.join(sweep_root, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(cfg), "runs": summary}, fh, indent=2, default=str)
        print(f"\n[camera_delay] summary written to {summary_path}")
    return 0 if all_runs_succeeded else 1


def main(argv: Optional[List[str]] = None) -> int:
    if os.environ.get(exp_config.ENV_CLIENT_FLAG) == "1":
        run_client()
        return 0
    return run_driver(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
