"""Configuration objects for the camera-delay experiment.

Two distinct configs live here:

* :class:`DelayConfig` -- client-side perturbation settings. Serialized to JSON
  by the driver and re-loaded inside the Isaac sim-client process (via the
  ``ROBODOJO_CAMERA_DELAY_CONFIG`` environment variable).
* :class:`RunConfig`   -- driver-side settings: which RoboDojo eval to launch and
  which delay values to sweep.

Kept intentionally dependency-light (PyYAML + dataclasses) so the module is
importable both in the driver process and under the Isaac Sim conda env.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

VALID_WARMUP_MODES = ("repeat_first", "passthrough")

# Environment variables shared between the driver and the sim-client process.
ENV_CLIENT_FLAG = "ROBODOJO_CAMERA_DELAY_CLIENT"
ENV_CONFIG_PATH = "ROBODOJO_CAMERA_DELAY_CONFIG"
ENV_FRAMES_OVERRIDE = "ROBODOJO_CAMERA_DELAY_FRAMES"
ENV_CLIENT_ENTRY = "ROBODOJO_CLIENT_ENTRY"

# Value the driver puts in ROBODOJO_CLIENT_ENTRY (relative to the RoboDojo root,
# because scripts/eval_policy.sh runs after `cd "$root_dir"`).
CLIENT_ENTRY_RELPATH = os.path.join("experiments", "camera_delay", "run_eval.py")


@dataclass
class DelayConfig:
    """Client-side RGB frame-delay settings."""

    delay_frames: int = 0
    warmup_mode: str = "repeat_first"
    cameras: Optional[List[str]] = None  # None => every camera in the observation
    record_policy_input: bool = False
    policy_input_dir: Optional[str] = None  # required when record_policy_input
    record_fps: float = 25.0

    def __post_init__(self) -> None:
        self.delay_frames = int(self.delay_frames)
        if self.delay_frames < 0:
            raise ValueError(f"delay_frames must be >= 0, got {self.delay_frames}")
        if self.warmup_mode not in VALID_WARMUP_MODES:
            raise ValueError(
                f"warmup_mode must be one of {VALID_WARMUP_MODES}, got {self.warmup_mode!r}"
            )
        if self.cameras is not None:
            if isinstance(self.cameras, str):
                raise ValueError("cameras must be a list of camera names (or null), not a string")
            self.cameras = [str(c) for c in self.cameras]
        self.record_fps = float(self.record_fps)
        if self.record_policy_input and not self.policy_input_dir:
            raise ValueError("record_policy_input=True requires policy_input_dir")

    # -- capabilities ------------------------------------------------------
    @property
    def delays_vision(self) -> bool:
        return self.delay_frames > 0

    @property
    def active(self) -> bool:
        """True when the client must install the observation wrapper."""
        return self.delays_vision or self.record_policy_input

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelayConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown DelayConfig keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "DelayConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_env(cls) -> "DelayConfig":
        """Rebuild the config inside the sim-client process.

        Reads the JSON written by the driver, then applies the
        ``ROBODOJO_CAMERA_DELAY_FRAMES`` override if present.
        """
        cfg_path = os.environ.get(ENV_CONFIG_PATH)
        if cfg_path and os.path.isfile(cfg_path):
            cfg = cls.from_json(cfg_path)
        else:
            cfg = cls()
        override = os.environ.get(ENV_FRAMES_OVERRIDE)
        if override not in (None, ""):
            cfg.delay_frames = int(override)
            cfg.__post_init__()
        return cfg


@dataclass
class RunConfig:
    """Driver-side settings for one camera-delay sweep."""

    # RoboDojo eval inputs (mirrors scripts/robodojo.sh eval flags).
    dataset: str = "RoboDojo"
    policy_dir: str = "XPolicyLab/policy/ACT"
    task: str = ""
    ckpt: str = ""
    env_cfg: str = "arx_x5"
    action_type: str = "joint"
    seed: int = 0
    eval_num: str = "1"
    policy_env: str = "RoboDojo"
    eval_env: str = "RoboDojo"
    policy_gpu: str = "0"
    env_gpu: str = "0"

    # Experiment variables.
    delays: List[int] = field(default_factory=lambda: [0, 1, 2, 4])
    warmup_mode: str = "repeat_first"
    cameras: Optional[List[str]] = None
    record_policy_input: bool = False

    # Runner plumbing.
    robodojo_sh: str = "scripts/robodojo.sh"
    results_dir: str = "experiments/camera_delay/results"
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.delays = [int(d) for d in self.delays]
        if not self.delays:
            raise ValueError("delays must contain at least one delay value")
        if any(d < 0 for d in self.delays):
            raise ValueError(f"delays must be >= 0, got {self.delays}")
        try:
            eval_num = int(self.eval_num)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"eval_num must be a positive integer, got {self.eval_num!r}") from exc
        if eval_num <= 0:
            raise ValueError(f"eval_num must be a positive integer, got {self.eval_num!r}")
        self.eval_num = str(eval_num)
        if self.warmup_mode not in VALID_WARMUP_MODES:
            raise ValueError(
                f"warmup_mode must be one of {VALID_WARMUP_MODES}, got {self.warmup_mode!r}"
            )
        if self.cameras is not None and isinstance(self.cameras, str):
            raise ValueError("cameras must be a list of camera names (or null), not a string")

    def delay_config(self, delay_frames: int, policy_input_dir: Optional[str]) -> DelayConfig:
        return DelayConfig(
            delay_frames=delay_frames,
            warmup_mode=self.warmup_mode,
            cameras=self.cameras,
            record_policy_input=self.record_policy_input,
            policy_input_dir=policy_input_dir,
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {path}, got {type(data).__name__}")
    return data


def load_run_config(
    base_path: str,
    sweep_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RunConfig:
    merged = load_yaml_file(base_path)
    if sweep_path:
        merged = _deep_merge(merged, load_yaml_file(sweep_path))
    if overrides:
        merged = _deep_merge(merged, {k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in dataclasses.fields(RunConfig)}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(f"Unknown RunConfig key(s) in YAML: {sorted(unknown)}")
    return RunConfig(**merged)
