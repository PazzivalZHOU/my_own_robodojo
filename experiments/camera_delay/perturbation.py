"""RGB camera frame-delay buffering for the camera-delay experiment.

The experiment intercepts :meth:`EvalEnv.get_obs_batch` -- the policy-facing
observation boundary -- and replaces only the per-camera ``vision`` frames with
frames from ``delay_frames`` observation calls ago. Robot proprioception
(``state`` / ``action``) is never touched.

The delay is a pure data-layer FIFO (a ``collections.deque`` per env and
camera); it does not alter simulator timing, physics stepping or rendering.

This module is intentionally free of Isaac/omni imports: it can be imported and
unit-tested with a stub environment. The optional policy-input recorder reuses
the core ``utils.save_file.VideoStreamWriter`` if it is importable and silently
degrades otherwise.
"""

from __future__ import annotations

import atexit
from collections import deque
from copy import deepcopy
import os
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

try:  # core helper; optional so the module works without it
    from utils.save_file import VideoStreamWriter
except Exception:  # pragma: no cover - defensive
    VideoStreamWriter = None  # type: ignore[assignment]


class VisionDelay:
    """Per-env, per-camera FIFO that returns a frame ``delay_frames`` calls old.

    Semantics:
      * ``delay_frames == 0`` -> identity (never constructed by :func:`install`).
      * ``delay_frames == d`` -> at call ``t``, returns the frame captured at
        call ``t - d`` (``deque(maxlen=d + 1)`` keeps exactly that window).
      * Warm-up (fewer than ``d + 1`` frames seen) follows ``warmup_mode``:
          - ``"repeat_first"``: hold the earliest captured frame.
          - ``"passthrough"``: return the current frame until the window fills.
    """

    def __init__(
        self,
        delay_frames: int,
        warmup_mode: str = "repeat_first",
        cameras: Optional[Sequence[str]] = None,
    ) -> None:
        self.delay_frames = int(delay_frames)
        self.warmup_mode = warmup_mode
        self._cameras = set(cameras) if cameras is not None else None
        # env_idx -> {camera_name -> deque(RGB array)}
        self._buffers: Dict[int, Dict[str, deque]] = {}

    def clear(self, env_idx: Optional[int] = None) -> None:
        if env_idx is None:
            self._buffers.clear()
        else:
            self._buffers.pop(env_idx, None)

    def push_and_peek(self, env_idx: Any, vision: Dict[str, Any]) -> Dict[str, Any]:
        """Store the current frames and return the delayed ``vision`` mapping."""
        if not isinstance(vision, dict):
            return vision
        out: Dict[str, Any] = {}
        for camera_name, frame in vision.items():
            if self._cameras is not None and camera_name not in self._cameras:
                out[camera_name] = frame  # camera not part of the experiment
                continue
            out[camera_name] = self._push_camera(env_idx, camera_name, frame)
        return out

    def _push_camera(self, env_idx: Any, camera_name: str, frame: Any) -> Any:
        if not isinstance(frame, dict) or "color" not in frame:
            return frame

        per_env = self._buffers.setdefault(env_idx, {})
        buf = per_env.get(camera_name)
        if buf is None:
            buf = deque(maxlen=self.delay_frames + 1)
            per_env[camera_name] = buf

        # Only RGB is the experimental variable. Depth, camera matrices, shape,
        # and any future per-camera metadata remain from the current observation.
        current_color = frame["color"]
        buf.append(deepcopy(current_color))

        if len(buf) < self.delay_frames + 1 and self.warmup_mode == "passthrough":
            delayed_color = current_color
        else:
            # buf[0] is the oldest retained color == `delay_frames` calls ago
            # once full, and the earliest color during warm-up.
            delayed_color = deepcopy(buf[0])

        out = deepcopy(frame)
        out["color"] = delayed_color
        return out


class PolicyInputRecorder:
    """Optionally records the exact frames handed to the policy (delayed).

    This is a debug aid, disabled by default. It streams one mp4 per
    ``(env_idx, camera)`` into ``out_dir`` and rotates files on episode reset.
    Every failure is swallowed after a one-time warning so recording can never
    break an evaluation run.
    """

    def __init__(self, out_dir: str, fps: float = 25.0, log: Callable[[str], None] = print) -> None:
        self.out_dir = out_dir
        self.fps = float(fps)
        self._log = log
        self._writers: Dict[tuple, Any] = {}
        self._episode = 0
        self._disabled = VideoStreamWriter is None
        self._warned = False
        os.makedirs(self.out_dir, exist_ok=True)
        if self._disabled:
            self._warn(
                "[camera_delay] policy-input recording disabled: "
                "utils.save_file.VideoStreamWriter unavailable."
            )

    # -- lifecycle ---------------------------------------------------------
    def start_new_episode(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close(announce=False)
            except Exception:
                try:
                    writer.abort()
                except Exception:
                    pass
        self._writers.clear()
        self._episode += 1

    def close_all(self) -> None:
        self.start_new_episode()

    # -- recording ---------------------------------------------------------
    def record(self, env_idx: Any, vision: Dict[str, Any]) -> None:
        if self._disabled or not isinstance(vision, dict):
            return
        for camera_name, camera_data in vision.items():
            color = camera_data.get("color") if isinstance(camera_data, dict) else None
            if color is None:
                continue
            color = np.ascontiguousarray(color)
            if color.ndim != 3 or color.shape[2] not in (3, 4):
                continue
            key = (env_idx, camera_name)
            writer = self._writers.get(key)
            if writer is None:
                writer = self._open_writer(env_idx, camera_name, color)
                if writer is None:
                    return
                self._writers[key] = writer
            try:
                writer.append(color)
            except Exception as exc:
                self._warn(f"[camera_delay] policy-input append failed ({camera_name}): {exc}")
                self._writers.pop(key, None)

    def _open_writer(self, env_idx: Any, camera_name: str, color: np.ndarray) -> Optional[Any]:
        height, width, channels = color.shape
        out_path = os.path.join(
            self.out_dir, f"env{env_idx}_ep{self._episode:04d}_{camera_name}.mp4"
        )
        try:
            writer = VideoStreamWriter(out_path, height, width, channels, fps=self.fps, is_rgb=True)
        except Exception as exc:
            self._warn(f"[camera_delay] policy-input recording disabled: {exc}")
            self._disabled = True
            return None
        self._log(f"[camera_delay] recording policy input -> {out_path}")
        return writer

    def _warn(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            self._log(message)


def install(env: Any, cfg: Any, log: Callable[[str], None] = print) -> Any:
    """Wrap ``env.get_obs_batch`` / ``env.reset`` with the delay perturbation.

    When ``cfg.active`` is False (``delay_frames == 0`` and no recording) the
    environment is returned completely untouched, which guarantees identical
    baseline behavior.
    """
    if not getattr(cfg, "active", False):
        log("[camera_delay] inactive (delay_frames=0, no recording); env left unmodified.")
        return env

    delay = VisionDelay(cfg.delay_frames, cfg.warmup_mode, cfg.cameras) if cfg.delays_vision else None
    recorder = None
    if cfg.record_policy_input:
        recorder = PolicyInputRecorder(cfg.policy_input_dir, fps=cfg.record_fps, log=log)
        atexit.register(recorder.close_all)

    orig_get_obs_batch = env.get_obs_batch
    orig_reset = env.reset

    def wrapped_get_obs_batch(env_idx_list=None, last_frame=False):
        data_list = orig_get_obs_batch(env_idx_list=env_idx_list, last_frame=last_frame)
        if last_frame:
            # Final video frame only: keep it current and do not advance buffers.
            return data_list
        for env_data in data_list:
            vision = env_data.get("vision")
            if not vision:
                continue
            if delay is not None:
                env_data["vision"] = delay.push_and_peek(env_data.get("env_idx"), vision)
            if recorder is not None:
                recorder.record(env_data.get("env_idx"), env_data["vision"])
        return data_list

    def wrapped_reset(*args, **kwargs):
        if delay is not None:
            delay.clear()
        if recorder is not None:
            recorder.start_new_episode()
        return orig_reset(*args, **kwargs)

    env.get_obs_batch = wrapped_get_obs_batch
    env.reset = wrapped_reset
    env._camera_delay = {"cfg": cfg, "delay": delay, "recorder": recorder}

    log(
        "[camera_delay] installed: "
        f"delay_frames={cfg.delay_frames} warmup={cfg.warmup_mode} "
        f"cameras={cfg.cameras or 'all'} record_policy_input={cfg.record_policy_input}"
    )
    return env
