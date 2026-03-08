#!/usr/bin/env python3
"""
Play back recorded LeRobot episodes with side-by-side camera views and motor positions.

Usage:
    playback mthirumalai/so101.pnp.1 --episodes 0
    playback mthirumalai/so101.pnp.1 --episodes 0,3,7

Controls during playback:
    ESC / q      — quit
    left arrow   — replay current episode from the beginning
    right arrow  — skip to next episode

After an episode finishes:
    left arrow   — replay
    right arrow  — next episode (or quit if last)
    ESC / q      — quit
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import torch

KEY_ESC = 27
# macOS OpenCV arrow keys after & 0xFF
KEY_LEFT = 2
KEY_RIGHT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play back LeRobot episodes.")
    parser.add_argument("dataset", help="HuggingFace repo_id or local path (e.g. mthirumalai/so101.pnp.1)")
    parser.add_argument("--episodes", type=str, default="0",
                        help="Comma-separated episode indices to play back (default: 0)")
    parser.add_argument("--root", type=str, default=None, help="Local dataset root (overrides default cache)")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS (default: dataset fps)")
    return parser.parse_args()


def is_quit(key: int) -> bool:
    return key == KEY_ESC or key == ord("q")


def is_left(key: int) -> bool:
    return key == KEY_LEFT


def is_right(key: int) -> bool:
    return key == KEY_RIGHT


def tensor_to_bgr(frame: torch.Tensor) -> np.ndarray:
    """Convert a CHW float [0,1] or HWC uint8 tensor to a BGR numpy array for OpenCV."""
    img = frame.cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):  # CHW
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def make_motor_panel(state: torch.Tensor, names: list[str] | None, width: int, height: int) -> np.ndarray:
    """Render motor positions as a dark panel with bar graphs and values."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    values = state.cpu().numpy().flatten().tolist()
    n = len(values)
    if n == 0:
        return panel

    margin = 10
    row_h = max(1, (height - 2 * margin) // n)
    bar_max_w = width - 2 * margin - 120

    for i, val in enumerate(values):
        y = margin + i * row_h + row_h // 2
        label = names[i] if (names and i < len(names)) else f"motor_{i}"
        if len(label) > 14:
            label = label[:13] + "…"

        clamped = max(-1.0, min(1.0, float(val)))
        bar_len = int(abs(clamped) * bar_max_w / 2)
        bar_x_center = margin + 110 + bar_max_w // 2
        color = (0, 200, 80) if clamped >= 0 else (0, 80, 200)
        if clamped >= 0:
            x1, x2 = bar_x_center, bar_x_center + bar_len
        else:
            x1, x2 = bar_x_center - bar_len, bar_x_center
        cv2.rectangle(panel, (x1, y - 6), (x2, y + 6), color, -1)
        cv2.line(panel, (bar_x_center, y - 8), (bar_x_center, y + 8), (80, 80, 80), 1)
        cv2.putText(panel, label, (margin, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(panel, f"{val:+.3f}", (margin + 110 + bar_max_w + 4, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 100), 1)

    return panel


def play_episode(dataset, ep_idx: int, frame_indices: list[int], camera_keys: list[str],
                 state_key: str | None, motor_names: list[str] | None,
                 fps: float, window: str, ep_num: int, total_eps: int) -> tuple[str, np.ndarray]:
    """
    Play one episode. Returns (action, last_composite) where action is:
        'quit'   — user pressed ESC/q
        'replay' — user pressed left arrow during playback
        'done'   — episode finished naturally
        'next'   — user pressed right arrow during playback
    """
    delay_ms = max(1, int(1000 / fps))
    num_frames = len(frame_indices)
    composite = np.zeros((240, 640, 3), dtype=np.uint8)

    for frame_idx in range(num_frames):
        t0 = time.time()
        item = dataset[frame_indices[frame_idx]]

        # Camera panels
        cam_panels = []
        for key in camera_keys:
            if key in item:
                cam_panels.append(tensor_to_bgr(item[key]))
        if not cam_panels:
            cam_panels = [np.zeros((240, 320, 3), dtype=np.uint8)]

        target_h = max(p.shape[0] for p in cam_panels)
        resized = []
        for p in cam_panels:
            if p.shape[0] != target_h:
                scale = target_h / p.shape[0]
                p = cv2.resize(p, (int(p.shape[1] * scale), target_h))
            resized.append(p)
        cameras_strip = np.concatenate(resized, axis=1)

        # Motor panel
        motor_panel = np.zeros((target_h, 10, 3), dtype=np.uint8)
        if state_key and state_key in item:
            state = item[state_key]
            if state.ndim > 1:
                state = state[-1]
            motor_panel = make_motor_panel(state, motor_names, width=360, height=target_h)

        # Overlay
        label = f"ep {ep_idx}  [{ep_num}/{total_eps}]  frame {frame_idx + 1}/{num_frames}"
        hint = "  [<] replay  [>] next  q quit"
        cv2.putText(cameras_strip, label + hint, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(cameras_strip, label + hint, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        composite = np.concatenate([cameras_strip, motor_panel], axis=1)
        cv2.imshow(window, composite)

        elapsed_ms = int((time.time() - t0) * 1000)
        key = cv2.waitKey(max(1, delay_ms - elapsed_ms)) & 0xFF
        if is_quit(key):
            return "quit", composite
        if is_left(key):
            return "replay", composite
        if is_right(key):
            return "next", composite

    return "done", composite


def wait_for_nav(window: str, ep_idx: int, ep_num: int, total_eps: int, composite: np.ndarray) -> str:
    """Show a 'finished' prompt and wait for left/right/quit."""
    overlay = composite.copy()
    msg = f"ep {ep_idx} done.  [<] replay  [>] next  q quit"
    cv2.putText(overlay, msg, (8, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(overlay, msg, (8, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imshow(window, overlay)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if is_quit(key):
            return "quit"
        if is_left(key):
            return "replay"
        if is_right(key):
            return "next"


def main() -> int:
    args = parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("ERROR: lerobot package not found. Activate your environment first.", file=sys.stderr)
        return 1

    # Parse episode list
    try:
        episode_list = [int(e.strip()) for e in args.episodes.split(",")]
    except ValueError:
        print("ERROR: --episodes must be comma-separated integers, e.g. --episodes 0,3,7", file=sys.stderr)
        return 1

    print(f"Loading dataset '{args.dataset}' …")
    try:
        dataset = LeRobotDataset(repo_id=args.dataset, root=args.root, episodes=episode_list)
    except Exception as e:
        print(f"ERROR loading dataset: {e}", file=sys.stderr)
        return 1

    total_eps = dataset.meta.total_episodes
    invalid = [e for e in episode_list if e >= total_eps]
    if invalid:
        print(f"ERROR: episodes {invalid} out of range (dataset has {total_eps} episodes).", file=sys.stderr)
        return 1

    fps = args.fps or dataset.meta.fps
    camera_keys = dataset.meta.camera_keys
    state_key = "observation.state" if "observation.state" in dataset.features else None
    motor_names: list[str] | None = None
    if state_key and "names" in dataset.meta.features.get(state_key, {}):
        raw = dataset.meta.features[state_key]["names"]
        if isinstance(raw, list):
            motor_names = [str(n) for n in raw]

    # Pre-build per-episode frame index lists
    all_indices = dataset.hf_dataset["episode_index"]
    ep_frame_map: dict[int, list[int]] = {e: [] for e in episode_list}
    for i, e in enumerate(all_indices):
        e = int(e)
        if e in ep_frame_map:
            ep_frame_map[e].append(i)

    for ep_idx, frames in ep_frame_map.items():
        if not frames:
            print(f"WARNING: No frames found for episode {ep_idx}, skipping.", file=sys.stderr)

    print(f"Episodes: {episode_list}  |  Cameras: {camera_keys}  |  FPS: {fps}")
    print("Controls: [<] replay  [>] next  q/ESC quit")

    window = "LeRobot Playback"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    ep_cursor = 0  # index into episode_list

    while ep_cursor < len(episode_list):
        ep_idx = episode_list[ep_cursor]
        frames = ep_frame_map[ep_idx]
        if not frames:
            ep_cursor += 1
            continue

        print(f"Playing episode {ep_idx} ({ep_cursor + 1}/{len(episode_list)}) …")
        action, composite = play_episode(
            dataset, ep_idx, frames, camera_keys, state_key, motor_names,
            fps, window, ep_cursor + 1, len(episode_list),
        )

        if action == "quit":
            break
        if action == "replay":
            continue  # ep_cursor unchanged → replay same episode
        if action == "next":
            ep_cursor += 1
            continue

        # action == "done": episode finished naturally — show nav prompt on last frame
        action = wait_for_nav(window, ep_idx, ep_cursor + 1, len(episode_list), composite)
        if action == "quit":
            break
        if action == "replay":
            continue
        # "next"
        ep_cursor += 1

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
