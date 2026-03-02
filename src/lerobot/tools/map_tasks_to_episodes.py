from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from ._dataset_cli import add_common_dataset_args, download_dataset_snapshot

DEFAULT_TASK_VARIANTS = [
    "Pick up the white lego block and place it in the stainless steel cup",
    "Move the white block to the silver cup",
    "Lift the lego block and drop it into the steel cup",
    "Transfer the white lego brick into the metal container",
    "Drop that white brick into the stainless steel mug",
    "Shift the white piece from the table to the steel cup",
    "Take the white lego and put it inside the silver-colored cup",
    "Relocate the white block to the chrome vessel",
    "Grab the white brick and set it in the metallic cup",
    "Place the white lego piece into the steel cup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign task variants per episode and rewrite task_index in data files")
    add_common_dataset_args(parser, default_repo_id="mthirumalai/so101.tp1.e161.c2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tasks-json", type=str, default=None, help="Path to JSON list of task strings")
    parser.add_argument("--upload", action="store_true", help="Upload modified files back to Hub")
    return parser.parse_args()


def load_episodes_df(root: Path) -> pd.DataFrame:
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise FileNotFoundError("No episode parquet files found in meta/episodes")
    return pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)


def load_task_variants(tasks_json: str | None) -> list[str]:
    if tasks_json is None:
        return DEFAULT_TASK_VARIANTS
    payload = json.loads(Path(tasks_json).read_text())
    if not isinstance(payload, list) or not payload or not all(isinstance(item, str) for item in payload):
        raise ValueError("--tasks-json must point to a JSON file containing a non-empty list of strings")
    return payload


def patch_data_task_indices(root: Path, episode_to_task_index: dict[int, int]) -> list[Path]:
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError("No data parquet files found in data/")

    updated_files: list[Path] = []
    for data_path in data_files:
        df = pd.read_parquet(data_path)
        if "episode_index" not in df.columns or "task_index" not in df.columns:
            raise ValueError(f"Missing required columns in {data_path}")

        new_task_index = df["episode_index"].astype(int).map(episode_to_task_index)
        mask = new_task_index.notna()
        if mask.any():
            df.loc[mask, "task_index"] = new_task_index[mask].astype("int64")
            df.to_parquet(data_path, index=False)
            updated_files.append(data_path)

    return updated_files


def upload_files(args: argparse.Namespace, root: Path, file_paths: list[Path], message_prefix: str) -> None:
    api = HfApi()
    for file_path in file_paths:
        rel = file_path.relative_to(root).as_posix()
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=f"{message_prefix}: {rel}",
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = download_dataset_snapshot(
        args,
        allow_patterns=["meta/*", "meta/episodes/*/*", "data/*/*"],
    )
    episodes_df = load_episodes_df(root)
    if "episode_index" not in episodes_df.columns:
        raise ValueError("Episode metadata is missing 'episode_index'")

    task_variants = load_task_variants(args.tasks_json)
    episode_indices = sorted(episodes_df["episode_index"].astype(int).unique().tolist())
    episode_to_task = {ep: random.choice(task_variants) for ep in episode_indices}

    unique_tasks = sorted(set(episode_to_task.values()))
    task_to_index = {task: i for i, task in enumerate(unique_tasks)}
    episode_to_task_index = {ep: task_to_index[episode_to_task[ep]] for ep in episode_indices}

    tasks_df = pd.DataFrame({"task_index": list(task_to_index.values())}, index=list(task_to_index.keys()))
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_df.to_parquet(tasks_path)

    updated_data_files = patch_data_task_indices(root, episode_to_task_index)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(unique_tasks)
    info_path.write_text(json.dumps(info, indent=4))

    print(f"Prepared updates for {args.repo_id}")
    print(f"- tasks: {len(unique_tasks)}")
    print(f"- episodes: {len(episode_indices)}")
    print(f"- rewritten data files: {len(updated_data_files)}")

    if args.upload:
        upload_files(args, root, [tasks_path, info_path, *updated_data_files], "Update task mapping")
        print("✅ Uploaded changes to Hub")
    else:
        print("ℹ️ Dry run only (no upload). Use --upload to push changes.")


if __name__ == "__main__":
    main()
