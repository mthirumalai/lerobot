from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ._dataset_cli import add_common_dataset_args, download_dataset_snapshot


def fail(message: str) -> None:
    raise ValueError(message)


def load_episodes_df(root: Path) -> pd.DataFrame:
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        fail("No meta/episodes parquet files found")
    return pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate episode/task index consistency for a LeRobot dataset")
    add_common_dataset_args(parser, default_repo_id="mthirumalai/so101.tp1.e161.c2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = download_dataset_snapshot(
        args,
        allow_patterns=["meta/*", "meta/episodes/*/*", "data/*/*"],
    )

    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        fail("meta/tasks.parquet not found")

    tasks_df = pd.read_parquet(tasks_path)
    if "task_index" not in tasks_df.columns:
        fail("meta/tasks.parquet must contain a 'task_index' column")

    if tasks_df.index.isnull().any():
        fail("meta/tasks.parquet has null task names in index")

    if not tasks_df.index.map(lambda v: isinstance(v, str)).all():
        fail("meta/tasks.parquet index must be task strings")

    if tasks_df["task_index"].isnull().any():
        fail("meta/tasks.parquet has null task_index values")

    if not tasks_df["task_index"].is_unique:
        fail("meta/tasks.parquet has duplicate task_index values")

    sorted_task_indices = sorted(tasks_df["task_index"].astype(int).tolist())
    expected_task_indices = list(range(len(tasks_df)))
    if sorted_task_indices != expected_task_indices:
        fail(
            "meta/tasks.parquet has non-contiguous task_index values. "
            f"Expected {expected_task_indices}, got {sorted_task_indices}."
        )

    task_index_to_name = dict(zip(tasks_df["task_index"].astype(int), tasks_df.index, strict=False))

    episodes_df = load_episodes_df(root)
    if "episode_index" not in episodes_df.columns:
        fail("meta/episodes parquet is missing 'episode_index'")

    all_episode_indices = set(episodes_df["episode_index"].astype(int).unique().tolist())
    found_episode_indices: set[int] = set()

    unknown_task_indices: set[int] = set()
    episodes_with_null_task: set[int] = set()

    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        fail("No data parquet files found")

    for data_path in data_files:
        frame_df = pd.read_parquet(data_path, columns=["episode_index", "task_index"])

        if frame_df["episode_index"].isnull().any():
            fail(f"{data_path} has null episode_index values")

        if frame_df["task_index"].isnull().any():
            eps = frame_df.loc[frame_df["task_index"].isnull(), "episode_index"].astype(int).unique().tolist()
            episodes_with_null_task.update(eps)
            continue

        frame_df["episode_index"] = frame_df["episode_index"].astype(int)
        frame_df["task_index"] = frame_df["task_index"].astype(int)

        found_episode_indices.update(frame_df["episode_index"].unique().tolist())
        unknown = set(frame_df["task_index"].unique().tolist()) - set(task_index_to_name.keys())
        unknown_task_indices.update(unknown)

    missing_episodes = sorted(all_episode_indices - found_episode_indices)

    if episodes_with_null_task:
        fail(f"Episodes with null task_index in data parquet: {sorted(episodes_with_null_task)}")
    if unknown_task_indices:
        fail(
            "Found task_index values in data parquet that are not present in meta/tasks.parquet: "
            f"{sorted(unknown_task_indices)}"
        )
    if missing_episodes:
        fail(f"Episodes listed in meta/episodes but missing from data parquet: {missing_episodes}")

    print("✅ Validation passed")
    print(f"- tasks defined: {len(task_index_to_name)}")
    print(f"- episodes found: {len(found_episode_indices)}")
    print(f"- data files checked: {len(data_files)}")


if __name__ == "__main__":
    main()
