from __future__ import annotations

import argparse

import pandas as pd

from ._dataset_cli import add_common_dataset_args, download_dataset_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and verify meta/tasks.parquet task_index health")
    add_common_dataset_args(parser, default_repo_id="mthirumalai/so101.tp1.e161.c2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = download_dataset_snapshot(args, allow_patterns=["meta/tasks.parquet"])
    tasks_path = root / "meta" / "tasks.parquet"

    df = pd.read_parquet(tasks_path)
    print(df)

    if "task_index" not in df.columns:
        raise ValueError("meta/tasks.parquet must contain task_index")

    if df["task_index"].isnull().any():
        raise ValueError("task_index contains null values")

    if not df["task_index"].is_unique:
        raise ValueError("task_index has duplicates")

    got = sorted(df["task_index"].astype(int).tolist())
    expected = list(range(len(df)))
    if got != expected:
        raise ValueError(f"task_index non-contiguous: expected {expected}, got {got}")

    if not all(isinstance(v, str) for v in df.index):
        raise ValueError("task names must be string index in tasks.parquet")

    print("✅ task_index mapping is valid")


if __name__ == "__main__":
    main()
