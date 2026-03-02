from __future__ import annotations

import argparse

import pandas as pd

from ._dataset_cli import add_common_dataset_args, download_dataset_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick tasks.parquet smoke-check tool")
    add_common_dataset_args(parser, default_repo_id="mthirumalai/so101.tp1.e161.c2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = download_dataset_snapshot(args, allow_patterns=["meta/tasks.parquet"])
    tasks_path = root / "meta" / "tasks.parquet"

    df = pd.read_parquet(tasks_path)
    print(df.head())

    if "task_index" not in df.columns:
        raise ValueError("meta/tasks.parquet missing task_index")
    if df.index.isnull().any():
        raise ValueError("Task names cannot be null")
    if not all(isinstance(v, str) for v in df.index):
        raise ValueError("Task names must be stored as string index")

    print("✅ temp check passed")


if __name__ == "__main__":
    main()
