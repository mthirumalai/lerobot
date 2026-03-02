from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from ._dataset_cli import add_common_dataset_args, download_dataset_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modify a single task string by task_index")
    add_common_dataset_args(parser, default_repo_id="mthirumalai/so101.tp1.e161.c2")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-text", type=str, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--out", type=str, default="tasks_modified.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = download_dataset_snapshot(args, allow_patterns=["meta/tasks.parquet"])
    tasks_path = root / "meta" / "tasks.parquet"

    tasks_df = pd.read_parquet(tasks_path)
    if "task_index" not in tasks_df.columns:
        raise ValueError("meta/tasks.parquet missing task_index")

    mask = tasks_df["task_index"].astype(int) == int(args.task_index)
    if not mask.any():
        raise ValueError(f"task_index {args.task_index} not found")

    row_pos = int(mask.to_numpy().nonzero()[0][0])
    old_name = tasks_df.index[row_pos]
    tasks_df = tasks_df.rename(index={old_name: args.task_text})

    out_path = Path(args.out).resolve()
    tasks_df.to_parquet(out_path)
    print(f"Wrote updated tasks parquet: {out_path}")

    if args.upload:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="meta/tasks.parquet",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=f"Update task string at task_index={args.task_index}",
        )
        print("✅ Uploaded meta/tasks.parquet")


if __name__ == "__main__":
    main()
