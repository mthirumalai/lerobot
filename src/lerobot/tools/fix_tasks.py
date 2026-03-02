from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from ._dataset_cli import add_common_dataset_args, download_dataset_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair malformed meta/tasks.parquet schema")
    add_common_dataset_args(parser, default_repo_id="mthirumalai/so101.tp1.e161.c2")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--out", type=str, default="tasks_fixed.parquet")
    return parser.parse_args()


def normalize_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    if "task_index" not in df.columns:
        raise ValueError("tasks parquet must include task_index column")

    if all(isinstance(v, str) for v in df.index) and not df.index.isnull().any():
        out = df.copy()
    elif "task" in df.columns:
        out = pd.DataFrame({"task_index": df["task_index"].astype(int).tolist()}, index=df["task"].astype(str))
    else:
        raise ValueError("Unable to determine task strings: index is invalid and 'task' column is missing")

    out["task_index"] = out["task_index"].astype(int)
    out = out[~out.index.duplicated(keep="first")].copy()
    out = out.sort_values("task_index")

    expected = list(range(len(out)))
    got = out["task_index"].tolist()
    if got != expected:
        remapped = {old: new for new, old in enumerate(got)}
        out["task_index"] = out["task_index"].map(remapped).astype(int)
        out = out.sort_values("task_index")

    return out


def main() -> None:
    args = parse_args()
    root = download_dataset_snapshot(args, allow_patterns=["meta/tasks.parquet"])
    tasks_path = root / "meta" / "tasks.parquet"

    df = pd.read_parquet(tasks_path)
    fixed = normalize_tasks_df(df)

    out_path = Path(args.out).resolve()
    fixed.to_parquet(out_path)
    print(f"Wrote repaired tasks parquet: {out_path}")
    print(f"- tasks: {len(fixed)}")
    print(f"- task_index range: {fixed['task_index'].min()}..{fixed['task_index'].max()}")

    if args.upload:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="meta/tasks.parquet",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message="Repair tasks.parquet schema and task_index mapping",
        )
        print("✅ Uploaded meta/tasks.parquet")


if __name__ == "__main__":
    main()
