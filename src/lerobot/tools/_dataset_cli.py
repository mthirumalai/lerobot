from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def add_common_dataset_args(parser: argparse.ArgumentParser, default_repo_id: str | None = None) -> None:
    parser.add_argument("--repo-id", default=default_repo_id, required=default_repo_id is None)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default="main")


def download_dataset_snapshot(args: argparse.Namespace, allow_patterns: list[str]) -> Path:
    return Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            allow_patterns=allow_patterns,
        )
    )
