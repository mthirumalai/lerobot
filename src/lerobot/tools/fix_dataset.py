from __future__ import annotations

import argparse

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/update a dataset tag")
    parser.add_argument("--repo-id", default="mthirumalai/so101.tp1.e161.c2")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--tag", default="v3.0")
    parser.add_argument("--revision", default=None, help="Optional source revision for tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_tag(
        repo_id=args.repo_id,
        tag=args.tag,
        repo_type=args.repo_type,
        revision=args.revision,
        exist_ok=True,
    )
    print(f"✅ Tag '{args.tag}' ensured for {args.repo_id}")


if __name__ == "__main__":
    main()
