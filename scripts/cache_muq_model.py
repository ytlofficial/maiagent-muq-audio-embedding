#!/usr/bin/env python3
"""Download a MuQ snapshot into a deterministic local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(args.output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"cached_muq_model={resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
