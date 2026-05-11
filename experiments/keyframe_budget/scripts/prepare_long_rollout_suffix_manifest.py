#!/usr/bin/env python3
"""Prepare a manifest of rollout directories for parallel suffix materialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def discover_rollout_dirs(input_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in input_root.rglob("full.mp4"):
        rollout_dir = p.parent
        if (rollout_dir / "chunk_boundaries.json").exists():
            out.append(rollout_dir)
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=str)
    parser.add_argument("--manifest_path", required=True, type=str)
    parser.add_argument("--max_rollouts", default=0, type=int)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    rollout_dirs = discover_rollout_dirs(input_root)
    if not rollout_dirs:
        raise RuntimeError(
            f"No rollout directories with full.mp4 + chunk_boundaries.json found under {input_root}"
        )
    if args.max_rollouts > 0:
        rollout_dirs = rollout_dirs[: args.max_rollouts]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for path in rollout_dirs:
            f.write(str(path))
            f.write("\n")

    print(f"[suffix-manifest] path={manifest_path}")
    print(f"[suffix-manifest] count={len(rollout_dirs)}")


if __name__ == "__main__":
    main()
