#!/usr/bin/env python3
"""Materialize fixed-window suffix clips from long-rollout outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torchvision.io import read_video, write_video


def latent_index_to_visible_index(latent_index: int, latent_to_visible_ratio: int) -> int:
    if latent_index < 0:
        raise ValueError(f"latent_index must be >= 0, got {latent_index}.")
    if latent_to_visible_ratio <= 0:
        raise ValueError(
            f"latent_to_visible_ratio must be positive, got {latent_to_visible_ratio}."
        )
    if latent_index == 0:
        return 0
    return 1 + (latent_index - 1) * latent_to_visible_ratio


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_rollout_dirs(input_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in input_root.rglob("full.mp4"):
        rollout_dir = p.parent
        if (rollout_dir / "chunk_boundaries.json").exists():
            out.append(rollout_dir)
    return sorted(set(out))


def _write_manifest(rollout_dirs: List[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for path in rollout_dirs:
            f.write(str(path))
            f.write("\n")


def _read_manifest(manifest_path: Path) -> List[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    out: List[Path] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Path(line))
    if not out:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return out


def _materialize_one_rollout(
    rollout_dir: Path,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
) -> Tuple[int, int]:
    full_video_path = rollout_dir / "full.mp4"
    boundaries_path = rollout_dir / "chunk_boundaries.json"
    meta_path = rollout_dir / "rollout_meta.json"

    boundaries = _load_json(boundaries_path)
    meta = _load_json(meta_path) if meta_path.exists() else {}

    latent_to_visible_ratio = int(
        boundaries.get("latent_to_visible_ratio", meta.get("latent_to_visible_ratio", 4))
    )
    suffix_window_latent = int(
        boundaries.get("suffix_window_latent", meta.get("suffix_window_latent", 32))
    )
    num_output_frames_latent = int(
        boundaries.get("num_output_frames_latent", meta.get("num_output_frames", 0))
    )
    chunks = boundaries.get("chunks", [])
    if not chunks:
        raise RuntimeError(f"No chunks found in {boundaries_path}")

    video, _, info = read_video(str(full_video_path), pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"Decoded empty video: {full_video_path}")
    total_visible_frames = int(video.shape[0])
    fps = float(info.get("video_fps", meta.get("fps", 16)))

    rel_rollout_dir = rollout_dir.relative_to(input_root)
    out_rollout_dir = output_root / rel_rollout_dir / "suffixes"
    out_rollout_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    suffix_manifest = {
        "source_rollout_dir": str(rollout_dir),
        "full_video_path": str(full_video_path),
        "suffix_window_latent": suffix_window_latent,
        "latent_to_visible_ratio": latent_to_visible_ratio,
        "num_output_frames_latent": num_output_frames_latent,
        "total_visible_frames": total_visible_frames,
        "clips": [],
    }

    for chunk in chunks:
        chunk_idx = int(chunk["chunk_idx"])
        latent_start = int(chunk["latent_start"])
        latent_end_for_suffix = min(latent_start + suffix_window_latent, num_output_frames_latent)
        visible_start = latent_index_to_visible_index(latent_start, latent_to_visible_ratio)
        visible_end = min(
            latent_index_to_visible_index(latent_end_for_suffix, latent_to_visible_ratio),
            total_visible_frames,
        )

        if visible_end <= visible_start:
            suffix_manifest["clips"].append(
                {
                    "chunk_idx": chunk_idx,
                    "status": "skipped_invalid_window",
                    "visible_start": visible_start,
                    "visible_end": visible_end,
                }
            )
            skipped += 1
            continue

        clip = video[visible_start:visible_end]
        if int(clip.shape[0]) < 2:
            suffix_manifest["clips"].append(
                {
                    "chunk_idx": chunk_idx,
                    "status": "skipped_too_short",
                    "visible_start": visible_start,
                    "visible_end": visible_end,
                }
            )
            skipped += 1
            continue

        out_path = out_rollout_dir / f"suffix_from_{chunk_idx:02d}.mp4"
        if out_path.exists() and not overwrite:
            suffix_manifest["clips"].append(
                {
                    "chunk_idx": chunk_idx,
                    "status": "exists",
                    "path": str(out_path),
                    "visible_start": visible_start,
                    "visible_end": visible_end,
                    "num_frames": int(clip.shape[0]),
                }
            )
            skipped += 1
            continue

        write_video(str(out_path), clip, fps=fps)
        suffix_manifest["clips"].append(
            {
                "chunk_idx": chunk_idx,
                "status": "created",
                "path": str(out_path),
                "visible_start": visible_start,
                "visible_end": visible_end,
                "num_frames": int(clip.shape[0]),
            }
        )
        created += 1

    manifest_path = out_rollout_dir / "suffix_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(suffix_manifest, f, indent=2, ensure_ascii=True, sort_keys=False)

    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--manifest_path", default="", type=str)
    parser.add_argument("--task_index", default=-1, type=int)
    parser.add_argument("--max_rollouts", default=0, type=int)
    parser.add_argument("--write_manifest_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    manifest_path: Optional[Path] = Path(args.manifest_path).resolve() if args.manifest_path else None
    if manifest_path is not None and manifest_path.exists():
        rollout_dirs = _read_manifest(manifest_path)
    else:
        rollout_dirs = _discover_rollout_dirs(input_root)
        if not rollout_dirs:
            raise RuntimeError(
                f"No rollout directories with full.mp4 + chunk_boundaries.json found under {input_root}"
            )
        if args.max_rollouts > 0:
            rollout_dirs = rollout_dirs[: args.max_rollouts]
        if manifest_path is not None:
            _write_manifest(rollout_dirs, manifest_path)

    if args.write_manifest_only:
        print(f"[suffix-materialization] manifest_written={manifest_path} entries={len(rollout_dirs)}")
        return

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "rollout_dirs_discovered": len(rollout_dirs),
        "clips_created": 0,
        "clips_skipped": 0,
        "rollout_summaries": [],
    }

    if args.task_index >= 0:
        if args.task_index >= len(rollout_dirs):
            raise IndexError(
                f"task_index {args.task_index} out of range for {len(rollout_dirs)} rollouts"
            )
        selected = [rollout_dirs[args.task_index]]
    else:
        selected = rollout_dirs

    for rollout_dir in selected:
        created, skipped = _materialize_one_rollout(
            rollout_dir=rollout_dir,
            input_root=input_root,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        summary["clips_created"] += created
        summary["clips_skipped"] += skipped
        summary["rollout_summaries"].append(
            {"rollout_dir": str(rollout_dir), "created": created, "skipped": skipped}
        )

    summary_dir = output_root / "_suffix_materialization"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if args.task_index >= 0:
        task_dir = summary_dir / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        summary_path = task_dir / f"task_{args.task_index:06d}.json"
    else:
        summary_path = summary_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True, sort_keys=False)

    print(
        "[suffix-materialization] "
        f"rollouts={summary['rollout_dirs_discovered']} "
        f"created={summary['clips_created']} skipped={summary['clips_skipped']}"
    )
    print(f"[suffix-materialization] summary={summary_path}")


if __name__ == "__main__":
    main()
