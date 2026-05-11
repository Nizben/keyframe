"""Chunk-boundary utilities for long-rollout metadata contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def latent_index_to_visible_index(latent_index: int, latent_to_visible_ratio: int) -> int:
    """
    WAN temporal mapping from latent index to decoded visible-frame index.
    First latent index maps to one visible frame, subsequent indices advance by ratio.
    """
    if latent_index < 0:
        raise ValueError(f"latent_index must be >= 0, got {latent_index}.")
    if latent_to_visible_ratio <= 0:
        raise ValueError(
            f"latent_to_visible_ratio must be positive, got {latent_to_visible_ratio}."
        )
    if latent_index == 0:
        return 0
    return 1 + (latent_index - 1) * latent_to_visible_ratio


def infer_chunk_latent_lengths(
    num_output_frames: int,
    num_frame_per_block: int,
    independent_first_frame: bool,
) -> List[int]:
    if num_output_frames <= 0:
        raise ValueError(f"num_output_frames must be > 0, got {num_output_frames}")
    if num_frame_per_block <= 0:
        raise ValueError(f"num_frame_per_block must be > 0, got {num_frame_per_block}")

    if independent_first_frame:
        if (num_output_frames - 1) % num_frame_per_block != 0:
            raise ValueError(
                "(num_output_frames - 1) must be divisible by num_frame_per_block "
                f"when independent_first_frame=true. Got {num_output_frames=} {num_frame_per_block=}"
            )
        lengths = [1]
        lengths.extend([num_frame_per_block] * ((num_output_frames - 1) // num_frame_per_block))
        return lengths

    if num_output_frames % num_frame_per_block != 0:
        raise ValueError(
            "num_output_frames must be divisible by num_frame_per_block "
            f"when independent_first_frame=false. Got {num_output_frames=} {num_frame_per_block=}"
        )
    return [num_frame_per_block] * (num_output_frames // num_frame_per_block)


def build_chunk_boundaries(
    num_output_frames: int,
    num_frame_per_block: int,
    independent_first_frame: bool,
    latent_to_visible_ratio: int,
    decoded_total_frames: int,
    fps: int,
    suffix_window_latent: int,
) -> Dict[str, object]:
    lengths = infer_chunk_latent_lengths(
        num_output_frames=num_output_frames,
        num_frame_per_block=num_frame_per_block,
        independent_first_frame=independent_first_frame,
    )
    chunks: List[Dict[str, int]] = []
    latent_cursor = 0
    for chunk_idx, chunk_len in enumerate(lengths):
        latent_start = latent_cursor
        latent_end = latent_cursor + chunk_len
        visible_start = latent_index_to_visible_index(latent_start, latent_to_visible_ratio)
        visible_end = latent_index_to_visible_index(latent_end, latent_to_visible_ratio)
        chunks.append(
            {
                "chunk_idx": chunk_idx,
                "latent_start": latent_start,
                "latent_end": latent_end,
                "visible_start": visible_start,
                "visible_end": visible_end,
            }
        )
        latent_cursor = latent_end

    return {
        "num_chunks": len(chunks),
        "num_output_frames_latent": num_output_frames,
        "decoded_total_frames": int(decoded_total_frames),
        "fps": int(fps),
        "latent_to_visible_ratio": int(latent_to_visible_ratio),
        "suffix_window_latent": int(suffix_window_latent),
        "chunks": chunks,
    }


def write_chunk_boundaries(path: str | Path, payload: Dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)
