"""Read and sample frames from source videos."""
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def open_shard(data: dict, shard: str):
    """Open a local video shard or stream it from the pinned dataset revision."""
    local = data["raw"] / shard
    if local.exists():
        return open(local, "rb")
    from huggingface_hub import HfFileSystem

    hub = data["source_hub"]
    return HfFileSystem().open(f"datasets/{hub['repo']}@{hub['revision']}/{shard}", "rb")


def read_frames(path: Path, fps: int, start: float | None = None,
                end: float | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (timestamp_ms, BGR frame) between `start` and `end` seconds at about `fps`.

    Timestamps are from the start of the video, so they rise across a whole recording.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    index = round(start * source_fps) if start else 0
    last = end * source_fps if end is not None else float("inf")
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    next_index = float(index)
    try:
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                return
            if index >= next_index:
                yield int(index / source_fps * 1000), frame
                next_index += source_fps / fps
            index += 1
    finally:
        cap.release()


