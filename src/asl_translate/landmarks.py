"""MediaPipe Holistic landmarks. The one extraction path for offline clips and the live stream."""
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from asl_translate.config import REPO_ROOT

# Versioned URL so the checkpoint stays pinned (Experiment Card field 12).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
MODEL_PATH = REPO_ROOT / "models" / "holistic_landmarker.task"

# Each frame becomes a (553, 3) array of normalized x, y, z. Undetected parts are NaN.
PARTS = {
    "pose": slice(0, 33),
    "face": slice(33, 511),
    "left_hand": slice(511, 532),
    "right_hand": slice(532, 553),
}
NUM_LANDMARKS = 553


def make_landmarker() -> vision.HolisticLandmarker:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def to_array(result) -> np.ndarray:
    frame = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    for name, part in PARTS.items():
        points = getattr(result, f"{name}_landmarks")
        if points:
            frame[part] = [(p.x, p.y, p.z) for p in points]
    return frame


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


def extract(frames: Iterable[tuple[int, np.ndarray]]) -> np.ndarray:
    """Landmarks for (timestamp_ms, BGR frame) pairs, shaped (frames, 553, 3)."""
    output = []
    with make_landmarker() as landmarker:
        for timestamp, frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            output.append(to_array(landmarker.detect_for_video(image, timestamp)))
    return np.stack(output) if output else np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32)
