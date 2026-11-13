"""Show example data and its landmarks. Run with `uv run python -m asl_translate.samples`.

Reads a few full validation videos from their shards, then cuts them at aligned timestamps.
Never touches test videos: the test set is used once (Experiment Card field 5).
"""
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asl_translate.config import REPO_ROOT, load_config
from asl_translate.data import DataManager, annotation_path, load_annotations, verify
from asl_translate.pose import PARTS, MediaPipeEstimator
from asl_translate.video import open_shard, read_frames

OUT_DIR = REPO_ROOT / "reports" / "samples"  # git-ignored: contains signers' faces
NUM_CLIPS, NUM_FRAMES = 4, 5
COLORS = {  # BGR
    "pose": (214, 120, 42),
    "face": (122, 175, 27),
    "left_hand": (52, 104, 235),
    "right_hand": (164, 123, 232),
}


def select_samples(df: pd.DataFrame, videos: dict) -> pd.DataFrame:
    """Pick validation sentences of varied duration from distinct indexed videos."""
    words = df["SENTENCE"].str.split().str.len()
    available = df[df["VIDEO_NAME"].isin(videos) & words.between(5, 25) & (df["DURATION"] > 0)]
    available = available.sort_values("DURATION").drop_duplicates("VIDEO_NAME")
    if available.empty:
        raise ValueError("No matching validation sentences in the video index")
    count = min(NUM_CLIPS, len(available))
    positions = ((np.arange(count) + 0.5) * len(available) / count).astype(int)
    return available.iloc[positions]


def pick(items: list, count: int = NUM_FRAMES) -> list:
    """Select evenly spaced items, repeating items if necessary."""
    return [items[i] for i in np.linspace(0, len(items) - 1, count).astype(int)]


def draw(canvas: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Draw one (553, 3) landmark frame onto a copy of `canvas`."""
    canvas = canvas.copy()
    height, width = canvas.shape[:2]
    for part, indices in PARTS.items():
        for x, y in frame[indices][:, :2]:
            if not np.isnan(x):
                # Sized for the grid, where a 1280 px frame displays at about 270 px.
                radius = width // 400 if part == "face" else width // 160
                cv2.circle(canvas, (int(x * width), int(y * height)), radius, COLORS[part], -1)
    return canvas


def crop(image: np.ndarray, points: np.ndarray, size: int = 112) -> np.ndarray:
    """Square crop around landmarks; black if the part wasn't detected."""
    if np.isnan(points).any():
        return np.zeros((size, size, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    xs, ys = points[:, 0] * width, points[:, 1] * height
    half = max(np.ptp(xs), np.ptp(ys)) * 0.7 + 1  # 0.7: 40% padding around the box
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, width))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, height))
    if x0 >= x1 or y0 >= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(image[y0:y1, x0:x1], (size, size))


def show_grid(rows: list[tuple[str, list[np.ndarray]]], out: Path, title: str) -> None:
    """Save labeled rows of BGR images as a grid."""
    fig, axes = plt.subplots(len(rows), NUM_FRAMES, figsize=(13, 2.4 * len(rows)), squeeze=False)
    for row, (label, images) in zip(axes, rows):
        for ax, image in zip(row, images):
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
        row[0].set_title(label, loc="left", fontsize=9, color="#1f1f1f")
    fig.suptitle(title, x=0.01, ha="left", color="#1f1f1f")
    # Not tight_layout: it shrinks the whole grid to fit a sentence wider than one frame.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.6 / (2.4 * len(rows)),
                        wspace=0.04, hspace=0.25)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    """Extract and visualize MediaPipe landmarks for validation examples."""
    data = load_config()["data"]
    root = data["root"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify([annotation_path(data, split) for split in ("train", "val")])
    videos = DataManager(data).load_video_index()["val"]
    chosen = select_samples(load_annotations(data, "val"), videos)
    (root / "samples").mkdir(parents=True, exist_ok=True)

    raw_rows, landmark_rows, stats = [], [], []
    with tempfile.TemporaryDirectory(prefix="asl-samples-") as temporary:
        for _, clip in chosen.iterrows():
            shard, member = videos[clip["VIDEO_NAME"]]
            path = Path(temporary) / "video.mp4"
            with (open_shard(data, shard) as remote,
                  zipfile.ZipFile(remote) as archive,
                  archive.open(member) as video,
                  path.open("wb") as output):
                shutil.copyfileobj(video, output, length=1 << 22)
            cap = cv2.VideoCapture(str(path))
            source = {
                "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
                "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            cap.release()

            sampled = list(read_frames(path, data["fps"], start=clip["START_REALIGNED"],
                                       end=clip["END_REALIGNED"]))
            if not sampled:
                print(f"Skipping {clip['SENTENCE_NAME']}: no frames in aligned time range")
                continue
            frames = [frame for _, frame in sampled]
            start = time.perf_counter()
            marks = MediaPipeEstimator().extract(iter(sampled))
            seconds = time.perf_counter() - start
            np.save(root / "samples" / f"{clip['SENTENCE_NAME']}.npy", marks)

            detected = {part: round(float((~np.isnan(marks[:, i, 0])).all(axis=1).mean()), 3)
                        for part, i in PARTS.items()}
            stats.append({
                "clip": clip["SENTENCE_NAME"],
                "sentence": clip["SENTENCE"],
                "csv_duration_s": round(clip["DURATION"], 2),
                "clip_duration_s": round(len(frames) / data["fps"], 2),
                "source": source,
                "array_shape": list(marks.shape),
                "array_mb": round(marks.nbytes / 1e6, 2),
                "extraction_fps": round(len(marks) / seconds, 1),
                "frames_detected": detected,
            })

            chosen_frames = pick(list(range(len(frames))))
            label = f"\"{clip['SENTENCE']}\" ({clip['DURATION']:.1f} s)"
            raw_rows.append((label, [frames[i] for i in chosen_frames]))
            crop_parts = ("left_hand", "right_hand", "face")
            landmark_rows += [
                (f"{clip['SENTENCE_NAME']}: landmarks on video",
                 [draw(frames[i], marks[i]) for i in chosen_frames]),
                ("What the GRU encoder sees: landmarks only",
                 [draw(np.zeros_like(frames[i]), marks[i]) for i in chosen_frames]),
                ("What DINOv2 sees: left hand, right hand, face crops",
                 [np.hstack([crop(frames[i], marks[i][PARTS[p]]) for p in crop_parts])
                  for i in chosen_frames]),
            ]
    if not raw_rows:
        raise RuntimeError("No sample videos contained frames in their aligned time ranges")
    show_grid(raw_rows, OUT_DIR / "clips.png", "Example val clips and their English translations")
    show_grid(landmark_rows[:6], OUT_DIR / "landmarks.png",
              "MediaPipe Holistic output (blue pose, green face, orange left hand, pink right hand)")
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    train = load_annotations(data, "train")
    print("\nRandom train rows:")
    with pd.option_context("display.max_colwidth", 90, "display.width", 140):
        print(train.sample(min(8, len(train)), random_state=0)[["SENTENCE_NAME", "SENTENCE"]].to_string(index=False))


if __name__ == "__main__":
    main()
