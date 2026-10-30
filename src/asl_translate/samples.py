"""Show example data and its landmarks. Run with `uv run python -m asl_translate.samples`.

Reads a few val clips out of the remote zip, so it never downloads the full 1.8 GB archive.
Never touches test clips: the test set is used once (Experiment Card field 5).
"""
import json
import time
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem

from asl_translate.config import REPO_ROOT, load_config
from asl_translate.landmarks import PARTS, extract, read_frames

OUT_DIR = REPO_ROOT / "reports" / "samples"  # git-ignored: contains signers' faces
NUM_CLIPS, NUM_FRAMES = 4, 5
COLORS = {  # BGR
    "pose": (214, 120, 42),
    "face": (122, 175, 27),
    "left_hand": (52, 104, 235),
    "right_hand": (164, 123, 232),
}


def fetch_clips(data: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Pick NUM_CLIPS val clips of varied length that exist in the zip, and fetch them."""
    entry = data["sources"]["clips"]["val"]
    hub = data["hub"]
    out_dir = data["root"] / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    with HfFileSystem().open(f"datasets/{hub['repo']}@{hub['revision']}/{entry['file']}", "rb") as f:
        archive = zipfile.ZipFile(f)
        members = {Path(n).stem: n for n in archive.namelist() if n.endswith(".mp4")}
        available = df[df["SENTENCE_NAME"].isin(members)]
        words = available["SENTENCE"].str.split().str.len()
        chosen = available[words.between(5, 25)].sort_values("DURATION")
        positions = np.linspace(0, len(chosen) - 1, NUM_CLIPS + 2)[1:-1].astype(int)
        chosen = chosen.iloc[positions]
        for name in chosen["SENTENCE_NAME"]:
            path = out_dir / f"{name}.mp4"
            if not path.exists():
                path.write_bytes(archive.read(members[name]))
    print(f"Zip has {len(members)} clips; CSV has {len(df)} sentences")
    print("In CSV, not zip:", sorted(set(df["SENTENCE_NAME"]) - set(members)))
    return chosen, len(members)


def pick(items: list, count: int = NUM_FRAMES) -> list:
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
    return cv2.resize(image[y0:y1, x0:x1], (size, size))


def show_grid(rows: list[tuple[str, list[np.ndarray]]], out: Path, title: str) -> None:
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
    data = load_config()["data"]
    root = data["root"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        root / data["sources"]["annotations"]["val"]["file"],
        sep="\t", quoting=3, keep_default_na=False,
    )
    df["DURATION"] = df["END_REALIGNED"] - df["START_REALIGNED"]
    chosen, _ = fetch_clips(data, df)

    raw_rows, landmark_rows, stats = [], [], []
    for _, clip in chosen.iterrows():
        path = root / "samples" / f"{clip['SENTENCE_NAME']}.mp4"
        cap = cv2.VideoCapture(str(path))
        source = {
            "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
            "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()

        frames = [frame for _, frame in read_frames(path, data["fps"])]
        start = time.perf_counter()
        marks = extract(read_frames(path, data["fps"]))
        seconds = time.perf_counter() - start
        np.save(root / "samples" / f"{clip['SENTENCE_NAME']}.npy", marks)

        detected = {part: round(float((~np.isnan(marks[:, i, 0])).all(axis=1).mean()), 3)
                    for part, i in PARTS.items()}
        stats.append({
            "clip": clip["SENTENCE_NAME"],
            "sentence": clip["SENTENCE"],
            "csv_duration_s": round(clip["DURATION"], 2),
            "clip_duration_s": round(source["frames"] / source["fps"], 2),
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

    show_grid(raw_rows, OUT_DIR / "clips.png", "Example val clips and their English translations")
    show_grid(landmark_rows[:6], OUT_DIR / "landmarks.png",
              "MediaPipe Holistic output (blue pose, green face, orange left hand, pink right hand)")
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    train = pd.read_csv(root / data["sources"]["annotations"]["train"]["file"],
                        sep="\t", quoting=3, keep_default_na=False)
    print("\nRandom train rows:")
    with pd.option_context("display.max_colwidth", 90, "display.width", 140):
        print(train.sample(8, random_state=0)[["SENTENCE_NAME", "SENTENCE"]].to_string(index=False))


if __name__ == "__main__":
    main()
