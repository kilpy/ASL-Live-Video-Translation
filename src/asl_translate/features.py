"""Extract sentence-level pose features from full videos.

Test a few: uv run python -m asl_translate.features extract --splits val --limit 5
Full run:   uv run python -m asl_translate.features extract --splits train val --workers 4
Smoke data: uv run python -m asl_translate.features smoke

Use --pose_estimator to choose MediaPipe, OpenPose, ViTPose, RTMW, or DWPose.
Sentence timings come from the realigned annotations. Videos are read from local shards
when available, or streamed from the pinned mirror. Completed sentences are skipped;
--offset and --limit allow chunked processing. The test split must be requested explicitly.

MediaPipe saves full 553-point interim landmarks and reduced 49-point processed features.
Other estimators save processed coordinates, detection masks, and confidence values in
source-specific folders. See pose.py for each estimator's landmark layout.
"""
import argparse
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from asl_translate.config import load_config
from asl_translate.data import DataManager, annotation_path, load_annotations, record, verify
from asl_translate.pose import interpolate
from asl_translate.video import open_shard

POSE = slice(0, 33)
LEFT_HAND, RIGHT_HAND = slice(511, 532), slice(532, 553)


# ---------------------------------------------------------------- model input


def reduce(holistic: np.ndarray, features: dict) -> dict[str, np.ndarray]:
    """(frames, 553, 3) holistic -> SHuBERT-style input: chosen pose points + both hands, x y.

    Returns `points` (frames, 7 + 21 + 21, 2) float16 and `detected` (frames, 3) bool for
    pose / left hand / right hand, recorded before interpolation fills the gaps.
    """
    pose = holistic[:, POSE][:, features["pose_points"]]
    parts = [pose, holistic[:, LEFT_HAND], holistic[:, RIGHT_HAND]]
    detected = np.stack([~np.isnan(p).any(axis=(1, 2)) for p in parts], axis=1)
    points = np.concatenate(parts, axis=1)[..., :2]  # x, y
    if features["interpolate_missing"]:
        points = interpolate(points)
    return {"points": points.astype(np.float16), "detected": detected}


# ---------------------------------------------------------------- extraction


def paths_for(data: dict, split: str, name: str, source: str = "mediapipe") -> tuple[Path, Path]:
    """Return the interim and processed feature paths for a sentence."""
    from asl_translate.pose import folder

    return (data["interim"] / ("holistic" if source == "mediapipe" else source) / split / f"{name}.npz",
            data["processed"] / folder(source) / split / f"{name}.npz")


def process_video(data: dict, split: str, video: str, shard: str, member: str,
                  sentences: list[tuple[str, float, float]], source: str = "mediapipe") -> tuple[str, int, float, str]:
    """Extract each requested sentence from one video and remove its temporary copy."""
    from asl_translate.pose import get_pose_estimator, prepare
    from asl_translate.video import read_frames

    started, frames = time.time(), 0
    tmp = None
    try:
        tmp_dir = data["interim"] / "videos"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f"{video}-", suffix=".mp4", dir=tmp_dir,
                                         delete=False) as temporary:
            tmp = Path(temporary.name)
        with open_shard(data, shard) as f, zipfile.ZipFile(f).open(member) as src:
            with open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        estimator = get_pose_estimator(source)
        for name, start, end in sentences:
            holistic_path, model_path = paths_for(data, split, name, source)
            sampled = read_frames(tmp, data["fps"], start=start, end=end)
            holistic = estimator.extract(sampled, data["interim"] / "frames")
            if len(holistic) == 0:  # segment shorter than one frame, or past the video's end
                print(f"skipped {name}: no frames between {start:.2f}s and {end:.2f}s",
                      flush=True)
                continue
            if source == "mediapipe":
                holistic_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(holistic_path, landmarks=holistic.astype(np.float16))
                result = reduce(holistic, data["features"])
            else:
                result = prepare(holistic)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            partial = model_path.with_suffix(".partial")
            try:
                with partial.open("wb") as out:
                    np.savez_compressed(out, **result)
                os.replace(partial, model_path)
            finally:
                partial.unlink(missing_ok=True)
            frames += len(holistic)
        return video, frames, time.time() - started, ""
    except Exception:
        return video, frames, time.time() - started, traceback.format_exc()
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def read_split(data: dict, split: str) -> list[str]:
    """Verify and read sentence IDs from a processed split list."""
    path = data["processed"] / "splits" / f"{split}.txt"
    verify([path])
    return path.read_text().split()


def extraction_info() -> dict:
    """Return the MediaPipe, OpenCV, and model versions used for extraction."""
    import cv2
    import mediapipe

    from asl_translate.pose import MODEL_URL

    return {"mediapipe": mediapipe.__version__, "opencv": cv2.__version__, "model": MODEL_URL}


def extract_splits(data: dict, splits: list[str], limit: int | None, workers: int,
                   offset: int = 0, source: str = "mediapipe") -> bool:
    """Process pending sentences by video and return whether all video jobs succeeded."""
    index = DataManager(data).load_video_index()
    jobs = []
    for split in splits:
        verify([annotation_path(data, split)])
        df = load_annotations(data, split).set_index("SENTENCE_NAME")
        names = read_split(data, split)[offset:offset + limit if limit is not None else None]
        todo = [n for n in names if not paths_for(data, split, n, source)[1].exists()]
        print(f"{split}: {len(names) - len(todo)} of {len(names)} sentences already done")
        rows = df.loc[todo]
        for video, group in rows.groupby("VIDEO_NAME"):
            shard, member = index[split][video]
            sentences = list(zip(group.index, group["START_REALIGNED"], group["END_REALIGNED"]))
            jobs.append((data, split, video, shard, member, sentences, source))

    total_frames, failures, started = 0, [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_video, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            video, frames, seconds, error = future.result()
            total_frames += frames
            rate = total_frames / (time.time() - started)
            status = "FAILED" if error else f"{frames} frames in {seconds:.0f}s"
            print(f"[{done}/{len(jobs)}] {video}: {status} | overall {rate:.1f} frames/s")
            if error:
                failures.append(f"{video}\n{error}")
    if failures:
        log = data["interim"] / "extraction_failures.txt"
        log.write_text("\n".join(failures))
        print(f"{len(failures)} videos failed; details in {log}. Rerun to retry them.")
    else:
        print(f"Done: {total_frames} frames in {(time.time() - started) / 60:.1f} min")
    return not failures


# ---------------------------------------------------------------- smoke subset


def build_smoke(data: dict, source: str = "mediapipe") -> None:
    """Copy selected train and validation features into smoke folders and record hashes."""
    from asl_translate.pose import folder
    dirs = []
    for split in ("train", "val"):
        names = read_split(data, f"smoke_{split}")
        out = data["processed"] / folder(source, smoke=True) / split
        missing = [n for n in names if not paths_for(data, split, n, source)[1].exists()]
        if missing:
            raise SystemExit(f"{len(missing)} smoke_{split} sentences not extracted yet, "
                             f"e.g. {missing[:3]}; run `features extract --splits {split}` first")
        if out.exists():
            unexpected = {p.name for p in out.iterdir()} - {f"{name}.npz" for name in names}
            if unexpected:
                raise SystemExit(f"Unexpected files in {out}; move them before building smoke data")
        out.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(paths_for(data, split, name, source)[1], out / f"{name}.npz")
        dirs.append(out)
        size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
        print(f"smoke/{split}: {len(names)} sentences, {size:.1f} MB")
    if source != "mediapipe":
        dirs.append(data["processed"] / folder(source) / "metadata.json")
    record(dirs, {"extraction": extraction_info()} if source == "mediapipe" else None)


def record_landmarks(data: dict, splits: list[str], source: str = "mediapipe") -> None:
    """Record hashes for extracted splits and, when applicable, their source metadata."""
    from asl_translate.pose import folder
    dirs = [data["processed"] / folder(source) / s for s in splits
            if (data["processed"] / folder(source) / s).exists()]
    if source != "mediapipe":
        dirs.append(data["processed"] / folder(source) / "metadata.json")
    record(dirs, {"extraction": extraction_info()} if source == "mediapipe" else None)


def main() -> None:
    """Run feature extraction or build the smoke subset from command-line options."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["extract", "smoke"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="process up to N sentences per split")
    parser.add_argument("--offset", type=int, default=0, help="skip N sentences per split")
    parser.add_argument("--workers", type=int, default=1)
    from asl_translate.pose import SOURCES
    parser.add_argument("--pose_estimator", choices=SOURCES, default="mediapipe")
    args = parser.parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--offset must be non-negative and --limit must be positive")
    data = load_config()["data"]
    if args.step == "extract":
        if args.pose_estimator != "mediapipe":
            from asl_translate.pose import check_metadata
            check_metadata(data, args.pose_estimator)
        succeeded = extract_splits(data, args.splits, args.limit, args.workers, args.offset, args.pose_estimator)
        if not succeeded:
            raise SystemExit("Extraction failed for one or more videos; rerun to retry")
        if args.limit is None and args.offset == 0:
            record_landmarks(data, args.splits, args.pose_estimator)
    else:
        build_smoke(data, args.pose_estimator)


if __name__ == "__main__":
    main()
