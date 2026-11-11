"""Extract landmarks for every sentence in the splits, cut from the full videos.

Test on a few:  uv run python -m asl_translate.features extract --splits val --limit 5
Full run:       uv run python -m asl_translate.features extract --splits train val --workers 4
Smoke subset:   uv run python -m asl_translate.features smoke

Each sentence is cut at START_REALIGNED..END_REALIGNED from its full frontal video, run through
the same MediaPipe path as the live stream (landmarks.py), and saved twice:
- data/interim/holistic/<split>/<SENTENCE_NAME>.npz   all 553 landmarks, x y z, NaN if missed
- data/processed/landmarks/<split>/<SENTENCE_NAME>.npz the model input (see `reduce`)
Videos come from data/raw/<shard> if that shard was downloaded, else streamed from the mirror.
Extraction is resumable: finished sentences are skipped. The test split is only extracted when
asked for explicitly, for the one final evaluation.
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
from asl_translate.dataset import (annotation_path, load_annotations, load_video_index, record,
                                   verify)

POSE = slice(0, 33)
LEFT_HAND, RIGHT_HAND = slice(511, 532), slice(532, 553)


# ---------------------------------------------------------------- model input


def interpolate(points: np.ndarray) -> np.ndarray:
    """Fill NaN frames linearly in time, per coordinate; edges take the nearest value.

    A part that is never detected in the sentence becomes zeros.
    """
    out = points.astype(np.float32).copy()
    flat = out.reshape(len(out), -1)
    t = np.arange(len(out))
    for column in range(flat.shape[1]):
        values = flat[:, column]
        seen = ~np.isnan(values)
        if seen.all():
            continue
        flat[:, column] = np.interp(t, t[seen], values[seen]) if seen.any() else 0.0
    return out


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
    from asl_translate.pose_sources import folder

    return (data["interim"] / ("holistic" if source == "mediapipe" else source) / split / f"{name}.npz",
            data["processed"] / folder(source) / split / f"{name}.npz")


def open_shard(data: dict, shard: str):
    local = data["raw"] / shard
    if local.exists():
        return open(local, "rb")
    from huggingface_hub import HfFileSystem

    hub = data["source_hub"]
    return HfFileSystem().open(f"datasets/{hub['repo']}@{hub['revision']}/{shard}", "rb")


def process_video(data: dict, split: str, video: str, shard: str, member: str,
                  sentences: list[tuple[str, float, float]], source: str = "mediapipe") -> tuple[str, int, float, str]:
    """Copy one full video out of its shard, extract every sentence in it, delete the copy."""
    from asl_translate.landmarks import extract, read_frames

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
        for name, start, end in sentences:
            holistic_path, model_path = paths_for(data, split, name, source)
            sampled = read_frames(tmp, data["fps"], start=start, end=end)
            if source == "mediapipe":
                holistic = extract(sampled)
            else:
                from asl_translate.pose_sources import extract as extract_pose
                holistic = extract_pose(source, sampled, data["interim"] / "frames")
            if len(holistic) == 0:  # segment shorter than one frame, or past the video's end
                print(f"skipped {name}: no frames between {start:.2f}s and {end:.2f}s",
                      flush=True)
                continue
            if source == "mediapipe":
                holistic_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(holistic_path, landmarks=holistic.astype(np.float16))
                result = reduce(holistic, data["features"])
            else:
                from asl_translate.pose_sources import prepare
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
    path = data["processed"] / "splits" / f"{split}.txt"
    verify([path])
    return path.read_text().split()


def extraction_info() -> dict:
    import cv2
    import mediapipe

    from asl_translate.landmarks import MODEL_URL

    return {"mediapipe": mediapipe.__version__, "opencv": cv2.__version__, "model": MODEL_URL}


def extract_splits(data: dict, splits: list[str], limit: int | None, workers: int,
                   offset: int = 0, source: str = "mediapipe") -> bool:
    index = load_video_index(data)
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
    """Copy the smoke sentences' model inputs into data/processed/smoke/ and hash both folders."""
    from asl_translate.pose_sources import folder
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
    from asl_translate.pose_sources import folder
    dirs = [data["processed"] / folder(source) / s for s in splits
            if (data["processed"] / folder(source) / s).exists()]
    if source != "mediapipe":
        dirs.append(data["processed"] / folder(source) / "metadata.json")
    record(dirs, {"extraction": extraction_info()} if source == "mediapipe" else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["extract", "smoke"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="process up to N sentences per split")
    parser.add_argument("--offset", type=int, default=0, help="skip N sentences per split")
    parser.add_argument("--workers", type=int, default=1)
    from asl_translate.pose_sources import SOURCES
    parser.add_argument("--source", choices=SOURCES, default="mediapipe")
    args = parser.parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--offset must be non-negative and --limit must be positive")
    data = load_config()["data"]
    if args.step == "extract":
        if args.source != "mediapipe":
            from asl_translate.pose_sources import check_metadata
            check_metadata(data, args.source)
        succeeded = extract_splits(data, args.splits, args.limit, args.workers, args.offset, args.source)
        if not succeeded:
            raise SystemExit("Extraction failed for one or more videos; rerun to retry")
        if args.limit is None and args.offset == 0:
            record_landmarks(data, args.splits, args.source)
    else:
        build_smoke(data, args.source)


if __name__ == "__main__":
    main()
