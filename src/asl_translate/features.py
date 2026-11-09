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
import shutil
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


def paths_for(data: dict, split: str, name: str) -> tuple[Path, Path]:
    return (data["interim"] / "holistic" / split / f"{name}.npz",
            data["processed"] / "landmarks" / split / f"{name}.npz")


def open_shard(data: dict, shard: str):
    local = data["raw"] / shard
    if local.exists():
        return open(local, "rb")
    from huggingface_hub import HfFileSystem

    hub = data["source_hub"]
    return HfFileSystem().open(f"datasets/{hub['repo']}@{hub['revision']}/{shard}", "rb")


def process_video(data: dict, split: str, video: str, shard: str, member: str,
                  sentences: list[tuple[str, float, float]]) -> tuple[str, int, float, str]:
    """Copy one full video out of its shard, extract every sentence in it, delete the copy."""
    from asl_translate.landmarks import extract, read_frames

    started, frames = time.time(), 0
    tmp = data["interim"] / "videos" / f"{video}.mp4"
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open_shard(data, shard) as f, zipfile.ZipFile(f).open(member) as src:
            with open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        for name, start, end in sentences:
            holistic_path, model_path = paths_for(data, split, name)
            holistic = extract(read_frames(tmp, data["fps"], start=start, end=end))
            if len(holistic) == 0:  # segment shorter than one frame, or past the video's end
                print(f"skipped {name}: no frames between {start:.2f}s and {end:.2f}s",
                      flush=True)
                continue
            holistic_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(holistic_path, landmarks=holistic.astype(np.float16))
            model_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(model_path, **reduce(holistic, data["features"]))
            frames += len(holistic)
        return video, frames, time.time() - started, ""
    except Exception:
        return video, frames, time.time() - started, traceback.format_exc()
    finally:
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


def extract_splits(data: dict, splits: list[str], limit: int | None, workers: int) -> None:
    index = load_video_index(data)
    jobs = []
    for split in splits:
        verify([annotation_path(data, split)])
        df = load_annotations(data, split).set_index("SENTENCE_NAME")
        names = read_split(data, split)[:limit]
        todo = [n for n in names if not paths_for(data, split, n)[1].exists()]
        print(f"{split}: {len(names) - len(todo)} of {len(names)} sentences already done")
        rows = df.loc[todo]
        for video, group in rows.groupby("VIDEO_NAME"):
            shard, member = index[split][video]
            sentences = list(zip(group.index, group["START_REALIGNED"], group["END_REALIGNED"]))
            jobs.append((data, split, video, shard, member, sentences))

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


# ---------------------------------------------------------------- smoke subset


def build_smoke(data: dict) -> None:
    """Copy the smoke sentences' model inputs into data/processed/smoke/ and hash both folders."""
    dirs = []
    for split in ("train", "val"):
        names = read_split(data, f"smoke_{split}")
        out = data["processed"] / "smoke" / split
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        missing = [n for n in names if not paths_for(data, split, n)[1].exists()]
        if missing:
            raise SystemExit(f"{len(missing)} smoke_{split} sentences not extracted yet, "
                             f"e.g. {missing[:3]}; run `features extract --splits {split}` first")
        for name in names:
            shutil.copy2(paths_for(data, split, name)[1], out / f"{name}.npz")
        dirs.append(out)
        size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
        print(f"smoke/{split}: {len(names)} sentences, {size:.1f} MB")
    record(dirs, {"extraction": extraction_info()})


def record_landmarks(data: dict, splits: list[str]) -> None:
    dirs = [data["processed"] / "landmarks" / s for s in splits
            if (data["processed"] / "landmarks" / s).exists()]
    record(dirs, {"extraction": extraction_info()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["extract", "smoke"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="first N sentences per split")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    data = load_config()["data"]
    if args.step == "extract":
        extract_splits(data, args.splits, args.limit, args.workers)
        if args.limit is None:
            record_landmarks(data, args.splits)
    else:
        build_smoke(data)


if __name__ == "__main__":
    main()