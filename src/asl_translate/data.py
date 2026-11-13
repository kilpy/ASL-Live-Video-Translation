"""Data preparation, feature transfer, and Parquet export."""
import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from asl_translate.config import MANIFEST_PATH, REPO_ROOT, load_config

# ---------------------------------------------------------------- manifest


def sha256(path: Path) -> str:
    """Hash of a file, or of a folder's file names and hashes (so one entry covers a folder).

    Folder entries are sorted by their POSIX path string, not by Path objects, because Windows
    compares paths case-insensitively and would combine the same files in a different order.
    """
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted((p for p in path.rglob("*") if p.is_file()),
                       key=lambda p: p.relative_to(path).as_posix())
        for f in files:
            digest.update(f"{f.relative_to(path).as_posix()}:{sha256(f)}\n".encode())
        return digest.hexdigest()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_key(path: Path) -> str:
    """Paths in the manifest are relative to data/, so the data root can move."""
    data = load_config()["data"]
    return PurePosixPath(Path(path).relative_to(data["root"])).as_posix()


def read_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Read the checksum manifest, returning an empty file mapping if it is absent."""
    return json.loads(path.read_text()) if path.exists() else {"files": {}}


def record(paths: list[Path], extra: dict | None = None, path: Path = MANIFEST_PATH) -> None:
    """Add or update hashes for `paths` (and any top-level `extra` fields) in the manifest."""
    manifest = read_manifest(path)
    manifest.update(extra or {})
    manifest.setdefault("files", {})
    for p in paths:
        manifest["files"][manifest_key(p)] = sha256(p)
    manifest["files"] = dict(sorted(manifest["files"].items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the same bytes on every OS, so hashes never depend on line endings
    path.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")


def verify(paths: list[Path], path: Path = MANIFEST_PATH) -> None:
    """Fail loudly if any file differs from the hash in the committed manifest."""
    files = read_manifest(path)["files"]
    for p in paths:
        key = manifest_key(p)
        if key not in files:
            raise SystemExit(f"{key} is not in {path.name}; rebuild it with asl_translate.data")
        if sha256(p) != files[key]:
            raise SystemExit(f"{key} does not match its hash in {path.name}: data has changed")


# ---------------------------------------------------------------- step 1: annotations


def tokenize(sentence: str) -> list[str]:
    """Extract lowercase word tokens from a sentence."""
    return re.findall(r"[a-z0-9']+", sentence.lower())


def annotation_path(data: dict, split: str) -> Path:
    """Return the annotation CSV path for a dataset split."""
    return data["raw"] / data["sources"]["annotations"][split]["file"]


def load_annotations(data: dict, split: str) -> pd.DataFrame:
    # QUOTE_NONE: sentences contain literal quotes, and the files are tab-separated
    """Load a split and add duration and token columns."""
    df = pd.read_csv(annotation_path(data, split), sep="\t", quoting=3, keep_default_na=False)
    df["DURATION"] = df["END_REALIGNED"] - df["START_REALIGNED"]
    df["TOKENS"] = df["SENTENCE"].map(tokenize)
    return df


# ---------------------------------------------------------------- step 2: video index


def video_index_path(data: dict) -> Path:
    """Return the path to the processed video-to-shard index."""
    return data["processed"] / "video_index.json"


# ---------------------------------------------------------------- step 3: splits


def build_vocab(train: pd.DataFrame, size: int) -> list[str]:
    """Top `size` words by train frequency; ties broken alphabetically so the order is fixed."""
    counts = Counter(w for tokens in train["TOKENS"] for w in tokens)
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]]


def covered(df: pd.DataFrame, vocab: set[str]) -> pd.DataFrame:
    """Usable sentences: non-empty, positive duration, every word in the vocabulary."""
    keep = df["TOKENS"].map(lambda t: bool(t) and set(t) <= vocab) & (df["DURATION"] > 0)
    return df[keep]


def pick_by_video(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Whole videos, most covered sentences first (ties in seeded random order), until >= n.

    Fewer videos means fewer downloads, at the cost of less signer and topic variety.
    """
    if n is None or n >= len(df):
        return df
    counts = df.groupby("VIDEO_NAME").size()
    rng = np.random.default_rng(seed)
    order = counts.index.to_numpy()[rng.permutation(len(counts))]
    ranked = counts.loc[order].sort_values(ascending=False, kind="stable")
    chosen = ranked.index[: int(np.searchsorted(ranked.cumsum().to_numpy(), n) + 1)]
    return df[df["VIDEO_NAME"].isin(chosen)]


def make_splits(frames: dict[str, pd.DataFrame], cfg: dict,
                available: dict[str, set[str]]) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Return the vocabulary and the chosen sentences for every split (plus smoke subsets)."""
    vocab = build_vocab(frames["train"], cfg["vocab_size"])
    seed = cfg["split_seed"]
    limits = {"train": cfg["n_train_sentences"], "val": cfg["n_val_sentences"], "test": None}
    chosen = {}
    for split, df in frames.items():
        usable = covered(df, set(vocab))
        usable = usable[usable["VIDEO_NAME"].isin(available[split])]
        chosen[split] = pick_by_video(usable, limits[split], seed).sort_values("SENTENCE_NAME")

    # Every split must come from different source videos (How2Sign's official split).
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = set(chosen[a]["VIDEO_ID"]) & set(chosen[b]["VIDEO_ID"])
        if shared:
            raise SystemExit(f"{a} and {b} share videos: {sorted(shared)[:5]}")

    for split in ("train", "val"):
        n = min(cfg["smoke"][split], len(chosen[split]))
        chosen[f"smoke_{split}"] = chosen[split].sample(n, random_state=seed).sort_values(
            "SENTENCE_NAME")
    return vocab, chosen


def convert_partition(source: str, source_dir: Path, output: Path) -> int:
    """Convert NPZ arrays losslessly, retaining their source-specific shape and dtype."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(source_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No feature files found in {source_dir}")
    fields = [pa.field("sentence_id", pa.string()),
              pa.field("points", pa.list_(pa.list_(pa.list_(pa.float16())))),
              pa.field("detected", pa.list_(pa.list_(pa.bool_())))]
    if source != "mediapipe":
        fields.append(pa.field("confidence", pa.list_(pa.list_(pa.float16()))))
    schema = pa.schema(fields)
    keys = ["points", "detected"] + ([] if source == "mediapipe" else ["confidence"])
    count = 49 if source == "mediapipe" else 65
    detected_count = 3 if source == "mediapipe" else 65
    output.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for start in range(0, len(files), 32):
            rows = []
            for path in files[start:start + 32]:
                with np.load(path, allow_pickle=False) as data:
                    points, detected = data["points"], data["detected"]
                    if (points.dtype != np.float16 or points.shape[1:] != (count, 2)
                            or detected.dtype != np.bool_
                            or detected.shape != (len(points), detected_count)):
                        raise ValueError(f"Unexpected shape or dtype in {path}")
                    if source != "mediapipe" and (data["confidence"].dtype != np.float16
                            or data["confidence"].shape != (len(points), count)):
                        raise ValueError(f"Unexpected confidence shape or dtype in {path}")
                    rows.append({"sentence_id": path.stem, **{k: data[k].tolist() for k in keys}})
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return len(files)


class DataManager:
    """Manage the configured dataset without changing its stored formats."""

    def __init__(self, data: dict):
        self.data = data

    def download_annotations(self, update_manifest: bool = True) -> None:
        """Fetch the re-aligned CSVs if missing.

        update_manifest=True records their hashes; training passes False, so a
        reproduction run only reads the committed manifest and verify() checks the files against it.
        """
        data = self.data
        from huggingface_hub import hf_hub_download

        hub = data["source_hub"]
        paths = []
        for split in data["splits"]:
            path = annotation_path(data, split)
            if not path.exists():
                hf_hub_download(repo_id=hub["repo"], repo_type="dataset", revision=hub["revision"],
                                filename=path.name, local_dir=data["raw"])
            paths.append(path)
        if update_manifest:
            source = {"repo": hub["repo"], "revision": hub["revision"], "license": "CC BY-NC 4.0",
                      "homepage": "https://how2sign.github.io/"}
            previous = read_manifest()
            retrieved = previous.get("retrieved") if previous.get("source") == source else None
            record(paths, {"source": source, "retrieved": retrieved or date.today().isoformat()})
            print(f"Annotations in {data['raw']}; hashes in {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    def build_video_index(self) -> None:
        """Read each shard's zip directory over HTTP (no video bytes) and map video -> shard."""
        data = self.data
        from huggingface_hub import HfApi, HfFileSystem

        hub = data["source_hub"]
        files = HfApi().list_repo_files(hub["repo"], repo_type="dataset", revision=hub["revision"])
        fs = HfFileSystem()
        index = {}
        for split in data["splits"]:
            folder = data["sources"]["videos"][split]["dir"]
            shards = sorted(f for f in files if f.startswith(f"{folder}/") and f.endswith(".zip"))
            if not shards:
                raise SystemExit(f"No shards under {folder}/ at revision {hub['revision']}")
            index[split] = {}
            for shard in shards:
                with fs.open(f"datasets/{hub['repo']}@{hub['revision']}/{shard}", "rb") as f:
                    for member in zipfile.ZipFile(f).namelist():
                        if member.endswith(".mp4"):
                            index[split][PurePosixPath(member).stem] = [shard, member]
                print(f"{shard}: {len(index[split])} {split} videos so far")
        path = video_index_path(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", newline="\n")
        record([path])
        sample = next(iter(index["val"]))
        total = sum(map(len, index.values()))
        print(f"Indexed {total} videos, e.g. {sample!r} -> {index['val'][sample]}")

    def load_video_index(self) -> dict[str, dict[str, list[str]]]:
        """Verify and load the video-to-shard index."""
        data = self.data
        path = video_index_path(data)
        verify([path])
        return json.loads(path.read_text())

    def build_splits(self) -> None:
        """Build and record the vocabulary and sentence lists for each split."""
        data = self.data
        verify([annotation_path(data, s) for s in data["splits"]])
        index = self.load_video_index()
        frames = {s: load_annotations(data, s) for s in data["splits"]}
        for split, df in frames.items():
            missing = set(df["VIDEO_NAME"]) - set(index[split])
            if missing:
                print(f"{split}: {len(missing)} videos in the CSV are not in the mirror, e.g. "
                      f"{sorted(missing)[:3]}; their sentences are skipped")
        available = {s: set(index[s]) for s in data["splits"]}
        vocab, chosen = make_splits(frames, data, available)

        out = data["processed"] / "splits"
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / "vocab.txt"]
        paths[0].write_text("\n".join(vocab) + "\n", newline="\n")
        for name, df in chosen.items():
            path = out / f"{name}.txt"
            path.write_text("\n".join(df["SENTENCE_NAME"]) + "\n", newline="\n")
            paths.append(path)
            print(f"{name}: {len(df)} sentences from {df['VIDEO_NAME'].nunique()} videos")
        record(paths)

    def download_features(self, source: str, splits: list[str], smoke: bool = False) -> list[Path]:
        """Download and verify the selected feature partitions at the pinned revision."""
        from huggingface_hub import snapshot_download

        from asl_translate.pose import folder

        data = self.data
        root = data["processed"]
        paths = [root / folder(source, smoke) / split for split in splits]
        hub = data["processed_hub"]
        if not hub.get("revision"):
            raise SystemExit("Set data.processed_hub.revision to the dataset commit to download")
        patterns = ["splits/*"] + [f"{folder(source, smoke)}/{split}/*.npz" for split in splits]
        metadata = root / folder(source) / "metadata.json"
        if source != "mediapipe":
            patterns.append(f"{folder(source)}/metadata.json")
        snapshot_download(hub["repo"], repo_type="dataset", revision=hub["revision"],
                          allow_patterns=patterns, local_dir=root)
        missing = [str(path) for path in paths if not path.is_dir() or not any(path.glob("*.npz"))]
        if missing:
            raise SystemExit(f"Feature files absent at pinned dataset revision: {missing}")
        verify([root / "splits" / "vocab.txt"] + paths +
               ([metadata] if source != "mediapipe" else []))
        return paths

    def export_parquet(self, mediapipe_root: Path, other_root: Path, output: Path) -> None:
        """Convert the existing feature partitions into Parquet files."""
        for source in ("mediapipe", "vitpose", "rtmw", "dwpose"):
            for split in ("train", "val", "test"):
                if source == "mediapipe" and split == "test":
                    continue
                source_dir = mediapipe_root / split if source == "mediapipe" else other_root / source / split
                path = output / source / f"{split}-00000.parquet"
                print(source, split, convert_partition(source, source_dir, path), flush=True)

    def upload_features(self, source: str, splits: list[str], smoke: bool = False):
        """Verify and publish processed features to the configured dataset."""
        from huggingface_hub import HfApi

        from asl_translate.pose import folder

        processed = self.data["processed"]
        parts = ["splits", "video_index.json"] + [f"{folder(source)}/{s}" for s in splits]
        if source != "mediapipe":
            parts.append(f"{folder(source)}/metadata.json")
        if smoke or (source == "mediapipe" and splits == ["train", "val"]):
            parts += [f"{folder(source, True)}/{s}" for s in ("train", "val")]
        paths = [processed / p for p in parts]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"Missing {missing}; run data.py and features.py first")
        verify(sorted((processed / "splits").glob("*.txt")) + paths[1:])
        hub = self.data["processed_hub"]
        patterns = [f"{p}/**" if (processed / p).is_dir() else p for p in parts]
        info = HfApi().upload_folder(
            folder_path=processed, repo_id=hub["repo"], repo_type="dataset",
            allow_patterns=patterns, ignore_patterns=[".cache/**"],
            commit_message=f"Upload {source} processed data",
        )
        print(f"Uploaded to https://huggingface.co/datasets/{hub['repo']}")
        print("Pin this in configs/config.yaml under data.processed_hub:")
        print(f"    revision: {info.oid}")
        return info


def main() -> None:
    """Run a data preparation, transfer, or export command."""
    from asl_translate.pose import SOURCES

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name in ("annotations", "index", "splits"):
        commands.add_parser(name)
    for name in ("download", "upload"):
        command = commands.add_parser(name)
        command.add_argument("--pose_estimator", choices=SOURCES, default="mediapipe")
        command.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                             default=["train", "val"])
        command.add_argument("--smoke", action="store_true")
    export = commands.add_parser("export-parquet")
    export.add_argument("--mediapipe-root", type=Path, required=True)
    export.add_argument("--other-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manager = DataManager(load_config()["data"])
    if args.command in (None, "annotations"):
        manager.download_annotations()
    if args.command in (None, "index"):
        manager.build_video_index()
    if args.command in (None, "splits"):
        manager.build_splits()
    if args.command == "download":
        for path in manager.download_features(args.pose_estimator, args.splits, args.smoke):
            print(path)
    elif args.command == "upload":
        manager.upload_features(args.pose_estimator, args.splits, args.smoke)
    elif args.command == "export-parquet":
        manager.export_parquet(args.mediapipe_root, args.other_root, args.output)


if __name__ == "__main__":
    main()
