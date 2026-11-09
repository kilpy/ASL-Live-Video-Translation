"""Download annotations, index the full videos, and build the splits.

Run every step:   uv run python -m asl_translate.dataset
Run one step:     uv run python -m asl_translate.dataset annotations|index|splits

Steps, in order (each reads only what the previous ones wrote):
1. annotations  re-aligned CSVs -> data/raw/, sha256 recorded in data/manifest.json
2. index        which shard zip holds each full video -> data/processed/video_index.json
3. splits       train-only vocabulary, covered sentences -> data/processed/splits/*.txt
"""
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
    """Hash of a file, or of a folder's file names and hashes (so one entry covers a folder)."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
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
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def verify(paths: list[Path], path: Path = MANIFEST_PATH) -> None:
    """Fail loudly if any file differs from the hash in the committed manifest."""
    files = read_manifest(path)["files"]
    for p in paths:
        key = manifest_key(p)
        if key not in files:
            raise SystemExit(f"{key} is not in {path.name}; rebuild it with asl_translate.dataset")
        if sha256(p) != files[key]:
            raise SystemExit(f"{key} does not match its hash in {path.name}: data has changed")


# ---------------------------------------------------------------- step 1: annotations


def tokenize(sentence: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", sentence.lower())


def annotation_path(data: dict, split: str) -> Path:
    return data["raw"] / data["sources"]["annotations"][split]["file"]


def load_annotations(data: dict, split: str) -> pd.DataFrame:
    # QUOTE_NONE: sentences contain literal quotes, and the files are tab-separated
    df = pd.read_csv(annotation_path(data, split), sep="\t", quoting=3, keep_default_na=False)
    df["DURATION"] = df["END_REALIGNED"] - df["START_REALIGNED"]
    df["TOKENS"] = df["SENTENCE"].map(tokenize)
    return df


def download_annotations(data: dict) -> None:
    from huggingface_hub import hf_hub_download

    hub = data["source_hub"]
    paths = []
    for split in data["splits"]:
        path = annotation_path(data, split)
        if not path.exists():
            hf_hub_download(repo_id=hub["repo"], repo_type="dataset", revision=hub["revision"],
                            filename=path.name, local_dir=data["raw"])
        paths.append(path)
    source = {"repo": hub["repo"], "revision": hub["revision"], "license": "CC BY-NC 4.0",
              "homepage": "https://how2sign.github.io/"}
    previous = read_manifest()
    retrieved = previous.get("retrieved") if previous.get("source") == source else None
    record(paths, {"source": source, "retrieved": retrieved or date.today().isoformat()})
    print(f"Annotations in {data['raw']}; hashes in {MANIFEST_PATH.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------- step 2: video index


def video_index_path(data: dict) -> Path:
    return data["processed"] / "video_index.json"


def build_video_index(data: dict) -> None:
    """Read each shard's zip directory over HTTP (no video bytes) and map video -> shard."""
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
    path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    record([path])
    sample = next(iter(index["val"]))
    total = sum(map(len, index.values()))
    print(f"Indexed {total} videos, e.g. {sample!r} -> {index['val'][sample]}")


def load_video_index(data: dict) -> dict[str, dict[str, list[str]]]:
    path = video_index_path(data)
    verify([path])
    return json.loads(path.read_text())


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


def build_splits(data: dict) -> None:
    verify([annotation_path(data, s) for s in data["splits"]])
    index = load_video_index(data)
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
    paths[0].write_text("\n".join(vocab) + "\n")
    for name, df in chosen.items():
        path = out / f"{name}.txt"
        path.write_text("\n".join(df["SENTENCE_NAME"]) + "\n")
        paths.append(path)
        print(f"{name}: {len(df)} sentences from {df['VIDEO_NAME'].nunique()} videos")
    record(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", nargs="?", default="all",
                        choices=["all", "annotations", "index", "splits"])
    step = parser.parse_args().step
    data = load_config()["data"]
    if step in ("all", "annotations"):
        download_annotations(data)
    if step in ("all", "index"):
        build_video_index(data)
    if step in ("all", "splits"):
        build_splits(data)


if __name__ == "__main__":
    main()