"""Download and verify processed features for selected sources and splits."""
import argparse

from huggingface_hub import snapshot_download

from asl_translate.config import load_config
from asl_translate.dataset import verify
from asl_translate.pose_sources import SOURCES, folder


def download(data: dict, source: str, splits: list[str], smoke: bool = False) -> list:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default="mediapipe")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        default=["train", "val"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for path in download(load_config()["data"], args.source, args.splits, args.smoke):
        print(path)


if __name__ == "__main__":
    main()
