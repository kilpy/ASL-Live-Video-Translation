"""Publish data/processed/ to our Hugging Face dataset. Run after extraction:

    uv run hf auth login        # once; needs write access to the org
    uv run python -m asl_translate.upload

Uploads splits, the video index, model-input landmarks, and the smoke subset. Never uploads
data/raw (third-party files) or data/interim (full landmarks, which include the face).
Prints the commit hash to pin as data.processed_hub.revision in configs/config.yaml.
"""
import argparse

from asl_translate.config import load_config
from asl_translate.dataset import verify
from asl_translate.pose_sources import SOURCES, folder

def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default="mediapipe")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        default=["train", "val"])
    parser.add_argument("--smoke", action="store_true", help="include train/val smoke data")
    args = parser.parse_args()
    data = load_config()["data"]
    processed = data["processed"]
    parts = ["splits", "video_index.json"] + [f"{folder(args.source)}/{s}" for s in args.splits]
    if args.source != "mediapipe":
        parts.append(f"{folder(args.source)}/metadata.json")
    if args.smoke or (args.source == "mediapipe" and args.splits == ["train", "val"]):
        parts += [f"{folder(args.source, True)}/{s}" for s in ("train", "val")]
    paths = [processed / p for p in parts]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}; run dataset.py and features.py first")
    split_files = sorted((processed / "splits").glob("*.txt"))
    verify(split_files + paths[1:])  # upload exactly what the manifest describes

    hub = data["processed_hub"]
    patterns = [f"{p}/**" if (processed / p).is_dir() else p for p in parts]
    info = HfApi().upload_folder(
        folder_path=processed, repo_id=hub["repo"], repo_type="dataset",
        allow_patterns=patterns, ignore_patterns=[".cache/**"],
        commit_message=f"Upload {args.source} processed data",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{hub['repo']}")
    print("Pin this in configs/config.yaml under data.processed_hub:")
    print(f"    revision: {info.oid}")


if __name__ == "__main__":
    main()
