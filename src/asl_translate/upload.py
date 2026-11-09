"""Publish data/processed/ to our Hugging Face dataset. Run after extraction:

    uv run hf auth login        # once; needs write access to the org
    uv run python -m asl_translate.upload

Uploads splits, the video index, model-input landmarks, and the smoke subset. Never uploads
data/raw (third-party files) or data/interim (full landmarks, which include the face).
Prints the commit hash to pin as data.processed_hub.revision in configs/config.yaml.
"""
from asl_translate.config import load_config
from asl_translate.dataset import read_manifest, verify

PARTS = ["splits", "video_index.json", "landmarks/train", "landmarks/val",
         "smoke/train", "smoke/val"]


def main() -> None:
    from huggingface_hub import HfApi

    data = load_config()["data"]
    processed = data["processed"]
    paths = [processed / p for p in PARTS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}; run dataset.py and features.py first")
    split_files = sorted((processed / "splits").glob("*.txt"))
    verify(split_files + paths[1:])  # upload exactly what the manifest describes

    hub = data["processed_hub"]
    patterns = [f"{p}/**" if (processed / p).is_dir() else p for p in PARTS]
    extraction = read_manifest().get("extraction", {})
    info = HfApi().upload_folder(
        folder_path=processed, repo_id=hub["repo"], repo_type="dataset",
        allow_patterns=patterns, ignore_patterns=[".cache/**"],
        commit_message=f"Upload processed data (MediaPipe {extraction.get('mediapipe', '?')})",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{hub['repo']}")
    print("Pin this in configs/config.yaml under data.processed_hub:")
    print(f"    revision: {info.oid}")


if __name__ == "__main__":
    main()