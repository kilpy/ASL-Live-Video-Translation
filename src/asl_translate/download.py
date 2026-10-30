"""Fetch How2Sign files. Run with `uv run python -m asl_translate.download annotations`."""
import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from asl_translate.config import load_config


def fetch(entry: dict, data: dict) -> Path:
    """Download one file, refusing to start if the disk can't hold it and its extraction."""
    root = data["root"]
    root.mkdir(parents=True, exist_ok=True)
    out = root / entry["file"]
    is_zip = out.suffix == ".zip"
    if out.exists() or (is_zip and out.with_suffix("").exists()):
        print(f"Already present: {out}")
        return out
    free_gb = shutil.disk_usage(root).free / 1e9
    needed_gb = entry["gb"] * 2 if is_zip else entry["gb"]
    if free_gb < needed_gb:
        raise SystemExit(f"{entry['file']} needs {needed_gb:.1f} GB, only {free_gb:.1f} GB free")
    hf_hub_download(
        repo_id=data["hub"]["repo"],
        repo_type="dataset",
        revision=data["hub"]["revision"],
        filename=entry["file"],
        local_dir=root,
    )
    if is_zip:
        with zipfile.ZipFile(out) as archive:
            archive.extractall(out.with_suffix(""))
        out.unlink()
    return out


def main() -> None:
    data = load_config()["data"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=list(data["sources"]))
    parser.add_argument("--split", choices=data["splits"], action="append", help="default: all")
    args = parser.parse_args()

    for split in args.split or data["splits"]:
        fetch(data["sources"][args.source][split], data)


if __name__ == "__main__":
    main()
