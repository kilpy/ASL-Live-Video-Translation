"""Train and evaluate one approach; log the result. The one command for the walkthrough:

    uv run python -m asl_translate.models.train --approach BASELINE --smoke

--smoke trains on the small committed-to-Hugging-Face subset on CPU (under 5 minutes, no token).
Without it, trains on every extracted train sentence and evaluates on val. The test split is
never read here: it is used once, at the end of the project.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from torch import nn
from torch.utils.data import DataLoader

from asl_translate.config import CONFIG_PATH, REPO_ROOT, get_device, load_config, set_seed
from asl_translate.dataset import (annotation_path, download_annotations, load_annotations,
                                      manifest_key, read_manifest, tokenize, verify)
from asl_translate.models.baseline import Seq2Seq
from asl_translate.pose_sources import SHOULDERS, folder

RESULTS = REPO_ROOT / "reports" / "results.jsonl"
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


# ---------------------------------------------------------------- data


def ensure_data(data: dict, smoke: bool, source: str = "mediapipe") -> list[Path]:
    """Fetch anything missing (annotations, splits, landmarks), then check every hash."""
    if not all(annotation_path(data, s).exists() for s in ("train", "val")):
        download_annotations(data, update_manifest=False)
    processed = data["processed"]
    folders = [f"{folder(source, smoke)}/{split}" for split in ("train", "val")]
    needed = [processed / "splits" / "vocab.txt"] + [processed / f for f in folders]
    if source != "mediapipe":
        needed.append(processed / folder(source) / "metadata.json")
    if not all(p.exists() for p in needed):
        from asl_translate.download_features import download

        download(data, source, ["train", "val"], smoke=smoke)
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"Missing feature data at pinned dataset revision: {missing}")
    verify([annotation_path(data, s) for s in ("train", "val")] + needed)
    return needed


def normalize(points: np.ndarray, source: str = "mediapipe") -> np.ndarray:
    """Center each frame on the shoulders' midpoint; scale by the sentence's shoulder width."""
    left, right = SHOULDERS[source]
    center = points[:, [left, right]].mean(axis=1, keepdims=True)
    width = np.linalg.norm(points[:, left] - points[:, right], axis=-1)
    scale = float(np.median(width[width > 0])) if (width > 0).any() else 1.0
    return (points - center) / scale


def load_split(folder: Path, sentences: dict[str, str], words: dict[str, int],
               max_frames: int, max_tokens: int, source: str = "mediapipe") -> list[tuple[np.ndarray, list[int], str]]:
    examples = []
    for path in sorted(folder.glob("*.npz"), key=lambda p: p.name):  # same order on every OS
        with np.load(path) as f:
            points, detected = f["points"].astype(np.float32), f["detected"].astype(np.float32)
        features = np.concatenate([normalize(points, source).reshape(len(points), -1), detected], axis=1)
        if len(features) > max_frames:  # keep the whole sentence, evenly subsampled
            features = features[np.linspace(0, len(features) - 1, max_frames).astype(int)]
        reference = " ".join(tokenize(sentences[path.stem]))
        ids = [words[BOS]] + [words.get(w, words[UNK]) for w in reference.split()]
        examples.append((features, (ids[: max_tokens - 1] + [words[EOS]]), reference))
    return examples


def collate(batch, pad_id: int):
    frames = [torch.from_numpy(f) for f, _, _ in batch]
    lengths = torch.tensor([len(f) for f in frames])
    padded = nn.utils.rnn.pad_sequence(frames, batch_first=True)
    frame_pad = torch.arange(padded.size(1))[None] >= lengths[:, None]
    tokens = nn.utils.rnn.pad_sequence([torch.tensor(t) for _, t, _ in batch],
                                       batch_first=True, padding_value=pad_id)
    return padded, frame_pad, tokens, [r for _, _, r in batch]


# ---------------------------------------------------------------- train and evaluate


def evaluate(model: Seq2Seq, loader: DataLoader, itos: list[str], words: dict[str, int],
             device: torch.device, max_tokens: int) -> dict:
    model.eval()
    hypotheses, references, confidences = [], [], []
    for frames, frame_pad, _, refs in loader:
        ids, logps = model.generate(frames.to(device), frame_pad.to(device), words[BOS],
                                    words[EOS], max_tokens)
        for row, row_logp in zip(ids.tolist(), logps.cpu()):
            n = row.index(words[EOS]) if words[EOS] in row else len(row)
            hypotheses.append(" ".join(itos[i] for i in row[:n]))
            # Experiment Card field 8: confidence = exp(mean log-probability of the words)
            confidences.append(float(row_logp[: n + 1].mean().exp()))
        references.extend(refs)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {"bleu": round(bleu.score, 2), "mean_confidence": round(float(np.mean(confidences)), 4),
            "examples": list(zip(references[:3], hypotheses[:3]))}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approach", default="BASELINE")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="default: first seed in config")
    parser.add_argument("--device", default=None, help="override training.device: cpu, mps, cuda, or auto")
    parser.add_argument("--source", choices=list(SHOULDERS), default="mediapipe")
    args = parser.parse_args()

    cfg = load_config(args.approach)
    seed = cfg["seeds"][0] if args.seed is None else args.seed
    set_seed(seed)
    train_cfg = {**cfg["training"], **(cfg["training"]["smoke"] if args.smoke else {})}
    device = get_device(args.device or train_cfg["device"])
    data, started = cfg["data"], time.time()
    commit = git("rev-parse", "HEAD")
    # the results log itself is expected to change between runs, so it doesn't count as dirty
    dirty = bool(git("status", "--porcelain", "--", ".", ":!reports/results.jsonl"))

    used = ensure_data(data, args.smoke, args.source)
    itos = [PAD, BOS, EOS, UNK] + (data["processed"] / "splits" / "vocab.txt").read_text().split()
    words = {w: i for i, w in enumerate(itos)}
    sentences = {}
    for split in ("train", "val"):
        df = load_annotations(data, split)
        sentences.update(zip(df["SENTENCE_NAME"], df["SENTENCE"]))
    root = data["processed"] / folder(args.source, args.smoke)
    splits = {s: load_split(root / s, sentences, words, train_cfg["max_frames"],
                            train_cfg["max_tokens"], args.source) for s in ("train", "val")}
    print(f"train {len(splits['train'])} / val {len(splits['val'])} sentences on {device}")

    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(splits[s], batch_size=train_cfg["batch_size"], shuffle=s == "train",
                             generator=generator, collate_fn=lambda b: collate(b, words[PAD]))
               for s in ("train", "val")}
    in_dim = splits["train"][0][0].shape[1]
    model = Seq2Seq(in_dim, len(itos), cfg["encoder"], cfg["decoder"], words[PAD]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=words[PAD],
                                  label_smoothing=train_cfg["label_smoothing"])

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for frames, frame_pad, tokens, _ in loaders["train"]:
            frames, frame_pad, tokens = frames.to(device), frame_pad.to(device), tokens.to(device)
            logits = model(frames, frame_pad, tokens)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch}: train loss {total / len(loaders['train']):.3f}")

    metrics = evaluate(model, loaders["val"], itos, words, device, train_cfg["max_tokens"])
    run_dir = cfg["run_dir"] / ("smoke" if args.smoke else "full") / f"seed_{seed}"
    if args.source != "mediapipe":
        run_dir = cfg["run_dir"] / args.source / ("smoke" if args.smoke else "full") / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    manifest = read_manifest()["files"]
    data_keys = sorted(manifest_key(p) for p in used)
    result = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approach": args.approach,
        "source": args.source,
        "smoke": args.smoke,
        "seed": seed,
        "split": "val",
        "n_train": len(splits["train"]),
        "n_eval": len(splits["val"]),
        "bleu": metrics["bleu"],
        "mean_confidence": metrics["mean_confidence"],
        "commit": commit,
        "dirty": dirty,
        # normalize line endings: Windows checkouts use CRLF, which would change a raw-byte hash
        "config_sha256": hashlib.sha256(
            CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "data_sha256": hashlib.sha256("".join(manifest[k] for k in data_keys).encode()).hexdigest(),
        "data": data_keys,
        "device": str(device),
        "hardware": f"{platform.platform()} {platform.processor()}",
        "torch": torch.__version__,
        "minutes": round((time.time() - started) / 60, 2),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    for reference, hypothesis in metrics["examples"]:
        print(f"  ref: {reference}\n  hyp: {hypothesis}")
    if dirty:
        print("Warning: uncommitted changes at start; commit before logging a result you will report.")


if __name__ == "__main__":
    main()
