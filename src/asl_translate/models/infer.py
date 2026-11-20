"""Load a trained checkpoint and translate one sentence of landmarks.

    from asl_translate.models.infer import Translator
    translator = Translator.load("models/BASELINE/full/seed_0/model.pt")
    text, confidence = translator.translate(points, detected)

`points` and `detected` are what `features.reduce` returns, so the webcam demo and offline
evaluation share the preprocessing used in training (`train.frame_features`).
"""
from pathlib import Path

import numpy as np
import torch

from asl_translate.config import get_device, load_config
from asl_translate.models.baseline import Seq2Seq
from asl_translate.models.train import BOS, EOS, PAD, UNK, frame_features


def _legacy_checkpoint(state_dict: dict, approach: str) -> dict:
    """Fill in metadata for checkpoints saved before train.py stored it (bare state_dict)."""
    cfg = load_config(approach)
    vocab = cfg["data"]["processed"] / "splits" / "vocab.txt"
    if not vocab.exists():
        raise SystemExit(f"Old-style checkpoint needs {vocab}; retrain or run a smoke train first.")
    return {"state_dict": state_dict, "itos": [PAD, BOS, EOS, UNK] + vocab.read_text().split(),
            "in_dim": state_dict["encoder.proj.0.weight"].shape[1],
            "encoder": cfg["encoder"], "decoder": cfg["decoder"],
            "max_frames": cfg["training"]["max_frames"], "max_tokens": cfg["training"]["max_tokens"],
            "source": "mediapipe", "fps": cfg["data"]["fps"], "features": cfg["data"]["features"]}


class Translator:
    """A trained Seq2Seq model plus the vocabulary and preprocessing settings it was trained with."""

    def __init__(self, checkpoint: dict, device: str = "auto"):
        self.device = get_device(device)
        self.itos = checkpoint["itos"]
        self.words = {w: i for i, w in enumerate(self.itos)}
        self.max_frames = checkpoint["max_frames"]
        self.max_tokens = checkpoint["max_tokens"]
        self.source = checkpoint["source"]
        self.fps = checkpoint["fps"]
        self.features = checkpoint["features"]
        self.model = Seq2Seq(checkpoint["in_dim"], len(self.itos), checkpoint["encoder"],
                             checkpoint["decoder"], self.max_tokens, self.words[PAD])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, device: str = "auto", approach: str = "BASELINE"):
        """Load a checkpoint written by train.py (new format or a bare state_dict)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" not in checkpoint:
            checkpoint = _legacy_checkpoint(checkpoint, approach)
        return cls(checkpoint, device)

    def translate(self, points: np.ndarray, detected: np.ndarray) -> tuple[str, float]:
        """(T, P, 2) points and (T, K) detection flags -> (sentence, confidence in [0, 1])."""
        if len(points) == 0:
            return "", 0.0
        features = frame_features(points, detected, self.max_frames, self.source)
        frames = torch.from_numpy(features)[None].to(self.device)                 # (1, T, F)
        frame_pad = torch.zeros(frames.shape[:2], dtype=torch.bool, device=self.device)
        ids, logps = self.model.generate(frames, frame_pad, self.words[BOS], self.words[EOS],
                                         self.max_tokens)
        row = ids[0].tolist()
        n = row.index(self.words[EOS]) if self.words[EOS] in row else len(row)
        # same confidence as train.evaluate: exp(mean log-probability, including EOS)
        confidence = float(logps[0, : n + 1].mean().exp())
        # greedy decoding can emit <pad>/<bos> from a weak model; they mean nothing to a reader
        hidden = {self.words[PAD], self.words[BOS]}
        return " ".join(self.itos[i] for i in row[:n] if i not in hidden), confidence
