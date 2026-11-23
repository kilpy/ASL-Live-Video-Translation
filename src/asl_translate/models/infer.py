"""Load a trained checkpoint and translate one sentence of landmarks.

    from asl_translate.models.infer import Translator
    translator = Translator.load("models/BASELINE/full/seed_0/model.pt")
    text, confidence = translator.translate(points, detected)

`points` and `detected` are what `features.reduce` returns (plus `dino.embed` crops for
models trained on the dinov2 source, see `Translator.needs_crops`), so the webcam demo and offline
evaluation share the preprocessing used in training (`train.frame_features`).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from asl_translate.config import get_device, load_config
from asl_translate.models.seq2seq import Seq2Seq
from asl_translate.models.train import BOS, EOS, PAD, SEPARATOR, UNK, frame_features, normalize


@dataclass
class Explanation:
    """A translation plus what the demo shows about how it was made."""
    words: list[str]
    probs: list[float]                     # probability of each chosen word
    topk: list[list[tuple[str, float]]]    # best alternatives at each word, chosen word first
    confidence: float
    attention: np.ndarray                  # (words, T) decoder cross-attention over input frames
    model_input: np.ndarray                # (T, P, 2) normalized points, as the model sees them
    detected: np.ndarray                   # (T, 3) pose / left hand / right hand flags


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
        self.separator = SEPARATOR[checkpoint.get("units", "words")]
        self.needs_crops = self.source == "dinov2"
        state = checkpoint["state_dict"]
        if "head.weight" in state:  # saved before the encoder/decoder registry
            state = {k if k.startswith("encoder.") else
                     "decoder." + k.replace("decoder.", "stack.", 1): v for k, v in state.items()}
        self.model = Seq2Seq(checkpoint["in_dim"], len(self.itos), checkpoint["encoder"],
                             checkpoint["decoder"], self.max_tokens, self.words[PAD])
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, device: str = "auto", approach: str = "BASELINE"):
        """Load a checkpoint written by train.py (new format or a bare state_dict)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" not in checkpoint:
            checkpoint = _legacy_checkpoint(checkpoint, approach)
        return cls(checkpoint, device)

    def translate(self, points: np.ndarray, detected: np.ndarray,
                  crops: np.ndarray | None = None) -> tuple[str, float]:
        """(T, P, 2) points and (T, K) detection flags -> (sentence, confidence in [0, 1])."""
        explanation = self.explain(points, detected, crops)
        return self.separator.join(explanation.words), explanation.confidence

    @torch.no_grad()
    def explain(self, points: np.ndarray, detected: np.ndarray, crops: np.ndarray | None = None,
                k: int = 3) -> Explanation:
        """Translate, and keep per-word probabilities, alternatives and cross-attention.

        Words are characters for a `units: chars` decoder (ByT5); join with `separator`.
        """
        if self.needs_crops and crops is None:
            raise ValueError("This model was trained on dinov2 features: pass dino.embed crops")
        total = len(points)
        model_input = normalize(points.astype(np.float32), self.source) if total else points
        if total == 0:
            return Explanation([], [], [], 0.0, np.zeros((0, 0)), model_input, detected)
        features = frame_features(points, detected, self.max_frames, self.source, crops)
        frames = torch.from_numpy(features)[None].to(self.device)                 # (1, T', F)
        frame_pad = torch.zeros(frames.shape[:2], dtype=torch.bool, device=self.device)
        ids, logps = self.model.generate(frames, frame_pad, self.words[BOS], self.words[EOS],
                                         self.max_tokens)
        row = ids[0].tolist()
        n = row.index(self.words[EOS]) if self.words[EOS] in row else len(row)
        # same confidence as train.evaluate: exp(mean log-probability, including EOS)
        confidence = float(logps[0, : n + 1].mean().exp())

        # One teacher-forced pass over the greedy output reproduces each step's distribution
        # (causal mask, eval mode) and lets us read the cross-attention for every word at once.
        tokens = torch.tensor([[self.words[BOS]] + row[:n]], device=self.device)  # (1, n + 1)
        logits, attention = self.model.decoder(tokens, self.model.encode(frames, frame_pad),
                                               frame_pad, attention=True)
        logits = logits[0, :n]
        attention = attention[0, :n].float().cpu().numpy()                         # (n, T')
        probs = logits.float().softmax(-1)                                         # (n, V)
        top_p, top_i = probs.topk(min(k, probs.size(-1)), dim=-1)

        # Map attention from the model frames back to every input frame. frame_features keeps
        # them all up to max_frames, else subsamples evenly; give each input frame its nearest.
        full = attention[:, np.round(np.linspace(0, len(features) - 1, total)).astype(int)]

        # greedy decoding can emit <pad>/<bos> from a weak model; they mean nothing to a reader
        hidden = {self.words[PAD], self.words[BOS]}
        keep = [i for i in range(n) if row[i] not in hidden]
        return Explanation(
            words=[self.itos[row[i]] for i in keep],
            probs=[float(probs[i, row[i]]) for i in keep],
            topk=[[(self.itos[j], float(p)) for j, p in zip(top_i[i].tolist(), top_p[i].tolist())]
                  for i in keep],
            confidence=confidence, attention=full[keep], model_input=model_input,
            detected=detected)
