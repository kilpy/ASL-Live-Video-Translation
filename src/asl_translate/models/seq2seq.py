"""Any encoder from encoders.py + any decoder from decoders.py, picked by config name.

Shapes: B batch, T frames, L tokens, F features per frame, D decoder width, V vocabulary
including special tokens. F depends on the feature source: MediaPipe uses 49 points x 2
coordinates plus 3 detection flags (101); dinov2 adds 3 x 384 crop features in front of those;
the other estimators use 65 or 67 points x 2 coordinates plus one detection flag per point.
"""
import torch
from torch import nn

from asl_translate.models.decoders import DECODERS
from asl_translate.models.encoders import ENCODERS


class Seq2Seq(nn.Module):
    """Encode frame features and decode text tokens autoregressively."""

    def __init__(self, in_dim: int, vocab_size: int, encoder: dict, decoder: dict,
                 max_tokens: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        # checkpoints from before the registry have no names: they are the BASELINE pair
        self.decoder = DECODERS[decoder.get("name", "transformer")](
            vocab_size, **{**decoder, "max_tokens": max_tokens, "pad_id": pad_id})
        self.encoder = ENCODERS[encoder.get("name", "pose_gru")](
            in_dim, self.decoder.d_model, **encoder)

    def encode(self, frames: torch.Tensor, frame_pad: torch.Tensor) -> torch.Tensor:
        """(B, T, F) frames and (B, T) padding (True = pad) -> (B, T, D) decoder memory."""
        return self.decoder.prepare(self.encoder(frames, frame_pad), frame_pad)

    def forward(self, frames: torch.Tensor, frame_pad: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        """Teacher forcing.

        tokens: (B, L) = BOS w1 ... wn EOS, padded.
        Returns (B, L-1, V): the prediction for tokens[:, 1:] from tokens[:, :-1].
        """
        return self.decoder(tokens[:, :-1], self.encode(frames, frame_pad), frame_pad)

    @torch.no_grad()
    def generate(self, frames: torch.Tensor, frame_pad: torch.Tensor, bos: int, eos: int,
                 max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding, one token per step.

        Returns ids (B, S) without BOS and their log-probabilities (B, S), S <= max_len - 1.
        """
        memory = self.encode(frames, frame_pad)                             # (B, T, D)
        tokens = torch.full((frames.size(0), 1), bos, dtype=torch.long,
                            device=frames.device)                           # (B, 1)
        logps = []
        finished = torch.zeros(frames.size(0), dtype=torch.bool, device=frames.device)  # (B,)
        # ponytail: re-decodes the whole prefix each step (no KV cache); fine at <= 256 tokens
        for _ in range(max_len - 1):
            logp = self.decoder(tokens, memory, frame_pad)[:, -1].log_softmax(-1)  # (B, V)
            best, next_token = logp.max(-1)                                 # (B,), (B,)
            next_token = next_token.masked_fill(finished, self.pad_id)      # stop finished rows
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)        # (B, step + 1)
            logps.append(best.masked_fill(finished, 0.0))
            finished |= next_token == eos
            if finished.all():
                break
        return tokens[:, 1:], torch.stack(logps, dim=1)                     # (B, S), (B, S)
