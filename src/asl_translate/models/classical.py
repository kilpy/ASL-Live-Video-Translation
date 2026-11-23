"""BASELINE model: pose GRU encoder + GRU decoder with attention over word tokens.

Shapes: B batch, T frames, L tokens, F features per frame, D decoder width, H GRU
hidden size per direction, V vocabulary including special tokens. F depends on the pose
estimator: MediaPipe uses 49 points x 2 coordinates plus 3 detection flags (101);
the other estimators use 65 or 67 points x 2 coordinates plus one detection flag per point.
"""
import math

import torch
from torch import nn


class PoseGRUEncoder(nn.Module):
    """Per-frame features -> bidirectional GRU -> one d_model vector per frame."""

    def __init__(self, in_dim: int, d_model: int, hidden: int, layers: int,
                 bidirectional: bool, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(d_model, hidden, layers, batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), d_model)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, F). Padded frames are zeros; the decoder masks them out of attention
        # and out of the initial-state average.
        """Encode frame features of shape (B, T, F) as (B, T, D)."""
        x = self.proj(frames)       # (B, T, D)
        hidden, _ = self.gru(x)     # (B, T, 2H): forward and backward states concatenated
        return self.out(hidden)     # (B, T, D): the "memory" the decoder attends to


class GRUAttentionDecoder(nn.Module):
    """Unidirectional GRU over word embeddings, then Luong-style attention over the frames.

    The recurrent state is initialised from the masked mean of the encoder memory. At each
    position the GRU output queries the memory; [output; context] is mixed by a linear layer
    before the vocabulary head. Attention is applied after the GRU (no input feeding), so
    teacher-forced training runs in one parallel pass and generation only needs to carry the
    GRU hidden state.
    """

    def __init__(self, d_model: int, layers: int, dropout: float):
        super().__init__()
        self.layers = layers
        self.init_state = nn.Linear(d_model, d_model)
        self.gru = nn.GRU(d_model, d_model, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.query = nn.Linear(d_model, d_model, bias=False)     # "general" attention score
        self.mix = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, frame_pad: torch.Tensor,
                hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, D) embedded tokens; memory: (B, T, D); frame_pad: (B, T) True = padding;
        hidden: (layers, B, D) or None to start from the memory summary.
        Returns features (B, L, D) and the final hidden state (layers, B, D)."""
        if hidden is None:
            valid = (~frame_pad).unsqueeze(-1).to(memory.dtype)                  # (B, T, 1)
            mean = (memory * valid).sum(1) / valid.sum(1).clamp(min=1.0)         # (B, D)
            hidden = torch.tanh(self.init_state(mean))                           # (B, D)
            hidden = hidden.unsqueeze(0).repeat(self.layers, 1, 1).contiguous()  # (layers, B, D)
        out, hidden = self.gru(self.dropout(x), hidden)                          # (B, L, D)
        scores = torch.bmm(self.query(out), memory.transpose(1, 2)) / self.scale  # (B, L, T)
        scores = scores.masked_fill(frame_pad[:, None, :], float("-inf"))
        context = torch.bmm(scores.softmax(-1), memory)                          # (B, L, D)
        feats = torch.tanh(self.mix(torch.cat([out, context], dim=-1)))          # (B, L, D)
        return self.dropout(feats), hidden


class Seq2Seq(nn.Module):
    """Encode landmark sequences and decode text tokens autoregressively."""
    def __init__(self, in_dim: int, vocab_size: int, encoder: dict, decoder: dict,
                 max_tokens: int, pad_id: int):
        super().__init__()
        d_model = decoder["d_model"]
        self.pad_id = pad_id
        self.encoder = PoseGRUEncoder(in_dim, d_model, encoder["hidden"], encoder["layers"],
                                      encoder["bidirectional"], encoder["dropout"])
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # id -> (D,)
        # The GRU encodes order itself, so there is no position embedding. max_tokens is kept
        # in the signature so the Transformer config still works. The decoder's "heads" and
        # "ff" entries are unused here.
        self.decoder = GRUAttentionDecoder(d_model, decoder["layers"], decoder["dropout"])
        self.head = nn.Linear(d_model, vocab_size)                          # (D,) -> (V,)

    def decode(self, tokens: torch.Tensor, memory: torch.Tensor, frame_pad: torch.Tensor,
               hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits for the next token at every position.

        tokens: (B, L) ids; memory: (B, T, D); frame_pad: (B, T) bool, True = padding;
        hidden: optional (layers, B, D) GRU state to resume from.
        Returns logits (B, L, V) and the new hidden state (layers, B, D).
        """
        x = self.embed(tokens)                                              # (B, L, D)
        out, hidden = self.decoder(x, memory, frame_pad, hidden)            # (B, L, D)
        return self.head(out), hidden                                       # (B, L, V)

    def forward(self, frames: torch.Tensor, frame_pad: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        """Teacher forcing.

        frames: (B, T, F); frame_pad: (B, T); tokens: (B, L) = BOS w1 ... wn EOS, padded.
        Returns (B, L-1, V): the prediction for tokens[:, 1:] from tokens[:, :-1].
        """
        memory = self.encoder(frames)                                       # (B, T, D)
        logits, _ = self.decode(tokens[:, :-1], memory, frame_pad)          # (B, L-1, V)
        return logits

    @torch.no_grad()
    def generate(self, frames: torch.Tensor, frame_pad: torch.Tensor, bos: int, eos: int,
                 max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding, one token per step.

        frames: (B, T, F); frame_pad: (B, T).
        Returns ids (B, S) without BOS and their log-probabilities (B, S), S <= max_len - 1.
        """
        memory = self.encoder(frames)                                       # (B, T, D)
        tokens = torch.full((frames.size(0), 1), bos, dtype=torch.long,
                            device=frames.device)                           # (B, 1)
        logps = []
        hidden = None                                                       # set on first step
        finished = torch.zeros(frames.size(0), dtype=torch.bool, device=frames.device)  # (B,)
        for _ in range(max_len - 1):
            # Only the newest token is fed; the GRU hidden state carries the history.
            logits, hidden = self.decode(tokens[:, -1:], memory, frame_pad, hidden)
            logp = logits[:, -1].log_softmax(-1)                            # (B, V)
            best, next_token = logp.max(-1)                                 # (B,), (B,)
            next_token = next_token.masked_fill(finished, self.pad_id)      # stop finished rows
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)        # (B, step + 1)
            logps.append(best.masked_fill(finished, 0.0))
            finished |= next_token == eos
            if finished.all():
                break
        return tokens[:, 1:], torch.stack(logps, dim=1)                     # (B, S), (B, S)