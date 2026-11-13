"""BASELINE model: pose GRU encoder + Transformer decoder over word tokens.

Shapes: B batch, T frames, L tokens, F features per frame, D decoder width, H GRU
hidden size per direction, V vocabulary including special tokens. F depends on the pose
estimator: MediaPipe uses 49 points x 2 coordinates plus 3 detection flags (101);
the other estimators use 65 or 67 points x 2 coordinates plus one detection flag per point.
"""
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
        # frames: (B, T, F). Padded frames are zeros; the decoder masks them out with
        # memory_key_padding_mask.
        """Encode frame features of shape (B, T, F) as (B, T, D)."""
        x = self.proj(frames)       # (B, T, D)
        hidden, _ = self.gru(x)     # (B, T, 2H): forward and backward states concatenated
        return self.out(hidden)     # (B, T, D): the "memory" the decoder attends to


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
        self.position = nn.Embedding(max_tokens, d_model)                   # position -> (D,)
        layer = nn.TransformerDecoderLayer(d_model, decoder["heads"], decoder["ff"],
                                           decoder["dropout"], batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder["layers"])
        self.head = nn.Linear(d_model, vocab_size)                          # (D,) -> (V,)

    def decode(self, tokens: torch.Tensor, memory: torch.Tensor,
               frame_pad: torch.Tensor) -> torch.Tensor:
        """Logits for the next token at every position.

        tokens: (B, L) ids; memory: (B, T, D); frame_pad: (B, T) bool, True = padding.
        Returns (B, L, V).
        """
        length = tokens.size(1)
        positions = torch.arange(length, device=tokens.device)              # (L,)
        x = self.embed(tokens) + self.position(positions)                   # (B, L, D)
        # (L, L) bool, True above the diagonal: position i cannot see positions after i
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                       device=tokens.device), diagonal=1)
        out = self.decoder(x, memory, tgt_mask=causal,
                           tgt_key_padding_mask=tokens == self.pad_id,      # (B, L)
                           memory_key_padding_mask=frame_pad)               # (B, T)
        return self.head(out)                                               # (B, L, V)

    def forward(self, frames: torch.Tensor, frame_pad: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        """Teacher forcing.

        frames: (B, T, F); frame_pad: (B, T); tokens: (B, L) = BOS w1 ... wn EOS, padded.
        Returns (B, L-1, V): the prediction for tokens[:, 1:] from tokens[:, :-1].
        """
        memory = self.encoder(frames)                                       # (B, T, D)
        return self.decode(tokens[:, :-1], memory, frame_pad)               # (B, L-1, V)

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
        finished = torch.zeros(frames.size(0), dtype=torch.bool, device=frames.device)  # (B,)
        for _ in range(max_len - 1):
            logp = self.decode(tokens, memory, frame_pad)[:, -1].log_softmax(-1)  # (B, V)
            best, next_token = logp.max(-1)                                 # (B,), (B,)
            next_token = next_token.masked_fill(finished, self.pad_id)      # stop finished rows
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)        # (B, step + 1)
            logps.append(best.masked_fill(finished, 0.0))
            finished |= next_token == eos
            if finished.all():
                break
        return tokens[:, 1:], torch.stack(logps, dim=1)                     # (B, S), (B, S)