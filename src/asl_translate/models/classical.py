"""CLASSICAL model: pose GRU encoder + GRU decoder with dot-product (Luong) attention.

Same interface as baseline.Seq2Seq, so train.py treats both alike. Shapes: B batch, T frames,
L tokens, F features per frame, E token embedding, Hd decoder hidden (also the encoder's output
size, which dot-product attention requires), V vocabulary incl. specials.
"""
import random

import torch
from torch import nn

from asl_translate.models.baseline import PoseGRUEncoder


class Decoder(nn.Module):
    """GRU decoder with dot-product (Luong) attention; generates one token per call."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, layers: int,
                 dropout: float, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hidden_dim, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token: torch.Tensor, hidden: torch.Tensor, memory: torch.Tensor,
                frame_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """token: (B,); hidden: (layers, B, Hd); memory: (B, T, Hd); frame_pad: (B, T) bool,
        True = padding. Returns logits (B, V), the new hidden state, and the attention weights
        over frames (B, T), kept for inspecting which frames each word came from."""
        embedded = self.dropout(self.embedding(token[:, None]))       # (B, 1, E)
        output, hidden = self.rnn(embedded, hidden)                   # (B, 1, Hd)
        scores = torch.bmm(output, memory.transpose(1, 2))            # (B, 1, T)
        scores = scores.masked_fill(frame_pad[:, None], float("-inf"))
        weights = scores.softmax(-1)                                  # (B, 1, T)
        context = torch.bmm(weights, memory)                          # (B, 1, Hd)
        combined = torch.tanh(self.attn_combine(torch.cat([output, context], dim=-1)))
        return self.out(combined.squeeze(1)), hidden, weights.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, encoder: dict, decoder: dict, pad_id: int):
        """
        Args:
            in_dim: features per pose frame.
            vocab_size: English tokens, including <pad>, <bos>, <eos> and <unk>.
            encoder: the `encoders.pose_gru` config block (hidden, layers, bidirectional, dropout).
            decoder: the `decoders.rnn` config block (emb_dim, hidden_dim, layers, dropout,
                teacher_forcing).
            pad_id: token id used to pad target sequences.
        """
        super().__init__()
        hidden = decoder["hidden_dim"]
        self.pad_id = pad_id
        self.teacher_forcing = decoder["teacher_forcing"]
        self.encoder = PoseGRUEncoder(in_dim, hidden, encoder["hidden"], encoder["layers"],
                                      encoder["bidirectional"], encoder["dropout"])
        self.decoder = Decoder(vocab_size, decoder["emb_dim"], hidden, decoder["layers"],
                               decoder["dropout"], pad_id)

    def start(self, frames: torch.Tensor,
              frame_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, and return the memory (B, T, Hd) plus a zero decoder state. Source
        information reaches the decoder only through attention."""
        memory = self.encoder(frames, frame_pad)                      # (B, T, Hd)
        rnn = self.decoder.rnn
        hidden = memory.new_zeros(rnn.num_layers, frames.size(0), rnn.hidden_size)
        return memory, hidden

    def forward(self, frames: torch.Tensor, frame_pad: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, F); frame_pad: (B, T); tokens: (B, L) = BOS w1 ... wn EOS, padded.
        Returns (B, L-1, V): the prediction for tokens[:, 1:]. While training, each step feeds
        the true previous token with probability `teacher_forcing`, else the model's own guess."""
        memory, hidden = self.start(frames, frame_pad)
        token, logits = tokens[:, 0], []
        for t in range(1, tokens.size(1)):
            step, hidden, _ = self.decoder(token, hidden, memory, frame_pad)
            logits.append(step)
            teacher = self.training and random.random() < self.teacher_forcing
            token = tokens[:, t] if teacher else step.argmax(-1)
        return torch.stack(logits, dim=1)                             # (B, L-1, V)

    @torch.no_grad()
    def generate(self, frames: torch.Tensor, frame_pad: torch.Tensor, bos: int, eos: int,
                 max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding. Returns ids (B, S) without BOS and their log-probabilities (B, S),
        S <= max_len - 1."""
        memory, hidden = self.start(frames, frame_pad)
        token = torch.full((frames.size(0),), bos, dtype=torch.long, device=frames.device)
        finished = torch.zeros_like(token, dtype=torch.bool)          # (B,)
        ids, logps = [], []
        for _ in range(max_len - 1):
            step, hidden, _ = self.decoder(token, hidden, memory, frame_pad)
            best, token = step.log_softmax(-1).max(-1)                # (B,), (B,)
            token = token.masked_fill(finished, self.pad_id)          # stop finished rows
            ids.append(token)
            logps.append(best.masked_fill(finished, 0.0))
            finished |= token == eos
            if finished.all():
                break
        return torch.stack(ids, dim=1), torch.stack(logps, dim=1)     # (B, S), (B, S)
