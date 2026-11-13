"""BASELINE model: pose GRU encoder + Transformer decoder over word tokens.

Shapes: B batch, T frames, L tokens, F features per frame (49 points x 2 coords + 3 detection
flags = 101), D d_model (256), H GRU hidden per direction (128), V vocabulary incl. specials.
"""
import torch
from torch import nn
import torch.nn.functional as F
import random

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
        x = self.proj(frames)       # (B, T, D)
        hidden, _ = self.gru(x)     # (B, T, 2H): forward and backward states concatenated
        return self.out(hidden)     # (B, T, D): the "memory" the decoder attends to


class Encoder(nn.Module):
    """GRU encoder: reads the source sequence and returns per-step outputs and a final hidden state."""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            emb_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths=None):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)

        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=src.size(1)
            )
        else:
            outputs, hidden = self.rnn(embedded)

        # outputs: (batch, src_len, hidden_dim), hidden: (num_layers, batch, hidden_dim)
        return outputs, hidden


class Decoder(nn.Module):
    """GRU decoder with dot-product (Luong) attention; generates one token per call."""

    def __init__(self, vocab_size, hidden_dim, emb_dim=256,num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            emb_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, encoder_outputs, src_mask=None):
        # token: (batch,)  hidden: (num_layers, batch, hidden_dim)
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))  # (batch, 1, emb_dim) Creates a matrix and masks some values
        output, hidden = self.rnn(embedded, hidden)                  # (batch, 1, hidden_dim)

        # Creates attentions scores for values
        scores = torch.bmm(output, encoder_outputs.transpose(1, 2))  # (batch, 1, src_len)
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)           # (batch, 1, hidden_dim)

        combined = torch.tanh(self.attn_combine(torch.cat([output, context], dim=-1)))
        logits = self.out(combined.squeeze(1))                       # (batch, vocab_size)
        return logits, hidden, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    """Builds and owns its Encoder and Decoder so their hidden_dim and num_layers always match."""

    def __init__(self, in_dim: int, vocab_size: int, encoder_dim: dict, decoder_dim: dict, pad_id: int):
        """
        Args:
            in_dim: Number of features in each pose frame. Sets the input size
                of the encoder's first linear layer.
            vocab_size: Number of tokens in the English (target) vocabulary, including
                special tokens such as <pad>, <sos> and <eos>. 
            encoder_dim: Keyword arguments unpacked into Encoder. The keys must match the
                parameter names in Encoder.__init__.
            decoder_dim: Keyword arguments unpacked into Decoder. The keys must match the 
                parameter names in Decoder.__init__. Its hidden_dim must equal the encoder's output size 
            pad_id: Token ID used for padding in the target sequences. 
        """
        # in_dim: features per pose frame (101); vocab_size: # of English tokens
        # encoder / decoder: keyword arguments for each class (emb_dim, hidden_dim, num_layers, dropout, ...)
        super().__init__()
        self.encoder = Encoder(in_dim, **encoder_dim)
        self.decoder = Decoder(vocab_size, **decoder_dim, pad_idx=pad_id)
        self.pad_id = pad_id

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        # src: (batch, src_len, features), tgt: (batch, tgt_len) beginning with <sos>
        # src_lengths: (batch,) number of real frames per sample (now required)
        encoder_outputs, _ = self.encoder(src, src_lengths)  # final hidden state is discarded

        # mask built from lengths, True for real frames and False for padding (added to to match pose data)
        src_mask = (
            torch.arange(src.size(1), device=src.device)[None, :]
            < src_lengths.to(src.device)[:, None]
        )  # (batch, src_len)

        # zero-initialized decoder state, so source information arrives only via attention which assists with some of RNN's weaknesses
        hidden = torch.zeros(
            self.decoder.rnn.num_layers, src.size(0), self.decoder.rnn.hidden_size,
            device=src.device,
        )

        token = tgt[:, 0]
        all_logits = []
        for t in range(1, tgt.size(1)):
            logits, hidden, _ = self.decoder(token, hidden, encoder_outputs, src_mask)
            all_logits.append(logits)
            use_teacher = self.training and random.random() < teacher_forcing_ratio
            token = tgt[:, t] if use_teacher else logits.argmax(dim=-1)

        return torch.stack(all_logits, dim=1)