"""Text decoders. Every decoder takes (vocab_size, max_tokens, pad_id, **its config block), sets
`d_model` (the width the encoder must produce) and implements:

    prepare(memory (B, T, D), frame_pad (B, T)) -> memory   # once per sentence
    forward(tokens (B, L), memory, frame_pad, attention=False)
        -> logits (B, L, V), or (logits, cross-attention (B, L, T)) when attention=True

To add one: write the class, register it in DECODERS, add a `decoders:` entry in the config.
"""
from contextlib import contextmanager

import torch
from torch import nn


class TransformerDecoder(nn.Module):
    """Learned positions + nn.TransformerDecoder over word ids."""

    def __init__(self, vocab_size: int, max_tokens: int, pad_id: int, d_model: int, layers: int,
                 heads: int, ff: int, dropout: float, **_):
        super().__init__()
        self.d_model, self.pad_id = d_model, pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # id -> (D,)
        self.position = nn.Embedding(max_tokens, d_model)                   # position -> (D,)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.stack = nn.TransformerDecoder(layer, layers)
        self.head = nn.Linear(d_model, vocab_size)                          # (D,) -> (V,)

    def prepare(self, memory, frame_pad):
        return memory

    def forward(self, tokens, memory, frame_pad, attention: bool = False):
        length = tokens.size(1)
        x = self.embed(tokens) + self.position(torch.arange(length, device=tokens.device))
        # (L, L) True above the diagonal: position i cannot see positions after i
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                       device=tokens.device), diagonal=1)
        with self._cross_attention(attention) as maps:
            out = self.stack(x, memory, tgt_mask=causal,
                             tgt_key_padding_mask=tokens == self.pad_id,
                             memory_key_padding_mask=frame_pad)
        logits = self.head(out)
        return (logits, torch.stack(maps).mean(0)) if attention else logits

    @contextmanager
    def _cross_attention(self, enabled: bool):
        """Collect each layer's cross-attention, (B, L, T) averaged over heads."""
        maps, handles = [], []
        if enabled:
            def want_weights(module, args, kwargs):
                return args, {**kwargs, "need_weights": True, "average_attn_weights": True}
            for layer in self.stack.layers:
                handles.append(layer.multihead_attn.register_forward_pre_hook(
                    want_weights, with_kwargs=True))
                handles.append(layer.multihead_attn.register_forward_hook(
                    lambda module, args, output: maps.append(output[1])))
        try:
            yield maps
        finally:
            for handle in handles:
                handle.remove()


class RNNDecoder(nn.Module):
    """GRU/LSTM over word ids with Luong (dot-product) attention over the encoder memory.

    The recurrent state starts from the mean of the memory; attention is applied to the RNN
    outputs, so teacher forcing stays one parallel pass.
    """

    def __init__(self, vocab_size: int, max_tokens: int, pad_id: int, d_model: int = 256,
                 hidden: int = 256, layers: int = 2, cell: str = "gru", attention: bool = True,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.d_model, self.layers, self.use_attention = d_model, layers, attention
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.init = nn.Linear(d_model, hidden * layers)
        self.rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell](
            d_model, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.query = nn.Linear(hidden, d_model)
        self.combine = nn.Sequential(nn.Linear(hidden + d_model, hidden), nn.Tanh(),
                                     nn.Dropout(dropout))
        self.head = nn.Linear(hidden, vocab_size)

    def prepare(self, memory, frame_pad):
        return memory

    def forward(self, tokens, memory, frame_pad, attention: bool = False):
        keep = (~frame_pad)[..., None].float()                                   # (B, T, 1)
        mean = (memory * keep).sum(1) / keep.sum(1).clamp(min=1)                 # (B, D)
        h0 = torch.tanh(self.init(mean)).unflatten(-1, (self.layers, -1)).transpose(0, 1)
        state = (h0.contiguous(), torch.zeros_like(h0)) if isinstance(self.rnn, nn.LSTM) \
            else h0.contiguous()
        out, _ = self.rnn(self.embed(tokens), state)                             # (B, L, H)
        weights = torch.zeros(*out.shape[:2], memory.size(1), device=out.device)
        context = torch.zeros(*out.shape[:2], self.d_model, device=out.device)
        if self.use_attention:
            scores = self.query(out) @ memory.transpose(1, 2)                    # (B, L, T)
            weights = scores.masked_fill(frame_pad[:, None], -torch.inf).softmax(-1)
            context = weights @ memory                                           # (B, L, D)
        logits = self.head(self.combine(torch.cat([out, context], -1)))
        return (logits, weights) if attention else logits


class ByT5Decoder(nn.Module):
    """Pretrained ByT5: its encoder reads the fused frame memory, its decoder writes bytes.

    Uses our character vocabulary ([PAD, BOS, EOS, UNK] + 256 byte values, see train.py), which
    maps onto ByT5's ids (0 pad = decoder start, 1 eos, 2 unk, byte b -> b + 3) by shifting one.
    """

    # our id -> ByT5 id, for inputs (clamped shift) and output columns (gathered)
    BYT5_IDS = [0, 0, 1, 2] + [b + 3 for b in range(256)]

    def __init__(self, vocab_size: int, max_tokens: int, pad_id: int,
                 checkpoint: str = "google/byt5-small", pretrained: bool = True,
                 frozen: bool = False, config: dict | None = None, **_):
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration

        if vocab_size != len(self.BYT5_IDS):
            raise ValueError("ByT5 needs `units: chars` on its decoder config")
        self.t5 = (T5ForConditionalGeneration.from_pretrained(checkpoint) if pretrained else
                   T5ForConditionalGeneration(T5Config(vocab_size=384, decoder_start_token_id=0,
                                                       **(config or {}))))
        self.t5.requires_grad_(not frozen)
        self.d_model = self.t5.config.d_model
        self.register_buffer("ids", torch.tensor(self.BYT5_IDS), persistent=False)

    def prepare(self, memory, frame_pad):
        return self.t5.encoder(inputs_embeds=memory,
                               attention_mask=(~frame_pad).long()).last_hidden_state

    def forward(self, tokens, memory, frame_pad, attention: bool = False):
        if attention:  # sdpa can't return weights
            self.t5.set_attn_implementation("eager")
        try:
            out = self.t5(encoder_outputs=(memory,), attention_mask=(~frame_pad).long(),
                          decoder_input_ids=(tokens - 1).clamp(min=0),
                          output_attentions=attention)
        finally:
            if attention:
                self.t5.set_attn_implementation("sdpa")
        logits = out.logits[..., self.ids]                                       # (B, L, V)
        if not attention:
            return logits
        return logits, torch.stack(out.cross_attentions).mean((0, 2))           # (B, L, T)


DECODERS = {"transformer": TransformerDecoder, "rnn": RNNDecoder, "byt5": ByT5Decoder}
