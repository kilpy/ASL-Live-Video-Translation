"""Frame encoders: (B, T, F) features + (B, T) padding mask -> (B, T, D) memory.

Every encoder takes (in_dim, d_model, **its config block) so `configs/config.yaml` picks one by
name. D is the decoder's width; the encoder's last layer is the fusion into it.
To add one: write the class, register it in ENCODERS, add an `encoders:` entry in the config.
"""
import torch
from torch import nn


class PoseGRUEncoder(nn.Module):
    """Per-frame features -> bidirectional GRU -> one d_model vector per frame."""

    def __init__(self, in_dim: int, d_model: int, hidden: int, layers: int,
                 bidirectional: bool, dropout: float, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(d_model, hidden, layers, batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), d_model)

    def forward(self, frames: torch.Tensor, frame_pad: torch.Tensor) -> torch.Tensor:
        # Padded frames are zeros; the decoder masks them out of attention.
        x = self.proj(frames)       # (B, T, D)
        hidden, _ = self.gru(x)     # (B, T, 2H): forward and backward states concatenated
        return self.out(hidden)     # (B, T, D)


class SHuBERTEncoder(nn.Module):
    """SHuBERT (Gueuwou et al., ACL 2025): four per-frame streams -> HuBERT-style transformer.

    Input layout per frame (train.frame_features with DINOv2 crops):
    [face, left hand, right hand DINOv2 CLS vectors (3 x dino_dim) | pose stream (the rest)].
    Each stream is projected to width/4 and concatenated, as in the paper; a convolutional
    positional embedding (HuBERT) precedes the transformer, and `out` fuses into d_model.
    """

    def __init__(self, in_dim: int, d_model: int, dino_dim: int = 384, width: int = 768,
                 layers: int = 12, heads: int = 12, ff: int = 3072, dropout: float = 0.1, **_):
        super().__init__()
        self.dino_dim = dino_dim
        pose_dim = in_dim - 3 * dino_dim
        if pose_dim <= 0:
            raise ValueError(f"SHuBERT needs DINOv2 crop features (in_dim {in_dim}); "
                             "train with --pose_estimator dinov2")
        self.streams = nn.ModuleList([nn.Linear(dino_dim, width // 4) for _ in range(3)]
                                     + [nn.Linear(pose_dim, width - 3 * (width // 4))])
        self.norm = nn.LayerNorm(width)
        # ponytail: kernel 31 at 15 fps (~2 s); HuBERT's 128 is for 50 Hz audio frames
        self.position = nn.Sequential(
            nn.Conv1d(width, width, 31, padding=15, groups=16), nn.GELU())
        layer = nn.TransformerEncoderLayer(width, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, d_model)  # the fusion layer into the decoder

    def forward(self, frames: torch.Tensor, frame_pad: torch.Tensor) -> torch.Tensor:
        crops = frames[..., : 3 * self.dino_dim].unflatten(-1, (3, self.dino_dim))  # (B, T, 3, C)
        parts = [crops[..., i, :] for i in range(3)] + [frames[..., 3 * self.dino_dim:]]
        x = self.norm(torch.cat([proj(p) for proj, p in zip(self.streams, parts)], -1))  # (B,T,W)
        x = x.masked_fill(frame_pad[..., None], 0.0)
        x = x + self.position(x.transpose(1, 2)).transpose(1, 2)
        return self.out(self.transformer(x, src_key_padding_mask=frame_pad))      # (B, T, D)


ENCODERS = {"pose_gru": PoseGRUEncoder, "shubert": SHuBERTEncoder}
