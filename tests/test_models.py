import pytest
import torch

from asl_translate.config import load_config
from asl_translate.models import baseline, classical
from asl_translate.models.train import build_model

B, T, F, L, V, PAD, BOS, EOS = 3, 12, 101, 7, 20, 0, 1, 2


def batch():
    lengths = torch.tensor([12, 8, 5])
    frame_pad = torch.arange(T)[None] >= lengths[:, None]
    frames = torch.randn(B, T, F).masked_fill(frame_pad[..., None], 0.0)
    tokens = torch.randint(3, V, (B, L))
    tokens[:, 0] = BOS
    return frames, frame_pad, tokens


@pytest.mark.parametrize("approach,cls", [("BASELINE", baseline.Seq2Seq),
                                          ("CLASSICAL", classical.Seq2Seq)])
def test_forward_and_generate_shapes(approach, cls):
    torch.manual_seed(0)
    model = build_model(load_config(approach), F, V, max_tokens=16, pad_id=PAD)
    assert isinstance(model, cls)
    frames, frame_pad, tokens = batch()
    assert model(frames, frame_pad, tokens).shape == (B, L - 1, V)
    model.eval()
    ids, logps = model.generate(frames, frame_pad, BOS, EOS, max_len=16)
    assert ids.shape == logps.shape and ids.shape[0] == B and ids.shape[1] <= 15
    assert torch.isfinite(logps).all()


def test_packed_encoder_ignores_padding():
    torch.manual_seed(0)
    encoder = baseline.PoseGRUEncoder(F, 32, 16, 2, True, 0.0).eval()
    frames = torch.randn(1, 5, F)
    padded = torch.cat([frames, torch.zeros(1, 4, F)], dim=1)
    alone = encoder(frames, torch.zeros(1, 5, dtype=torch.bool))
    with_pad = encoder(padded, torch.arange(9)[None] >= 5)[:, :5]
    assert torch.allclose(alone, with_pad, atol=1e-6)
