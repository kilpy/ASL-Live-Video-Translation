import numpy as np
import pytest
import torch

from asl_translate.models.decoders import ByT5Decoder
from asl_translate.models.seq2seq import Seq2Seq
from asl_translate.models.train import frame_features

POSE = 49 * 2 + 3
ENCODERS = {
    "pose_gru": {"hidden": 8, "layers": 1, "bidirectional": True, "dropout": 0.0},
    "shubert": {"dino_dim": 4, "width": 16, "layers": 1, "heads": 2, "ff": 32, "dropout": 0.0},
}
DECODERS = {
    "transformer": ({"d_model": 16, "layers": 1, "heads": 2, "ff": 32, "dropout": 0.0}, 10),
    "rnn": ({"d_model": 16, "hidden": 8, "layers": 2, "dropout": 0.0}, 10),
    "byt5": ({"pretrained": False, "config": {"d_model": 16, "d_kv": 4, "d_ff": 32,
                                              "num_layers": 1, "num_heads": 2}},
             len(ByT5Decoder.BYT5_IDS)),
}


@pytest.mark.parametrize("decoder", DECODERS)
@pytest.mark.parametrize("encoder", ENCODERS)
def test_every_encoder_decoder_pair_trains_and_generates(encoder, decoder):
    if decoder == "byt5":
        pytest.importorskip("transformers")
    torch.manual_seed(0)
    dec_cfg, vocab = DECODERS[decoder]
    in_dim = POSE + 3 * 4
    model = Seq2Seq(in_dim, vocab, {"name": encoder, **ENCODERS[encoder]},
                    {"name": decoder, **dec_cfg}, max_tokens=8, pad_id=0)
    frames = torch.randn(2, 12, in_dim)
    frame_pad = torch.arange(12)[None] >= torch.tensor([[12], [7]])  # second row: 5 pad frames
    tokens = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
    logits = model(frames, frame_pad, tokens)
    assert logits.shape == (2, 3, vocab)
    logits.logsumexp(-1).sum().backward()  # gradients reach the encoder
    assert any(p.grad is not None for p in model.encoder.parameters())

    model.eval()
    ids, logps = model.generate(frames, frame_pad, bos=1, eos=2, max_len=6)
    assert ids.shape == logps.shape and ids.size(1) <= 5
    _, attention = model.decoder(tokens, model.encode(frames, frame_pad), frame_pad,
                                 attention=True)
    assert attention.shape == (2, 4, 12)
    assert attention[1, :, 7:].abs().max() < 1e-6  # padded frames get no attention


def test_shubert_rejects_landmark_only_features():
    with pytest.raises(ValueError, match="dinov2"):
        Seq2Seq(POSE, 10, {"name": "shubert", "dino_dim": 384},
                {"name": "transformer", **DECODERS["transformer"][0]}, 8, 0)


def test_frame_features_puts_crops_first():
    points, detected = np.ones((5, 49, 2), np.float16), np.ones((5, 3), bool)
    crops = np.full((5, 3, 4), 7, np.float16)
    features = frame_features(points, detected, 10, crops=crops)
    assert features.shape == (5, 12 + POSE) and (features[:, :12] == 7).all()
