import numpy as np
import torch

from asl_translate.demo import PANEL_WIDTH, Session, draw_text
from asl_translate.models.infer import Translator
from asl_translate.models.seq2seq import Seq2Seq
from asl_translate.models.train import BOS, EOS, PAD, UNK, frame_features

FEATURES = {"pose_points": [0, 11, 12, 13, 14, 15, 16], "interpolate_missing": True}
ENCODER = {"hidden": 8, "layers": 1, "bidirectional": True, "dropout": 0.0}
DECODER = {"d_model": 16, "layers": 1, "heads": 2, "ff": 32, "dropout": 0.0}


def sentence(frames: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.random((frames, 49, 2)).astype(np.float16), np.ones((frames, 3), dtype=bool)


def test_frame_features_shape_and_subsampling():
    points, detected = sentence(40)
    assert frame_features(points, detected, max_frames=100).shape == (40, 49 * 2 + 3)
    assert frame_features(points, detected, max_frames=10).shape == (10, 49 * 2 + 3)


ITOS = [PAD, BOS, EOS, UNK, "hello", "world"]


def save_tiny(tmp_path):
    torch.manual_seed(0)
    model = Seq2Seq(49 * 2 + 3, len(ITOS), ENCODER, DECODER, max_tokens=8, pad_id=0)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "itos": ITOS, "in_dim": 49 * 2 + 3,
                "encoder": ENCODER, "decoder": DECODER, "max_frames": 20, "max_tokens": 8,
                "source": "mediapipe", "fps": 15, "features": FEATURES}, path)
    return path


def test_translator_round_trips_a_saved_checkpoint(tmp_path):
    itos = ITOS
    translator = Translator.load(save_tiny(tmp_path), device="cpu")
    text, confidence = translator.translate(*sentence(30))
    assert set(text.split()) <= set(itos[3:])
    assert 0.0 <= confidence <= 1.0
    assert translator.translate(np.empty((0, 49, 2)), np.empty((0, 3))) == ("", 0.0)


def test_explain_aligns_words_attention_and_frames(tmp_path):
    translator = Translator.load(save_tiny(tmp_path), device="cpu")
    points, detected = sentence(50)  # > max_frames, so the model sees a subsample
    expl = translator.explain(points, detected)
    n = len(expl.words)
    assert len(expl.probs) == len(expl.topk) == n
    assert expl.attention.shape == (n, 50) and (expl.attention >= 0).all()
    assert expl.model_input.shape == (50, 49, 2)
    for word, p, alts in zip(expl.words, expl.probs, expl.topk):
        assert alts[0] == (word, p) and [q for _, q in alts] == sorted([q for _, q in alts])[::-1]
    assert translator.translate(points, detected) == (" ".join(expl.words), expl.confidence)


def test_session_renders_live_review_and_missing_landmarks(tmp_path):
    session = Session(Translator.load(save_tiny(tmp_path), device="cpu"))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    holistic = np.full((553, 3), np.nan, dtype=np.float32)
    assert session.render(frame).shape[:2] == (240, 320 + PANEL_WIDTH)  # nothing detected yet
    session.holistic = holistic
    session.toggle()
    session.toggle()
    assert session.expl is None and "short" in session.message[0]
    session.toggle()
    session.buffer = [np.random.default_rng(i).random((553, 3), dtype=np.float32)
                      for i in range(20)]
    session.toggle()
    session.select(1)
    canvas = session.render(frame)
    assert canvas.shape[0] >= 240 and canvas.shape[1] == 320 + PANEL_WIDTH
    draw_text(frame, [("word " * 40, (0, 255, 0))])
    assert frame.any()
