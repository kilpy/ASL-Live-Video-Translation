import numpy as np
import torch

from asl_translate.demo import draw_landmarks, draw_text
from asl_translate.models.baseline import Seq2Seq
from asl_translate.models.infer import Translator
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


def test_translator_round_trips_a_saved_checkpoint(tmp_path):
    itos = [PAD, BOS, EOS, UNK, "hello", "world"]
    model = Seq2Seq(49 * 2 + 3, len(itos), ENCODER, DECODER, max_tokens=8, pad_id=0)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "itos": itos, "in_dim": 49 * 2 + 3,
                "encoder": ENCODER, "decoder": DECODER, "max_frames": 20, "max_tokens": 8,
                "source": "mediapipe", "fps": 15, "features": FEATURES}, path)

    translator = Translator.load(path, device="cpu")
    text, confidence = translator.translate(*sentence(30))
    assert set(text.split()) <= set(itos[3:])
    assert 0.0 <= confidence <= 1.0
    assert translator.translate(np.empty((0, 49, 2)), np.empty((0, 3))) == ("", 0.0)


def test_overlay_handles_missing_landmarks_and_long_text():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    holistic = np.full((553, 3), np.nan, dtype=np.float32)
    holistic[0] = (0.5, 0.5, 0.0)
    draw_landmarks(frame, holistic, FEATURES["pose_points"])
    draw_text(frame, [("word " * 40, (0, 255, 0))])
    assert frame.any()
