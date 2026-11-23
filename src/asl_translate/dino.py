"""DINOv2 crop features, SHuBERT's per-frame input: face, left hand and right hand vectors.

Crops come from MediaPipe Holistic landmarks; each crop becomes DINOv2-small's 384-d CLS
vector. Needs the `deep` dependency group (transformers).
"""
from functools import lru_cache
from itertools import batched

import cv2
import numpy as np
import torch

from asl_translate.config import get_device
from asl_translate.pose import PARTS

MODEL = "facebook/dinov2-small"
CROP_PARTS = ("face", "left_hand", "right_hand")
CROP_SIZE = 224  # 16 x 16 patches of 14 px
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet, as DINOv2 was trained


def crop(image: np.ndarray, points: np.ndarray, size: int = 112) -> np.ndarray:
    """Square crop around landmarks; black if the part wasn't detected."""
    if np.isnan(points).any():
        return np.zeros((size, size, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    xs, ys = points[:, 0] * width, points[:, 1] * height
    half = max(np.ptp(xs), np.ptp(ys)) * 0.7 + 1  # 0.7: 40% padding around the box
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, width))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, height))
    if x0 >= x1 or y0 >= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(image[y0:y1, x0:x1], (size, size))


@lru_cache(maxsize=1)
def _model(device: str):
    from transformers import AutoModel

    return AutoModel.from_pretrained(MODEL).eval().to(device)


@torch.no_grad()
def embed(frames, holistic: np.ndarray, device: str = "auto", batch: int = 32) -> np.ndarray:
    """(timestamp, BGR frame) pairs and their (T, 553, 3) landmarks -> (T, 3, 384) float16.

    A part that wasn't detected gets a zero vector, SHuBERT's convention for missing hands.
    """
    device = get_device(device)
    model = _model(str(device))
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    rows = []
    for chunk in batched(zip(frames, holistic), batch):
        crops = np.stack([crop(frame, marks[PARTS[p], :2], CROP_SIZE)
                          for (_, frame), marks in chunk for p in CROP_PARTS])   # (3N, S, S, 3)
        x = torch.from_numpy(crops[..., ::-1].copy()).to(device).permute(0, 3, 1, 2) / 255
        cls = model(pixel_values=(x - mean) / std).pooler_output                  # (3N, 384)
        rows.append(cls.view(len(chunk), 3, -1).cpu().numpy())
    if not rows:
        return np.zeros((0, 3, model.config.hidden_size), dtype=np.float16)
    vectors = np.concatenate(rows)
    missing = np.stack([np.isnan(holistic[:, PARTS[p], :2]).any(axis=(1, 2))
                        for p in CROP_PARTS], 1)
    vectors[missing] = 0.0
    return vectors.astype(np.float16)
