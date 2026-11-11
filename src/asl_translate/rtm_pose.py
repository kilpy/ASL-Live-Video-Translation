"""RTMW and DWPose whole-body inference with CPU detection and Core ML pose."""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from asl_translate.config import REPO_ROOT


def model_paths(source: str) -> tuple[Path, Path]:
    model_dir = Path(os.environ.get("ASL_RTM_MODEL_DIR", REPO_ROOT / "models"))
    return model_dir / "yolox-tiny.onnx", model_dir / f"{source}.onnx"


@lru_cache(maxsize=2)
def _model(source: str):
    import onnxruntime as ort
    from rtmlib import YOLOX, RTMPose

    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("Core ML provider unavailable; install a Core ML-enabled onnxruntime")
    det, pose = model_paths(source)
    detector = YOLOX(str(det), model_input_size=(416, 416), backend="onnxruntime", device="cpu")
    estimator = RTMPose(str(pose), model_input_size=(192, 256), backend="onnxruntime", device="mps")
    if "CoreMLExecutionProvider" not in estimator.session.get_providers():
        raise RuntimeError("Pose model did not initialize the Core ML provider")
    return detector, estimator


def extract_wholebody(source: str, frames) -> np.ndarray:
    detector, estimator = _model(source)
    rows = []
    for _, frame in frames:
        h, w = frame.shape[:2]
        boxes = detector(frame)
        if len(boxes) == 0:
            rows.append(np.full((65, 3), np.nan, dtype=np.float32))
            continue
        # The largest detected person is the signer in the frontal recordings.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best = int(np.argmax(areas))
        points, scores = estimator(frame, bboxes=boxes[best:best + 1])
        if points.shape != (1, 133, 2) or scores.shape != (1, 133):
            raise ValueError(f"Unexpected whole-body shapes: {points.shape}, {scores.shape}")
        keep = np.r_[0:23, 91:133]
        coords = points[0, keep].astype(np.float32)
        coords /= [w, h]
        rows.append(np.column_stack((coords, scores[0, keep])).astype(np.float32))
    return np.stack(rows) if rows else np.empty((0, 65, 3), dtype=np.float32)
