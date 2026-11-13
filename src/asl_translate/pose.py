"""Pose estimators and extraction metadata for all supported sources."""
import json
import os
import subprocess
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from functools import lru_cache
from importlib.metadata import distribution, version
from pathlib import Path

import cv2
import numpy as np

from asl_translate.config import REPO_ROOT, get_device
from asl_translate.data import sha256

SOURCES = ("mediapipe", "openpose", "vitpose", "rtmw", "dwpose")
SHOULDERS = {"mediapipe": (1, 2), "openpose": (5, 2), "vitpose": (5, 6),
             "rtmw": (5, 6), "dwpose": (5, 6)}


def extraction_metadata(data: dict, source: str) -> dict:
    """Fingerprint all model weights before resuming a source's extraction."""
    if source == "openpose":
        exe = os.environ.get("ASL_OPENPOSE_BIN")
        models = os.environ.get("ASL_OPENPOSE_MODELS")
        if not exe or not models:
            raise ValueError("Set ASL_OPENPOSE_BIN and ASL_OPENPOSE_MODELS")
        files = {"binary": Path(exe),
                 "body_weights": Path(models) / "pose/body_25/pose_iter_584000.caffemodel",
                 "body_config": Path(models) / "pose/body_25/pose_deploy.prototxt",
                 "hand_weights": Path(models) / "hand/pose_iter_102000.caffemodel",
                 "hand_config": Path(models) / "hand/pose_deploy.prototxt"}
        runtime = {"backend": "official OpenPose BODY_25 + hands"}
    elif source == "vitpose":
        model = os.environ.get("ASL_VITPOSE_MODEL")
        detector = os.environ.get("ASL_VITPOSE_DETECTOR")
        if not model or not detector:
            raise ValueError("Set ASL_VITPOSE_MODEL and ASL_VITPOSE_DETECTOR")
        files = {"pose_weights": Path(model), "detector_weights": Path(detector)}
        runtime = {"backend": "easy_ViTPose wholebody small + YOLOv8",
                   "easy_vitpose": version("easy-vitpose"), "torch": version("torch"),
                   "ultralytics": version("ultralytics"),
                   "device": str(get_device(os.environ.get("ASL_VITPOSE_DEVICE", "auto")))}
        direct_url = distribution("easy-vitpose").read_text("direct_url.json")
        if direct_url:
            runtime["easy_vitpose_source"] = json.loads(direct_url)
    elif source in ("rtmw", "dwpose"):
        detector, model = RTMEstimator.model_paths(source)
        files = {"pose_weights": model, "detector_weights": detector}
        runtime = {"backend": "rtmlib YOLOX-tiny CPU + wholebody CoreML",
                   "rtmlib": version("rtmlib"), "onnxruntime": version("onnxruntime"),
                   "detector_device": "cpu", "pose_device": "coreml"}
    else:
        raise ValueError(source)
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {source} model files: {missing}")
    code_hashes = {"pose": sha256(Path(__file__)),
                   "video": sha256(Path(__file__).with_name("video.py")),
                   "features": sha256(Path(__file__).with_name("features.py"))}
    return {"source": source, "fps": data["fps"], "source_revision": data["source_hub"]["revision"],
            "runtime": runtime, "sha256": {key: sha256(path) for key, path in files.items()},
            "code_sha256": code_hashes,
            "confidence_threshold": 0.05}


def check_metadata(data: dict, source: str) -> Path:
    """Create extraction metadata or reject a run that differs from existing metadata."""
    root = data["processed"] / folder(source)
    path = root / "metadata.json"
    metadata = extraction_metadata(data, source)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != metadata:
            raise ValueError(f"{path} does not match the selected model/runtime; use a new data root")
    else:
        if root.exists() and any(root.rglob("*.npz")):
            raise ValueError(f"Existing {source} features have no metadata: {root}")
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def folder(source: str, smoke: bool = False) -> str:
    """Return the processed feature folder for a source and optional smoke subset."""
    if source not in SOURCES:
        raise ValueError(f"Unknown feature source: {source}")
    kind = "smoke" if smoke else "landmarks"
    return kind if source == "mediapipe" else f"{kind}/{source}"


def interpolate(points: np.ndarray) -> np.ndarray:
    """Fill NaN frames linearly in time, per coordinate; edges take the nearest value.

    A part that is never detected in the sentence becomes zeros.
    """
    out = points.astype(np.float32).copy()
    flat = out.reshape(len(out), -1)
    t = np.arange(len(out))
    for column in range(flat.shape[1]):
        values = flat[:, column]
        seen = ~np.isnan(values)
        if seen.all():
            continue
        flat[:, column] = np.interp(t, t[seen], values[seen]) if seen.any() else 0.0
    return out


def prepare(raw: np.ndarray, threshold: float = 0.05) -> dict[str, np.ndarray]:
    """Interpolate missing coordinates, keeping confidence and per-point detection masks."""
    scores = np.nan_to_num(raw[..., 2], nan=0.0).clip(0, 1).astype(np.float16)
    detected = (scores >= threshold) & np.isfinite(raw[..., :2]).all(axis=-1)
    points = np.where(detected[..., None], raw[..., :2], np.nan)
    return {"points": interpolate(points).astype(np.float16), "detected": detected,
            "confidence": scores}


# Versioned URL so the checkpoint stays pinned (Experiment Card field 12).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
MODEL_PATH = REPO_ROOT / "models" / "holistic_landmarker.task"

# Each frame becomes a (553, 3) array of normalized x, y, z. Undetected parts are NaN.
PARTS = {
    "pose": slice(0, 33),
    "face": slice(33, 511),
    "left_hand": slice(511, 532),
    "right_hand": slice(532, 553),
}
NUM_LANDMARKS = 553


class PoseEstimator(ABC):
    """Extract one sentence's landmarks, preserving each estimator's native layout."""

    @abstractmethod
    def extract(self, frames, work_dir: Path | None = None) -> np.ndarray:
        """Extract landmarks for timestamped BGR frames."""


class MediaPipeEstimator(PoseEstimator):
    """Extract MediaPipe Holistic pose, face, and hand landmarks."""

    @staticmethod
    def _make_landmarker():
        from mediapipe.tasks.python import BaseOptions, vision

        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
        )
        return vision.HolisticLandmarker.create_from_options(options)

    @staticmethod
    def to_array(result) -> np.ndarray:
        """Convert a MediaPipe result to a 553-by-3 array, using NaN for missing parts."""
        frame = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
        for name, part in PARTS.items():
            points = getattr(result, f"{name}_landmarks")
            if points:
                frame[part] = [(p.x, p.y, p.z) for p in points]
        return frame

    def extract(self, frames, work_dir: Path | None = None) -> np.ndarray:
        """Extract full MediaPipe landmarks from timestamped BGR frames."""
        import mediapipe as mp

        output = []
        with self._make_landmarker() as landmarker:
            for timestamp, frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                output.append(self.to_array(landmarker.detect_for_video(image, timestamp)))
        return np.stack(output) if output else np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32)


class OpenPoseEstimator(PoseEstimator):
    """Extract BODY_25 and hand landmarks using the OpenPose executable."""
    def extract(self, frames, work_dir: Path | None = None) -> np.ndarray:
        """Run OpenPose on one bounded sentence's sampled frames."""
        executable = os.environ.get("ASL_OPENPOSE_BIN")
        model_dir = os.environ.get("ASL_OPENPOSE_MODELS")
        if not executable or not model_dir:
            raise RuntimeError("Set ASL_OPENPOSE_BIN and ASL_OPENPOSE_MODELS")
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="asl-openpose-", dir=work_dir) as tmp:
            root = Path(tmp)
            images, output = root / "frames", root / "json"
            images.mkdir()
            count = 0
            for _, frame in frames:
                if not cv2.imwrite(str(images / f"{count:08d}.png"), frame):
                    raise OSError("Could not write temporary frame for OpenPose")
                count += 1
            if not count:
                return np.empty((0, 67, 3), dtype=np.float32)
            subprocess.run([executable, "--image_dir", str(images), "--write_json", str(output),
                            "--model_folder", str(Path(model_dir).resolve()) + os.sep,
                            "--model_pose", "BODY_25", "--hand",
                            "--number_people_max", "1", "--keypoint_scale", "3", "--display", "0",
                            "--render_pose", "0"], check=True)
            rows = []
            for i in range(count):
                path = output / f"{i:08d}_keypoints.json"
                if not path.is_file():
                    raise RuntimeError(f"OpenPose did not produce {path.name}")
                people = json.loads(path.read_text())["people"]
                if not people:
                    rows.append(np.full((67, 3), np.nan, dtype=np.float32))
                    continue
                person = people[0]
                parts = [np.asarray(person[key], dtype=np.float32).reshape(-1, 3) for key in
                         ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d")]
                if [len(p) for p in parts] != [25, 21, 21]:
                    raise ValueError(f"Unexpected OpenPose keypoint lengths: {[len(p) for p in parts]}")
                rows.append(np.concatenate(parts))
            return np.stack(rows)


class ViTPoseEstimator(PoseEstimator):
    """Extract whole-body and hand landmarks using ViTPose and a person detector."""

    @staticmethod
    @lru_cache(maxsize=1)
    def _model(model: str, detector: str, device: str):
        from easy_ViTPose import VitInference

        return VitInference(model=model, yolo=detector, model_name="s", dataset="wholebody",
                            device=device, is_video=True, single_pose=True)

    def extract(self, frames, work_dir: Path | None = None) -> np.ndarray:
        """Extract 23 body/foot and 42 hand points, omitting detailed face points."""
        model = os.environ.get("ASL_VITPOSE_MODEL")
        detector = os.environ.get("ASL_VITPOSE_DETECTOR")
        if not model or not detector:
            raise RuntimeError("Set ASL_VITPOSE_MODEL and ASL_VITPOSE_DETECTOR")
        device = str(get_device(os.environ.get("ASL_VITPOSE_DEVICE", "auto")))
        estimator = self._model(model, detector, device)
        estimator.reset()
        result = []
        for _, frame in frames:
            h, w = frame.shape[:2]
            poses = estimator.inference(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not poses:
                result.append(np.full((65, 3), np.nan, dtype=np.float32))
                continue
            # easy_ViTPose returns y, x, score. Choose the highest detector confidence.
            person = max(poses, key=lambda key: estimator._scores_bbox[key])
            raw = np.asarray(poses[person], dtype=np.float32)
            if raw.shape != (133, 3):
                raise ValueError(f"Expected 133 whole-body keypoints, got {raw.shape}")
            raw = raw[np.r_[0:23, 91:133]][:, [1, 0, 2]]
            raw[:, :2] /= [w, h]
            result.append(raw)
        return np.stack(result) if result else np.empty((0, 65, 3), dtype=np.float32)


class RTMEstimator(PoseEstimator):
    """Extract RTMW or DWPose landmarks using a CPU detector and Core ML pose model."""
    def __init__(self, source: str):
        if source not in ("rtmw", "dwpose"):
            raise ValueError(f"Unknown RTM estimator: {source}")
        self.source = source

    @staticmethod
    def model_paths(source: str) -> tuple[Path, Path]:
        """Return the configured detector and pose model paths."""
        model_dir = Path(os.environ.get("ASL_RTM_MODEL_DIR", REPO_ROOT / "models"))
        return model_dir / "yolox-tiny.onnx", model_dir / f"{source}.onnx"

    @staticmethod
    @lru_cache(maxsize=2)
    def _model(source: str):
        import onnxruntime as ort
        from rtmlib import YOLOX, RTMPose

        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Core ML provider unavailable; install a Core ML-enabled onnxruntime")
        det, pose = RTMEstimator.model_paths(source)
        detector = YOLOX(str(det), model_input_size=(416, 416), backend="onnxruntime", device="cpu")
        estimator = RTMPose(str(pose), model_input_size=(192, 256), backend="onnxruntime", device="mps")
        if "CoreMLExecutionProvider" not in estimator.session.get_providers():
            raise RuntimeError("Pose model did not initialize the Core ML provider")
        return detector, estimator

    def extract(self, frames, work_dir: Path | None = None) -> np.ndarray:
        """Return 65 normalized body, foot, and hand points with confidence per frame."""
        detector, estimator = self._model(self.source)
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


def get_pose_estimator(source: str) -> PoseEstimator:
    """Create the estimator corresponding to a feature source."""
    if source == "mediapipe":
        return MediaPipeEstimator()
    if source == "openpose":
        return OpenPoseEstimator()
    if source == "vitpose":
        return ViTPoseEstimator()
    if source in ("rtmw", "dwpose"):
        return RTMEstimator(source)
    raise ValueError(f"Unknown feature source: {source}")

