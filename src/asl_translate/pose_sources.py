"""Adapters for OpenPose BODY_25+hands and ViTPose COCO-WholeBody.

Both adapters return normalized image coordinates and confidence in native order.
Facial-detail landmarks are omitted, consistent with the published dataset.
"""
import json
import os
import subprocess
import tempfile
from functools import lru_cache
from importlib.metadata import distribution, version
from pathlib import Path

import cv2
import numpy as np

from asl_translate.config import get_device
from asl_translate.dataset import sha256
from asl_translate.features import interpolate

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
        from asl_translate.rtm_pose import model_paths

        detector, model = model_paths(source)
        files = {"pose_weights": model, "detector_weights": detector}
        runtime = {"backend": "rtmlib YOLOX-tiny CPU + wholebody CoreML",
                   "rtmlib": version("rtmlib"), "onnxruntime": version("onnxruntime"),
                   "detector_device": "cpu", "pose_device": "coreml"}
    else:
        raise ValueError(source)
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {source} model files: {missing}")
    code_hashes = {"pose_sources": sha256(Path(__file__)),
                   "features": sha256(Path(__file__).with_name("features.py"))}
    if source in ("rtmw", "dwpose"):
        code_hashes["rtm_pose"] = sha256(Path(__file__).with_name("rtm_pose.py"))
    return {"source": source, "fps": data["fps"], "source_revision": data["source_hub"]["revision"],
            "runtime": runtime, "sha256": {key: sha256(path) for key, path in files.items()},
            "code_sha256": code_hashes,
            "confidence_threshold": 0.05}


def check_metadata(data: dict, source: str) -> Path:
    root = data["processed"] / folder(source)
    path = root / "metadata.json"
    metadata = extraction_metadata(data, source)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != metadata:
            from asl_translate.pose_compat import compatible_vitpose_metadata

            if compatible_vitpose_metadata(previous, metadata):
                return path
            raise ValueError(f"{path} does not match the selected model/runtime; use a new data root")
    else:
        if root.exists() and any(root.rglob("*.npz")):
            raise ValueError(f"Existing {source} features have no metadata: {root}")
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def folder(source: str, smoke: bool = False) -> str:
    if source not in SOURCES:
        raise ValueError(f"Unknown feature source: {source}")
    kind = "smoke" if smoke else "landmarks"
    return kind if source == "mediapipe" else f"{kind}/{source}"


def prepare(raw: np.ndarray, threshold: float = 0.05) -> dict[str, np.ndarray]:
    """Interpolate missing coordinates, keeping confidence and per-point detection masks."""
    scores = np.nan_to_num(raw[..., 2], nan=0.0).clip(0, 1).astype(np.float16)
    detected = (scores >= threshold) & np.isfinite(raw[..., :2]).all(axis=-1)
    points = np.where(detected[..., None], raw[..., :2], np.nan)
    return {"points": interpolate(points).astype(np.float16), "detected": detected,
            "confidence": scores}


def openpose(frames, executable: str, model_dir: str, work_dir: Path | None = None) -> np.ndarray:
    """Run the official OpenPose executable on one bounded sentence's sampled frames."""
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


@lru_cache(maxsize=1)
def _vit_model(model: str, detector: str, device: str):
    from easy_ViTPose import VitInference

    return VitInference(model=model, yolo=detector, model_name="s", dataset="wholebody",
                        device=device, is_video=True, single_pose=True)


def vitpose(frames, model: str, detector: str, device: str) -> np.ndarray:
    """ViTPose whole-body points: 23 body/foot followed by 42 hands; omit 68 face points."""
    estimator = _vit_model(model, detector, device)
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


def extract(source: str, frames, work_dir: Path | None = None) -> np.ndarray:
    if source == "openpose":
        exe = os.environ.get("ASL_OPENPOSE_BIN")
        models = os.environ.get("ASL_OPENPOSE_MODELS")
        if not exe or not models:
            raise RuntimeError("Set ASL_OPENPOSE_BIN and ASL_OPENPOSE_MODELS")
        return openpose(frames, exe, models, work_dir)
    if source == "vitpose":
        model = os.environ.get("ASL_VITPOSE_MODEL")
        detector = os.environ.get("ASL_VITPOSE_DETECTOR")
        if not model or not detector:
            raise RuntimeError("Set ASL_VITPOSE_MODEL and ASL_VITPOSE_DETECTOR")
        device = str(get_device(os.environ.get("ASL_VITPOSE_DEVICE", "auto")))
        return vitpose(frames, model, detector, device)
    if source in ("rtmw", "dwpose"):
        from asl_translate.rtm_pose import extract_wholebody

        return extract_wholebody(source, frames)
    raise ValueError(f"Unknown feature source: {source}")
