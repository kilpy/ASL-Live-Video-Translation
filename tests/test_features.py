import json
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asl_translate import features
from asl_translate.data import record, verify
from asl_translate.features import LEFT_HAND, RIGHT_HAND, reduce
from asl_translate.pose import interpolate

FEATURES = {"pose_points": [0, 11, 12, 13, 14, 15, 16], "interpolate_missing": True}


def test_interpolate_fills_gaps_linearly_and_edges_with_nearest():
    points = np.array([np.nan, 1.0, np.nan, 3.0, np.nan]).reshape(5, 1, 1)
    assert interpolate(points).ravel().tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_interpolate_turns_a_never_seen_part_into_zeros():
    assert (interpolate(np.full((4, 2, 2), np.nan)) == 0).all()


def test_reduce_keeps_seven_pose_points_and_both_hands_in_xy():
    holistic = np.random.default_rng(0).random((6, 553, 3)).astype(np.float32)
    holistic[2, LEFT_HAND] = np.nan
    holistic[:, RIGHT_HAND] = np.nan
    out = reduce(holistic, FEATURES)
    assert out["points"].shape == (6, 7 + 21 + 21, 2)
    assert out["points"].dtype == np.float16
    assert not np.isnan(out["points"]).any()
    assert out["detected"][:, 0].all()                           # pose always seen
    assert out["detected"][:, 1].tolist() == [1, 1, 0, 1, 1, 1]  # left hand missing in frame 2
    assert not out["detected"][:, 2].any()                       # right hand never seen
    assert np.allclose(out["points"][:, 0], holistic[:, 0, :2], atol=1e-3)  # nose first


def test_folder_hash_changes_when_any_file_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("ASL_DATA_ROOT", str(tmp_path))
    folder = tmp_path / "processed" / "smoke" / "val"
    folder.mkdir(parents=True)
    (folder / "a.npz").write_bytes(b"1")
    manifest = tmp_path / "manifest.json"
    record([folder], path=manifest)
    verify([folder], path=manifest)
    (folder / "b.npz").write_bytes(b"2")
    try:
        verify([folder], path=manifest)
    except SystemExit:
        return
    raise AssertionError("an added file should change the folder hash")


def test_chunk_does_not_record_incomplete_split(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["features", "extract", "--offset", "10", "--limit", "5"])
    monkeypatch.setattr(features, "load_config", lambda: {"data": {}})
    monkeypatch.setattr(features, "extract_splits", lambda *args: calls.append(args) or True)
    monkeypatch.setattr(features, "record_landmarks", lambda *_: calls.append("record"))
    features.main()
    assert calls == [({}, ["train", "val"], 5, 1, 10, "mediapipe")]


def test_native_pose_sources_keep_dimensions_and_fill_missing_values():
    from asl_translate.pose import prepare

    for n in (65, 67):
        raw = np.ones((3, n, 3), dtype=np.float32)
        raw[1, 0] = np.nan
        features = prepare(raw)
        assert features["points"].shape == (3, n, 2)
        assert features["detected"].shape == (3, n)
        assert not features["detected"][1, 0]
        assert features["confidence"][1, 0] == 0
        assert np.isfinite(features["points"]).all()


def test_openpose_adapter_keeps_body_and_both_hands(monkeypatch):
    from pathlib import Path

    from asl_translate.pose import OpenPoseEstimator

    def run(command, check):
        output = Path(command[command.index("--write_json") + 1])
        output.mkdir()
        person = {"pose_keypoints_2d": [0.1, 0.2, 0.9] * 25,
                  "hand_left_keypoints_2d": [0.3, 0.4, 0.8] * 21,
                  "hand_right_keypoints_2d": [0.5, 0.6, 0.7] * 21}
        (output / "00000000_keypoints.json").write_text(json.dumps({"people": [person]}))
        (output / "00000001_keypoints.json").write_text(json.dumps({"people": []}))
        assert check

    monkeypatch.setattr("asl_translate.pose.subprocess.run", run)
    monkeypatch.setenv("ASL_OPENPOSE_BIN", "openpose")
    monkeypatch.setenv("ASL_OPENPOSE_MODELS", "models")
    frames = [(0, np.zeros((8, 8, 3), dtype=np.uint8)),
              (1, np.zeros((8, 8, 3), dtype=np.uint8))]
    raw = OpenPoseEstimator().extract(frames)
    assert raw.shape == (2, 67, 3)
    assert_allclose(raw[0, 25], [0.3, 0.4, 0.8])
    assert_allclose(raw[0, 46], [0.5, 0.6, 0.7])
    assert np.isnan(raw[1]).all()


def test_source_normalization_uses_native_shoulder_indices():
    from asl_translate.models.train import normalize

    for source, count, shoulders in (("openpose", 67, (5, 2)), ("vitpose", 65, (5, 6))):
        points = np.zeros((2, count, 2), dtype=np.float32)
        points[:, shoulders[0]] = [2, 4]
        points[:, shoulders[1]] = [4, 4]
        out = normalize(points, source)
        assert_allclose(out[0, shoulders[0]], [-0.5, 0])
        assert_allclose(out[0, shoulders[1]], [0.5, 0])


def test_resume_refuses_changed_extraction_metadata(tmp_path, monkeypatch):
    from asl_translate import pose as pose_sources

    data = {"processed": tmp_path}
    monkeypatch.setattr(pose_sources, "extraction_metadata", lambda *_: {"weights": "first"})
    path = pose_sources.check_metadata(data, "vitpose")
    assert path.exists()
    monkeypatch.setattr(pose_sources, "extraction_metadata", lambda *_: {"weights": "second"})
    with pytest.raises(ValueError, match="does not match"):
        pose_sources.check_metadata(data, "vitpose")


def test_vitpose_adapter_selects_person_and_converts_coordinates(monkeypatch):
    from asl_translate import pose as pose_sources

    class Estimator:
        def __init__(self):
            self._scores_bbox = {0: 0.3, 1: 0.9}

        def reset(self):
            pass

        def inference(self, _frame):
            person = np.tile([10.0, 20.0, 0.8], (133, 1))
            return {0: person * 0, 1: person}

    monkeypatch.setenv("ASL_VITPOSE_MODEL", "pose.pth")
    monkeypatch.setenv("ASL_VITPOSE_DETECTOR", "detector.pt")
    monkeypatch.setenv("ASL_VITPOSE_DEVICE", "cpu")
    monkeypatch.setattr(pose_sources.ViTPoseEstimator, "_model", staticmethod(lambda *_: Estimator()))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = pose_sources.ViTPoseEstimator().extract([(0, frame)])
    assert out.shape == (1, 65, 3)
    assert_allclose(out[0, 0], [0.1, 0.1, 0.8])


def test_feature_download_uses_selected_source_and_split(tmp_path, monkeypatch):
    import huggingface_hub

    from asl_translate import data as data_module

    calls = []

    def snapshot(_repo, **kwargs):
        calls.append(kwargs["allow_patterns"])
        path = tmp_path / "landmarks/openpose/test/example.npz"
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(data_module, "verify", lambda paths: calls.append(paths))
    data = {"processed": tmp_path, "processed_hub": {"repo": "example/repo", "revision": "abc"}}
    result = data_module.DataManager(data).download_features("openpose", ["test"])
    assert result == [tmp_path / "landmarks/openpose/test"]
    assert "landmarks/openpose/test/*.npz" in calls[0]
    assert tmp_path / "landmarks/openpose/metadata.json" in calls[1]


def test_training_reuses_feature_downloader_when_data_is_missing(tmp_path, monkeypatch):
    from asl_translate.models import train

    calls = []
    data = {"processed": tmp_path / "processed"}
    annotations = tmp_path / "annotations.csv"
    annotations.touch()
    monkeypatch.setattr(train, "annotation_path", lambda *_: annotations)
    monkeypatch.setattr(train, "verify", lambda paths: calls.append(("verify", paths)))

    def download(_data, source, splits, smoke):
        calls.append((source, splits, smoke))
        for split in splits:
            (data["processed"] / "smoke" / source / split).mkdir(parents=True)
        vocab = data["processed"] / "splits" / "vocab.txt"
        vocab.parent.mkdir(parents=True)
        vocab.touch()
        (data["processed"] / "landmarks" / source).mkdir(parents=True)
        (data["processed"] / "landmarks" / source / "metadata.json").touch()

    monkeypatch.setattr(train.DataManager, "download_features", download)
    needed = train.ensure_data(data, smoke=True, source="rtmw")
    assert calls[0] == ("rtmw", ["train", "val"], True)
    assert len(needed) == 4
    assert calls[1][0] == "verify"


def test_wholebody_adapter_selects_signer_and_handles_missing_frames(monkeypatch):
    from asl_translate import pose as rtm_pose

    class Detector:
        def __init__(self):
            self.calls = 0

        def __call__(self, frame):
            self.calls += 1
            return np.array([[0, 0, 10, 10], [0, 0, 40, 20]]) if self.calls == 1 else []

    def estimator(frame, bboxes):
        assert bboxes.tolist() == [[0, 0, 40, 20]]
        return np.full((1, 133, 2), [20, 10]), np.full((1, 133), 0.8)

    monkeypatch.setattr(rtm_pose.RTMEstimator, "_model", staticmethod(lambda _: (Detector(), estimator)))
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    result = rtm_pose.RTMEstimator("rtmw").extract([(0, frame), (1, frame)])
    assert result.shape == (2, 65, 3)
    np.testing.assert_allclose(result[0], np.tile([0.5, 0.5, 0.8], (65, 1)))
    assert np.isnan(result[1]).all()


def test_legacy_metadata_requires_new_data_root(tmp_path, monkeypatch):
    from asl_translate import pose

    root = tmp_path / "landmarks" / "rtmw"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"source": "rtmw", "code_sha256": {"old": "hash"}}))
    monkeypatch.setattr(pose, "extraction_metadata", lambda *_: {"source": "rtmw", "code_sha256": {"new": "hash"}})
    with pytest.raises(ValueError, match="use a new data root"):
        pose.check_metadata({"processed": tmp_path}, "rtmw")


def test_pose_estimator_factory_dispatches_all_sources():
    from asl_translate.pose import (
        MediaPipeEstimator,
        OpenPoseEstimator,
        RTMEstimator,
        ViTPoseEstimator,
        get_pose_estimator,
    )

    for name, expected in (("mediapipe", MediaPipeEstimator), ("openpose", OpenPoseEstimator),
                           ("vitpose", ViTPoseEstimator), ("rtmw", RTMEstimator),
                           ("dwpose", RTMEstimator)):
        assert isinstance(get_pose_estimator(name), expected)
    with pytest.raises(ValueError, match="Unknown"):
        get_pose_estimator("unknown")


def test_estimators_require_model_configuration(monkeypatch):
    from asl_translate import pose

    monkeypatch.delenv("ASL_OPENPOSE_BIN", raising=False)
    monkeypatch.delenv("ASL_VITPOSE_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="ASL_OPENPOSE_BIN"):
        pose.OpenPoseEstimator().extract([])
    with pytest.raises(RuntimeError, match="ASL_VITPOSE_MODEL"):
        pose.ViTPoseEstimator().extract([])


def test_extract_cli_uses_pose_estimator_option(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["features", "extract", "--pose_estimator", "rtmw", "--limit", "1"])
    monkeypatch.setattr(features, "load_config", lambda: {"data": {}})
    monkeypatch.setattr("asl_translate.pose.check_metadata", lambda *args: calls.append(args))
    monkeypatch.setattr(features, "extract_splits", lambda *args: calls.append(args) or True)
    features.main()
    assert calls[0] == ({}, "rtmw")
    assert calls[1][-1] == "rtmw"


def test_process_video_uses_estimator_and_preserves_saved_layout(tmp_path, monkeypatch):
    import zipfile

    import cv2

    from asl_translate import pose

    raw = tmp_path / "raw"
    raw.mkdir()
    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    assert writer.isOpened()
    for _ in range(48):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    with zipfile.ZipFile(raw / "shard.zip", "w") as archive:
        archive.write(video, "video.mp4")

    calls = []

    class Estimator:
        def extract(self, frames, work_dir):
            frames = list(frames)
            calls.append((len(frames), work_dir))
            return np.full((len(frames), 65, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(pose, "get_pose_estimator", lambda source: Estimator())
    data = {"raw": raw, "interim": tmp_path / "interim", "processed": tmp_path / "processed",
            "fps": 15, "features": FEATURES}
    _, frames, _, error = features.process_video(
        data, "val", "video", "shard.zip", "video.mp4", [("sentence", 0.0, 1.0)], "rtmw")
    assert not error
    assert frames == calls[0][0] == 15
    with np.load(tmp_path / "processed/landmarks/rtmw/val/sentence.npz") as result:
        assert result["points"].shape == (frames, 65, 2)
        assert result["detected"].shape == (frames, 65)
        assert result["confidence"].shape == (frames, 65)
