import json
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from asl_translate import features
from asl_translate.dataset import record, verify
from asl_translate.features import LEFT_HAND, RIGHT_HAND, interpolate, reduce

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
    from asl_translate.pose_sources import prepare

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

    from asl_translate.pose_sources import openpose

    def run(command, check):
        output = Path(command[command.index("--write_json") + 1])
        output.mkdir()
        person = {"pose_keypoints_2d": [0.1, 0.2, 0.9] * 25,
                  "hand_left_keypoints_2d": [0.3, 0.4, 0.8] * 21,
                  "hand_right_keypoints_2d": [0.5, 0.6, 0.7] * 21}
        (output / "00000000_keypoints.json").write_text(json.dumps({"people": [person]}))
        (output / "00000001_keypoints.json").write_text(json.dumps({"people": []}))
        assert check

    monkeypatch.setattr("asl_translate.pose_sources.subprocess.run", run)
    frames = [(0, np.zeros((8, 8, 3), dtype=np.uint8)),
              (1, np.zeros((8, 8, 3), dtype=np.uint8))]
    raw = openpose(frames, "openpose", "models")
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
    from asl_translate import pose_sources

    data = {"processed": tmp_path}
    monkeypatch.setattr(pose_sources, "extraction_metadata", lambda *_: {"weights": "first"})
    path = pose_sources.check_metadata(data, "vitpose")
    assert path.exists()
    monkeypatch.setattr(pose_sources, "extraction_metadata", lambda *_: {"weights": "second"})
    with pytest.raises(ValueError, match="does not match"):
        pose_sources.check_metadata(data, "vitpose")


def test_vitpose_adapter_selects_person_and_converts_coordinates(monkeypatch):
    from asl_translate import pose_sources

    class Estimator:
        def __init__(self):
            self._scores_bbox = {0: 0.3, 1: 0.9}

        def reset(self):
            pass

        def inference(self, _frame):
            person = np.tile([10.0, 20.0, 0.8], (133, 1))
            return {0: person * 0, 1: person}

    monkeypatch.setattr(pose_sources, "_vit_model", lambda *_: Estimator())
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = pose_sources.vitpose([(0, frame)], "pose.pth", "detector.pt", "cpu")
    assert out.shape == (1, 65, 3)
    assert_allclose(out[0, 0], [0.1, 0.1, 0.8])


def test_feature_download_uses_selected_source_and_split(tmp_path, monkeypatch):
    from asl_translate import download_features

    calls = []

    def snapshot(_repo, **kwargs):
        calls.append(kwargs["allow_patterns"])
        path = tmp_path / "landmarks/openpose/test/example.npz"
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr(download_features, "snapshot_download", snapshot)
    monkeypatch.setattr(download_features, "verify", lambda paths: calls.append(paths))
    data = {"processed": tmp_path, "processed_hub": {"repo": "example/repo", "revision": "abc"}}
    result = download_features.download(data, "openpose", ["test"])
    assert result == [tmp_path / "landmarks/openpose/test"]
    assert "landmarks/openpose/test/*.npz" in calls[0]
    assert tmp_path / "landmarks/openpose/metadata.json" in calls[1]


def test_training_reuses_feature_downloader_when_data_is_missing(tmp_path, monkeypatch):
    from asl_translate import download_features
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

    monkeypatch.setattr(download_features, "download", download)
    needed = train.ensure_data(data, smoke=True, source="rtmw")
    assert calls[0] == ("rtmw", ["train", "val"], True)
    assert len(needed) == 4
    assert calls[1][0] == "verify"


def test_wholebody_adapter_selects_signer_and_handles_missing_frames(monkeypatch):
    from asl_translate import rtm_pose

    class Detector:
        def __init__(self):
            self.calls = 0

        def __call__(self, frame):
            self.calls += 1
            return np.array([[0, 0, 10, 10], [0, 0, 40, 20]]) if self.calls == 1 else []

    def estimator(frame, bboxes):
        assert bboxes.tolist() == [[0, 0, 40, 20]]
        return np.full((1, 133, 2), [20, 10]), np.full((1, 133), 0.8)

    monkeypatch.setattr(rtm_pose, "_model", lambda _: (Detector(), estimator))
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    result = rtm_pose.extract_wholebody("rtmw", [(0, frame), (1, frame)])
    assert result.shape == (2, 65, 3)
    np.testing.assert_allclose(result[0], np.tile([0.5, 0.5, 0.8], (65, 1)))
    assert np.isnan(result[1]).all()


def test_existing_vitpose_metadata_accepts_only_the_verified_code_migration():
    from asl_translate.pose_compat import (
        ADDED_RTM_SOURCES,
        LEGACY_VITPOSE,
        compatible_vitpose_metadata,
    )

    previous = {"source": "vitpose", "code_sha256": LEGACY_VITPOSE, "fps": 15}
    current = {"source": "vitpose", "code_sha256": ADDED_RTM_SOURCES, "fps": 15}
    assert compatible_vitpose_metadata(previous, current)
    assert not compatible_vitpose_metadata(previous, {**current, "fps": 30})
    assert not compatible_vitpose_metadata(previous, {**current, "code_sha256": {}})
