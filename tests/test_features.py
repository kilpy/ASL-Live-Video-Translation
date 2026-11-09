import numpy as np

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