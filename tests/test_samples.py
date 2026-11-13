import json
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from asl_translate import samples


def test_samples_use_indexed_validation_videos_and_aligned_ranges(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    assert writer.isOpened()
    for i in range(72):
        writer.write(np.full((64, 64, 3), i, dtype=np.uint8))
    writer.release()
    with zipfile.ZipFile(raw / "shard.zip", "w") as archive:
        archive.write(video, "one.mp4")
        archive.write(video, "two.mp4")

    rows = [{"VIDEO_NAME": name, "SENTENCE_NAME": name + "_sentence",
             "SENTENCE": "one two three four five", "DURATION": 1.0,
             "START_REALIGNED": 1.0, "END_REALIGNED": 2.0} for name in ("one", "two")]
    frame = pd.DataFrame(rows)
    videos = {name: ["shard.zip", name + ".mp4"] for name in ("one", "two")}
    data = {"root": tmp_path, "raw": raw, "fps": 15}
    seen_timestamps = []

    def extract(frames):
        timestamps = [time for time, _ in frames]
        seen_timestamps.append(timestamps)
        return np.full((len(timestamps), 553, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(samples, "OUT_DIR", tmp_path / "reports")
    monkeypatch.setattr(samples, "load_config", lambda: {"data": data})
    monkeypatch.setattr(samples, "verify", lambda _: None)
    monkeypatch.setattr(samples, "annotation_path", lambda *_: tmp_path / "annotations.csv")
    monkeypatch.setattr(samples.DataManager, "load_video_index", lambda self: {"val": videos})
    monkeypatch.setattr(samples, "load_annotations", lambda *_: frame)
    monkeypatch.setattr(samples, "MediaPipeEstimator", lambda: type("FakeEstimator", (), {"extract": staticmethod(extract)})())

    samples.main()

    assert len(seen_timestamps) == 2
    assert all(times[0] == 1000 and times[-1] <= 2000 for times in seen_timestamps)
    assert len(list((tmp_path / "samples").glob("*.npy"))) == 2
    stats = json.loads((tmp_path / "reports" / "stats.json").read_text())
    assert {item["clip"] for item in stats} == {"one_sentence", "two_sentence"}
    assert (tmp_path / "reports" / "clips.png").is_file()
    assert (tmp_path / "reports" / "landmarks.png").is_file()
    assert not list((tmp_path / "samples").glob("*.mp4"))


def test_select_samples_rejects_missing_videos():
    frame = pd.DataFrame([{"VIDEO_NAME": "missing", "SENTENCE": "one two three four five",
                           "DURATION": 1.0}])
    with pytest.raises(ValueError, match="No matching validation sentences"):
        samples.select_samples(frame, {})
