from types import SimpleNamespace

import cv2
import numpy as np

from asl_translate.landmarks import NUM_LANDMARKS, PARTS, read_frames, to_array


def test_to_array_places_parts_and_marks_missing_as_nan():
    point = SimpleNamespace(x=0.1, y=0.2, z=0.3)
    result = SimpleNamespace(
        pose_landmarks=[point] * 33,
        face_landmarks=[point] * 478,
        left_hand_landmarks=[],
        right_hand_landmarks=[point] * 21,
    )
    frame = to_array(result)
    assert frame.shape == (NUM_LANDMARKS, 3)
    assert np.isnan(frame[PARTS["left_hand"]]).all()
    assert not np.isnan(np.delete(frame, PARTS["left_hand"], axis=0)).any()
    assert np.allclose(frame[PARTS["right_hand"]], (0.1, 0.2, 0.3))


def test_read_frames_resamples_24fps_source_to_15fps_with_rising_timestamps(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    for _ in range(48):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    timestamps = [t for t, _ in read_frames(path, 15)]
    assert len(timestamps) == 30
    assert timestamps == sorted(set(timestamps))
