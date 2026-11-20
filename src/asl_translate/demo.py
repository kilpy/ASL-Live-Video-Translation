"""Live webcam ASL-to-English demo, one sentence at a time (push-to-sign).

    uv run python -m asl_translate.demo                      # newest models/**/model.pt
    uv run python -m asl_translate.demo --checkpoint models/BASELINE/full/seed_0/model.pt
    uv run python -m asl_translate.demo --input clip.mp4 --headless   # translate a file

Webcam: press SPACE to start a sentence and SPACE again to translate it; q or Esc quits.
A file is treated as one sentence and translated at the end.

The model was trained on whole How2Sign sentences, so the demo segments by hand rather than
decoding a sliding window. Frames are landmarked at the training frame rate (`data.fps`) and
go through the same `reduce` -> `frame_features` steps as training. The image is mirrored
for display only, after landmarking, so left and right hands match the training data.
"""
import argparse
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np

from asl_translate.config import REPO_ROOT
from asl_translate.features import LEFT_HAND, POSE, RIGHT_HAND, reduce
from asl_translate.models.infer import Translator
from asl_translate.pose import MediaPipeEstimator

MIN_SECONDS = 0.5  # shorter recordings are almost certainly an accidental double tap
FONT = cv2.FONT_HERSHEY_SIMPLEX


def newest_checkpoint() -> Path:
    """Return the most recently written models/**/model.pt."""
    checkpoints = sorted((REPO_ROOT / "models").glob("**/model.pt"),
                         key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise SystemExit("No checkpoint under models/. Train one first, e.g.\n"
                         "  uv run python -m asl_translate.models.train --smoke")
    return checkpoints[-1]


def draw_landmarks(frame: np.ndarray, holistic: np.ndarray, pose_points: list[int]) -> None:
    """Draw the points the model sees: chosen upper-body points and both hands."""
    h, w = frame.shape[:2]
    pose = holistic[POSE][pose_points]
    for points, color in ((pose, (255, 200, 0)), (holistic[LEFT_HAND], (0, 255, 0)),
                          (holistic[RIGHT_HAND], (0, 128, 255))):
        for x, y in points[:, :2]:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1)


def draw_text(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Draw lines of text on a dark band at the bottom of the frame."""
    h, w = frame.shape[:2]
    scale, step = 0.7, 30
    wrapped = [(part, color) for text, color in lines
               for part in (textwrap.wrap(text, max(20, w // 15)) or [""])]
    top = h - step * len(wrapped) - 10
    band = frame[top:h].copy()
    frame[top:h] = (band * 0.4).astype(frame.dtype)
    for i, (text, color) in enumerate(wrapped):
        cv2.putText(frame, text, (10, top + step * (i + 1) - 5), FONT, scale, color, 2,
                    cv2.LINE_AA)


def translate(translator: Translator, buffer: list[np.ndarray]) -> tuple[str, float]:
    """Translate buffered (553, 3) holistic frames as one sentence."""
    reduced = reduce(np.stack(buffer), translator.features)
    return translator.translate(reduced["points"], reduced["detected"])


def main() -> None:
    """Run the push-to-sign webcam loop, or translate a video file."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="default: newest models/**/model.pt")
    parser.add_argument("--input", default="0", help="camera index or video file (default: 0)")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--headless", action="store_true", help="no window (video files only)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or newest_checkpoint()
    translator = Translator.load(checkpoint, args.device)
    if translator.source != "mediapipe":
        raise SystemExit(f"{checkpoint} was trained on {translator.source} landmarks; "
                         "the demo extracts MediaPipe landmarks.")
    print(f"Loaded {checkpoint} on {translator.device}")

    is_file = not args.input.isdigit()
    if args.headless and not is_file:
        raise SystemExit("--headless needs --input <video file>")
    cap = cv2.VideoCapture(args.input if is_file else int(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")

    period = 1.0 / translator.fps
    pose_points = translator.features["pose_points"]
    buffer: list[np.ndarray] = []
    recording, started = is_file, time.monotonic()
    last_sample, last_holistic = -np.inf, None
    result = ("Press SPACE to start signing", (255, 255, 255))
    window = "ASL translation (SPACE sign / translate, q quit)"

    with MediaPipeEstimator._make_landmarker() as landmarker:
        import mediapipe as mp

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # seconds since start: video time for files, wall clock for the webcam
                now = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file
                       else time.monotonic() - started)
                if now - last_sample >= period - 1e-3:  # subsample to the training frame rate
                    last_sample = now
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    last_holistic = MediaPipeEstimator.to_array(
                        landmarker.detect_for_video(image, int(now * 1000)))
                    if recording:
                        buffer.append(last_holistic)

                if args.headless:
                    continue
                if last_holistic is not None:
                    draw_landmarks(frame, last_holistic, pose_points)
                if not is_file:
                    frame = cv2.flip(frame, 1)  # mirror for the signer, after landmarking
                status = (f"REC {len(buffer) * period:4.1f}s  (SPACE to translate)", (0, 0, 255))
                draw_text(frame, [status, result] if recording else [result])
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" ") and not is_file:
                    if not recording:
                        buffer, recording = [], True
                        result = ("Signing...", (200, 200, 200))
                    else:
                        recording = False
                        if len(buffer) < MIN_SECONDS * translator.fps:
                            result = ("Too short, press SPACE to try again", (0, 200, 255))
                            continue
                        text, confidence = translate(translator, buffer)
                        print(f"[{len(buffer) * period:.1f}s, confidence {confidence:.2f}] {text}")
                        result = (f"{text}  ({confidence:.0%})", (0, 255, 0))
        finally:
            cap.release()
            cv2.destroyAllWindows()

    if is_file:
        if not buffer:
            raise SystemExit(f"No frames read from {args.input}")
        text, confidence = translate(translator, buffer)
        print(f"[{len(buffer) * period:.1f}s, {len(buffer)} frames, "
              f"confidence {confidence:.2f}] {text}")


if __name__ == "__main__":
    main()
