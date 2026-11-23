"""Live webcam ASL-to-English demo, one sentence at a time (push-to-sign), with interpretability.

    uv run python -m asl_translate.demo                      # newest models/**/model.pt
    uv run python -m asl_translate.demo --checkpoint models/BASELINE/full/seed_0/model.pt
    uv run python -m asl_translate.demo --input clip.mp4 --headless   # translate a file

Webcam: SPACE starts a sentence and SPACE again translates it; while signing, a provisional
translation refreshes every --preview-every seconds. Afterwards a/d step through the words:
the side panel shows each word's probability and top alternatives, the strip at the bottom
shows which frames the decoder attended to for it, and the inset shows the model's input
(normalized skeleton) at the word's most-attended frame. q or Esc quits.
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
from asl_translate.features import reduce
from asl_translate.models.infer import Explanation, Translator
from asl_translate.models.train import normalize
from asl_translate.pose import MediaPipeEstimator

MIN_SECONDS = 0.5  # shorter recordings are almost certainly an accidental double tap
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_WIDTH, INSET = 320, 180

# Bones over the 49 reduced points: 7 pose points (nose, shoulders, elbows, wrists),
# then the left hand (7..27) and right hand (28..48) in MediaPipe's 21-point layout.
HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
        (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
        (17, 18), (18, 19), (19, 20)]
BONES = [[(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)],
         [(a + 7, b + 7) for a, b in HAND], [(a + 28, b + 28) for a, b in HAND]]
PARTS = ["pose", "L hand", "R hand"]           # order of the model's 3 detection flags
COLORS = [(255, 200, 0), (0, 255, 0), (0, 128, 255)]  # BGR
GRAY = (110, 110, 110)


def newest_checkpoint() -> Path:
    """Return the most recently written models/**/model.pt."""
    checkpoints = sorted((REPO_ROOT / "models").glob("**/model.pt"),
                         key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise SystemExit("No checkpoint under models/. Train one first, e.g.\n"
                         "  uv run python -m asl_translate.models.train --smoke")
    return checkpoints[-1]


def load_translator(checkpoint: Path | None, device: str) -> Translator:
    """Load a checkpoint (default: newest) and check it was trained on MediaPipe landmarks."""
    checkpoint = checkpoint or newest_checkpoint()
    translator = Translator.load(checkpoint, device)
    if translator.source not in ("mediapipe", "dinov2"):
        raise SystemExit(f"{checkpoint} was trained on {translator.source} landmarks; "
                         "the demo extracts MediaPipe landmarks.")
    print(f"Loaded {checkpoint} on {translator.device}")
    return translator


# ---------------------------------------------------------------- drawing


def draw_skeleton(image: np.ndarray, points: np.ndarray, detected: np.ndarray, to_px) -> None:
    """Draw the bones of each detected part; `to_px` maps a point to integer pixels."""
    for bones, color, ok in zip(BONES, COLORS, detected):
        if not ok:
            continue
        for a, b in bones:
            cv2.line(image, to_px(points[a]), to_px(points[b]), color, 2, cv2.LINE_AA)
            cv2.circle(image, to_px(points[b]), 3, color, -1)


def draw_status(image: np.ndarray, detected: np.ndarray) -> None:
    """Chips for the 3 detection flags the model receives; gray = not detected."""
    x = 10
    for name, color, ok in zip(PARTS, COLORS, detected):
        label = f"{name} {'ok' if ok else '--'}"
        (w, h), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(image, (x - 4, 8), (x + w + 4, 14 + h), (0, 0, 0), -1)
        cv2.putText(image, label, (x, 10 + h), FONT, 0.5, color if ok else GRAY, 1, cv2.LINE_AA)
        x += w + 16


def draw_model_input(image: np.ndarray, points: np.ndarray, detected: np.ndarray) -> None:
    """Inset (top left before mirroring) with the normalized skeleton the model is given."""
    inset = image[:INSET, :INSET]
    inset[:] = 20
    # normalized: shoulder midpoint at 0, shoulder width 1
    to_px = lambda p: (int(INSET / 2 + p[0] * INSET / 5), int(INSET * 0.35 + p[1] * INSET / 5))
    draw_skeleton(inset, np.nan_to_num(points), detected, to_px)
    cv2.rectangle(inset, (0, 0), (INSET - 1, INSET - 1), GRAY, 1)


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


def draw_panel(height: int, expl: Explanation | None, selected: int, live: bool) -> np.ndarray:
    """Side panel: each word with its probability bar, then the selected word's alternatives."""
    panel = np.full((height, PANEL_WIDTH, 3), 30, dtype=np.uint8)
    put = lambda text, y, color=(230, 230, 230), s=0.55, x=10: cv2.putText(
        panel, text, (x, y), FONT, s, color, 1, cv2.LINE_AA)
    if not expl or not expl.words:
        put("no translation yet", 30, GRAY)
        return panel
    put(f"{'PREVIEW' if live else 'TRANSLATION'}  confidence {expl.confidence:.0%}", 25,
        (0, 200, 255) if live else (0, 255, 0))
    y = 50
    for i, (word, p) in enumerate(zip(expl.words, expl.probs)):
        if y > height - 110:  # leave room for the alternatives
            put("...", y, GRAY)
            break
        if i == selected and not live:
            cv2.rectangle(panel, (4, y - 15), (PANEL_WIDTH - 4, y + 5), (80, 80, 80), -1)
        cv2.rectangle(panel, (165, y - 11), (165 + int(p * 145), y), (0, 180, 0), -1)
        put(word[:11], y)
        put(f"{p:.0%}", y, s=0.45, x=120)
        y += 22
    if not live:
        y = height - 85
        put(f"alternatives for '{expl.words[selected]}':", y, GRAY, 0.5)
        for word, p in expl.topk[selected]:
            y += 22
            put(f"  {word[:18]:<18} {p:.0%}", y, s=0.5)
    return panel


def draw_attention(width: int, expl: Explanation, selected: int, live: bool) -> np.ndarray:
    """Heatmap strip: one row per word, one column per buffered frame (time left to right)."""
    row_h = 16
    att = expl.attention / np.maximum(expl.attention.max(axis=1, keepdims=True), 1e-9)
    heat = cv2.resize((att * 255).astype(np.uint8), (width, row_h * len(att)),
                      interpolation=cv2.INTER_NEAREST)
    strip = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    for i, word in enumerate(expl.words):
        cv2.putText(strip, word[:14], (4, row_h * (i + 1) - 4), FONT, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)
    if not live:
        cv2.rectangle(strip, (0, row_h * selected), (width - 1, row_h * (selected + 1) - 1),
                      (255, 255, 255), 1)
    return strip


# ---------------------------------------------------------------- session


class Session:
    """Push-to-sign state shared by the OpenCV window and the Gradio app."""

    def __init__(self, translator: Translator, preview_every: float = 1.0):
        self.translator, self.preview_every = translator, preview_every
        self.period = 1.0 / translator.fps
        self.landmarker = None  # created on the first frame
        self.buffer: list[np.ndarray] = []
        self.crops: list[np.ndarray] = []  # DINOv2 vectors, only for dinov2 models (SHuBERT)
        self.recording, self.expl, self.selected = False, None, 0
        self.holistic = None
        self.last_sample = self.last_preview = -np.inf
        self.message = ("Press SPACE to start signing", (255, 255, 255))

    def feed(self, frame: np.ndarray, now: float) -> None:
        """Landmark a BGR frame at `now` seconds, subsampled to the training frame rate."""
        if now - self.last_sample < self.period - 1e-3:
            return
        import mediapipe as mp

        self.last_sample = now
        if self.landmarker is None:
            self.landmarker = MediaPipeEstimator._make_landmarker()
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.holistic = MediaPipeEstimator.to_array(
            self.landmarker.detect_for_video(image, int(now * 1000)))
        if self.recording:
            self.buffer.append(self.holistic)
            if self.translator.needs_crops:
                from asl_translate.dino import embed
                self.crops.append(embed([(0, frame)], self.holistic[None],
                                        str(self.translator.device))[0])
            if (self.preview_every and now - self.last_preview >= self.preview_every
                    and self.long_enough()):
                # ponytail: synchronous; move to a worker thread if previews stutter the video
                self.last_preview, self.expl = now, self.explain()

    def long_enough(self) -> bool:
        return len(self.buffer) >= MIN_SECONDS * self.translator.fps

    def explain(self) -> Explanation:
        """Translate the buffered (553, 3) holistic frames as one sentence."""
        reduced = reduce(np.stack(self.buffer), self.translator.features)
        crops = np.stack(self.crops) if self.translator.needs_crops else None
        return self.translator.explain(reduced["points"], reduced["detected"], crops)

    def toggle(self) -> None:
        """SPACE: start a sentence, or stop and translate it."""
        if not self.recording:
            self.buffer, self.crops, self.recording, self.expl = [], [], True, None
            self.message = ("Signing...", (200, 200, 200))
            return
        self.recording = False
        if not self.long_enough():
            self.expl = None
            self.message = ("Too short, press SPACE to try again", (0, 200, 255))
            return
        self.expl, self.selected = self.explain(), 0
        text = self.translator.separator.join(self.expl.words)
        print(f"[{len(self.buffer) * self.period:.1f}s, confidence {self.expl.confidence:.2f}] "
              f"{text}")
        self.message = ("a/d: inspect words   SPACE: sign again", (200, 200, 200))

    def select(self, step: int) -> None:
        """Move the inspected word by `step`."""
        if self.expl and self.expl.words and not self.recording:
            self.selected = (self.selected + step) % len(self.expl.words)

    def render(self, frame: np.ndarray, mirror: bool = True) -> np.ndarray:
        """Video with skeleton, model-input inset and status, plus side panel and attention strip."""
        frame, h, w = frame.copy(), *frame.shape[:2]
        detected = np.zeros(3, dtype=bool)
        expl, live = self.expl, self.recording
        if self.holistic is not None:
            reduced = reduce(self.holistic[None], self.translator.features)
            points, detected = reduced["points"][0].astype(np.float32), reduced["detected"][0]
            draw_skeleton(frame, points, detected, lambda p: (int(p[0] * w), int(p[1] * h)))
            inset = (normalize(points[None], self.translator.source)[0], detected)
        if expl and expl.words and not live:  # reviewing: the selected word's peak frame
            peak = int(expl.attention[self.selected].argmax())
            inset = (expl.model_input[peak], expl.detected[peak])
        if self.holistic is not None or (expl and expl.words and not live):
            draw_model_input(frame, *inset)
        if mirror:
            frame = cv2.flip(frame, 1)  # mirror for the signer, after landmarking
        draw_status(frame, detected)
        lines = [self.message]
        if live:
            lines.insert(0, (f"REC {len(self.buffer) * self.period:4.1f}s  (SPACE to translate)",
                             (0, 0, 255)))
        draw_text(frame, lines)
        canvas = np.hstack([frame, draw_panel(h, expl, self.selected, live)])
        if expl and expl.words:
            canvas = np.vstack([canvas, draw_attention(canvas.shape[1], expl, self.selected, live)])
        return canvas


# ---------------------------------------------------------------- main


def main() -> None:
    """Run the push-to-sign webcam loop, or translate a video file."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="default: newest models/**/model.pt")
    parser.add_argument("--input", default="0", help="camera index or video file (default: 0)")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--headless", action="store_true", help="no window (video files only)")
    parser.add_argument("--preview-every", type=float, default=1.0,
                        help="seconds between provisional translations while signing (0: off)")
    args = parser.parse_args()

    session = Session(load_translator(args.checkpoint, args.device), args.preview_every)
    is_file = not args.input.isdigit()
    if args.headless and not is_file:
        raise SystemExit("--headless needs --input <video file>")
    cap = cv2.VideoCapture(args.input if is_file else int(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")

    if is_file:
        session.toggle()  # a file is one sentence
        session.preview_every = 0
    started = time.monotonic()
    window = "ASL translation (SPACE sign / translate, a/d words, q quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # seconds since start: video time for files, wall clock for the webcam
            now = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file
                   else time.monotonic() - started)
            session.feed(frame, now)
            if args.headless:
                continue
            cv2.imshow(window, session.render(frame, mirror=not is_file))
            key = cv2.waitKey(1) & 0xFF
            # closing the window with X would otherwise just reopen it on the next imshow
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" ") and not is_file:
                session.toggle()
            elif key in (ord("a"), ord("d")):
                session.select(-1 if key == ord("a") else 1)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if session.landmarker is not None:
            session.landmarker.close()

    if is_file:
        if not session.buffer:
            raise SystemExit(f"No frames read from {args.input}")
        session.toggle()
        if session.expl:
            for word, p, alts in zip(session.expl.words, session.expl.probs, session.expl.topk):
                print(f"  {word:<15} {p:.2f}   alternatives: "
                      + ", ".join(f"{w} {q:.2f}" for w, q in alts[1:]))


if __name__ == "__main__":
    main()
