"""Browser version of the push-to-sign demo, same views as `asl_translate.demo`.

    uv run --group demo python -m asl_translate.app [--checkpoint ...] [--device cuda]
"""
import argparse
import time
from pathlib import Path

import cv2

from asl_translate.demo import Session, load_translator


def main() -> None:
    """Serve the webcam demo on a local Gradio page."""
    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preview-every", type=float, default=1.0)
    parser.add_argument("--share", action="store_true", help="public Gradio link")
    args = parser.parse_args()

    # ponytail: one session for the whole server (a local, single-signer demo);
    # use gr.State per browser tab if several people ever sign at once.
    session = Session(load_translator(args.checkpoint, args.device), args.preview_every)
    started = time.monotonic()

    def step(rgb):
        if rgb is None:
            return None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        session.feed(bgr, time.monotonic() - started)
        return cv2.cvtColor(session.render(bgr), cv2.COLOR_BGR2RGB)

    with gr.Blocks(title="ASL translation") as ui:
        gr.Markdown("**Start** begins a sentence, press it again to translate. Then step through "
                    "the words to see their probability, alternatives and attention.")
        with gr.Row():
            # unmirrored capture so left/right hands match training; render() mirrors for display
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", scale=1,
                           webcam_options=gr.WebcamOptions(mirror=False))
            out = gr.Image(label="What the model sees", scale=2)
        with gr.Row():
            sign = gr.Button("Start / translate", variant="primary")
            prev = gr.Button("◀ previous word")
            nxt = gr.Button("next word ▶")
        cam.stream(step, cam, out, stream_every=0.05, time_limit=None, concurrency_limit=1)
        sign.click(session.toggle)
        prev.click(lambda: session.select(-1))
        nxt.click(lambda: session.select(1))
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
