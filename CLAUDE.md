# ASL Live Video Translation

Real-time ASL-to-text translation from webcam video (CS6170 capstone).

## Commands
- `uv sync` installs deps (Python 3.12, pinned in `.python-version`)
- `uv run pytest` runs tests
- `uv run ruff check .` lints
- `uv run python -m asl_translate.capture` webcam smoke test
- `uv run python -m asl_translate.demo` live push-to-sign translation (SPACE start/stop, live preview every `--preview-every` s, a/d step through words: probability, top-3, cross-attention strip, model-input inset) with the newest `models/**/model.pt`; `--input clip.mp4 --headless` translates a file
- `uv run --group demo python -m asl_translate.app` same demo in the browser (Gradio)

- `uv run python -m asl_translate.data annotations` fetches How2Sign annotations
- `uv run python -m asl_translate.eda` writes `reports/eda/`
- DEEP (SHuBERT + ByT5) needs `--group deep`: `uv run --group deep python -m asl_translate.features extract --pose_estimator mediapipe dinov2`, then `uv run --group deep python -m asl_translate.models.train --approach DEEP --device cuda`
- `uv run python -m asl_translate.samples` shows aligned validation excerpts and their MediaPipe landmarks in `reports/samples/` (git-ignored: signers' faces)

## Layout
- `src/asl_translate/` package code
- `configs/config.yaml` shared `data` block, pluggable `encoders`/`decoders`, and the CLASSICAL, BASELINE and DEEP approaches that pair them
- `src/asl_translate/models/encoders.py` / `decoders.py` registries (any encoder pairs with any decoder); `seq2seq.py` wires them; `dino.py` is the DINOv2 crop extractor (feature source `dinov2`)
- `tests/` pytest tests
- `data/`, `models/`, `runs/` git-ignored datasets, weights and experiment outputs
- `Submissions Context/` course submissions; the Experiment Card in `Architecture_Seminar.pdf` is the spec

## Notes
- The HMM fallback in the Experiment Card is dropped. The three architectures are CLASSICAL (RNN decoder), BASELINE (transformer decoder) and DEEP.
- Dev machine: Windows 11, RTX 3080. PyTorch comes from the `cu132` index on Windows/Linux (PyPI build on macOS); train with `--device cuda`.
- System Python is 3.14, which MediaPipe/PyTorch don't support. Always go through `uv run`.
