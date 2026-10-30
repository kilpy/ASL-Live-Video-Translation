# ASL Live Video Translation

Real-time ASL-to-text translation from webcam video (CS6170 capstone).

## Commands
- `uv sync` installs deps (Python 3.12, pinned in `.python-version`)
- `uv run pytest` runs tests
- `uv run ruff check .` lints
- `uv run python -m asl_translate.capture` webcam smoke test

- `uv run python -m asl_translate.download <annotations|clips> [--split val]` fetches How2Sign
- `uv run python -m asl_translate.eda` writes `reports/eda/`
- `uv run python -m asl_translate.samples` shows val clips and their MediaPipe landmarks in `reports/samples/` (git-ignored: signers' faces)

## Layout
- `src/asl_translate/` package code
- `configs/config.yaml` shared `data` block, pluggable `encoders`/`decoders`, and the CLASSICAL, BASELINE and DEEP approaches that pair them
- `tests/` pytest tests
- `data/`, `models/`, `runs/` git-ignored datasets, weights and experiment outputs
- `Submissions Context/` course submissions; the Experiment Card in `Architecture_Seminar.pdf` is the spec

## Notes
- The HMM fallback in the Experiment Card is dropped. The three architectures are CLASSICAL (RNN decoder), BASELINE (transformer decoder) and DEEP.
- Dev machine: Windows 11, RTX 3080. PyTorch is not yet a dependency; add it with the CUDA index when the model work starts.
- System Python is 3.14, which MediaPipe/PyTorch don't support. Always go through `uv run`.
