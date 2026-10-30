# ASL Live Video Translation

CS6170 Capstone: real-time ASL-to-text translation from a webcam feed.

## Setup (Windows)

Requires [uv](https://docs.astral.sh/uv/). It fetches Python 3.12 automatically.

```bash
uv sync
uv run pytest
uv run python -m asl_translate.capture   # webcam smoke test
```

Lint: `uv run ruff check .`

`data/` and `models/` are git-ignored; keep datasets and weights there.

## Data

```bash
uv run python -m asl_translate.download annotations        # ~6 MB, enough for EDA
uv run python -m asl_translate.download clips --split val  # 1.7 GB frontal clips
uv run python -m asl_translate.eda                         # writes reports/eda/
```

Files come from [`aipieces/How2Sign`](https://huggingface.co/datasets/aipieces/How2Sign), an
unofficial Hugging Face mirror of [How2Sign](https://how2sign.github.io/) pinned to one revision in
`configs/config.yaml`, because Google Drive's quotas blocked the official links. The official
CC BY-NC 4.0, research-only terms still apply.
Set `ASL_DATA_ROOT` to keep data somewhere other than `data/how2sign`.

## Experiments

`configs/config.yaml` defines named `encoders` and `decoders`, and each approach pairs one of each:

| Approach | Encoder | Decoder |
|---|---|---|
| `CLASSICAL` | Pose GRU | RNN |
| `BASELINE` | Pose GRU | Transformer |
| `DEEP` | DINOv2 + SHuBERT | ByT5 |

`load_config("BASELINE")` returns the approach with its encoder and decoder blocks resolved, the
shared `data` block and its run directory, `runs/<APPROACH>/`.
