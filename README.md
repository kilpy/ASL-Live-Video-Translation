# ASL Live Video Translation

CS6170 Capstone: ASL-to-English sentence translation from video, built toward real-time webcam use.
We compare three approaches on the same data, splits, preprocessing, and metric (Experiment Card,
`deliverables/Architecture_Seminar.pdf`). This repository currently runs the **BASELINE**.

## Run the baseline (smoke test)

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the pinned packages from
`uv.lock`. From the repository root:

```bash
uv sync
uv run python -m asl_translate.models.train --smoke
```

This downloads the re-aligned annotations (about 6 MB) and the smoke subset (under 1 MB) from public
Hugging Face datasets, so it needs internet but no account or token. It then checks every file
against `data/manifest.json`, trains for 3 epochs on 50 sentences on CPU, evaluates on 10 val
sentences, and appends one line to `reports/results.jsonl`. It takes under a minute, and running it
twice gives the same numbers.

The full run trains on every extracted train sentence and evaluates on val:

```bash
uv run python -m asl_translate.models.train --device cpu   # exactly reproducible; omit for mps/cuda
```

Tests and lint: `uv run pytest` and `uv run ruff check .`

## Reproducibility

| What | Where |
|---|---|
| Environment | `pyproject.toml` + `uv.lock` (Python 3.12, every package pinned) |
| Seeds | `set_seed()` in `config.py`, called at the start of `train.py`; seeds in `configs/config.yaml` |
| Source data | `aipieces/How2Sign` mirror, pinned to one revision in `configs/config.yaml` |
| Processed data | our dataset, pinned to one revision in `configs/config.yaml` |
| Data hashes | `data/manifest.json` (committed); `verify()` checks it every time data is read |
| Split | `make_splits()` in `dataset.py`; split files in `data/processed/splits/` |
| Results | `reports/results.jsonl`: metric, commit, dirty flag, config hash, data hash, device |

The smoke test is exactly reproducible on CPU. Full runs on Apple MPS can differ slightly between
repeats because some MPS kernels (for example the embedding backward pass) are not deterministic;
the comparison table reports the spread across 3 seeds for this reason.

## Results

| Approach | Run | Train | Val | BLEU | Mean confidence | Device | Commit |
|---|---|---|---|---|---|---|---|
| BASELINE | Smoke | 50 | 10 | 0.00 | 0.09 | CPU | `011e59b` |
| BASELINE | Full, seed 0 | 1,994 | 298 | 1.29 | 0.46 | CPU | `ad04be0` |

Every number traces to a line in `reports/results.jsonl` with its commit and data hash. The BASELINE
produces fluent instructional English that is mostly unrelated to the signing: the decoder learns
the language of How2Sign, but the pose encoder contributes little. Confidence (0.47) far exceeds
accuracy, so the model is badly calibrated. Next: val-loss early stopping, BLEURT, seeds 1 and 2,
and pretrained SHuBERT features (DEEP).


## Data

[How2Sign](https://how2sign.github.io/) (CC BY-NC 4.0, research use only; license and citation in
`references/how2sign/`). The official links hit Google Drive quotas, so we use the unofficial
[`aipieces/How2Sign`](https://huggingface.co/datasets/aipieces/How2Sign) mirror.

We cut every sentence from the **full frontal videos at the manually re-aligned timestamps**. The
pre-cut clips use the original speech timing, so their boundaries do not match the signing.

The full pipeline, which the smoke test does not need:

```bash
uv run python -m asl_translate.dataset                  # annotations, video index, splits
uv run python -m asl_translate.features extract --workers 4   # landmarks for train and val
uv run python -m asl_translate.features smoke           # smoke subset
uv run python -m asl_translate.upload                   # publish; prints the revision to pin
```

Processed data is public at
[`ASL-Live-Video-Translation/how2sign-realigned-mediapipe`](https://huggingface.co/datasets/ASL-Live-Video-Translation/how2sign-realigned-mediapipe).

**Split.** How2Sign's official train, val, and test splits, filtered to sentences whose words are
all in a 1,000-word vocabulary built from train only. Train keeps whole videos, densest first, up
to 2,000 sentences; val (299 sentences, 84 videos) and test (427 sentences, 117 videos) keep every
covered sentence. Test is never read by `train.py`; it is used once, at the end.

**Features.** MediaPipe Holistic at 15 fps (matching SHuBERT): 7 upper-body points and both hands,
x and y, with missing frames interpolated, plus per-frame detection flags. Each frame is centered on
the shoulders and scaled by shoulder width.

## Leakage risks

- **Shared signers.** The same signers appear in every split, so scores may overstate performance on
  new signers. This is How2Sign's design; we name it rather than fix it.
- **Repeated sentences.** Some sentence texts appear in more than one split (see
  `reports/eda/summary.json`). No source video is shared: `make_splits()` stops if one is.
- **Vocabulary and normalization.** The vocabulary comes from train only. Normalization is computed
  per sentence from its own frames, so no statistics cross splits.

## Known limitations

- 139 train videos are missing from the mirror; their sentences are skipped.
- 5 segments under half a second produced no frames, likely because they fall at or past the end of
  the video, and are excluded.
- Train comes from relatively few videos to limit downloads, which reduces topic and signer variety.
- The 1,000-word filter keeps under a fifth of How2Sign's sentences and favors short ones.
- BLEU is computed on lowercased, tokenized text. BLEURT, the planned primary metric, is the next step.

## Progress on the other approaches

- **CLASSICAL** (pose GRU + RNN decoder): shares this data and training loop; needs an RNN decoder
  module plugged in through `configs/config.yaml`.
- **DEEP** (DINOv2 + SHuBERT + ByT5): full landmarks, including face, are kept in `data/interim/` for
  the SHuBERT-style hand and face crops; feature extraction and fine-tuning are not started.

## Repository layout

Follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/):
`data/` (raw, interim, processed; git-ignored except the manifest), `models/` (checkpoints,
git-ignored), `reports/` (EDA and results), `references/` (license, citation), `deliverables/`
(course submissions), `src/asl_translate/` (code; `models/` holds the model code and `train.py`).