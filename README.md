# ASL Live Video Translation

ASL-to-English sentence translation from video, developed toward real-time webcam use. The current
baseline uses a pose GRU encoder and Transformer decoder.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync
uv run python -m asl_translate.models.train --smoke
```

The smoke run downloads public annotations and a small feature subset, verifies their checksums,
trains on 50 sentences, and evaluates on 10 validation sentences. Internet access is needed; a
Hugging Face account is not required.

## Key commands

```bash
# Train and evaluate on the full MediaPipe train and validation partitions
uv run python -m asl_translate.models.train --device cpu

# Download and train with another published pose source
uv run python -m asl_translate.data download --pose_estimator rtmw --splits train val
uv run python -m asl_translate.models.train --pose_estimator rtmw

# Run tests and lint
uv run pytest
uv run ruff check .
```

Set `ASL_DATA_ROOT` to store data elsewhere. Configuration and dataset revisions are in
[configs/config.yaml](configs/config.yaml); checksums are in [data/manifest.json](data/manifest.json).
Training records run details in [reports/results.jsonl](reports/results.jsonl). CPU runs are
reproducible; results on accelerated devices can vary slightly.

## Data

The processed [How2Sign feature dataset](https://huggingface.co/datasets/ASL-Live-Video-Translation/how2sign-realigned-mediapipe)
is based on full frontal videos and realigned sentence timings. How2Sign is licensed for
non-commercial research use (CC BY-NC 4.0).

Load the Parquet features with Hugging Face Datasets (`uv sync --group parquet`):

```python
from datasets import load_dataset

dataset = load_dataset(
    "ASL-Live-Video-Translation/how2sign-realigned-mediapipe",
    "rtmw",  # or "mediapipe", "vitpose", or "dwpose"
)
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]  # not available for mediapipe
```

Pass `split="train"`, `"validation"`, or `"test"` to load a single split. Rows contain a sentence
ID, landmark coordinates, and detection flags; ViTPose, RTMW, and DWPose also include confidence
scores. Translation text is not included. The pose sources use different landmark layouts, so
train a separate model for each.

To generate features, prepare the data and run extraction (the test split is opt-in):

```bash
export ASL_DATA_ROOT=/path/to/data  # optional; may be on an external drive
uv run python -m asl_translate.data
uv run python -m asl_translate.data videos --splits train val test
uv run python -m asl_translate.features extract --pose_estimator mediapipe vitpose rtmw dwpose --splits train val
```

The `videos` command caches the pinned source ZIPs under `raw/`; repeat it to resume downloads.
Extraction uses each cached video for all selected estimators, running them sequentially and
skipping features already saved for each. Without cached ZIPs, videos are streamed. The default
split lists are still the filtered baseline subset. Use `--offset` and `--limit` for chunks;
run again without them to record the completed split.
For RTMW or DWPose, install `uv sync --group rtm` and place `yolox-tiny.onnx` and
`rtmw.onnx` or `dwpose.onnx` in `models/` (or set `ASL_RTM_MODEL_DIR`). The model files are
`end2end.onnx` in the official [detector](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip),
[RTMW](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip), and
[DWPose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.zip) archives.
For ViTPose, install `uv sync --group vitpose`, download the [pose and detector weights](https://huggingface.co/JunkyByte/easy_ViTPose/tree/e83805274e89428969355ec4afffcbc413e79188),
and set `ASL_VITPOSE_MODEL` and `ASL_VITPOSE_DETECTOR` to their paths. OpenPose requires its
[executable and models](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md),
configured with `ASL_OPENPOSE_BIN` and `ASL_OPENPOSE_MODELS`.

Completed features can be published with `uv run python -m asl_translate.data upload --pose_estimator rtmw`.
Existing non-MediaPipe extractions cannot be resumed if their recorded model, runtime, or code
fingerprints change.

## Results

| Run | Train | Validation | BLEU | Mean confidence | Device | Commit |
|---|---:|---:|---:|---:|---|---|
| Baseline smoke | 50 | 10 | 0.00 | 0.09 | CPU | `e60575c` |
| Baseline full, seed 0 | 1,994 | 298 | 1.29 | 0.46 | CPU | `ad04be0` |

These are recorded MediaPipe baseline runs, not results for the other pose sources. The full model
produces mostly unrelated translations despite relatively high confidence; further evaluation and
model improvements are needed.

## Limitations

The dataset shares signers across splits, and some sentence text repeats across splits. Missing
source videos and out-of-range segments leave gaps in the extracted features. The vocabulary filter
and selection of whole videos also limit the training data's variety. Training evaluates validation;
the test split is reserved for final evaluation.
