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
| BASELINE | Full, seed 0, encoder skips padding | 1,994 | 298 | 0.67 | 0.45 | CPU | `359410b` |

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

Extraction can be interrupted and resumed: completed sentence files are skipped. To work through
bounded batches, use `--splits train --offset 0 --limit 100`, then increase `--offset` by 100 for
each batch. After all batches, run `extract --splits train` once without either option to retry any
missing sentences and record the completed split hash. Set `ASL_DATA_ROOT` to move the raw,
interim, and processed data directories; the committed manifest remains in this repository.

To inspect validation examples, run `uv run python -m asl_translate.samples` after the annotations
and video index are available. It reads up to four full validation videos from their shards, cuts
the aligned sentence ranges, and writes visualizations to `reports/samples/`.

Processed data is public at
[`ASL-Live-Video-Translation/how2sign-realigned-mediapipe`](https://huggingface.co/datasets/ASL-Live-Video-Translation/how2sign-realigned-mediapipe).

**Split.** How2Sign's official train, val, and test splits, filtered to sentences whose words are
all in a 1,000-word vocabulary built from train only. Train keeps whole videos, densest first, up
to 2,000 sentences; val (299 sentences, 84 videos) and test (427 sentences, 117 videos) keep every
covered sentence. Test is never read by `train.py`; it is used once, at the end.

**Features.** MediaPipe Holistic at 15 fps (matching SHuBERT): 7 upper-body points and both hands,
x and y, with missing frames interpolated, plus per-frame detection flags. Each frame is centered on
the shoulders and scaled by shoulder width.

### Additional pose sources

The extractor supports `--source mediapipe` (default), `openpose`, `vitpose`, `rtmw`, or `dwpose`. All use the same
sentence IDs, re-aligned timestamps, 15 fps sampling, splits, and interpolation. OpenPose stores
BODY_25 followed by 21 points per hand (67 total). ViTPose stores COCO-WholeBody's 23 body/foot
points followed by 21 points per hand (65 total); its 68 detailed face points are omitted. RTMW
and DWPose use the same 65-point COCO-WholeBody layout. The additional sources
store normalized x/y, per-point `confidence`, and per-point `detected` flags in each NPZ. The
existing MediaPipe files remain at `landmarks/{split}`; new files go in
`landmarks/{source}/{split}`. Native body keypoints can include eyes and ears.

RTMW and DWPose use YOLOX-tiny for CPU person detection and Core ML for pose inference on macOS;
some pose operations may also run on CPU. Install the optional runtime with
`uv sync --group rtm` (include `--group vitpose` to keep that optional runtime installed too).
Put the ONNX files named `yolox-tiny.onnx`, `rtmw.onnx`, and `dwpose.onnx` in `models/`, or
set `ASL_RTM_MODEL_DIR` to their directory. The checkpoints are the `end2end.onnx` files in the
official [YOLOX-tiny](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip),
[RTMW lightweight](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip), and
[DWPose small](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.zip) archives.
For example, run `uv run --group rtm python -m asl_translate.features extract --source rtmw --splits val --limit 1`.

Before extracting ViTPose, install the optional pinned runtime and fetch the two checkpoints:

```bash
uv sync --group vitpose
mkdir -p models
curl -fL -o models/vitpose-s-wholebody.pth https://huggingface.co/JunkyByte/easy_ViTPose/resolve/e83805274e89428969355ec4afffcbc413e79188/torch/wholebody/vitpose-s-wholebody.pth
curl -fL -o models/yolov8n.pt https://huggingface.co/JunkyByte/easy_ViTPose/resolve/e83805274e89428969355ec4afffcbc413e79188/yolov8/yolov8n.pt
export ASL_VITPOSE_MODEL="$PWD/models/vitpose-s-wholebody.pth"
export ASL_VITPOSE_DETECTOR="$PWD/models/yolov8n.pt"
uv run --group vitpose python -m asl_translate.features extract --source vitpose --splits val --limit 1
```

ViTPose automatically selects MPS when available; `ASL_VITPOSE_DEVICE=cpu` overrides it. The
YOLO person detector runs before ViTPose. OpenPose needs a separately built official executable
and its BODY_25 and hand models; set `ASL_OPENPOSE_BIN` to the executable and
`ASL_OPENPOSE_MODELS` to its model directory. Its hand detector runs after body detection. Consult
the [OpenPose installation instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md)
and its non-commercial research license. A CPU-only Apple Silicon build was checked on this Mac;
the Toshiba installation uses the [Apple Silicon OpenPose/Caffe fork](https://github.com/franzcrs/openpose-with-caffe-for-MacM1/tree/16ca125687a08d2aa31449aa2945e92f144848d4)
at commit `16ca125687a08d2aa31449aa2945e92f144848d4`. Configure it with:

```bash
export ASL_OPENPOSE_BIN=/Volumes/Toshiba/temp/asl-openpose-runtime/build/examples/openpose/openpose.bin
export ASL_OPENPOSE_MODELS=/Volumes/Toshiba/temp/asl-openpose-runtime/source/models
uv run python -m asl_translate.features extract --source openpose --splits val --limit 1
```

The drive must be mounted at the same path for this build's linked libraries to work.

For a full run, remove `--limit 1`; the default splits are train and val, and test is opt-in with
`--splits test`. Start with one worker on an 18 GB Mac. Each new source records model checksums and
runtime details in `landmarks/{source}/metadata.json`; a resume refuses changed settings. To keep
the raw videos and intermediate files on an external drive, set `ASL_DATA_ROOT` before initializing
the dataset. Existing annotations, split files, and the video index must be available under that data
root or rebuilt with `python -m asl_translate.dataset`.

After a full source extraction, `features smoke --source vitpose` creates the smoke subset, and
`upload --source vitpose --smoke` publishes that source when you choose to run it. The equivalent
OpenPose commands use `--source openpose`. Once the dataset revision in `configs/config.yaml` and
`data/manifest.json` include the new files, downstream users can run:

```bash
uv run python -m asl_translate.download_features --source vitpose --splits train val
uv run python -m asl_translate.download_features --source rtmw --splits train val test
uv run python -m asl_translate.download_features --source dwpose --splits test
uv run python -m asl_translate.models.train --source vitpose
```

Choose `mediapipe`, `vitpose`, `rtmw`, or `dwpose` with `--source`, and one or more
partitions with `--splits`. The downloader fetches only the selected source and partitions,
along with their split lists and source metadata. The dataset split lists include 10 train and
one validation IDs whose annotated ranges are beyond the available videos; their feature files
are absent for every source.

The same features are also published as Parquet configurations usable directly with Hugging Face
Datasets (`uv sync --group parquet`):

```python
from datasets import load_dataset

dataset = load_dataset(
    "ASL-Live-Video-Translation/how2sign-realigned-mediapipe",
    "rtmw",  # or "mediapipe", "vitpose", or "dwpose"
)

train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]
```

Choose `mediapipe`, `vitpose`, `rtmw`, or `dwpose` as the configuration. MediaPipe has no
published test split. To load just one split, pass `split="train"`, `split="validation"`, or
`split="test"` to `load_dataset`. Rows preserve NPZ `points` and `detected` arrays and include `sentence_id`;
the other three sources also have `confidence`. The rows do not contain translation text.
To regenerate the Parquet files from a local MediaPipe dataset checkout and the other sources'
processed landmarks, run:

```bash
uv run --group parquet python -m asl_translate.export_parquet \
  --mediapipe-root /path/to/dataset/landmarks \
  --other-root /path/to/data/processed/landmarks \
  --output /path/to/dataset/parquet
```

The model derives its input width from the selected features and uses source-specific shoulder
indices. Train a separate model per source: existing MediaPipe checkpoints cannot be reused for
different point layouts. Check the pinned dataset revision for availability of additional sources.

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

- **CLASSICAL** (pose GRU + GRU decoder with dot-product attention, `models/classical.py`): shares
  this data, encoder and training loop; run with `--approach CLASSICAL`. Not yet trained in full.
- **Encoder padding.** The shared pose GRU encoder now skips padded frames (before, its backward
  direction ran through the batch padding). BASELINE was rerun after this change (`359410b`, same
  data): BLEU 1.29 to 0.67. Both are near zero, and an MPS repeat of the old code alone gave 0.63,
  so seeds 1 and 2 are needed before reading anything into the difference.
- **DEEP** (DINOv2 + SHuBERT + ByT5): full landmarks, including face, are kept in `data/interim/` for
  the SHuBERT-style hand and face crops; feature extraction and fine-tuning are not started.

## Repository layout

Follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/):
`data/` (raw, interim, processed; git-ignored except the manifest), `models/` (checkpoints,
git-ignored), `reports/` (EDA and results), `references/` (license, citation), `deliverables/`
(course submissions), `src/asl_translate/` (code; `models/` holds the model code and `train.py`).
