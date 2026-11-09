# How2Sign Data License
This project uses the How2Sign dataset (https://how2sign.github.io/).

How2Sign is © its authors and released under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/

- Use is non-commercial (academic course project, CS6170).
- Files are fetched from an unofficial Hugging Face mirror, `aipieces/How2Sign`,
  pinned to the revision in `configs/config.yaml`. The original license terms still apply.

## Changes made to the original data

Files committed under `data/processed/` are derived from How2Sign:

- `splits/`: sentence IDs from the official train/val/test splits, filtered to
  sentences covered by a 1,000-word vocabulary built from the train split.
- `smoke/`: MediaPipe Holistic pose and hand landmarks extracted from a small
  number of clips. No video frames or face landmarks are included.

These derived files are shared under the same CC BY-NC 4.0 terms.
Please cite the original authors (see `how2sign_citation.bib`).