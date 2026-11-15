# Baseline Code Walk

After our first meeting, our made the following changes:

1. Added thorough comments to the baseline code. Commit 75c5326 on seq2seq branch.
2. Sourced pose estimations from additional sources ("mediapipe", "openpose", "vitpose", "rtmw", "dwpose"), rather thann just media pipe. Commmit 29af8ec on main branch.

## Smoke Test
- To run the smoke test, run: `uv sync && uv run python -m asl_translate.models.train --smoke`
- This is defined in the readme.

## Environment
The environment is commited through the `pyproject.toml` and the `uv.lock` files.

## Seeds
The seed is automatically set to 0 when running the code, for reproducibility.

## Data Versioning
We are versioning all of our data through hugging face. This allows git style tracking of changes (with commits and dates) to our dataset and file storage, as well as manifest versions that we specify in /configs/config.yaml.

## Split Definition
We are following the how2sign dataset's predefined split for now, but the suggestion to use k-fold cross validation is currently being investigated as suggested by the professor. The splits are defined on our hugging face at: https://huggingface.co/datasets/ASL-Live-Video-Translation/how2sign-realigned-mediapipe/tree/main

## The Baseline Results
Results for the baseline are logged in /reports/results.json. Each log is associated with a commit hash, a config hash, and a data hash.

## The Smoke Test
The smoke test runs almost instantly and is system agnostic. The command listed above will reproduce the smoke test on cpu.

## Security
No environment variables, secrets, credentials, or restricted data have been uploaded into this repository.

## Reproducibility on a Fresh Machine
Reproducability was confirmed by all team members (Commit 6f77f24). Slight variations existed due to uncontrollable system differences. To resolve this we could use a containerized approach. 