"""Load configs/config.yaml, optionally resolved for one approach."""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config(approach: str | None = None, path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(os.environ.get("ASL_DATA_ROOT", cfg["data"]["root"]))
    cfg["data"]["root"] = root if root.is_absolute() else REPO_ROOT / root
    if approach is None:
        return cfg
    if approach not in cfg["approaches"]:
        raise ValueError(f"Unknown approach {approach!r}; choose from {list(cfg['approaches'])}")
    spec = cfg["approaches"][approach]
    return {
        "name": approach,
        "data": cfg["data"],
        "seeds": cfg["seeds"],
        "run_dir": REPO_ROOT / "runs" / approach,
        **spec,
        "encoder": {"name": spec["encoder"], **cfg["encoders"][spec["encoder"]]},
        "decoder": {"name": spec["decoder"], **cfg["decoders"][spec["decoder"]]},
    }
