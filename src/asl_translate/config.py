"""Load configs/config.yaml, optionally resolved for one approach, and set seeds."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"


def load_config(approach: str | None = None, path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(os.environ.get("ASL_DATA_ROOT", cfg["data"]["root"]))
    root = root if root.is_absolute() else REPO_ROOT / root
    # CCDS data layout: raw is never edited, interim is cached, processed is what models read.
    cfg["data"]["root"] = root
    cfg["data"]["raw"] = root / "raw"
    cfg["data"]["interim"] = root / "interim"
    cfg["data"]["processed"] = root / "processed"
    if approach is None:
        return cfg
    if approach not in cfg["approaches"]:
        raise ValueError(f"Unknown approach {approach!r}; choose from {list(cfg['approaches'])}")
    spec = cfg["approaches"][approach]
    return {
        "name": approach,
        "data": cfg["data"],
        "seeds": cfg["seeds"],
        "training": cfg["training"],
        "metrics": cfg["metrics"],
        "run_dir": REPO_ROOT / "models" / approach,
        **spec,
        "encoder": {"name": spec["encoder"], **cfg["encoders"][spec["encoder"]]},
        "decoder": {"name": spec["decoder"], **cfg["decoders"][spec["decoder"]]},
    }


def set_seed(seed: int) -> None:
    """Seed every source of randomness we use. Called at the start of every entry point."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

def get_device(preference: str = "auto"):
    """Return a torch.device: `preference` if given, else mps, then cuda, then cpu."""
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")