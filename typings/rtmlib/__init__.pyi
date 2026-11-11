"""Project-used rtmlib interfaces with the human detector's actual return type.

rtmlib's YOLOX annotation describes its multiclass mode, but this project uses
the default human mode, which returns a NumPy array of bounding boxes.
"""
from typing import Any

import numpy as np

class YOLOX:
    def __init__(self, onnx_model: str, model_input_size: tuple[int, int] = ...,
                 *, backend: str = ..., device: str = ...) -> None: ...
    def __call__(self, image: np.ndarray) -> np.ndarray: ...

class RTMPose:
    session: Any
    def __init__(self, onnx_model: str, model_input_size: tuple[int, int] = ...,
                 *, backend: str = ..., device: str = ...) -> None: ...
    def __call__(self, image: np.ndarray, bboxes: np.ndarray = ...) -> tuple[np.ndarray, np.ndarray]: ...
