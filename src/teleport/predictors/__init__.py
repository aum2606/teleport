from __future__ import annotations

from typing import Callable

from teleport.predictors.base import Predictor
from teleport.predictors.order0 import Order0Predictor
from teleport.predictors.ppm import PPMPredictor
from teleport.predictors.rnn import PretrainedRNNPredictor, RNNPredictor
from teleport.predictors.uniform import UniformPredictor

PREDICTORS: dict[str, Callable[[], Predictor]] = {
    "uniform": UniformPredictor,
    "order0": Order0Predictor,
    "ppm2": lambda: PPMPredictor(max_order=2),
    "ppm3": lambda: PPMPredictor(max_order=3),
    "rnn": RNNPredictor,
    "rnn_pretrained": PretrainedRNNPredictor,
}

__all__ = [
    "Predictor",
    "UniformPredictor",
    "Order0Predictor",
    "PPMPredictor",
    "RNNPredictor",
    "PretrainedRNNPredictor",
    "PREDICTORS",
]
