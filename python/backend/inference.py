"""Inference with the trained global model."""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

import config
from backend.io_utils import load_json
from backend.preprocessing import preprocess_pil


def model_available() -> bool:
    return config.GLOBAL_MODEL_PATH.exists() and config.MODEL_CONFIG_PATH.exists()


def centralized_available() -> bool:
    return config.CENTRALIZED_MODEL_PATH.exists() and config.MODEL_CONFIG_PATH.exists()


def load_global_model():
    """Load the saved global model and the config it was trained with."""
    import keras
    cfg = load_json(config.MODEL_CONFIG_PATH)
    model = keras.saving.load_model(config.GLOBAL_MODEL_PATH, compile=False)
    return model, cfg


def load_centralized_model():
    """Centralized baseline (same architecture and preprocessing), used for comparison on the Diagnose page."""
    import keras
    return keras.saving.load_model(config.CENTRALIZED_MODEL_PATH, compile=False)


def _negative_class(class_names: list):
    """Best-effort: class name that denotes 'no tumour' (matched by wording, never assumed)."""
    for name in class_names:
        if re.search(r"\b(no|non|normal|healthy|negative|notumou?r)\b|notumou?r", name.lower().replace("_", " ")):
            return name
    return None


def predict(model, cfg: dict, image: Image.Image) -> dict:
    x = preprocess_pil(image, cfg["img_size"], cfg["grayscale"])
    out = _predict_array(model, cfg, x)
    out["original_size"] = list(image.size)
    return out


def _predict_array(model, cfg: dict, x) -> dict:
    probs = model(x, training=False).numpy()[0]
    names = cfg["class_names"]
    order = np.argsort(-probs)
    top, second = int(order[0]), int(order[1]) if len(order) > 1 else None
    confidence = float(probs[top])
    margin = float(probs[top] - probs[second]) if second is not None else confidence
    negative = _negative_class(names)
    low = confidence < config.LOW_CONFIDENCE_THRESHOLD or margin < config.LOW_MARGIN_THRESHOLD
    return {
        "predicted_class": names[top],
        "confidence": confidence,
        "margin_to_runner_up": margin,
        "runner_up": names[second] if second is not None else None,
        "probabilities": {names[i]: float(probs[i]) for i in order},
        "low_confidence": bool(low),
        "negative_class": negative,
        "screen_positive": (names[top] != negative) if negative else None,
        "input_size": cfg["img_size"],
        "grayscale": cfg["grayscale"],
        "class_index": top,
    }
