"""Evaluation: predictions -> accuracy, loss, precision / recall / F1, confusion matrix."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support



def predict_probs(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Softmax probabilities for a uint8 image array, in inference mode."""
    out = [model(X[i:i + batch_size].astype("float32"), training=False).numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out) if out else np.zeros((0, model.output_shape[-1]))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list) -> dict:
    """All metrics are computed from predictions - nothing is hardcoded."""
    labels = list(range(len(class_names)))
    y_pred = probs.argmax(axis=1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    picked = np.clip(probs[np.arange(len(y_true)), y_true], 1e-7, 1.0)
    return {
        "num_samples": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "loss": float(-np.log(picked).mean()),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "per_class": {
            name: {"precision": float(p_c[i]), "recall": float(r_c[i]), "f1": float(f_c[i]), "support": int(s_c[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
    }


def evaluate_model(model, X: np.ndarray, y: np.ndarray, class_names: list, batch_size: int = 64) -> dict:
    return compute_metrics(y, predict_probs(model, X, batch_size), class_names)


def wilson_interval(correct: int, n: int, z: float = 1.96):
    """95 % Wilson score interval for an accuracy measured on ``n`` test images."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
