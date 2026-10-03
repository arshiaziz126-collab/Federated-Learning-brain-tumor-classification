"""Centralized baseline: all training data pooled into one place.

Uses the same images (union of the hospitals' train / validation splits), the same
architecture, optimiser, learning rate, batch size and the same held-out test set as the
federated run. The only difference is that the data is pooled.
"""
from __future__ import annotations

import time

from backend.evaluation import evaluate_model
from backend.local_training import make_dataset
from backend.model import build_model, compile_model


def train_centralized(X_train, y_train, X_val, y_val, test_data, class_names, *, arch: str, input_shape,
                      epochs: int, batch_size: int, learning_rate: float, seed: int, log=print):
    Xte, yte = test_data
    model = build_model(arch, input_shape, len(class_names))
    compile_model(model, learning_rate)

    history, best = [], {"epoch": 0, "val_acc": -1.0, "val_loss": float("inf"), "weights": None}
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        h = model.fit(make_dataset(X_train, y_train, batch_size, shuffle=True, seed=seed * 1000 + ep),
                      epochs=1, shuffle=False, verbose=0)
        val = evaluate_model(model, X_val, y_val, class_names)
        test = evaluate_model(model, Xte, yte, class_names)
        history.append({"epoch": ep, "train_loss": float(h.history["loss"][-1]),
                        "train_accuracy": float(h.history["accuracy"][-1]),
                        "val_loss": float(val["loss"]), "val_accuracy": float(val["accuracy"]),
                        "test_accuracy": test["accuracy"], "test_loss": test["loss"]})
        if val["accuracy"] > best["val_acc"] or (val["accuracy"] == best["val_acc"] and val["loss"] < best["val_loss"]):
            best = {"epoch": ep, "val_acc": float(val["accuracy"]), "val_loss": float(val["loss"]),
                    "weights": model.get_weights()}
        log(f"  epoch {ep:>2}/{epochs}  val acc {val['accuracy']:.3f}  loss {val['loss']:.3f}  |  test acc {test['accuracy']:.3f}")

    model.set_weights(best["weights"])
    final = evaluate_model(model, Xte, yte, class_names)
    return model, history, best["epoch"], final, time.perf_counter() - t0
