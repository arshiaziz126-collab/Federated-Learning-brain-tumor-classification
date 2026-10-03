"""A hospital (federated client).

A ``HospitalClient`` owns its images and its own copy of the model. The only
things it accepts from / returns to the outside world are ``bytes`` messages
built by ``backend.communication.pack``: model parameters plus scalar metrics.
No image or image-derived array ever leaves this class.
"""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from backend import communication
from backend.dataset import load_client_arrays
from backend.evaluation import predict_probs
from backend.model import build_model, compile_model, get_parameters, set_parameters


def make_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False, seed: int | None = None):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


class HospitalClient:
    def __init__(self, entry: dict, class_names: list, model_cfg: dict, seed: int):
        self.client_id = entry["client_id"]
        self.name = entry["name"]
        self.seed = seed
        self.class_names = class_names
        size, gray = model_cfg["img_size"], model_cfg["grayscale"]
        self._X_train, self._y_train = load_client_arrays(entry, "train", class_names, size, gray)
        self._X_val, self._y_val = load_client_arrays(entry, "val", class_names, size, gray)
        self.model = build_model(model_cfg["arch"], model_cfg["input_shape"], len(class_names))

    # --- public facts the hospital reports about itself (no image data) ------
    @property
    def num_train(self) -> int:
        return int(len(self._y_train))

    @property
    def num_val(self) -> int:
        return int(len(self._y_val))

    def class_counts(self, split: str = "train") -> dict:
        y = self._y_train if split == "train" else self._y_val
        return {c: int((y == i).sum()) for i, c in enumerate(self.class_names)}

    def train_arrays(self):
        """Used ONLY by the centralized baseline, which deliberately pools all data."""
        return self._X_train, self._y_train

    def val_arrays(self):
        """Used ONLY by the centralized baseline, which deliberately pools all data."""
        return self._X_val, self._y_val

    # --- federated protocol --------------------------------------------------
    def local_update(self, global_payload: bytes, *, round_idx: int, epochs: int, batch_size: int,
                     learning_rate: float, mu: float = 0.0) -> bytes:
        """Receive global model -> train locally -> return parameters + scalar metrics.

        mu = 0 is plain FedAvg local training. mu > 0 is FedProx (Li et al., 2020): the local
        loss gets a proximal term (mu / 2) * ||w - w_global||^2 that keeps the hospital's model
        close to the global one and limits client drift under non-IID data.
        """
        params, _ = communication.unpack(global_payload)
        set_parameters(self.model, params)

        t0 = time.perf_counter()
        ds = make_dataset(self._X_train, self._y_train, batch_size, shuffle=True,
                          seed=self.seed * 1000 + round_idx)
        if mu > 0:
            train_loss, train_acc = self._fit_fedprox(ds, params, epochs, learning_rate, mu)
        else:
            compile_model(self.model, learning_rate)
            hist = self.model.fit(ds, epochs=epochs, shuffle=False, verbose=0)
            train_loss, train_acc = float(hist.history["loss"][-1]), float(hist.history["accuracy"][-1])
        seconds = time.perf_counter() - t0

        val = self._score()
        meta = {
            "client_id": self.client_id, "round": round_idx, "num_samples": self.num_train,
            "epochs": epochs,
            "train_loss": train_loss, "train_accuracy": train_acc, "mu": float(mu),
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "train_seconds": float(seconds),
        }
        return communication.pack(get_parameters(self.model), meta)

    def _fit_fedprox(self, ds, global_params: list, epochs: int, learning_rate: float, mu: float):
        """Local training with the FedProx proximal term. Returns last-epoch (loss, accuracy)."""
        import keras
        model = self.model
        opt = keras.optimizers.Adam(learning_rate)          # fresh optimiser every round, as for FedAvg
        anchors = [tf.constant(p) for p in global_params]    # same order as trainable_variables
        ce = keras.losses.SparseCategoricalCrossentropy()

        @tf.function
        def step(x, y):
            with tf.GradientTape() as tape:
                probs = model(x, training=True)
                task = ce(y, probs)
                prox = tf.add_n([tf.reduce_sum(tf.square(w - a)) for w, a in zip(model.trainable_variables, anchors)])
                loss = task + 0.5 * mu * prox
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(probs, -1), tf.cast(y, tf.int64)), tf.float32))
            return task, correct, tf.cast(tf.shape(y)[0], tf.float32)

        loss_sum = correct_sum = n_sum = 0.0
        for _ in range(epochs):
            loss_sum = correct_sum = n_sum = 0.0
            for x, y in ds:
                task, correct, n = step(x, y)
                loss_sum += float(task) * float(n)
                correct_sum += float(correct)
                n_sum += float(n)
        return loss_sum / max(n_sum, 1.0), correct_sum / max(n_sum, 1.0)

    def _score(self) -> dict:
        """Loss / accuracy of the current local model on the local validation split."""
        probs = predict_probs(self.model, self._X_val)
        picked = np.clip(probs[np.arange(len(self._y_val)), self._y_val], 1e-7, 1.0)
        return {"loss": float(-np.log(picked).mean()), "accuracy": float((probs.argmax(1) == self._y_val).mean()),
                "num_samples": self.num_val}

    def evaluate_global(self, global_payload: bytes) -> dict:
        """Score the received global model on this hospital's local validation data.

        Only three scalars leave the hospital: loss, accuracy and sample count.
        """
        params, _ = communication.unpack(global_payload)
        set_parameters(self.model, params)
        return self._score()
