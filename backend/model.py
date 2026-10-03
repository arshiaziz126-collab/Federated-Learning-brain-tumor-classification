"""Model definition.

To try a different network, write another ``build_*`` function, add it to
``MODEL_REGISTRY`` and pass ``--arch <name>`` to ``federated_train.py``.
The rest of the pipeline only relies on three things:

* the model takes raw 0-255 pixels ``(H, W, C)``,
* it ends in a softmax over the classes,
* ``get_parameters`` / ``set_parameters`` (below) define what a hospital sends.

Group normalisation is used instead of batch normalisation on purpose: batch
statistics are computed per hospital and average poorly across non-IID clients.
"""
from __future__ import annotations

import keras
from keras import layers


def build_cnn(input_shape, num_classes: int, augment: bool = True, name: str = "neurofed_cnn"):
    inputs = keras.Input(shape=input_shape, name="mri")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    if augment:   # only active during training
        x = layers.RandomRotation(0.04, name="aug_rotate")(x)
        x = layers.RandomZoom(0.08, name="aug_zoom")(x)
        x = layers.RandomTranslation(0.04, 0.04, name="aug_shift")(x)

    for i, filters in enumerate((16, 32, 64, 128), start=1):
        for j in (1, 2) if i > 1 else (1,):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}_{j}")(x)
            x = layers.GroupNormalization(groups=8, name=f"gn{i}_{j}")(x)
            x = layers.ReLU(name=f"relu{i}_{j}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return keras.Model(inputs, outputs, name=name)


MODEL_REGISTRY = {"cnn": build_cnn}


def build_model(arch: str, input_shape, num_classes: int, augment: bool = True):
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](input_shape, num_classes, augment=augment)


def compile_model(model, learning_rate: float):
    """Fresh optimiser every call: a hospital starts each round with clean optimiser state."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- what is exchanged between hospitals and server --------------------------
def get_parameters(model) -> list:
    """Trainable parameters as a list of numpy arrays (the model 'update')."""
    return [v.numpy() for v in model.trainable_variables]


def set_parameters(model, params: list) -> None:
    variables = model.trainable_variables
    if len(variables) != len(params):
        raise ValueError(f"Parameter count mismatch: model has {len(variables)}, received {len(params)}")
    for var, value in zip(variables, params):
        var.assign(value)


def count_parameters(model) -> int:
    return int(sum(v.numpy().size for v in model.trainable_variables))


def architecture_summary(model) -> dict:
    """Compact, JSON-friendly description of the network (shown on the About page)."""
    layer_rows = []
    for layer in model.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        layer_rows.append({"name": layer.name, "type": layer.__class__.__name__,
                           "output_shape": str(getattr(layer, "output", None).shape[1:]) if hasattr(layer, "output") else "",
                           "params": int(layer.count_params())})
    return {"name": model.name, "trainable_parameters": count_parameters(model), "layers": layer_rows}
