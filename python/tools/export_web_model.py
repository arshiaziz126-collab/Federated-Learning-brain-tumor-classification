"""Export the trained Keras models for the browser version (web/).

Writes, for each model, a JSON manifest and a little-endian float32 .bin file with every
weight in layer order. web/js/model.js runs the same network in plain JavaScript.

    python tools/export_web_model.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SUPPORTED = {"Conv2D", "GroupNormalization", "ReLU", "MaxPooling2D", "GlobalAveragePooling2D", "Dense"}
SKIP = {"InputLayer", "Rescaling", "RandomRotation", "RandomZoom", "RandomTranslation", "Dropout",
        "RandomFlip", "RandomContrast"}


def export(model_path: Path, name: str) -> dict:
    import keras
    model = keras.saving.load_model(model_path, compile=False)
    layers, blobs, offset = [], [], 0
    for layer in model.layers:
        kind = layer.__class__.__name__
        if kind in SKIP:
            if kind == "Rescaling":
                layers.append({"type": "Rescaling", "scale": float(layer.scale), "offset": float(layer.offset)})
            continue
        if kind not in SUPPORTED:
            raise ValueError(f"Layer {layer.name} ({kind}) is not supported by web/js/model.js")
        cfg = layer.get_config()
        entry = {"type": kind, "name": layer.name}
        if kind == "Conv2D":
            if cfg["padding"] != "same" or tuple(cfg["strides"]) != (1, 1) or cfg["activation"] != "linear":
                raise ValueError(f"{layer.name}: only 'same' padding, stride 1, linear activation supported")
            entry["useBias"] = bool(cfg["use_bias"])
        if kind == "GroupNormalization":
            entry.update(groups=cfg["groups"], epsilon=cfg["epsilon"])
        if kind == "MaxPooling2D":
            if tuple(cfg["pool_size"]) != (2, 2) or tuple(cfg["strides"]) != (2, 2) or cfg["padding"] != "valid":
                raise ValueError(f"{layer.name}: only 2x2/2 valid pooling supported")
        if kind == "Dense":
            entry["activation"] = cfg["activation"]
        entry["weights"] = []
        for w in layer.get_weights():
            a = np.asarray(w, dtype="<f4")
            entry["weights"].append({"shape": list(a.shape), "offset": offset, "length": int(a.size)})
            blobs.append(a.ravel())
            offset += a.size
        layers.append(entry)
    data = np.concatenate(blobs).astype("<f4")
    (WEB / "models" / f"{name}.bin").write_bytes(data.tobytes())
    manifest = {"inputShape": list(model.input_shape[1:]), "layers": layers, "totalFloats": int(offset)}
    (WEB / "models" / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _strip_local_paths(obj):
    """Drop absolute paths of the training machine before publishing the results on the web."""
    if isinstance(obj, dict):
        return {k: _strip_local_paths(v) for k, v in obj.items()
                if not (isinstance(v, str) and (":\\" in v or v.startswith(("/home/", "/Users/", "/tmp/"))))}
    if isinstance(obj, list):
        return [_strip_local_paths(v) for v in obj]
    return obj


def _copy_json(src: Path, dst: Path):
    dst.write_text(json.dumps(_strip_local_paths(json.loads(src.read_text())), indent=1))


def main():
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    export(ROOT / "models" / "global_model.keras", "federated")
    if (ROOT / "models" / "centralized_model.keras").exists():
        export(ROOT / "models" / "centralized_model.keras", "centralized")
    # results the dashboard pages read
    for f in ["model_config.json"]:
        _copy_json(ROOT / "models" / f, WEB / "data" / f)
    for f in ["metrics.json", "federated_metrics.json", "centralized_metrics.json", "federated_history.json",
              "client_metrics.json", "communication_metrics.json", "privacy_report.json", "dataset_report.json",
              "experiment_results.json"]:
        p = ROOT / "outputs" / f
        if p.exists():
            _copy_json(p, WEB / "data" / f)
    fp = ROOT / "results_fedprox" / "outputs"
    if fp.exists():
        for f in ["federated_metrics.json", "federated_history.json", "client_metrics.json"]:
            _copy_json(fp / f, WEB / "data" / f"fedprox_{f}")
    print("Exported models and results to web/")


if __name__ == "__main__":
    main()
