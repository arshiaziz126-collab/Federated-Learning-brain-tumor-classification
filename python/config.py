"""Central configuration for NeuroFed AI.

Everything that is a *path*, a *default hyper-parameter* or a *file name* lives
here so the rest of the code base never hardcodes them. Nothing in this file is
a result: every number shown in the application is computed by
``federated_train.py`` and read back from ``outputs/``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- folders ---------------------------------------------------------------
DATA_DIR = ROOT / "data" / "brain_tumor_dataset"       # raw Kaggle download (you place it here)
CLIENTS_DIR = ROOT / "data" / "federated_clients"       # one isolated folder per hospital (generated)
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# --- model artefacts -------------------------------------------------------
GLOBAL_MODEL_PATH = MODELS_DIR / "global_model.keras"
CENTRALIZED_MODEL_PATH = MODELS_DIR / "centralized_model.keras"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# --- output files (all written by federated_train.py) -----------------------
OUT = {
    "dataset_report": OUTPUTS_DIR / "dataset_report.json",
    "partition": OUTPUTS_DIR / "partition_manifest.json",
    "clients": OUTPUTS_DIR / "client_metrics.json",
    "history": OUTPUTS_DIR / "federated_history.json",
    "federated": OUTPUTS_DIR / "federated_metrics.json",
    "centralized": OUTPUTS_DIR / "centralized_metrics.json",
    "communication": OUTPUTS_DIR / "communication_metrics.json",
    "privacy": OUTPUTS_DIR / "privacy_report.json",
    "experiment": OUTPUTS_DIR / "experiment_results.json",
    "metrics": OUTPUTS_DIR / "metrics.json",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Folder names that mark a pre-made split inside the dataset (case-insensitive).
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "test": "test", "testing": "test",
    "val": "val", "valid": "val", "validation": "val",
}

# --- default training hyper-parameters (all overridable from the CLI) --------
DEFAULTS = {
    "num_clients": 3,
    "rounds": 15,
    "local_epochs": 2,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "img_size": 128,
    "grayscale": True,
    "arch": "cnn",
    "partition": "dirichlet",   # "dirichlet" (non-IID) or "iid"
    "alpha": 0.5,               # Dirichlet concentration: smaller = more heterogeneous
    "val_fraction": 0.15,       # per-hospital local validation share
    "test_fraction": 0.15,      # only used when the dataset has no test folder
    "client_fraction": 1.0,     # share of hospitals selected each round
    "seed": 42,
    "dedupe": True,
    "algorithm": "fedavg",      # "fedavg" or "fedprox"
    "mu": 0.01,                 # FedProx proximal strength (ignored for fedavg)
}

# Inference: softmax probability below this is flagged as low confidence in the UI.
LOW_CONFIDENCE_THRESHOLD = 0.70
LOW_MARGIN_THRESHOLD = 0.15

MEDICAL_DISCLAIMER = (
    "This system is a research prototype and is not a clinical diagnostic tool. "
    "It requires clinical validation before real-world medical use."
)


def client_id(index: int) -> str:
    """0 -> 'hospital_a'."""
    return f"hospital_{chr(97 + index)}"


def client_name(index: int) -> str:
    """0 -> 'Hospital A'."""
    return f"Hospital {chr(65 + index)}"
