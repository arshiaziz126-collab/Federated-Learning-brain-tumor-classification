"""Read the files written by federated_train.py. The UI never computes results itself."""
from __future__ import annotations

import streamlit as st

import config
from backend.io_utils import load_json

KEYS = ["metrics", "history", "clients", "communication", "experiment", "federated", "centralized",
        "privacy", "partition", "dataset_report"]


@st.cache_data(show_spinner=False)
def _read(path_str: str, mtime: float):
    return load_json(path_str)


def _load(key: str):
    path = config.OUT[key]
    if not path.exists():
        return None
    return _read(str(path), path.stat().st_mtime)


def load_results() -> dict:
    """Dict with one entry per output file (None when the file does not exist yet)."""
    res = {k: _load(k) for k in KEYS}
    cfg_path = config.MODEL_CONFIG_PATH
    res["model_config"] = _read(str(cfg_path), cfg_path.stat().st_mtime) if cfg_path.exists() else None
    return res


def is_trained(res: dict) -> bool:
    return bool(res.get("metrics") and res.get("history") and config.GLOBAL_MODEL_PATH.exists())
