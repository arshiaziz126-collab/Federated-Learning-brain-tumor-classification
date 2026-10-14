"""Privacy checks and the privacy report.

What this prototype implements (and verifies here):
  * data locality   - every hospital trains from its own folder only,
  * no image transfer - server messages are validated to be parameter tensors,
  * client isolation - hospital datasets are disjoint (by file and by content hash),
  * no test leakage  - held-out test images are not present in any hospital.

What it does NOT implement (stated explicitly in the report and the UI):
  differential privacy, secure aggregation, encryption of updates, homomorphic
  encryption. Model updates can still leak information about training data.
"""
from __future__ import annotations

import numpy as np


class PrivacyViolation(RuntimeError):
    pass


def validate_update(params: list, reference_shapes: list) -> None:
    """Server-side check on an incoming message: it must look exactly like model parameters.

    Rejects anything whose tensor count/shapes differ from the global model or that
    is not floating point - e.g. an image batch would fail this check.
    """
    if len(params) != len(reference_shapes):
        raise PrivacyViolation(f"Unexpected number of tensors: {len(params)} != {len(reference_shapes)}")
    for i, (p, shape) in enumerate(zip(params, reference_shapes)):
        if tuple(p.shape) != tuple(shape):
            raise PrivacyViolation(f"Tensor {i} has shape {p.shape}, expected {tuple(shape)}")
        if not np.issubdtype(p.dtype, np.floating):
            raise PrivacyViolation(f"Tensor {i} is not floating point ({p.dtype})")


def audit_partition(manifest: dict) -> dict:
    """Verify hospital isolation from the partition manifest."""
    clients = manifest["clients"]
    hashes = {}
    for c in clients:
        hashes[c["client_id"]] = {f["md5"] for s in ("train", "val") for f in c[s]["files"]}
    ids = list(hashes)
    overlaps = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            overlaps[f"{a} & {b}"] = len(hashes[a] & hashes[b])
    test_hashes = {f["md5"] for f in manifest["test"]["files"]}
    test_overlap = {cid: len(h & test_hashes) for cid, h in hashes.items()}

    train_val_overlap = {}
    for c in clients:
        t = {f["md5"] for f in c["train"]["files"]}
        v = {f["md5"] for f in c["val"]["files"]}
        train_val_overlap[c["client_id"]] = len(t & v)

    checks = [
        {"check": "Hospital datasets are disjoint (no image content shared between hospitals)",
         "passed": all(v == 0 for v in overlaps.values()), "detail": overlaps},
        {"check": "Held-out test images are not present in any hospital",
         "passed": all(v == 0 for v in test_overlap.values()), "detail": test_overlap},
        {"check": "Local validation images are not used for local training",
         "passed": all(v == 0 for v in train_val_overlap.values()), "detail": train_val_overlap},
        {"check": "Each hospital has its own physical data folder",
         "passed": len({c["dir"] for c in clients}) == len(clients),
         "detail": {c["client_id"]: c["dir"] for c in clients}},
    ]
    return {"checks": checks, "all_passed": all(c["passed"] for c in checks)}


def build_privacy_report(manifest: dict, comm_summary: dict, payload_tensors: int,
                         payload_shapes: list, updates_validated: int) -> dict:
    audit = audit_partition(manifest)
    return {
        "principle": "Raw patient MRI data never leaves the hospital.",
        "raw_data_location": {c["name"]: c["dir"] for c in manifest["clients"]},
        "raw_data_bytes_by_hospital": comm_summary["raw_data_bytes_by_client"],
        "raw_images_per_hospital": {c["name"]: c["train"]["count"] + c["val"]["count"] for c in manifest["clients"]},
        "shared_with_server": {
            "model_parameter_tensors": payload_tensors,
            "model_parameters": comm_summary["model_parameters"],
            "update_bytes": comm_summary["avg_update_bytes"],
            "per_update_metadata": ["round number", "sample count", "scalar training/validation loss and accuracy"],
            "server_side_payload_validation": {
                "updates_validated": updates_validated,
                "rule": "tensor count, shapes and floating-point dtype must match the global model",
            },
        },
        "stays_local": ["MRI image files", "patient-level raw data", "local train/validation datasets",
                        "per-image predictions and labels"],
        "audit": audit,
        "mechanisms": [
            {"name": "Data locality (train where the data lives)", "implemented": True},
            {"name": "Federated averaging (only parameters aggregated)", "implemented": True},
            {"name": "Server-side payload validation", "implemented": True},
            {"name": "Differential privacy", "implemented": False},
            {"name": "Secure aggregation", "implemented": False},
            {"name": "Encryption of updates in transit", "implemented": False},
            {"name": "Homomorphic encryption", "implemented": False},
        ],
        "limitations": [
            "Hospitals are simulated as separate folders and objects inside one Python process; "
            "no real network or separate machines are involved.",
            "Model updates are exchanged in plain form. Published research shows that updates can leak "
            "information about training data, and this prototype does not defend against that.",
            "Isolation is verified on file content (MD5), so visually similar but non-identical images "
            "may still appear in different hospitals.",
        ],
    }
