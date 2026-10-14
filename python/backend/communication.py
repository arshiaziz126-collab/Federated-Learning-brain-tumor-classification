"""Communication between hospitals and the aggregation server.

The simulation runs in one process, but hospitals and server never share Python
objects: everything that crosses the boundary is serialised to ``bytes`` by
``pack`` and rebuilt by ``unpack``. The byte counts reported by the app are the
real lengths of these messages, not estimates.

A message contains only:
  * model parameters (float arrays), and
  * a small JSON header (round number, sample count, scalar training metrics).
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def pack(params: list, meta: dict | None = None) -> bytes:
    arrays = {f"p{i:03d}": np.asarray(p) for i, p in enumerate(params)}
    header = np.frombuffer(json.dumps(meta or {}).encode("utf-8"), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez(buf, __meta__=header, **arrays)
    return buf.getvalue()


def unpack(payload: bytes):
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:     # pickle disabled: arrays only
        meta = json.loads(bytes(z["__meta__"].tolist()).decode("utf-8"))
        keys = sorted(k for k in z.files if k.startswith("p"))
        params = [z[k] for k in keys]
    return params, meta


def parameters_nbytes(params: list) -> int:
    """Raw tensor size (no container overhead)."""
    return int(sum(np.asarray(p).nbytes for p in params))


def directory_size(path) -> int:
    return int(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()))


class CommunicationTracker:
    """Records every message with its round, direction and byte length."""

    def __init__(self):
        self.events = []      # {"round", "client", "kind", "bytes"}

    def log(self, round_idx: int, client_id: str, kind: str, nbytes: int):
        # kinds: train_downlink | train_uplink | eval_downlink
        self.events.append({"round": round_idx, "client": client_id, "kind": kind, "bytes": int(nbytes)})

    def per_round(self) -> list:
        rows, cumulative = [], 0
        rounds = sorted({e["round"] for e in self.events})
        for r in rounds:
            ev = [e for e in self.events if e["round"] == r]
            down = sum(e["bytes"] for e in ev if e["kind"] == "train_downlink")
            up = sum(e["bytes"] for e in ev if e["kind"] == "train_uplink")
            ev_down = sum(e["bytes"] for e in ev if e["kind"] == "eval_downlink")
            cumulative += down + up
            rows.append({"round": r, "downlink_bytes": down, "uplink_bytes": up,
                         "eval_downlink_bytes": ev_down, "training_bytes": down + up,
                         "cumulative_training_bytes": cumulative})
        return rows

    def summary(self, *, num_clients: int, raw_bytes_by_client: dict, model_parameters: int,
                model_payload_bytes: int) -> dict:
        totals = defaultdict(int)
        for e in self.events:
            totals[e["kind"]] += e["bytes"]
        training_total = totals["train_downlink"] + totals["train_uplink"]
        raw_total = int(sum(raw_bytes_by_client.values()))
        n_up = sum(1 for e in self.events if e["kind"] == "train_uplink")
        avg_update = totals["train_uplink"] / n_up if n_up else 0
        rounds = len({e["round"] for e in self.events})
        per_round = self.per_round()
        avg_round = training_total / rounds if rounds else 0

        def ratio(a, b):
            return (a / b) if b else None

        return {
            "num_clients": num_clients,
            "rounds": rounds,
            "model_parameters": model_parameters,
            "model_payload_bytes": int(model_payload_bytes),
            "avg_update_bytes": float(avg_update),
            "raw_data_bytes_total": raw_total,
            "raw_data_bytes_by_client": {k: int(v) for k, v in raw_bytes_by_client.items()},
            "training_uplink_bytes": int(totals["train_uplink"]),
            "training_downlink_bytes": int(totals["train_downlink"]),
            "total_training_transfer_bytes": int(training_total),
            "evaluation_downlink_bytes": int(totals["eval_downlink"]),
            "total_transfer_incl_evaluation_bytes": int(training_total + totals["eval_downlink"]),
            "avg_training_bytes_per_round": float(avg_round),
            "ratios": {
                "update_to_total_raw": ratio(avg_update, raw_total),
                "round_transfer_to_total_raw": ratio(avg_round, raw_total),
                "total_training_transfer_to_total_raw": ratio(training_total, raw_total),
            },
            "centralized_alternative_bytes": raw_total,
            "notes": [
                "Message sizes are the real serialised lengths of the parameter payloads exchanged in the simulation.",
                "Raw data size is the on-disk size of the image files held by the hospitals (compressed JPEG/PNG).",
                "Evaluation traffic (clients scoring the global model on local validation data) is reported separately.",
            ],
            "per_round": per_round,
        }
