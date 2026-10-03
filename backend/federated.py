"""Federated Averaging (FedAvg) and the federated training loop.

Protocol for every round:

  1. Server packs the global parameters into bytes and sends them to the selected hospitals.
  2. Each hospital trains locally on its own images and sends back parameters + scalar metrics.
  3. Server validates the messages and computes the sample-count-weighted average:

         w_global = sum_k (n_k * w_k) / sum_k n_k

  4. Every hospital scores the new global model on its own local validation data and
     returns only (loss, accuracy, count); these are averaged into the federated validation score.
  5. The independent held-out test set is scored by the evaluator (for reporting only).

The best round is chosen on the *federated validation* score, never on the test set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from backend import communication
from backend.evaluation import evaluate_model
from backend.model import get_parameters, set_parameters
from backend.privacy import validate_update


@dataclass
class ClientUpdate:
    client_id: str
    num_samples: int
    parameters: list
    metrics: dict


def fedavg(updates: list) -> list:
    """Sample-count-weighted average of client parameters."""
    if not updates:
        raise ValueError("FedAvg needs at least one client update")
    total = float(sum(u.num_samples for u in updates))
    averaged = []
    for layer in range(len(updates[0].parameters)):
        acc = np.zeros_like(updates[0].parameters[layer], dtype=np.float64)
        for u in updates:
            acc += (u.num_samples / total) * u.parameters[layer].astype(np.float64)
        averaged.append(acc.astype(np.float32))
    return averaged


def run_federated_training(clients: list, server_model, class_names: list, test_data: tuple,
                           *, rounds: int, local_epochs: int, batch_size: int, learning_rate: float,
                           client_fraction: float, seed: int, tracker, mu: float = 0.0, log=print):
    """Run FedAvg (mu = 0) or FedProx (mu > 0; aggregation is the same weighted average).

    Returns (history, best_params, best_round, stats)."""
    Xte, yte = test_data
    rng = np.random.default_rng(seed)
    global_params = get_parameters(server_model)
    reference_shapes = [p.shape for p in global_params]
    n_select = max(1, int(round(client_fraction * len(clients))))

    history, best = [], {"round": 0, "val_acc": -1.0, "val_loss": float("inf"), "params": None}
    updates_validated = 0
    t_start = time.perf_counter()

    for r in range(1, rounds + 1):
        t_round = time.perf_counter()
        selected_idx = sorted(rng.choice(len(clients), size=n_select, replace=False).tolist())
        selected = [clients[i] for i in selected_idx]

        # 1) broadcast
        down = communication.pack(global_params, {"round": r})
        for c in selected:
            tracker.log(r, c.client_id, "train_downlink", len(down))

        # 2) local training at each hospital
        received, client_rows = [], []
        for c in clients:
            row = {"client_id": c.client_id, "name": c.name, "num_train": c.num_train, "num_val": c.num_val,
                   "participated": c in selected}
            if c in selected:
                up = c.local_update(down, round_idx=r, epochs=local_epochs, batch_size=batch_size,
                                    learning_rate=learning_rate, mu=mu)
                tracker.log(r, c.client_id, "train_uplink", len(up))
                params, meta = communication.unpack(up)
                validate_update(params, reference_shapes)
                updates_validated += 1
                received.append(ClientUpdate(c.client_id, int(meta["num_samples"]), params, meta))
                row.update({"local_train_loss": meta["train_loss"], "local_train_accuracy": meta["train_accuracy"],
                            "local_val_loss": meta["val_loss"], "local_val_accuracy": meta["val_accuracy"],
                            "train_seconds": meta["train_seconds"], "update_bytes": len(up)})
            client_rows.append(row)

        # 3) aggregate
        global_params = fedavg(received)
        set_parameters(server_model, global_params)
        weights = {u.client_id: u.num_samples / sum(x.num_samples for x in received) for u in received}
        for row in client_rows:
            row["aggregation_weight"] = weights.get(row["client_id"])

        # 4) federated evaluation: every hospital scores the new global model locally
        eval_payload = communication.pack(global_params, {"round": r, "purpose": "evaluation"})
        evals = {}
        for c in clients:
            tracker.log(r, c.client_id, "eval_downlink", len(eval_payload))
            evals[c.client_id] = c.evaluate_global(eval_payload)
        n_val = sum(e["num_samples"] for e in evals.values())
        fed_val_loss = sum(e["loss"] * e["num_samples"] for e in evals.values()) / n_val
        fed_val_acc = sum(e["accuracy"] * e["num_samples"] for e in evals.values()) / n_val
        for row in client_rows:
            e = evals[row["client_id"]]
            row["global_on_local_val_loss"] = e["loss"]
            row["global_on_local_val_accuracy"] = e["accuracy"]

        # 5) held-out test set (reporting only)
        test = evaluate_model(server_model, Xte, yte, class_names)
        record = {
            "round": r, "participating_clients": [c.client_id for c in selected],
            "clients": client_rows,
            "global_val_loss": fed_val_loss, "global_val_accuracy": fed_val_acc,
            "global_test": {k: test[k] for k in ("loss", "accuracy", "precision_macro", "recall_macro", "f1_macro")},
            "round_seconds": time.perf_counter() - t_round,
        }
        history.append(record)

        better = fed_val_acc > best["val_acc"] or (fed_val_acc == best["val_acc"] and fed_val_loss < best["val_loss"])
        if better:
            best = {"round": r, "val_acc": fed_val_acc, "val_loss": fed_val_loss, "params": [p.copy() for p in global_params]}
        log(f"  round {r:>2}/{rounds}  fed-val acc {fed_val_acc:.3f}  loss {fed_val_loss:.3f}  |  "
            f"test acc {test['accuracy']:.3f}  |  {record['round_seconds']:.0f}s")

    stats = {"training_seconds": time.perf_counter() - t_start, "updates_validated": updates_validated,
             "clients_per_round": n_select}
    return history, best["params"], best["round"], stats
