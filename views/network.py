"""Page 3 - Federated network."""
from __future__ import annotations

import streamlit as st

from backend.io_utils import format_bytes
from ui.components import (badge, callout, card, class_bar, empty_state, flow_diagram, kv_rows, num, page_header, pct,
                           section, simple_table)
from ui.data import is_trained, load_results


def render():
    page_header("Federated Network", "Hospitals train on their own data; only model updates travel to the aggregation "
                                     "server. Every value below was recorded during training.")
    res = load_results()
    if not is_trained(res):
        empty_state("Federated network results")
        return

    clients, history, m = res["clients"], res["history"], res["metrics"]
    classes = m["class_names"]
    local_epochs = res["experiment"]["federated"]["local_epochs"]

    flow_diagram(
        [(c["name"], f"{c['num_train']:,} training samples") for c in clients],
        [("Local training", f"{local_epochs} epochs per round", False),
         ("Model updates", f"{format_bytes(m['avg_update_bytes'])} per hospital", False),
         ("FedAvg aggregation", "weighted by samples", False),
         ("Global model", f"{m['rounds']} rounds", True)])

    section("Hospitals", "Class mix of each hospital's training data and how its local model performed at the selected round.")
    cols = st.columns(len(clients), gap="medium")
    for col, c in zip(cols, clients):
        b = c["at_best_round"]
        body = class_bar(c["class_counts_train"], classes) + kv_rows([
            ("Training samples", f"{c['num_train']:,}"),
            ("Local validation samples", f"{c['num_val']:,}"),
            ("Raw data kept on site", format_bytes(c["raw_data_bytes"])),
            ("Local training", c["status"]),
            ("Local model accuracy", pct(b["local_val_accuracy"])),
            ("Local model loss", num(b["local_val_loss"], 3)),
            ("Global model on local data", pct(b["global_on_local_val_accuracy"])),
            ("Update size", format_bytes(c["avg_update_bytes"])),
            ("FedAvg weight", pct(b["aggregation_weight"])),
        ])
        sub = badge("Participating" if c["rounds_participated"] else "Not selected",
                    "" if c["rounds_participated"] else "off")
        col.markdown(card(c["name"], sub, body), unsafe_allow_html=True)
    st.caption(f"Local model figures are from round {m['best_round']} (the round kept as the global model), measured on each "
               "hospital's own validation images. Class bars show training data only.")

    section("Round explorer", "Inspect what every hospital reported in a chosen round.")
    r = st.slider("Round", 1, len(history), m["best_round"])
    rec = history[r - 1]
    rows = []
    for row in rec["clients"]:
        if row["participated"]:
            rows.append([row["name"], badge("Participated"), pct(row["local_train_accuracy"]), num(row["local_train_loss"], 3),
                         pct(row["local_val_accuracy"]), num(row["local_val_loss"], 3),
                         pct(row["global_on_local_val_accuracy"]), format_bytes(row["update_bytes"]),
                         pct(row["aggregation_weight"]), f"{row['train_seconds']:.0f} s"])
        else:
            rows.append([row["name"], badge("Not selected", "off"), "-", "-", "-", "-",
                         pct(row["global_on_local_val_accuracy"]), "-", "-", "-"])
    simple_table(["Hospital", "Status", "Local train acc.", "Local train loss", "Local val acc.", "Local val loss",
                  "Global model val acc.", "Update size", "FedAvg weight", "Train time"], rows, num_cols=(2, 3, 4, 5, 6, 7, 8, 9))
    st.caption(f"After aggregating round {r}: federated validation accuracy {pct(rec['global_val_accuracy'])}, "
               f"held-out test accuracy {pct(rec['global_test']['accuracy'])}.")

    section("How FedAvg weights the hospitals")
    total = sum(c["num_train"] for c in clients)
    st.code("global_model = sum over hospitals of (n_k / N) * weights_k", language="text")
    simple_table(["Hospital", "Training samples (n_k)", "Share of N"],
                 [[c["name"], f"{c['num_train']:,}", pct(c["num_train"] / total)] for c in clients], num_cols=(1, 2))
    st.write("")
    callout("Only model parameters and a few scalar metrics leave a hospital. Images, labels and per-image outputs stay in "
            "the hospital's own folder.", "info")
