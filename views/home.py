"""Page 1 - Home / dashboard."""
from __future__ import annotations

import streamlit as st

import config
from backend.io_utils import format_bytes
from ui.components import callout, disclaimer, flow_html, pct, section, tiles
from ui.data import is_trained, load_results
from ui.theme import html


def render():
    res = load_results()
    trained = is_trained(res)
    m = res["metrics"] or {}
    clients = res["clients"] or []
    exp = res["experiment"] or {}

    # ---- hero with the live federation flow ------------------------------------------------
    if trained:
        hospitals = [(c["name"], f"{c['num_train'] + c['num_val']:,} images stay local") for c in clients]
        stages = [
            ("Local training", f"{exp['federated']['local_epochs']} epochs per round", False),
            ("Model updates", f"{format_bytes(m['avg_update_bytes'])} each", False),
            ("FedAvg", "weighted by sample count", False),
            ("Global model", f"round {m['best_round']} selected", True),
        ]
    else:
        n = config.DEFAULTS["num_clients"]
        hospitals = [(config.client_name(i), "local data") for i in range(n)]
        stages = [("Local training", "at each hospital", False), ("Model updates", "parameters only", False),
                  ("FedAvg", "weighted average", False), ("Global model", "not trained yet", True)]

    html(f"""
    <div class="nf-hero">
      <h1>Early Detection for a Better Tomorrow</h1>
      <p class="sub">Privacy-preserving brain tumour screening powered by Federated Learning.</p>
      <div class="pledge">Raw patient MRI data never leaves the hospital.</div>
      {flow_html(hospitals, stages, dark=True)}
    </div>""")

    if not trained:
        callout("<b>No trained model yet.</b> Every figure on this dashboard is read from files written by the training "
                "script. Add the dataset to <code>data/brain_tumor_dataset/</code> and run "
                "<code>python federated_train.py</code>, then reload.", "warn")
        return

    # ---- real project metrics ---------------------------------------------------------------
    tiles([
        ("Global model status", m["model_status"], f"Best round {m['best_round']} of {m['rounds']}; trained {m['generated_at'][:10]}"),
        ("Hospitals / clients", str(m["num_clients"]), "simulated, each with an isolated data folder"),
        ("Federated rounds", str(m["rounds"]), f"{exp['federated']['local_epochs']} local epochs per round"),
    ])
    st.write("")
    tiles([
        ("Test accuracy", pct(m["test_accuracy"]), f"{m['test_images']:,} held-out images; macro-F1 {pct(m['test_f1_macro'])}"),
        ("Classes", str(m["num_classes"]), ", ".join(m["class_names"])),
        ("Communication efficiency", pct(m["round_transfer_to_raw_ratio"]),
         f"of the raw data volume moves per round ({format_bytes(m['avg_update_bytes'])} per update vs "
         f"{format_bytes(m['raw_data_bytes_total'])} of images)"),
    ])

    section("What this prototype demonstrates")
    c1, c2 = st.columns(2)
    c1.markdown(
        f"""<div class="nf-card"><h4>Training without pooling images</h4>
        <p>Each hospital trains a copy of the global model on its own MRI folder. Only model parameters are sent
        back and combined with sample-count-weighted FedAvg. Instead of transferring raw MRI data, participating
        hospitals exchange model updates.</p>
        <p>The data split is <b>{'non-IID (Dirichlet)' if m['partition_mode'] == 'dirichlet' else 'IID'}</b>, so hospitals
        see different class mixes, as they would in practice.</p></div>""".replace("\n", " "),
        unsafe_allow_html=True)
    c2.markdown(
        """<div class="nf-card"><h4>AI-assisted early screening research prototype</h4>
        <p>The Diagnose page classifies an uploaded MRI slice with the trained global model and reports the class
        probabilities. It predicts the dataset's class labels only; it does not estimate tumour stage, grade or
        growth rate.</p>
        <p>Compare federated and centralized training on the same data in <b>Centralized vs Federated</b>.</p></div>""".replace("\n", " "),
        unsafe_allow_html=True)
    st.write("")
    disclaimer()
