"""Page 7 - About / methodology."""
from __future__ import annotations

from html import escape

import streamlit as st

import config
from backend.io_utils import format_bytes
from ui.components import callout, disclaimer, page_header, pct, simple_table
from ui.data import load_results


def render():
    page_header("About and Methodology", "NeuroFed AI: Federated Learning approach for brain tumour classification with "
                                         "privacy preservation. A research prototype.")
    res = load_results()
    rep, exp, cfgm = res["dataset_report"], res["experiment"], res["model_config"]
    cfg = (exp or {}).get("config", {})

    t1, t2, t3, t4, t5 = st.tabs(["Objective", "Federated learning", "Data and model", "Training and evaluation",
                                  "Limitations and future scope"])

    with t1:
        st.markdown("""
#### Project objective
Build and evaluate a brain tumour MRI classifier trained with **Federated Learning**, so that several hospitals can
contribute to one model without sharing patient images. The work covers local hospital training, FedAvg aggregation,
communication cost, privacy analysis and a measured comparison with centralized training.

#### Problem statement
Medical imaging models improve with more data, but MRI scans are sensitive patient records and are often bound by
regulation and hospital policy. Pooling images in one place creates a privacy risk and is frequently not permitted.
The question this project studies: *can hospitals that keep their data on site jointly train a useful classifier, and
what does that cost in accuracy and communication?*

#### Early-screening angle
The system is an **AI-assisted early screening research prototype**: it flags which class of MRI an image most resembles.
It does not predict tumour stage, grade or growth rate.
""")
        disclaimer()

    with t2:
        st.markdown("""
#### Federated learning
Instead of moving data to the model, the model moves to the data. A central server holds a global model and repeats
a round of communication:

1. The server sends the current global model to the hospitals.
2. Each hospital trains it locally on its own images for a few epochs.
3. Each hospital sends back its updated parameters (never images).
4. The server combines them into a new global model.

#### Federated Averaging (FedAvg)
The server computes a weighted average of the hospital parameters, weighting each hospital by its number of training samples:
""")
        st.code("w_global = sum_k ( n_k / N ) * w_k        N = sum_k n_k", language="text")
        st.markdown("""
#### Hospital / client architecture
Hospital A, B and C are simulated clients. Each has its own folder of images, its own copy of the model and its own
local train / validation split. A hospital only communicates through serialised messages containing parameters and a few
scalar metrics. The server validates each message before averaging.

#### Privacy preservation
The privacy mechanism implemented here is **data locality**: raw images stay in the hospital and only model updates are
shared. Differential privacy, secure aggregation and encryption are not implemented; see the Privacy & Security page for
the exact status and limitations.
""")

    with t3:
        st.markdown("#### Dataset")
        if rep:
            st.markdown(f"Images are read from `data/brain_tumor_dataset/` (found: {escape(rep['structure'])}). "
                        f"Classes are detected from folder names, not hardcoded.")
            simple_table(["Class", "Images (valid)"], [[escape(c), f"{rep['class_counts'][c]:,}"] for c in rep["class_names"]],
                         num_cols=(1,))
            d = rep["duplicates"]
            st.caption(f"{rep['valid_images']:,} readable images, {rep['corrupted_images']} corrupted. Exact duplicate copies found: "
                       f"{d['redundant_copies']} ({d['groups_spanning_splits']} groups span train and test). "
                       f"Total size {format_bytes(rep['total_bytes'])}.")
            if exp:
                cl = exp["partition"]["cleaning"]
                if cl["enabled"]:
                    st.caption(f"Cleaning before splitting: {cl['dropped_exact_duplicates']} duplicate copies and "
                               f"{cl['dropped_conflicting_label_images']} label-conflicting images removed, so no image appears in "
                               "more than one hospital or in both training and test data.")
        else:
            st.markdown("Run `python inspect_dataset.py` or `python federated_train.py` to inspect the dataset; "
                        "its statistics appear here.")

        st.markdown("#### CNN model")
        if cfgm:
            arch = cfgm["architecture"]
            st.markdown(f"A compact convolutional network ({arch['trainable_parameters']:,} trainable parameters): stacked "
                        "convolution blocks with group normalisation and pooling, global average pooling, a small dense layer "
                        "and a softmax over the classes. Group normalisation replaces batch normalisation because batch statistics "
                        "average poorly across hospitals with different data. The network is defined in `backend/model.py` and "
                        "can be replaced without changing the federated code.")
            with st.expander("Layer list"):
                simple_table(["Layer", "Type", "Output shape", "Parameters"],
                             [[escape(l["name"]), escape(l["type"]), escape(l["output_shape"]), f"{l['params']:,}"] for l in arch["layers"]],
                             num_cols=(3,))
        else:
            st.markdown("Model details appear here after training.")

    with t4:
        st.markdown("#### Training methodology")
        if cfg:
            simple_table(["Setting", "Value"], [
                ["Hospitals", str(cfg["num_clients"])],
                ["Data split", ("Non-IID, Dirichlet alpha " + str(cfg["alpha"])) if cfg["partition"] == "dirichlet" else "IID (stratified)"],
                ["Rounds x local epochs", f"{cfg['rounds']} x {cfg['local_epochs']}"],
                ["Hospitals per round", pct(cfg["client_fraction"], 0)],
                ["Optimiser", f"Adam, learning rate {cfg['lr']}, batch size {cfg['batch_size']} (fresh optimiser each round)"],
                ["Input", f"{cfg['img_size']}x{cfg['img_size']}, {'grayscale' if cfg['grayscale'] else 'RGB'}"],
                ["Seed", str(cfg["seed"])],
            ])
        else:
            st.markdown("Settings of the last training run appear here after training.")
        st.markdown("""
Each hospital keeps a local validation split. After each round the hospitals score the new global model on that split and
report only loss and accuracy; the round with the best federated validation accuracy becomes the deployed global model.
Training runs in `federated_train.py`, separately from this app.

#### Evaluation methodology
The final model is scored once on a held-out test set that belongs to no hospital and is never used to select rounds.
Accuracy, macro precision, recall and F1, loss and a confusion matrix are computed from the model's predictions. Accuracy
is reported with a 95% Wilson interval. A centralized baseline with the same architecture, optimiser and amount of
training is scored on the same test set.
""")

    with t5:
        st.markdown("""
#### Limitations
- Research prototype; not clinically validated and not a diagnostic tool.
- Hospitals are simulated inside one process on one machine; no real network, devices or institutional data agreements.
- The dataset is public and 2D; results may not transfer to real hospital data, scanners or protocols.
- Model updates are unprotected. No differential privacy, secure aggregation or encryption is implemented.
- Duplicate detection is exact (byte-identical); near-duplicate images can still exist across splits.
- Results come from single runs; repeated runs with different seeds are needed to judge small differences.

#### Future scope
- Differential privacy and secure aggregation, with the accuracy cost measured.
- Algorithms designed for non-IID data (for example FedProx) and comparison across several heterogeneity levels.
- Update compression to reduce communication cost.
- Deployment across real machines and evaluation on multi-centre data.
- Repeated experiments with confidence intervals across seeds.
""")
        callout(escape(config.MEDICAL_DISCLAIMER))
