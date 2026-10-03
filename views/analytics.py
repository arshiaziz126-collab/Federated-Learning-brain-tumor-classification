"""Page 4 - Federated analytics (all charts drawn from federated_history.json etc.)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.io_utils import format_bytes
from ui.components import (class_color, confusion_table, empty_state, page_header, pct, section, show_chart,
                           simple_table, style_fig, tiles)
from ui.data import is_trained, load_results
from ui.theme import FOREST, GOLD, SAGE, TEAL

MB = 1024 ** 2


def render():
    page_header("Federated Analytics", "Training behaviour round by round, per hospital, and in communication cost.")
    res = load_results()
    if not is_trained(res):
        empty_state("Federated analytics")
        return

    hist, clients, comm, fed, m = res["history"], res["clients"], res["communication"], res["federated"], res["metrics"]
    classes = m["class_names"]
    rounds = [h["round"] for h in hist]
    ci = fed["test_accuracy_ci95"]

    tiles([
        ("Rounds completed", str(len(hist)), f"best round {fed['best_round']} (by federated validation)"),
        ("Test accuracy", pct(fed["test"]["accuracy"]), f"95% interval {pct(ci[0])} to {pct(ci[1])}"),
        ("Macro F1", f"{fed['test']['f1_macro']:.3f}",
         f"precision {fed['test']['precision_macro']:.3f}, recall {fed['test']['recall_macro']:.3f}"),
        ("Training traffic", format_bytes(comm["total_training_transfer_bytes"]),
         f"{comm['rounds']} rounds, {comm['num_clients']} hospitals, both directions"),
    ])

    # ---- global performance ---------------------------------------------------------------------
    section("Global model by round", "Federated validation is scored by the hospitals on their own local validation data; "
                                     "the held-out test set is reported for reference and never used to pick the model.")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=rounds, y=[h["global_val_accuracy"] * 100 for h in hist], name="Federated validation", mode="lines+markers", line=dict(color=TEAL, width=3))
    fig.add_scatter(x=rounds, y=[h["global_test"]["accuracy"] * 100 for h in hist], name="Held-out test", mode="lines+markers", line=dict(color=GOLD, width=2, dash="dot"))
    fig.add_vline(x=fed["best_round"], line=dict(color=SAGE, dash="dash"), annotation_text="selected", annotation_position="top")
    with c1:
        st.markdown("**Accuracy by round**")
        show_chart(style_fig(fig, ytitle="Accuracy (%)", xtitle="Round"))
    fig = go.Figure()
    fig.add_scatter(x=rounds, y=[h["global_val_loss"] for h in hist], name="Federated validation", mode="lines+markers", line=dict(color=TEAL, width=3))
    fig.add_scatter(x=rounds, y=[h["global_test"]["loss"] for h in hist], name="Held-out test", mode="lines+markers", line=dict(color=GOLD, width=2, dash="dot"))
    with c2:
        st.markdown("**Loss by round**")
        show_chart(style_fig(fig, ytitle="Cross-entropy loss", xtitle="Round"))

    # ---- client performance ---------------------------------------------------------------------
    section("Hospital performance by round", "Solid: the hospital's local model after local training. Dashed: the aggregated "
                                              "global model scored on that hospital's local validation images.")
    c1, c2 = st.columns(2)
    fig_a, fig_b = go.Figure(), go.Figure()
    for i, c in enumerate(clients):
        rows = [next(r for r in h["clients"] if r["client_id"] == c["client_id"]) for h in hist]
        colr = class_color(i)
        fig_a.add_scatter(x=[h["round"] for h, r in zip(hist, rows) if r["participated"]],
                          y=[r["local_val_accuracy"] * 100 for r in rows if r["participated"]],
                          name=c["name"], mode="lines+markers", line=dict(color=colr, width=2.5))
        fig_a.add_scatter(x=rounds, y=[r["global_on_local_val_accuracy"] * 100 for r in rows], name=f"{c['name']} (global model)",
                          mode="lines", line=dict(color=colr, width=1.5, dash="dash"), showlegend=False)
        fig_b.add_scatter(x=[h["round"] for h, r in zip(hist, rows) if r["participated"]],
                          y=[r["local_val_loss"] for r in rows if r["participated"]],
                          name=c["name"], mode="lines+markers", line=dict(color=colr, width=2.5))
    with c1:
        st.markdown("**Validation accuracy per hospital**")
        show_chart(style_fig(fig_a, ytitle="Accuracy (%)", xtitle="Round"))
    with c2:
        st.markdown("**Local validation loss per hospital**")
        show_chart(style_fig(fig_b, ytitle="Cross-entropy loss", xtitle="Round"))

    # ---- data distribution ----------------------------------------------------------------------
    section("Data distribution", "How the images were split between hospitals, and the class balance of the whole dataset.")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for i, cls in enumerate(classes):
        fig.add_bar(x=[c["name"] for c in clients], y=[c["class_counts_train"][cls] for c in clients], name=cls,
                    marker_color=class_color(i))
    fig.update_layout(barmode="stack")
    with c1:
        st.markdown("**Training images per hospital, by class**")
        show_chart(style_fig(fig, ytitle="Images", xtitle=None))
    part = res["partition"]
    fig = go.Figure()
    pooled = [sum(c["class_counts_train"][cls] + c["class_counts_val"][cls] for c in clients) for cls in classes]
    fig.add_bar(x=classes, y=pooled, name="Hospitals (train + local validation)", marker_color=TEAL)
    fig.add_bar(x=classes, y=[part["test"]["class_counts"][cls] for cls in classes], name="Held-out test set", marker_color=GOLD)
    fig.update_layout(barmode="group")
    with c2:
        st.markdown("**Class distribution of the dataset used**")
        show_chart(style_fig(fig, ytitle="Images"))

    # ---- communication --------------------------------------------------------------------------
    section("Communication overhead", "Real serialised message sizes. Downlink: global model to hospitals. "
                                      "Uplink: hospital updates to the server.")
    pr = comm["per_round"]
    fig = go.Figure()
    fig.add_bar(x=[p["round"] for p in pr], y=[p["downlink_bytes"] / MB for p in pr], name="Downlink (MB)", marker_color=SAGE)
    fig.add_bar(x=[p["round"] for p in pr], y=[p["uplink_bytes"] / MB for p in pr], name="Uplink (MB)", marker_color=TEAL)
    fig.add_scatter(x=[p["round"] for p in pr], y=[p["cumulative_training_bytes"] / MB for p in pr], name="Cumulative (MB)",
                    mode="lines+markers", line=dict(color=FOREST, width=2.5), yaxis="y2")
    fig.update_layout(barmode="stack", yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Cumulative MB", rangemode="tozero", automargin=True))
    st.markdown("**Data transferred per round**")
    show_chart(style_fig(fig, ytitle="MB per round", xtitle="Round", height=360))
    st.caption(f"Raw images held by the hospitals: {format_bytes(comm['raw_data_bytes_total'])}. Total federated training "
               f"traffic: {format_bytes(comm['total_training_transfer_bytes'])} "
               f"({pct(comm['ratios']['total_training_transfer_to_total_raw'])} of the raw data volume). "
               f"Evaluation broadcasts ({format_bytes(comm['evaluation_downlink_bytes'])}) are counted separately.")

    # ---- final model detail ---------------------------------------------------------------------
    section("Final global model on the held-out test set", f"Round {fed['best_round']}, {fed['test']['num_samples']:,} test images.")
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.markdown("**Per-class metrics**")
        pc = fed["test"]["per_class"]
        simple_table(["Class", "Precision", "Recall", "F1", "Images"],
                     [[k, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", f"{v['support']:,}"] for k, v in pc.items()],
                     num_cols=(1, 2, 3, 4))
    with c2:
        st.markdown("**Confusion matrix**")
        confusion_table(fed["test"]["confusion_matrix"], fed["test"]["class_names"])

    # ---- table + download -----------------------------------------------------------------------
    section("Round-by-round results")
    df = pd.DataFrame([{
        "Round": h["round"], "Hospitals participating": len(h["participating_clients"]),
        "Fed. validation accuracy": round(h["global_val_accuracy"], 4), "Fed. validation loss": round(h["global_val_loss"], 4),
        "Test accuracy": round(h["global_test"]["accuracy"], 4), "Test loss": round(h["global_test"]["loss"], 4),
        "Test macro-F1": round(h["global_test"]["f1_macro"], 4), "Round time (s)": round(h["round_seconds"], 1),
    } for h in hist])
    st.dataframe(df, hide_index=True)
    st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"), "federated_rounds.csv", "text/csv")
