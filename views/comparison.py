"""Page 6 - Centralized vs federated."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from backend.io_utils import format_bytes
from ui.components import callout, empty_state, page_header, pct, section, show_chart, simple_table, style_fig
from ui.data import is_trained, load_results
from ui.theme import GOLD, TEAL


def _diff(a: float, b: float, pts: bool = True) -> str:
    d = a - b
    return f"{d * 100:+.2f} pts" if pts else f"{d:+.4f}"


def render():
    page_header("Centralized vs Federated", "The same model, data and held-out test set, trained two ways. "
                                            "Figures are measured, and neither approach is declared better.")
    res = load_results()
    if not is_trained(res):
        empty_state("The comparison experiment")
        return

    section("How the approaches differ")
    simple_table(["Feature", "Centralized", "Federated"], [
        ["Raw data location", "Central repository", "Local hospital"],
        ["Local training", "No", "Yes"],
        ["Model aggregation", "No", "FedAvg"],
        ["Raw MRI transfer", "Yes (all images pooled)", "No"],
        ["Multiple clients", "No", "Yes"],
    ])

    exp, cen, fed, comm = res["experiment"], res["centralized"], res["federated"], res["communication"]
    if not cen:
        st.write("")
        callout("The centralized baseline was skipped in this run (<code>--skip-centralized</code>). "
                "Re-run <code>python federated_train.py</code> without that flag to fill in this comparison.", "warn")
        return

    ft, ct = fed["test"], cen["test"]
    f_ci, c_ci = fed["test_accuracy_ci95"], cen["test_accuracy_ci95"]

    section("Measured results", f"Held-out test set: {ft['num_samples']:,} images that no hospital or baseline trained on.")
    rows = [
        ["Accuracy", pct(ct["accuracy"], 2), pct(ft["accuracy"], 2), _diff(ft["accuracy"], ct["accuracy"])],
        ["95% interval for accuracy", f"{pct(c_ci[0])} to {pct(c_ci[1])}", f"{pct(f_ci[0])} to {pct(f_ci[1])}", "-"],
        ["Precision (macro)", f"{ct['precision_macro']:.4f}", f"{ft['precision_macro']:.4f}", _diff(ft["precision_macro"], ct["precision_macro"], False)],
        ["Recall (macro)", f"{ct['recall_macro']:.4f}", f"{ft['recall_macro']:.4f}", _diff(ft["recall_macro"], ct["recall_macro"], False)],
        ["F1-score (macro)", f"{ct['f1_macro']:.4f}", f"{ft['f1_macro']:.4f}", _diff(ft["f1_macro"], ct["f1_macro"], False)],
        ["Test loss", f"{ct['loss']:.4f}", f"{ft['loss']:.4f}", _diff(ft["loss"], ct["loss"], False)],
        ["Training rounds / epochs", f"{cen['epochs']} epochs (best: {cen['best_epoch']})",
         f"{fed['rounds']} rounds x {fed['local_epochs']} local epochs (best: round {fed['best_round']})", "-"],
        ["Participating clients", "1 central environment", f"{comm['num_clients']} hospitals", "-"],
        ["Raw MRI data transferred", format_bytes(comm["centralized_alternative_bytes"]), "0 B", "-"],
        ["Model traffic (training)", "0 B", format_bytes(comm["total_training_transfer_bytes"]), "-"],
        ["Training time (this machine)", f"{cen['training_seconds'] / 60:.1f} min", f"{fed['training_seconds'] / 60:.1f} min", "-"],
    ]
    simple_table(["Metric", "Centralized", "Federated", "Federated minus centralized"], rows, num_cols=(1, 2, 3))
    st.caption("Precision, recall and F1 are macro-averaged over classes. Raw data transferred for the centralized case is the "
               "on-disk size of the images that would be pooled centrally.")

    section("Test-set metrics side by side")
    metrics = [("Accuracy", "accuracy"), ("Precision", "precision_macro"), ("Recall", "recall_macro"), ("F1", "f1_macro")]
    fig = go.Figure()
    fig.add_bar(x=[n for n, _ in metrics], y=[ct[k] * 100 for _, k in metrics], name="Centralized", marker_color=GOLD)
    fig.add_bar(x=[n for n, _ in metrics], y=[ft[k] * 100 for _, k in metrics], name="Federated", marker_color=TEAL)
    fig.update_layout(barmode="group")
    lo = min([ct[k] for _, k in metrics] + [ft[k] for _, k in metrics])
    fig.update_yaxes(range=[max(0, (lo - 0.1)) * 100, 100])
    show_chart(style_fig(fig, ytitle="Percent (axis starts above zero)", height=340))

    section("Learning curves", "Validation accuracy on the same pooled validation images. Federated rounds are placed at "
                               "epoch-equivalents (rounds x local epochs) so both curves use the same amount of training.")
    hist = res["history"]
    fig = go.Figure()
    fig.add_scatter(x=[e["epoch"] for e in cen["history"]], y=[e["val_accuracy"] * 100 for e in cen["history"]],
                    name="Centralized (per epoch)", mode="lines+markers", line=dict(color=GOLD, width=2.5))
    fig.add_scatter(x=[h["round"] * fed["local_epochs"] for h in hist], y=[h["global_val_accuracy"] * 100 for h in hist],
                    name="Federated (per round)", mode="lines+markers", line=dict(color=TEAL, width=2.5))
    show_chart(style_fig(fig, ytitle="Validation accuracy (%)", xtitle="Epoch-equivalents of training", height=340))

    section("Per-class F1 on the test set")
    names = ft["class_names"]
    simple_table(["Class", "Centralized", "Federated", "Federated minus centralized"],
                 [[n, f"{ct['per_class'][n]['f1']:.3f}", f"{ft['per_class'][n]['f1']:.3f}",
                   _diff(ft["per_class"][n]["f1"], ct["per_class"][n]["f1"], False)] for n in names], num_cols=(1, 2, 3))

    section("How to read this")
    overlap = not (f_ci[1] < c_ci[0] or c_ci[1] < f_ci[0])
    notes = [
        "This is one training run with one random seed and one data split. Repeating it with other seeds would show how "
        "much these numbers move by chance.",
        ("The 95% intervals for accuracy overlap, so the accuracy difference is small compared with the uncertainty from "
         "the size of the test set." if overlap else
         "The 95% intervals for accuracy do not overlap on this test set; a single run still cannot separate the effect "
         "of the training method from run-to-run variation."),
        f"The hospital split is {'non-IID (Dirichlet, alpha ' + str(exp['config']['alpha']) + ')' if exp['config']['partition'] == 'dirichlet' else 'IID'}. "
        "Results change with how unevenly the data is spread between hospitals.",
        ("Both runs use the same architecture, optimiser and total amount of training per image (epochs equal rounds x local "
         "epochs), so differences reflect the training setup, not the model." if cen["epochs"] == fed["rounds"] * fed["local_epochs"] else
         f"The centralized baseline trained for {cen['epochs']} epochs versus {fed['rounds'] * fed['local_epochs']} "
         "epoch-equivalents for federated training, so the amount of training differs between the two."),
    ]
    callout("<br>".join(f"• {n}" for n in notes))
