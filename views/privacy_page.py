"""Page 5 - Privacy & security."""
from __future__ import annotations

from html import escape

import streamlit as st

from backend.io_utils import format_bytes
from ui.components import (badge, callout, card, empty_state, kv_rows, pct, privacy_diagram, section, simple_table, tiles)
from ui.data import is_trained, load_results
from ui.theme import html


def render():
    res = load_results()
    html('<div class="nf-pagehead"><h1>Your data stays where it belongs.</h1>'
         '<p>Privacy &amp; security in NeuroFed AI: what never leaves a hospital, what is shared, and what this prototype '
         'does not protect against.</p></div>')
    if not is_trained(res):
        empty_state("The privacy report")
        return

    priv, comm, clients = res["privacy"], res["communication"], res["clients"]
    n_updates = comm["num_clients"] * comm["rounds"]

    callout("Raw Patient MRI Data Never Leaves the Hospital.", "strong")

    section("Data flow", "Dashed boxes are inside a hospital. Only the model update crosses the boundary.")
    privacy_diagram(
        [(c["name"], f"{c['num_train'] + c['num_val']:,} images, {format_bytes(c['raw_data_bytes'])}") for c in clients],
        f"{format_bytes(comm['avg_update_bytes'])} of parameters",
        f"weighted average over {comm['num_clients']} hospitals")

    st.write("")
    callout(f"Instead of transferring <b>{format_bytes(comm['raw_data_bytes_total'])}</b> of raw MRI data, the "
            f"{comm['num_clients']} participating hospitals exchanged model updates of "
            f"<b>{format_bytes(comm['avg_update_bytes'])}</b> each over {comm['rounds']} rounds "
            f"(<b>{format_bytes(comm['total_training_transfer_bytes'])}</b> in total).", "info")

    section("What stays local, what is shared")
    c1, c2 = st.columns(2, gap="medium")
    local_body = "".join(f"<p>{escape(x)}</p>" for x in priv["stays_local"]) + kv_rows(
        [(name, loc) for name, loc in priv["raw_data_location"].items()])
    c1.markdown(card("Stays in the hospital", badge("Never transmitted", ""), local_body), unsafe_allow_html=True)
    sh = priv["shared_with_server"]
    shared_body = kv_rows([
        ("Model parameters", f"{sh['model_parameters']:,} values in {sh['model_parameter_tensors']} tensors"),
        ("Size of one update", format_bytes(sh["update_bytes"])),
        ("Updates received", f"{n_updates:,}"),
    ]) + "<p><b>Sent with each update:</b> " + escape(", ".join(sh["per_update_metadata"])) + ".</p>" + (
        "<p>The server checks every incoming message: tensor count, shapes and floating-point type must match the "
        f"global model ({sh['server_side_payload_validation']['updates_validated']:,} updates validated).</p>")
    c2.markdown(card("Shared with the server", badge("Model updates only", "warn"), shared_body), unsafe_allow_html=True)

    section("The numbers", "Computed from the files and messages of the actual run.")
    r = comm["ratios"]
    tiles([
        ("Raw dataset held by hospitals", format_bytes(comm["raw_data_bytes_total"]), f"{sum(priv['raw_images_per_hospital'].values()):,} images"),
        ("Model update size", format_bytes(comm["avg_update_bytes"]), f"{pct(r['update_to_total_raw'], 2)} of the raw dataset"),
        ("Total training traffic", format_bytes(comm["total_training_transfer_bytes"]),
         f"{pct(r['total_training_transfer_to_total_raw'])} of the raw dataset size"),
    ])
    st.write("")
    tiles([
        ("Hospitals", str(comm["num_clients"]), "each with its own folder"),
        ("Communication rounds", str(comm["rounds"]), f"{format_bytes(comm['avg_training_bytes_per_round'])} per round, both directions"),
        ("Central raw-image repository", "None", "the server never receives image files"),
    ])
    if r["total_training_transfer_to_total_raw"] and r["total_training_transfer_to_total_raw"] > 1:
        callout("Over this many rounds the model traffic is larger than a one-off upload of the images would be. "
                "Federated learning is chosen here for data locality, not to save bandwidth.", "warn")

    section("Automated checks on this run")
    rows = []
    for ck in priv["audit"]["checks"]:
        rows.append([escape(ck["check"]), badge("Passed") if ck["passed"] else badge("Failed", "warn")])
    simple_table(["Check", "Result"], rows)

    section("Security mechanisms", "Only what is implemented is claimed.")
    simple_table(["Mechanism", "Status"],
                 [[escape(x["name"]), badge("Implemented") if x["implemented"] else badge("Not implemented", "off")]
                  for x in priv["mechanisms"]])
    st.write("")
    callout("<b>Limitations.</b><br>" + "<br>".join(f"• {escape(x)}" for x in priv["limitations"]), "warn")
