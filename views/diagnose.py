"""Page 2 - Diagnose: prediction and HL7 FHIR R4 / printable report."""
from __future__ import annotations

from html import escape

import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from backend import fhir, inference
from backend.report import build_report_html
from ui.components import callout, disclaimer, empty_state, kv_rows, page_header, prob_rows, section
from ui.theme import compact, html


@st.cache_resource(show_spinner="Loading the global model")
def _load_model(mtime: float):
    return inference.load_global_model()


@st.cache_resource(show_spinner="Loading the centralized model")
def _load_centralized(mtime: float):
    return inference.load_centralized_model()


def _chip(text: str, kind: str = "") -> str:
    return f'<span class="nf-chip {kind}">{escape(text)}</span>'


def render():
    if not inference.model_available():
        page_header("Diagnose", "Classify a brain MRI slice and export the result as an HL7 FHIR R4 report.")
        empty_state("The trained global model")
        return

    fed_model, cfg = _load_model(config.GLOBAL_MODEL_PATH.stat().st_mtime)
    algo = "FedProx" if cfg.get("algorithm") == "fedprox" else "FedAvg"
    html(f'<div class="nf-statusbar"><span class="crumb">Workspace / <b>Diagnose</b></span>'
         f'<span class="right">{_chip("Model loaded", "ok")}'
         f'<span>Federated round {escape(str(cfg.get("best_round", "-")))} · {algo} · '
         f'{cfg.get("trainable_parameters", 0):,} parameters</span></span></div>')
    page_header("Diagnose", "Classify a brain MRI slice and export the result as an HL7 FHIR R4 report. "
                            "Research prototype, not a clinical diagnosis.")

    kinds = ["Federated"] + (["Centralized"] if inference.centralized_available() else [])
    top_l, top_r = st.columns([1.6, 1])
    with top_l:
        upload = st.file_uploader("MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                                  help="A single 2D MRI slice. The file is processed in memory and not stored.")
    with top_r:
        kind = st.radio("Model", kinds, horizontal=True,
                        help="Federated: trained across hospitals without sharing images. "
                             "Centralized: baseline trained on pooled data, for comparison.")

    if upload is None:
        callout("Upload an MRI image to see the predicted class and create the FHIR report.", "info")
        disclaimer()
        return
    try:
        image = Image.open(upload)
        image.load()
    except (UnidentifiedImageError, OSError):
        callout("This file could not be read as an image. Upload a JPG, PNG, BMP or TIFF file.", "warn")
        return

    if kind == "Centralized":
        model = _load_centralized(config.CENTRALIZED_MODEL_PATH.stat().st_mtime)
    else:
        model = fed_model
    result = inference.predict(model, cfg, image)
    names = cfg["class_names"]

    # --- input + prediction --------------------------------------------------------------------
    left, right = st.columns([1, 1.55], gap="medium")
    with left:
        with st.container(border=True):
            html('<div class="nf-label">Input image</div>')
            st.image(image.convert("RGB") if image.mode not in ("L", "RGB") else image, width="stretch")
            html(kv_rows([
                ("File", upload.name),
                ("Original size", f"{image.size[0]} × {image.size[1]} px"),
                ("Model input", f"{result['input_size']} × {result['input_size']}, "
                                f"{'grayscale' if result['grayscale'] else 'RGB'}"),
                ("Uploaded data", "Not stored"),
            ]))

    review = _chip("Low confidence", "warn") if result["low_confidence"] else _chip("Requires clinician review", "warn")
    with right:
        st.markdown(compact(f"""
        <div class="nf-card">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:.6rem;flex-wrap:wrap">
            <span class="nf-label">Prediction · {escape(kind.lower())} model</span>{review}
          </div>
          <div class="nf-result">{escape(result['predicted_class'])}</div>
          <div style="font-size:14px;color:var(--muted);margin-bottom:.7rem">Confidence
            <span class="mono" style="color:var(--ink);font-weight:600">{result['confidence'] * 100:.1f}%</span>
            · uncalibrated softmax probability</div>
          <div class="nf-label" style="margin-bottom:.2rem">Class probabilities</div>
          {prob_rows(result['probabilities'], names)}
        </div>"""), unsafe_allow_html=True)
        if result["low_confidence"]:
            callout(f"<b>Low confidence.</b> The top class leads {escape(result['runner_up'] or 'the next class')} by "
                    f"only {result['margin_to_runner_up'] * 100:.1f} points. The image may be unlike the training data.",
                    "warn")
        elif result["screen_positive"] is True:
            callout("An abnormal class was predicted. In a real setting this result would be reviewed by a "
                    "qualified clinician before any decision.", "info")
        elif result["screen_positive"] is False:
            callout("No tumour class was predicted. This does not rule out disease.", "info")

    # --- FHIR report ---------------------------------------------------------------------------------
    section("FHIR R4 report", "Export the result for hospital information systems as an HL7 FHIR R4 bundle, or as a "
                              "printable report. Both are marked preliminary and identify the patient by pseudonym only.")
    f_left, f_right = st.columns([1, 1.15], gap="medium")
    with f_left:
        pseudonym = st.text_input("Patient pseudonym", value="CASE-0001",
                                  help="Use a pseudonym only. Never enter a real name or ID number.")
        model_info = {"kind": kind.lower(), "algorithm": algo, "best_round": cfg.get("best_round"),
                      "trained_at": cfg.get("trained_at"), "img_size": cfg["img_size"]}
        report_html = build_report_html(result=result, model_kind=kind.lower(), pseudonym=pseudonym,
                                        file_name=upload.name, model_info=model_info)
        bundle = fhir.build_bundle(result=result, model_kind=kind.lower(), model_info=model_info,
                                   pseudonym=pseudonym, report_html=report_html)
        rows = "".join(
            f'<tr><td class="code">{escape(e["resource"]["resourceType"])}</td><td>{escape(_describe(e["resource"]))}</td>'
            f'<td class="num">{_chip("Preliminary", "warn") if e["resource"].get("status") == "preliminary" else _chip("Ready", "ok")}</td></tr>'
            for e in bundle["entry"])
        html(f'<div class="nf-wrap"><table class="nf-table"><tr><th>Resource</th><th>Content</th><th class="num">Status</th></tr>'
             f'{rows}</table></div>')
        st.write("")
        stem = upload.name.rsplit(".", 1)[0]
        b1, b2 = st.columns(2)
        with b1:
            st.download_button("Download FHIR bundle", fhir.to_json(bundle), file_name=f"{stem}_fhir.json",
                               mime="application/fhir+json", type="primary", width="stretch")
        with b2:
            st.download_button("Printable report", report_html, file_name=f"{stem}_report.html",
                               mime="text/html", width="stretch")
        st.caption("Open the report in a browser and use Print → Save as PDF.")
    with f_right:
        html('<div class="nf-label" style="margin-bottom:.35rem">Bundle preview</div>')
        preview = {**bundle, "entry": [_strip(e) for e in bundle["entry"]]}
        html(f'<pre class="nf-code" style="max-height:420px;overflow:auto">{escape(fhir.to_json(preview))}</pre>')

    st.write("")
    disclaimer()


def _describe(res: dict) -> str:
    t = res["resourceType"]
    if t == "Patient":
        return "Pseudonymous identifier only"
    if t == "Observation":
        return f"Class '{res['valueCodeableConcept']['text']}' + {len(res.get('component', []))} probabilities"
    if t == "DiagnosticReport":
        c = res["code"]["coding"][0]
        return f"LOINC {c['code']} · {c['display']}"
    return t


def _strip(entry: dict) -> dict:
    res = dict(entry["resource"])
    if "presentedForm" in res:
        res["presentedForm"] = [{**f, "data": "<base64 HTML report omitted in preview>"} for f in res["presentedForm"]]
    if "extension" in res:
        res["extension"] = [{**x, "valueString": "<model details>"} for x in res["extension"]]
    return {**entry, "resource": res}
