// NeuroFed AI - static web dashboard. Every number is read from the JSON files in /data,
// written by the training run (tools/export_web_model.py). The model runs in the browser.
import { lineChart, barChart } from "./charts.js";
import { loadModel, predictProbs, rgbaToL, resizeLikePillow, summarise } from "./model.js";
import { buildBundle, buildReportHtml, previewBundle, DISCLAIMER } from "./fhir.js";

// ---------------- helpers ----------------
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x, d = 1) => (x == null ? "-" : `${(x * 100).toFixed(d)}%`);
const num = (x, d = 0) => (x == null ? "-" : Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const bytes = (n) => {
  if (n == null) return "-";
  let v = Number(n);
  for (const u of ["B", "KB", "MB", "GB", "TB"]) {
    if (Math.abs(v) < 1000 || u === "TB") return u === "B" ? `${v.toFixed(0)} B` : `${v.toFixed(2)} ${u}`;
    v /= 1000;
  }
};
const COLORS = ["#F2F2F2", "#8C8C8C", "#BDBDBD", "#5A5A5A", "#5A5A5A", "#C9A9A6"];
const color = (i) => COLORS[i % COLORS.length];
const chip = (t, k = "") => `<span class="chip ${k}">${esc(t)}</span>`;
const callout = (body, k = "") => `<div class="callout ${k}">${body}</div>`;
const header = (t, s) => `<div class="pagehead"><h1>${esc(t)}</h1><p>${esc(s)}</p></div>`;
const section = (t, c = "") => `<h2 class="sec">${esc(t)}</h2>${c ? `<p class="sec-cap">${esc(c)}</p>` : ""}`;
const kv = (rows) => rows.map(([k, v]) => `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("");
const tiles = (items, cols = items.length) => `<div class="grid g${cols}">${items.map(([l, v, c]) =>
  `<div class="tile"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="c">${esc(c)}</div></div>`).join("")}</div>`;
function table(head, rows, numCols = []) {
  const th = head.map((h, i) => `<th class="${numCols.includes(i) ? "num" : ""}">${esc(h)}</th>`).join("");
  const tr = rows.map((r) => `<tr>${r.map((c, i) => `<td class="${numCols.includes(i) ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("");
  return `<div class="tw"><table class="t"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}
const disclaimer = () => callout(esc(DISCLAIMER));
function flow(hospitals, stages, dark = false) {
  const hosp = hospitals.map(([n, c]) => `<div class="node"><b>${esc(n)}</b><span>${esc(c)}</span></div>`).join("");
  const parts = [`<div class="col">${hosp}</div>`];
  for (const [t, c, solid] of stages) {
    parts.push('<div class="link"></div>');
    parts.push(`<div class="col"><div class="node${solid ? " solid" : ""}"><b>${esc(t)}</b><span>${esc(c)}</span></div></div>`);
  }
  return `<div class="flow${dark ? " on-dark" : ""}">${parts.join("")}</div>`;
}
function classBar(counts, classes) {
  const total = classes.reduce((a, c) => a + (counts[c] || 0), 0) || 1;
  const segs = classes.map((c, i) => `<span style="width:${((counts[c] || 0) / total * 100).toFixed(2)}%;background:${color(i)}" title="${esc(c)}: ${counts[c] || 0}"></span>`).join("");
  const leg = classes.map((c, i) => `<div><i style="background:${color(i)}"></i>${esc(c)} ${num(counts[c] || 0)}</div>`).join("");
  return `<div class="stack">${segs}</div><div class="legend">${leg}</div>`;
}
function confusion(cm, names) {
  const max = Math.max(...cm.flat()) || 1;
  const head = `<th>True \\ predicted</th>${names.map((n) => `<th class="num">${esc(n)}</th>`).join("")}`;
  const rows = cm.map((row, i) => `<tr><td><b>${esc(names[i])}</b></td>${row.map((v) => {
    const a = v / max;
    return `<td class="cm" style="background:rgba(242,242,242,${(a * 0.85).toFixed(3)});color:${a > 0.5 ? "#0A0A0A" : "#F2F2F2"}">${num(v)}</td>`;
  }).join("")}</tr>`).join("");
  return `<div class="tw"><table class="t cm"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ---------------- data ----------------
const FILES = {
  metrics: "metrics", federated: "federated_metrics", centralized: "centralized_metrics", history: "federated_history",
  clients: "client_metrics", communication: "communication_metrics", privacy: "privacy_report", dataset: "dataset_report",
  experiment: "experiment_results", modelConfig: "model_config",
  fedprox: "fedprox_federated_metrics", fedproxHistory: "fedprox_federated_history", fedproxClients: "fedprox_client_metrics",
};
let DATA = null;
async function loadData() {
  if (DATA) return DATA;
  const entries = await Promise.all(Object.entries(FILES).map(async ([k, f]) => {
    try { const r = await fetch(`data/${f}.json`); return [k, r.ok ? await r.json() : null]; } catch { return [k, null]; }
  }));
  DATA = Object.fromEntries(entries);
  return DATA;
}

// ---------------- pages ----------------
function pageHome(d) {
  const m = d.metrics, clients = d.clients, exp = d.experiment;
  const hospitals = clients.map((c) => [c.name, `${num(c.num_train + c.num_val)} images stay local`]);
  const stages = [["Local training", `${exp.federated.local_epochs} epochs per round`], ["Model updates", `${bytes(m.avg_update_bytes)} each`],
    ["FedAvg", "weighted by sample count"], ["Global model", `round ${m.best_round} selected`, true]];
  return `
  <div class="hero">
    <h1>Early detection for a better tomorrow</h1>
    <p class="sub">Privacy-preserving brain tumour screening powered by federated learning.</p>
    <div class="pledge">Raw patient MRI data never leaves the hospital.</div>
    ${flow(hospitals, stages, true)}
  </div>
  ${tiles([
    ["Global model status", m.model_status, `Best round ${m.best_round} of ${m.rounds}; trained ${String(m.generated_at).slice(0, 10)}`],
    ["Hospitals", String(m.num_clients), "simulated, each with an isolated data folder"],
    ["Federated rounds", String(m.rounds), `${exp.federated.local_epochs} local epochs per round`],
  ], 3)}
  <div style="height:16px"></div>
  ${tiles([
    ["Test accuracy", pct(m.test_accuracy), `${num(m.test_images)} held-out images; macro-F1 ${pct(m.test_f1_macro)}`],
    ["Classes", String(m.num_classes), m.class_names.join(", ")],
    ["Communication per round", pct(m.round_transfer_to_raw_ratio), `of the raw data volume (${bytes(m.avg_update_bytes)} per update vs ${bytes(m.raw_data_bytes_total)} of images)`],
  ], 3)}
  ${section("What this prototype demonstrates")}
  <div class="grid g2">
    <div class="card"><h4>Training without pooling images</h4>
      <p>Each hospital trains a copy of the global model on its own MRI folder. Only model parameters are sent back and combined with sample-weighted FedAvg.</p>
      <p>The data split is <b>${m.partition_mode === "dirichlet" ? "non-IID (Dirichlet)" : "IID"}</b>, so hospitals see different class mixes, as they would in practice.</p></div>
    <div class="card"><h4>AI-assisted early screening</h4>
      <p>The Diagnose page classifies an MRI slice with the trained model, right in your browser, and exports the result as an HL7 FHIR R4 report. It predicts the dataset's class labels only, not tumour stage or grade.</p>
      <p>Compare federated and centralized training on the same data in <a href="#/comparison">Centralized vs federated</a>.</p></div>
  </div>
  <div style="height:14px"></div>${disclaimer()}`;
}

function pageNetwork(d) {
  const { clients, history, metrics: m } = d;
  const classes = m.class_names, E = d.experiment.federated.local_epochs;
  const cards = clients.map((c) => {
    const b = c.at_best_round;
    return `<div class="card"><h4>${esc(c.name)}</h4><div class="sub">${c.rounds_participated ? chip("Participating") : chip("Not selected", "off")}</div>
      ${classBar(c.class_counts_train, classes)}
      ${kv([["Training samples", num(c.num_train)], ["Local validation samples", num(c.num_val)], ["Raw data kept on site", bytes(c.raw_data_bytes)],
        ["Local model accuracy", pct(b.local_val_accuracy)], ["Local model loss", num(b.local_val_loss, 3)],
        ["Global model on local data", pct(b.global_on_local_val_accuracy)], ["Update size", bytes(c.avg_update_bytes)], ["FedAvg weight", pct(b.aggregation_weight)]])}</div>`;
  }).join("");
  const total = clients.reduce((a, c) => a + c.num_train, 0);
  return `${header("Federated network", "Hospitals train on their own data; only model updates travel to the aggregation server. Every value below was recorded during training.")}
  ${flow(clients.map((c) => [c.name, `${num(c.num_train)} training samples`]),
    [["Local training", `${E} epochs per round`], ["Model updates", `${bytes(m.avg_update_bytes)} per hospital`], ["FedAvg aggregation", "weighted by samples"], ["Global model", `${m.rounds} rounds`, true]])}
  ${section("Hospitals", "Class mix of each hospital's training data and how its local model performed at the selected round.")}
  <div class="grid g3">${cards}</div>
  <p class="cap">Local model figures are from round ${m.best_round} (the round kept as the global model), measured on each hospital's own validation images.</p>
  ${section("Round explorer", "Inspect what every hospital reported in a chosen round.")}
  <label class="label" for="round">Round <b id="roundv">${m.best_round}</b> of ${history.length}</label>
  <input id="round" type="range" min="1" max="${history.length}" value="${m.best_round}">
  <div id="roundtable" style="margin-top:10px"></div>
  ${section("How FedAvg weights the hospitals")}
  <pre class="json" style="max-height:none">global_model = sum over hospitals of (n_k / N) * weights_k</pre>
  <div style="height:10px"></div>
  ${table(["Hospital", "Training samples (n_k)", "Share of N"], clients.map((c) => [esc(c.name), num(c.num_train), pct(c.num_train / total)]), [1, 2])}
  ${callout("Only model parameters and a few scalar metrics leave a hospital. Images, labels and per-image outputs stay in the hospital's own folder.", "info")}`;
}
function bindNetwork(d) {
  const draw = (r) => {
    const rec = d.history[r - 1];
    $("#roundv").textContent = r;
    const rows = rec.clients.map((row) => row.participated
      ? [esc(row.name), chip("Participated"), pct(row.local_train_accuracy), num(row.local_train_loss, 3), pct(row.local_val_accuracy), num(row.local_val_loss, 3),
        pct(row.global_on_local_val_accuracy), bytes(row.update_bytes), pct(row.aggregation_weight), `${row.train_seconds.toFixed(0)} s`]
      : [esc(row.name), chip("Not selected", "off"), "-", "-", "-", "-", pct(row.global_on_local_val_accuracy), "-", "-", "-"]);
    $("#roundtable").innerHTML = table(["Hospital", "Status", "Local train acc.", "Local train loss", "Local val acc.", "Local val loss", "Global model val acc.", "Update size", "FedAvg weight", "Train time"], rows, [2, 3, 4, 5, 6, 7, 8, 9])
      + `<p class="cap">After aggregating round ${r}: federated validation accuracy ${pct(rec.global_val_accuracy)}, held-out test accuracy ${pct(rec.global_test.accuracy)}.</p>`;
  };
  const s = $("#round");
  s.addEventListener("input", () => draw(+s.value));
  draw(+s.value);
}

function pageAnalytics(d) {
  const { history: hist, clients, communication: comm, federated: fed, metrics: m } = d;
  const classes = m.class_names, ci = fed.test_accuracy_ci95, best = fed.best_round;
  const rounds = hist.map((h) => h.round);
  const accChart = lineChart({ title: "Accuracy by round", xLabel: "Round", yLabel: "Accuracy (%)", yMin: 0, yMax: 100, xTicks: rounds,
    marks: [{ x: best, label: "selected" }],
    series: [{ name: "Federated validation", color: "#F2F2F2", points: hist.map((h) => [h.round, h.global_val_accuracy * 100]) },
      { name: "Held-out test", color: "#8C8C8C", dash: "5 4", points: hist.map((h) => [h.round, h.global_test.accuracy * 100]) }] });
  const lossChart = lineChart({ title: "Loss by round", xLabel: "Round", yLabel: "Cross-entropy loss", yMin: 0, xTicks: rounds,
    series: [{ name: "Federated validation", color: "#F2F2F2", points: hist.map((h) => [h.round, h.global_val_loss]) },
      { name: "Held-out test", color: "#8C8C8C", dash: "5 4", points: hist.map((h) => [h.round, h.global_test.loss]) }] });
  const perHosp = (key, title, yLabel, yMax) => lineChart({ title, xLabel: "Round", yLabel, yMin: 0, yMax, xTicks: rounds,
    series: clients.map((c, i) => ({ name: c.name, color: color(i), points: hist.map((h) => {
      const r = h.clients.find((x) => x.client_id === c.client_id); return r && r.participated ? [h.round, key === "loss" ? r.local_val_loss : r.local_val_accuracy * 100] : null;
    }).filter(Boolean) })) });
  const dist = barChart({ title: "Training images per hospital, by class", categories: clients.map((c) => c.name), stacked: true, yLabel: "Images",
    series: classes.map((cls, i) => ({ name: cls, color: color(i), values: clients.map((c) => c.class_counts_train[cls]) })) });
  const testCounts = classes.map((c) => fed.test.per_class[c].support);
  const pooled = classes.map((cls) => clients.reduce((a, c) => a + c.class_counts_train[cls] + c.class_counts_val[cls], 0));
  const balance = barChart({ title: "Class distribution of the data used", categories: classes, yLabel: "Images",
    series: [{ name: "Hospitals (train + local validation)", color: "#F2F2F2", values: pooled }, { name: "Held-out test set", color: "#8C8C8C", values: testCounts }] });
  const pr = comm.per_round, MB = 1024 ** 2;
  const commChart = barChart({ title: "Data transferred per round (MB)", categories: pr.map((p) => String(p.round)), stacked: true, yLabel: "MB per round",
    series: [{ name: "Downlink (global model to hospitals)", color: "#5A5A5A", values: pr.map((p) => p.downlink_bytes / MB) },
      { name: "Uplink (hospital updates)", color: "#F2F2F2", values: pr.map((p) => p.uplink_bytes / MB) }] });
  const pc = fed.test.per_class;
  return `${header("Federated analytics", "Training behaviour round by round, per hospital, and in communication cost.")}
  ${tiles([
    ["Rounds completed", String(hist.length), `best round ${best} (by federated validation)`],
    ["Test accuracy", pct(fed.test.accuracy), `95% interval ${pct(ci[0])} to ${pct(ci[1])}`],
    ["Macro F1", fed.test.f1_macro.toFixed(3), `precision ${fed.test.precision_macro.toFixed(3)}, recall ${fed.test.recall_macro.toFixed(3)}`],
    ["Training traffic", bytes(comm.total_training_transfer_bytes), `${comm.rounds} rounds, ${comm.num_clients} hospitals, both directions`],
  ], 4)}
  ${section("Global model by round", "Federated validation is scored by the hospitals on their own validation data; the held-out test set is reported for reference and never used to pick the model.")}
  <div class="grid g2">${accChart}${lossChart}</div>
  ${section("Hospital performance by round", "Each hospital's local model after local training, on its own validation images.")}
  <div class="grid g2">${perHosp("acc", "Validation accuracy per hospital", "Accuracy (%)", 100)}${perHosp("loss", "Local validation loss per hospital", "Cross-entropy loss")}</div>
  ${section("Data distribution", "How the images were split between hospitals, and the class balance of the whole dataset.")}
  <div class="grid g2">${dist}${balance}</div>
  ${section("Communication overhead", "Real serialised message sizes.")}
  ${commChart}
  <p class="cap">Raw images held by the hospitals: ${bytes(comm.raw_data_bytes_total)}. Total federated training traffic: ${bytes(comm.total_training_transfer_bytes)} (${pct(comm.ratios.total_training_transfer_to_total_raw)} of the raw data volume). Evaluation broadcasts (${bytes(comm.evaluation_downlink_bytes)}) are counted separately.</p>
  ${section("Final global model on the held-out test set", `Round ${best}, ${num(fed.test.num_samples)} test images.`)}
  <div class="grid g2">
    <div>${table(["Class", "Precision", "Recall", "F1", "Images"], Object.entries(pc).map(([k, v]) => [esc(k), v.precision.toFixed(3), v.recall.toFixed(3), v.f1.toFixed(3), num(v.support)]), [1, 2, 3, 4])}</div>
    <div>${confusion(fed.test.confusion_matrix, fed.test.class_names)}</div>
  </div>
  ${section("Round-by-round results")}
  ${table(["Round", "Hospitals", "Fed. val. acc.", "Fed. val. loss", "Test acc.", "Test loss", "Test macro-F1", "Round time"],
    hist.map((h) => [h.round, h.participating_clients.length, pct(h.global_val_accuracy, 2), h.global_val_loss.toFixed(4), pct(h.global_test.accuracy, 2), h.global_test.loss.toFixed(4), h.global_test.f1_macro.toFixed(4), `${h.round_seconds.toFixed(1)} s`]), [0, 1, 2, 3, 4, 5, 6, 7])}
  <div style="margin-top:10px"><button class="btn" id="csv">Download as CSV</button></div>`;
}
function bindAnalytics(d) {
  $("#csv").addEventListener("click", () => {
    const head = "Round,Hospitals participating,Fed. validation accuracy,Fed. validation loss,Test accuracy,Test loss,Test macro-F1,Round time (s)";
    const rows = d.history.map((h) => [h.round, h.participating_clients.length, h.global_val_accuracy.toFixed(4), h.global_val_loss.toFixed(4),
      h.global_test.accuracy.toFixed(4), h.global_test.loss.toFixed(4), h.global_test.f1_macro.toFixed(4), h.round_seconds.toFixed(1)].join(","));
    download("federated_rounds.csv", [head, ...rows].join("\n"), "text/csv");
  });
}

function pagePrivacy(d) {
  const { privacy: priv, communication: comm, clients } = d;
  const n = clients.length, sh = priv.shared_with_server, r = comm.ratios;
  const cells = ['<div class="hd" style="grid-column:1/3;grid-row:1">Inside each hospital (never leaves)</div>',
    '<div class="hd" style="grid-column:4;grid-row:1">Sent to the server</div>', '<div class="hd" style="grid-column:6;grid-row:1">Central server</div>'];
  clients.forEach((c, i) => {
    const row = i + 2;
    cells.push(`<div class="cell local" style="grid-column:1;grid-row:${row}"><b>${esc(c.name)}: raw MRI</b><span>${num(c.num_train + c.num_val)} images, ${bytes(c.raw_data_bytes)}</span></div>`);
    cells.push(`<div class="cell local" style="grid-column:2;grid-row:${row}"><b>Local training</b><span>on local data only</span></div>`);
    cells.push(`<div class="arrow" style="grid-column:3;grid-row:${row}"></div>`);
    cells.push(`<div class="cell" style="grid-column:4;grid-row:${row}"><b>Model update</b><span>${bytes(comm.avg_update_bytes)} of parameters</span></div>`);
    cells.push(`<div class="arrow" style="grid-column:5;grid-row:${row}"></div>`);
  });
  cells.push(`<div class="cell server" style="grid-column:6;grid-row:2 / span ${n}"><b>FedAvg aggregation</b><span>weighted average over ${comm.num_clients} hospitals</span><b style="margin-top:12px">Global model</b><span>returned to every hospital</span></div>`);
  const nUpdates = comm.num_clients * comm.rounds;
  const totalImgs = Object.values(priv.raw_images_per_hospital).reduce((a, b) => a + b, 0);
  return `<div class="pagehead"><h1>Your data stays where it belongs.</h1><p>Privacy and security in NeuroFed AI: what never leaves a hospital, what is shared, and what this prototype does not protect against.</p></div>
  ${callout("Raw patient MRI data never leaves the hospital.", "strong")}
  ${section("Data flow", "Dashed boxes are inside a hospital. Only the model update crosses the boundary.")}
  <div class="priv">${cells.join("")}</div>
  ${callout(`Instead of transferring <b>${bytes(comm.raw_data_bytes_total)}</b> of raw MRI data, the ${comm.num_clients} hospitals exchanged model updates of <b>${bytes(comm.avg_update_bytes)}</b> each over ${comm.rounds} rounds (<b>${bytes(comm.total_training_transfer_bytes)}</b> in total).`, "info")}
  ${section("What stays local, what is shared")}
  <div class="grid g2">
    <div class="card"><h4>Stays in the hospital</h4><div class="sub">${chip("Never transmitted")}</div>${priv.stays_local.map((x) => `<p>${esc(x)}</p>`).join("")}${kv(Object.entries(priv.raw_data_location))}</div>
    <div class="card"><h4>Shared with the server</h4><div class="sub">${chip("Model updates only", "warn")}</div>
      ${kv([["Model parameters", `${num(sh.model_parameters)} values in ${sh.model_parameter_tensors} tensors`], ["Size of one update", bytes(sh.update_bytes)], ["Updates received", num(nUpdates)]])}
      <p><b>Sent with each update:</b> ${esc(sh.per_update_metadata.join(", "))}.</p>
      <p>The server checks every incoming message: tensor count, shapes and floating-point type must match the global model (${num(sh.server_side_payload_validation.updates_validated)} updates validated).</p></div>
  </div>
  ${section("The numbers", "Computed from the files and messages of the actual run.")}
  ${tiles([["Raw dataset held by hospitals", bytes(comm.raw_data_bytes_total), `${num(totalImgs)} images`],
    ["Model update size", bytes(comm.avg_update_bytes), `${pct(r.update_to_total_raw, 2)} of the raw dataset`],
    ["Total training traffic", bytes(comm.total_training_transfer_bytes), `${pct(r.total_training_transfer_to_total_raw)} of the raw dataset size`]], 3)}
  ${section("Automated checks on this run")}
  ${table(["Check", "Result"], priv.audit.checks.map((c) => [esc(c.check), c.passed ? chip("Passed", "ok") : chip("Failed", "warn")]))}
  ${section("Security mechanisms", "Only what is implemented is claimed.")}
  ${table(["Mechanism", "Status"], priv.mechanisms.map((x) => [esc(x.name), x.implemented ? chip("Implemented", "ok") : chip("Not implemented", "off")]))}
  ${callout(`<b>Limitations.</b><ul>${priv.limitations.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`, "warn")}`;
}

function pageComparison(d) {
  const { experiment: exp, centralized: cen, federated: fed, communication: comm, history: hist } = d;
  const ft = fed.test, ct = cen.test, fci = fed.test_accuracy_ci95, cci = cen.test_accuracy_ci95;
  const diff = (a, b, pts = true) => { const x = a - b; return pts ? `${x * 100 >= 0 ? "+" : ""}${(x * 100).toFixed(2)} pts` : `${x >= 0 ? "+" : ""}${x.toFixed(4)}`; };
  const rows = [
    ["Accuracy", pct(ct.accuracy, 2), pct(ft.accuracy, 2), diff(ft.accuracy, ct.accuracy)],
    ["95% interval for accuracy", `${pct(cci[0])} to ${pct(cci[1])}`, `${pct(fci[0])} to ${pct(fci[1])}`, "-"],
    ["Precision (macro)", ct.precision_macro.toFixed(4), ft.precision_macro.toFixed(4), diff(ft.precision_macro, ct.precision_macro, false)],
    ["Recall (macro)", ct.recall_macro.toFixed(4), ft.recall_macro.toFixed(4), diff(ft.recall_macro, ct.recall_macro, false)],
    ["F1-score (macro)", ct.f1_macro.toFixed(4), ft.f1_macro.toFixed(4), diff(ft.f1_macro, ct.f1_macro, false)],
    ["Test loss", ct.loss.toFixed(4), ft.loss.toFixed(4), diff(ft.loss, ct.loss, false)],
    ["Training", `${cen.epochs} epochs (best: ${cen.best_epoch})`, `${fed.rounds} rounds x ${fed.local_epochs} local epochs (best: round ${fed.best_round})`, "-"],
    ["Raw MRI data transferred", bytes(comm.centralized_alternative_bytes), "0 B", "-"],
    ["Model traffic (training)", "0 B", bytes(comm.total_training_transfer_bytes), "-"],
    ["Training time (this machine)", `${(cen.training_seconds / 60).toFixed(1)} min`, `${(fed.training_seconds / 60).toFixed(1)} min`, "-"],
  ];
  const metrics = [["Accuracy", "accuracy"], ["Precision", "precision_macro"], ["Recall", "recall_macro"], ["F1", "f1_macro"]];
  const side = barChart({ title: "Test-set metrics side by side", categories: metrics.map((x) => x[0]), yLabel: "Percent", yMax: 100,
    series: [{ name: "Centralized", color: "#8C8C8C", values: metrics.map(([, k]) => ct[k] * 100) }, { name: "Federated (FedAvg)", color: "#F2F2F2", values: metrics.map(([, k]) => ft[k] * 100) }] });
  const curves = lineChart({ title: "Validation accuracy by amount of training", xLabel: "Epoch-equivalents (federated: rounds x local epochs)", yLabel: "Validation accuracy (%)", yMin: 0, yMax: 100,
    series: [{ name: "Centralized (per epoch)", color: "#8C8C8C", points: cen.history.map((e) => [e.epoch, e.val_accuracy * 100]) },
      { name: "Federated (per round)", color: "#F2F2F2", points: hist.map((h) => [h.round * fed.local_epochs, h.global_val_accuracy * 100]) }] });
  const names = ft.class_names;
  let fedprox = "";
  if (d.fedprox) {
    const p = d.fedprox, pt = p.test, pci = p.test_accuracy_ci95;
    const recall = (t, c) => t.per_class[c]?.recall;
    const hospC = (cl) => cl?.find((c) => c.name === "Hospital C")?.at_best_round?.global_on_local_val_accuracy;
    fedprox = `${section("FedAvg vs FedProx", `Same hospitals, data split, model and training budget. FedProx adds a proximal term (mu = ${p.mu ?? 0.01}) that keeps each hospital's model close to the global model.`)}
    ${table(["Metric", "FedAvg", "FedProx"], [
      ["Accuracy", pct(ft.accuracy), pct(pt.accuracy)],
      ["95% interval", `${pct(fci[0])} to ${pct(fci[1])}`, `${pct(pci[0])} to ${pct(pci[1])}`],
      ["Macro F1", pct(ft.f1_macro), pct(pt.f1_macro)],
      ["Meningioma recall", pct(recall(ft, "meningioma")), pct(recall(pt, "meningioma"))],
      ["Hospital C: global model on its validation data", pct(hospC(d.clients)), pct(hospC(d.fedproxClients))],
      ["Selected round", String(fed.best_round), String(p.best_round)],
    ], [1, 2])}
    ${callout("The intervals overlap, so FedProx does not close the gap here. Limiting client drift cannot recover knowledge that only one hospital holds: almost all meningioma images are at Hospital C.", "info")}`;
  }
  const overlap = !(fci[1] < cci[0] || cci[1] < fci[0]);
  const notes = [
    "This is one training run with one random seed and one data split.",
    overlap ? "The 95% intervals for accuracy overlap, so the difference is small compared with the uncertainty from the size of the test set."
      : "The 95% intervals for accuracy do not overlap on this test set; a single run still cannot separate the training method from run-to-run variation.",
    `The hospital split is ${exp.config.partition === "dirichlet" ? `non-IID (Dirichlet, alpha ${exp.config.alpha})` : "IID"}. Results change with how unevenly the data is spread between hospitals.`,
    "Both runs use the same architecture, optimiser and total amount of training per image, so differences reflect the training setup, not the model.",
  ];
  return `${header("Centralized vs federated", "The same model, data and held-out test set, trained two ways.")}
  ${section("How the approaches differ")}
  ${table(["Feature", "Centralized", "Federated"], [["Raw data location", "Central repository", "Local hospital"], ["Local training", "No", "Yes"],
    ["Model aggregation", "No", "FedAvg"], ["Raw MRI transfer", "Yes (all images pooled)", "No"], ["Multiple clients", "No", "Yes"]])}
  ${section("Measured results", `Held-out test set: ${num(ft.num_samples)} images that no hospital or baseline trained on.`)}
  ${table(["Metric", "Centralized", "Federated", "Federated minus centralized"], rows, [1, 2, 3])}
  <div class="grid g2" style="margin-top:16px">${side}${curves}</div>
  ${section("Per-class F1 on the test set")}
  ${table(["Class", "Centralized", "Federated", "Difference"], names.map((n) => [esc(n), ct.per_class[n].f1.toFixed(3), ft.per_class[n].f1.toFixed(3), diff(ft.per_class[n].f1, ct.per_class[n].f1, false)]), [1, 2, 3])}
  ${fedprox}
  ${section("How to read this")}
  ${callout(`<ul>${notes.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`)}`;
}

function pageAbout(d) {
  const rep = d.dataset, exp = d.experiment, cfgm = d.modelConfig, cfg = exp?.config || {};
  const cl = exp?.partition?.cleaning;
  const tabs = [
    ["Objective", `<h3>Project objective</h3><p>Build and evaluate a brain tumour MRI classifier trained with federated learning, so that several hospitals can contribute to one model without sharing patient images. The work covers local hospital training, FedAvg aggregation, communication cost, privacy analysis, a comparison with centralized training and HL7 FHIR export.</p>
      <h3>Problem statement</h3><p>Medical imaging models improve with more data, but MRI scans are sensitive records bound by regulation and hospital policy. The question this project studies: can hospitals that keep their data on site jointly train a useful classifier, and what does that cost in accuracy and communication?</p>
      <h3>Early-screening angle</h3><p>The system is an AI-assisted early screening research prototype: it flags which class of MRI an image most resembles. It does not predict tumour stage, grade or growth rate.</p>`],
    ["Federated learning", `<h3>How it works</h3><ol><li>The server sends the current global model to the hospitals.</li><li>Each hospital trains it locally on its own images for a few epochs.</li><li>Each hospital sends back its updated parameters, never images.</li><li>The server combines them into a new global model.</li></ol>
      <h3>Federated averaging (FedAvg)</h3><p>The server computes a weighted average of the hospital parameters, weighting each hospital by its number of training samples.</p><pre class="json" style="max-height:none">w_global = sum_k ( n_k / N ) * w_k        N = sum_k n_k</pre>
      <h3>Privacy</h3><p>The privacy mechanism implemented here is data locality. Differential privacy, secure aggregation and encryption are not implemented; see Privacy &amp; security for the exact status.</p>`],
    ["Data and model", `<h3>Dataset</h3>${rep ? `<p>Public Kaggle Brain Tumor MRI dataset (${esc(rep.structure)}). Classes are detected from folder names.</p>
      ${table(["Class", "Images"], rep.class_names.map((c) => [esc(c), num(rep.class_counts[c])]), [1])}
      <p class="cap">${num(rep.valid_images)} readable images, ${rep.corrupted_images} corrupted. Exact duplicate copies found: ${rep.duplicates.redundant_copies} (${rep.duplicates.groups_spanning_splits} groups span train and test).${cl && cl.enabled ? ` Removed before splitting: ${cl.dropped_exact_duplicates} duplicate copies and ${cl.dropped_conflicting_label_images} label-conflicting images.` : ""}</p>` : ""}
      <h3>CNN model</h3>${cfgm ? `<p>A compact convolutional network (${num(cfgm.architecture.trainable_parameters)} trainable parameters): convolution blocks with group normalisation and pooling, global average pooling, a small dense layer and a softmax. Group normalisation replaces batch normalisation because batch statistics average poorly across hospitals with different data.</p>
      ${table(["Layer", "Type", "Output shape", "Parameters"], cfgm.architecture.layers.map((l) => [esc(l.name), esc(l.type), esc(l.output_shape), num(l.params)]), [3])}` : ""}
      <h3>Running in your browser</h3><p>On this website the trained model runs in plain JavaScript inside your browser. The image is converted and resized exactly as in training, so predictions match the Python model. Uploaded images are never sent to a server.</p>`],
    ["Training and evaluation", `<h3>Training setup</h3>${table(["Setting", "Value"], [["Hospitals", String(cfg.num_clients)],
      ["Data split", cfg.partition === "dirichlet" ? `Non-IID, Dirichlet alpha ${cfg.alpha}` : "IID (stratified)"], ["Rounds x local epochs", `${cfg.rounds} x ${cfg.local_epochs}`],
      ["Optimiser", `Adam, learning rate ${cfg.lr}, batch size ${cfg.batch_size}`], ["Input", `${cfg.img_size}x${cfg.img_size}, ${cfg.grayscale ? "grayscale" : "RGB"}`], ["Seed", String(cfg.seed)]])}
      <h3>Evaluation</h3><p>Each hospital keeps a local validation split; the round with the best federated validation accuracy becomes the global model. The final model is scored once on a held-out test set that belongs to no hospital. Accuracy is reported with a 95% Wilson interval, and a centralized baseline with the same architecture and amount of training is scored on the same test set.</p>`],
    ["Limitations and future scope", `<h3>Limitations</h3><ul><li>Research prototype; not clinically validated and not a diagnostic tool.</li><li>Hospitals are simulated on one machine.</li><li>The dataset is public and 2D; results may not transfer to real hospital data.</li><li>Model updates are unprotected: no differential privacy, secure aggregation or encryption.</li><li>Results come from single runs with one seed.</li></ul>
      <h3>Future scope</h3><ul><li>Differential privacy and secure aggregation, with the accuracy cost measured.</li><li>SCAFFOLD, personalised and clustered federated learning for non-IID data.</li><li>Repeated experiments across seeds and heterogeneity levels.</li><li>Deployment across real machines and multi-centre data.</li></ul>${disclaimer()}`],
  ];
  return `${header("About and methodology", "Federated learning for brain tumour classification with privacy preservation. A research prototype.")}
  <div class="tabs" role="tablist">${tabs.map(([t], i) => `<button role="tab" aria-selected="${i === 0}" data-tab="${i}">${esc(t)}</button>`).join("")}</div>
  ${tabs.map(([, body], i) => `<div class="prose ${i ? "hidden" : ""}" data-panel="${i}" role="tabpanel">${body}</div>`).join("")}`;
}
function bindAbout() {
  document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-tab]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    document.querySelectorAll("[data-panel]").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== b.dataset.tab));
  }));
}

// ---------------- diagnose ----------------
const MODELS = {};
async function getModel(kind) {
  if (!MODELS[kind]) MODELS[kind] = loadModel("models", kind);
  return MODELS[kind];
}
const state = { kind: "federated", file: null, gray: null, size: null, url: null };

function pageDiagnose(d) {
  const cfg = d.modelConfig, algo = cfg.algorithm === "fedprox" ? "FedProx" : "FedAvg";
  return `<div class="statusbar"><span>Workspace / <b>Diagnose</b></span><span class="right">${chip("Runs in your browser", "ok")}<span>Federated round ${cfg.best_round}, ${algo}, ${num(cfg.trainable_parameters)} parameters</span></span></div>
  ${header("Diagnose", "Classify a brain MRI slice and export the result as an HL7 FHIR R4 report. The image is analysed in your browser and is never uploaded.")}
  <div class="grid g-diag" style="align-items:start">
    <div>
      <label class="drop" id="drop" tabindex="0">
        <input id="file" type="file" accept="image/jpeg,image/png,image/bmp,image/webp" class="hidden">
        <b id="droptitle">Choose an MRI image</b><span>or drop it here. JPG, PNG, BMP or WebP.</span>
      </label>
      <div style="margin-top:14px"><div class="label" style="margin-bottom:6px">Model</div>
        <div class="seg" role="group" aria-label="Model"><button data-kind="federated" aria-pressed="true">Federated</button><button data-kind="centralized" aria-pressed="false">Centralized</button></div>
        <p class="cap">Federated: trained across hospitals without sharing images. Centralized: baseline trained on pooled data, for comparison.</p></div>
      <div id="imgcard" class="card hidden" style="margin-top:14px"><div class="label" style="margin-bottom:8px">Input image</div><img id="preview" class="preview-img" alt="Uploaded MRI slice"><div id="meta" style="margin-top:8px"></div></div>
    </div>
    <div id="resultcol">${callout("Choose an MRI image to see the predicted class and create the FHIR report.", "info")}${disclaimer()}</div>
  </div>
  <div id="fhir"></div>`;
}

async function decode(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "none", colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const c = document.createElement("canvas");
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const { data } = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close?.();
  return { gray: resizeLikePillow(rgbaToL(data, c.width, c.height), c.width, c.height, 128, 128), size: [c.width, c.height] };
}

async function runPrediction(d) {
  if (!state.gray) return;
  const col = $("#resultcol");
  col.innerHTML = `<div class="card"><span class="spinner"></span>Loading the ${state.kind} model and analysing the image…</div>`;
  try {
    const model = await getModel(state.kind);
    await new Promise((r) => setTimeout(r, 20));
    const probs = predictProbs(model, state.gray);
    const res = { ...summarise([...probs], d.modelConfig.class_names), original_size: state.size };
    renderResult(d, res);
  } catch (e) {
    col.innerHTML = callout(`The model could not be loaded: ${esc(e.message)}. Check your connection and reload the page.`, "warn");
  }
}

function renderResult(d, res) {
  const top = Math.max(...Object.values(res.probabilities));
  const probs = Object.entries(res.probabilities).map(([n, p]) =>
    `<div class="prob"><span style="font-weight:${p === top ? 600 : 400}">${esc(n)}</span><div class="bar"><i style="width:${Math.max(p * 100, 0.6).toFixed(1)}%;background:${p === top ? "#F2F2F2" : "#4A4A4A"}"></i></div><em>${(p * 100).toFixed(1)}%</em></div>`).join("");
  let note = "";
  if (res.low_confidence) note = callout(`<b>Low confidence.</b> The model is only ${(res.confidence * 100).toFixed(1)}% sure; the next class is ${esc(res.runner_up || "-")}. The image may be unlike the training data.`, "warn");
  else if (res.screen_positive === true) note = callout("An abnormal class was predicted. In a real setting this result would be reviewed by a qualified clinician before any decision.", "info");
  else if (res.screen_positive === false) note = callout("No tumour class was predicted. This does not rule out disease.", "info");
  $("#resultcol").innerHTML = `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap"><span class="label">Prediction, ${esc(state.kind)} model</span>${res.low_confidence ? chip("Low confidence", "warn") : chip("Requires clinician review", "warn")}</div>
    <div class="result">${esc(res.predicted_class)}</div>
    <div style="font-size:14px;color:var(--muted);margin-bottom:10px">Confidence <b class="mono" style="color:var(--ink)">${(res.confidence * 100).toFixed(1)}%</b>, uncalibrated softmax probability</div>
    <div class="label" style="margin-bottom:2px">Class probabilities</div>${probs}</div>${note}${disclaimer()}`;
  renderFhir(d, res);
}

function renderFhir(d, res) {
  const cfg = d.modelConfig;
  const algo = state.kind === "federated" ? (cfg.algorithm === "fedprox" ? "FedProx" : "FedAvg") : "centralized";
  const modelInfo = { kind: state.kind, algorithm: algo, best_round: cfg.best_round, trained_at: cfg.trained_at, img_size: cfg.img_size, runtime: "browser (JavaScript)" };
  const wrap = $("#fhir");
  wrap.innerHTML = `${section("FHIR R4 report", "Export the result for hospital information systems as an HL7 FHIR R4 bundle, or as a printable report. Both are marked preliminary and identify the patient by pseudonym only.")}
    <div class="grid g2" style="align-items:start">
      <div><label class="label" for="pseudo">Patient pseudonym</label><input id="pseudo" class="input" value="CASE-0001" style="margin:6px 0 4px">
        <p class="cap" style="margin-top:0">Use a pseudonym only. Never enter a real name or ID number.</p>
        <div id="restable" style="margin:12px 0"></div>
        <div style="display:flex;gap:10px;flex-wrap:wrap"><button class="btn primary" id="dlfhir">Download FHIR bundle</button><button class="btn" id="dlreport">Printable report</button></div>
        <p class="cap">Open the report in a browser and use Print, then Save as PDF.</p></div>
      <div><div class="label" style="margin-bottom:6px">Bundle preview</div><pre class="json" id="jsonprev"></pre></div>
    </div>`;
  const stem = (state.file?.name || "mri").replace(/\.[^.]+$/, "");
  const build = () => {
    const pseudonym = $("#pseudo").value.trim();
    const reportHtml = buildReportHtml({ result: res, modelKind: state.kind, pseudonym, fileName: state.file?.name || "image", modelInfo });
    const bundle = buildBundle({ result: res, modelKind: state.kind, modelInfo, pseudonym, reportHtml });
    return { bundle, reportHtml };
  };
  const refresh = () => {
    const { bundle } = build();
    $("#jsonprev").textContent = JSON.stringify(previewBundle(bundle), null, 2);
    const describe = (r) => r.resourceType === "Patient" ? "Pseudonymous identifier only"
      : r.resourceType === "Observation" ? `Class '${r.valueCodeableConcept.text}' + ${r.component.length} probabilities`
      : `LOINC ${r.code.coding[0].code}, ${r.code.coding[0].display}`;
    $("#restable").innerHTML = table(["Resource", "Content", "Status"], bundle.entry.map((e) => [`<span class="mono">${esc(e.resource.resourceType)}</span>`, esc(describe(e.resource)),
      e.resource.status === "preliminary" ? chip("Preliminary", "warn") : chip("Ready", "ok")]), [2]);
  };
  $("#pseudo").addEventListener("input", refresh);
  $("#dlfhir").addEventListener("click", () => download(`${stem}_fhir.json`, JSON.stringify(build().bundle, null, 2), "application/fhir+json"));
  $("#dlreport").addEventListener("click", () => download(`${stem}_report.html`, build().reportHtml, "text/html"));
  refresh();
}

function bindDiagnose(d) {
  const input = $("#file"), drop = $("#drop");
  const handle = async (file) => {
    if (!file) return;
    if (!/^image\/(jpeg|png|bmp|webp)$/.test(file.type)) {
      $("#resultcol").innerHTML = callout("This file type is not supported here. Use a JPG, PNG, BMP or WebP image.", "warn");
      return;
    }
    try {
      const { gray, size } = await decode(file);
      Object.assign(state, { file, gray, size });
      if (state.url) URL.revokeObjectURL(state.url);
      state.url = URL.createObjectURL(file);
      $("#preview").src = state.url;
      $("#meta").innerHTML = kv([["File", file.name], ["Original size", `${size[0]} x ${size[1]} px`], ["Model input", "128 x 128, grayscale"], ["Uploaded data", "Stays in your browser"]]);
      $("#imgcard").classList.remove("hidden");
      $("#droptitle").textContent = "Choose another image";
      runPrediction(d);
    } catch {
      $("#resultcol").innerHTML = callout("This file could not be read as an image. Try another JPG or PNG file.", "warn");
    }
  };
  input.addEventListener("change", () => handle(input.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => handle(e.dataTransfer.files[0]));
  document.querySelectorAll("[data-kind]").forEach((b) => b.addEventListener("click", () => {
    state.kind = b.dataset.kind;
    document.querySelectorAll("[data-kind]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    if (state.gray) runPrediction(d);
  }));
  getModel("federated").catch(() => {}); // warm up the default model
  if (state.gray) {
    $("#preview").src = state.url;
    $("#meta").innerHTML = kv([["File", state.file.name], ["Original size", `${state.size[0]} x ${state.size[1]} px`], ["Model input", "128 x 128, grayscale"], ["Uploaded data", "Stays in your browser"]]);
    $("#imgcard").classList.remove("hidden");
    runPrediction(d);
  }
}

// ---------------- router ----------------
const ROUTES = {
  "": ["Home", pageHome], diagnose: ["Diagnose", pageDiagnose, bindDiagnose], network: ["Federated network", pageNetwork, bindNetwork],
  analytics: ["Federated analytics", pageAnalytics, bindAnalytics], privacy: ["Privacy & security", pagePrivacy],
  comparison: ["Centralized vs federated", pageComparison], about: ["About", pageAbout, bindAbout],
};
async function render() {
  const key = location.hash.replace(/^#\/?/, "");
  const route = ROUTES[key] ? key : "";
  const [title, page, bind] = ROUTES[route];
  document.querySelectorAll(".nav a").forEach((a) => (a.getAttribute("href") === `#/${route}` ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  document.querySelector(".side").classList.remove("open");
  document.title = `${title} | NeuroFed AI`;
  const app = $("#app");
  try {
    const d = await loadData();
    if (!d.metrics || !d.federated) { app.innerHTML = callout("The training results could not be loaded. Run tools/export_web_model.py and redeploy.", "warn"); return; }
    app.innerHTML = page(d);
    bind?.(d);
  } catch (e) {
    app.innerHTML = callout(`Something went wrong while drawing this page: ${esc(e.message)}`, "warn");
  }
  window.scrollTo(0, 0);
  $("#app").focus({ preventScroll: true });
}
document.querySelector(".menu-btn").addEventListener("click", () => {
  const s = document.querySelector(".side");
  s.classList.toggle("open");
  document.querySelector(".menu-btn").setAttribute("aria-expanded", String(s.classList.contains("open")));
});
window.addEventListener("hashchange", render);
render();
