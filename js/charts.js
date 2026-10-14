// Minimal SVG charts in the dashboard's palette (no external library).
const W = 640, H = 300, M = { l: 52, r: 16, t: 14, b: 40 };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function niceTicks(min, max, n = 5) {
  const span = max - min || 1;
  const step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function frame({ xTicks, yTicks, xs, ys, xLabel, yLabel, xFmt = (v) => v, yFmt = (v) => v }) {
  let g = "";
  for (const t of yTicks) {
    const y = ys(t);
    g += `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" stroke="#262626"/>`;
    g += `<text x="${M.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#9A9A9A">${esc(yFmt(t))}</text>`;
  }
  for (const t of xTicks) {
    g += `<text x="${xs(t)}" y="${H - M.b + 16}" text-anchor="middle" font-size="11" fill="#9A9A9A">${esc(xFmt(t))}</text>`;
  }
  g += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="#3A3A3A"/>`;
  if (xLabel) g += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="#9A9A9A">${esc(xLabel)}</text>`;
  if (yLabel) g += `<text transform="translate(13 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#9A9A9A">${esc(yLabel)}</text>`;
  return g;
}

function legend(series) {
  return `<div class="legend">${series.map((s) => `<div><i style="background:${s.color}"></i>${esc(s.name)}</div>`).join("")}</div>`;
}

export function lineChart({ title, series, xLabel, yLabel, yMin, yMax, xTicks, marks = [], yFmt }) {
  const allX = series.flatMap((s) => s.points.map((p) => p[0]));
  const allY = series.flatMap((s) => s.points.map((p) => p[1]));
  const x0 = Math.min(...allX), x1 = Math.max(...allX);
  const y0 = yMin ?? Math.min(...allY), y1 = yMax ?? Math.max(...allY);
  const xs = (v) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const ys = (v) => H - M.b - ((v - y0) / (y1 - y0 || 1)) * (H - M.t - M.b);
  let g = frame({ xTicks: xTicks || niceTicks(x0, x1, 8), yTicks: niceTicks(y0, y1, 5), xs, ys, xLabel, yLabel, yFmt });
  for (const s of series) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
    if (s.dots !== false) for (const p of s.points) g += `<circle cx="${xs(p[0])}" cy="${ys(p[1])}" r="2.6" fill="${s.color}"><title>${esc(s.name)}: ${p[1].toFixed(2)}</title></circle>`;
  }
  for (const mk of marks) {
    g += `<line x1="${xs(mk.x)}" x2="${xs(mk.x)}" y1="${M.t}" y2="${H - M.b}" stroke="#6B6B6B" stroke-dasharray="4 4"/>`;
    g += `<text x="${xs(mk.x) + 5}" y="${M.t + 11}" font-size="11" fill="#9A9A9A">${esc(mk.label)}</text>`;
  }
  return `<div class="chart">${title ? `<h5>${esc(title)}</h5>` : ""}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title || "Line chart")}">${g}</svg>${legend(series)}</div>`;
}

export function barChart({ title, categories, series, stacked = false, yLabel, yMin = 0, yMax, yFmt }) {
  const totals = categories.map((_, i) => (stacked ? series.reduce((a, s) => a + s.values[i], 0) : Math.max(...series.map((s) => s.values[i]))));
  const top = yMax ?? Math.max(...totals) * 1.08;
  const xs0 = (i) => M.l + (i + 0.5) * ((W - M.l - M.r) / categories.length);
  const ys = (v) => H - M.b - ((v - yMin) / (top - yMin || 1)) * (H - M.t - M.b);
  const band = (W - M.l - M.r) / categories.length;
  let g = frame({ xTicks: categories.map((_, i) => i), yTicks: niceTicks(yMin, top, 5), xs: xs0, ys, yLabel, xFmt: (i) => categories[i], yFmt });
  categories.forEach((cat, i) => {
    if (stacked) {
      let acc = yMin;
      const bw = band * 0.5;
      for (const s of series) {
        const v = s.values[i];
        g += `<rect x="${xs0(i) - bw / 2}" y="${ys(acc + v)}" width="${bw}" height="${Math.max(0, ys(acc) - ys(acc + v))}" fill="${s.color}"><title>${esc(cat)}, ${esc(s.name)}: ${v}</title></rect>`;
        acc += v;
      }
    } else {
      const bw = (band * 0.7) / series.length;
      series.forEach((s, j) => {
        const v = s.values[i], x = xs0(i) - (band * 0.7) / 2 + j * bw;
        g += `<rect x="${x}" y="${ys(v)}" width="${bw - 2}" height="${Math.max(0, ys(yMin) - ys(v))}" fill="${s.color}"><title>${esc(cat)}, ${esc(s.name)}: ${typeof v === "number" ? v.toFixed(2) : v}</title></rect>`;
      });
    }
  });
  return `<div class="chart">${title ? `<h5>${esc(title)}</h5>` : ""}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title || "Bar chart")}">${g}</svg>${legend(series)}</div>`;
}
