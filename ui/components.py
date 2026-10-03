"""Reusable HTML/Plotly building blocks. All numbers come from the arguments - none are defined here."""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

import config
from ui.theme import CLASS_COLORS, FOREST, TEAL, compact, html

FONT = "Source Sans 3, Segoe UI, sans-serif"


# --- formatting ------------------------------------------------------------------------------
def pct(x, digits: int = 1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def num(x, digits: int = 0) -> str:
    return "-" if x is None else f"{x:,.{digits}f}"


def class_color(index: int) -> str:
    return CLASS_COLORS[index % len(CLASS_COLORS)]


# --- structure --------------------------------------------------------------------------------
def page_header(title: str, subtitle: str) -> None:
    html(f'<div class="nf-pagehead"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>')


def section(title: str, caption: str | None = None) -> None:
    html(f'<div class="nf-h">{escape(title)}</div>' + (f'<div class="nf-hcap">{escape(caption)}</div>' if caption else ""))


def tile(label: str, value: str, caption: str = "") -> str:
    return compact(f'<div class="nf-tile"><div class="l">{escape(label)}</div><div class="v">{escape(value)}</div>'
                   f'<div class="c">{escape(caption)}</div></div>')


def tiles(items: list) -> None:
    """items: list of (label, value, caption)."""
    cols = st.columns(len(items))
    for col, (label, value, caption) in zip(cols, items):
        col.markdown(tile(label, value, caption), unsafe_allow_html=True)


def callout(text: str, kind: str = "") -> None:
    html(f'<div class="nf-callout {kind}">{text}</div>')


def badge(text: str, kind: str = "") -> str:
    return f'<span class="nf-badge {kind}">{escape(text)}</span>'


def kv_rows(rows: list) -> str:
    return "".join(f'<div class="nf-kv"><span class="k">{escape(str(k))}</span><span class="val">{escape(str(v))}</span></div>'
                   for k, v in rows)


def card(title: str, subtitle: str, body_html: str) -> str:
    return compact(f'<div class="nf-card"><h4>{escape(title)}</h4><div class="sub">{subtitle}</div>{body_html}</div>')


def empty_state(what: str) -> None:
    """Shown when results are missing. Tells the user exactly how to produce them."""
    callout(f"<b>{escape(what)}</b> has not been generated yet. The dashboard only displays results produced by the "
            "training script, so run it first.", "warn")
    st.markdown("**1.** Put the Kaggle dataset in `data/brain_tumor_dataset/`  \n"
                "**2.** Inspect it (optional): `python inspect_dataset.py`  \n"
                "**3.** Train: `python federated_train.py`  \n"
                "**4.** Reload this page.")


def disclaimer() -> None:
    callout(escape(config.MEDICAL_DISCLAIMER), "")


# --- class distribution -----------------------------------------------------------------------
def class_bar(counts: dict, class_names: list) -> str:
    total = sum(counts.get(c, 0) for c in class_names) or 1
    segs = "".join(
        f'<span style="width:{counts.get(c, 0) / total * 100:.2f}%;background:{class_color(i)}" '
        f'title="{escape(c)}: {counts.get(c, 0)}"></span>' for i, c in enumerate(class_names))
    legend = "".join(f'<div><i style="background:{class_color(i)}"></i>{escape(c)} {counts.get(c, 0):,}</div>'
                     for i, c in enumerate(class_names))
    return f'<div class="nf-stack">{segs}</div><div class="nf-legend">{legend}</div>'


# --- diagrams -----------------------------------------------------------------------------------
def flow_html(hospitals: list, stages: list, dark: bool = False) -> str:
    """hospitals: [(name, caption)]; stages: [(title, caption, solid?)]. Hospitals feed the first stage."""
    hosp = "".join(f'<div class="nf-node"><b>{escape(n)}</b><span>{escape(c)}</span></div>' for n, c in hospitals)
    parts = [f'<div class="col">{hosp}</div>']
    for title, cap, solid in stages:
        parts.append('<div class="nf-link"></div>')
        parts.append(f'<div class="col"><div class="nf-node{" solid" if solid else ""}"><b>{escape(title)}</b>'
                     f'<span>{escape(cap)}</span></div></div>')
    return compact(f'<div class="nf-flow{" on-dark" if dark else ""}">{"".join(parts)}</div>')


def flow_diagram(hospitals: list, stages: list, dark: bool = False) -> None:
    html(flow_html(hospitals, stages, dark))


def prob_rows(probabilities: dict, class_names: list) -> str:
    """Horizontal probability bars, in the order given by ``probabilities``."""
    out = []
    top = max(probabilities.values()) if probabilities else 0
    for name, p in probabilities.items():
        colour = FOREST if p == top else "#9A9188"
        weight = "600" if p == top else "400"
        out.append(f'<div class="nf-prob"><span style="font-weight:{weight}">{escape(name)}</span>'
                   f'<div class="bar"><i style="width:{max(p * 100, 0.6):.1f}%;background:{colour}"></i></div>'
                   f'<em>{p * 100:.1f}%</em></div>')
    return "".join(out)


def privacy_diagram(rows: list, update_caption: str, server_caption: str) -> None:
    """rows: [(hospital name, local image caption)]. Draws the data-locality boundary."""
    n = len(rows)
    cells = ['<div class="hd" style="grid-column:1/3;grid-row:1">Inside each hospital (never leaves)</div>',
             '<div class="hd" style="grid-column:4;grid-row:1">Sent to the server</div>',
             '<div class="hd" style="grid-column:6;grid-row:1">Central server</div>']
    for i, (name, cap) in enumerate(rows):
        r = i + 2
        cells.append(f'<div class="cell local" style="grid-column:1;grid-row:{r}"><b>{escape(name)}: raw MRI</b><span>{escape(cap)}</span></div>')
        cells.append(f'<div class="cell local" style="grid-column:2;grid-row:{r}"><b>Local training</b><span>on local data only</span></div>')
        cells.append(f'<div class="arrow" style="grid-column:3;grid-row:{r};width:100%"></div>')
        cells.append(f'<div class="cell" style="grid-column:4;grid-row:{r}"><b>Model update</b><span>{escape(update_caption)}</span></div>')
        cells.append(f'<div class="arrow" style="grid-column:5;grid-row:{r};width:100%"></div>')
    cells.append(f'<div class="cell server" style="grid-column:6;grid-row:2 / span {n}"><b>FedAvg aggregation</b>'
                 f'<span>{escape(server_caption)}</span><b style="margin-top:.9rem">Global model</b>'
                 f'<span>returned to every hospital</span></div>')
    html(f'<div class="nf-priv">{"".join(cells)}</div>')


def confusion_table(cm: list, class_names: list) -> None:
    """Confusion matrix as a shaded table (rows = true class, columns = predicted class)."""
    head = "".join(f"<th class='num'>{escape(c)}</th>" for c in class_names)
    body = []
    for i, row in enumerate(cm):
        total = sum(row) or 1
        cells = "".join(
            f"<td class='cm' style='background:rgba(23,107,91,{0.06 + 0.7 * v / total:.2f});"
            f"color:{'#fff' if v / total > 0.5 else '#1E1B18'}'>{v}</td>" for v in row)
        body.append(f"<tr><td><b>{escape(class_names[i])}</b></td>{cells}</tr>")
    html(f'<div class="nf-wrap"><table class="nf-table nf-cm"><tr><th>True \\ Predicted</th>{head}</tr>{"".join(body)}</table></div>')


def simple_table(headers: list, rows: list, num_cols: tuple = ()) -> None:
    th = "".join(f"<th class='{'num' if i in num_cols else ''}'>{escape(h)}</th>" for i, h in enumerate(headers))
    trs = "".join("<tr>" + "".join(
        f"<td class='{'num' if i in num_cols else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    html(f'<div class="nf-wrap"><table class="nf-table"><tr>{th}</tr>{trs}</table></div>')


# --- charts -------------------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 340, legend: bool = True, ytitle: str | None = None,
              xtitle: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=12, r=16, t=44, b=12), font=dict(family=FONT, size=13, color="#211D1A"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        colorway=[FOREST, "#9C7A3C", "#5E6670", TEAL, "#B08D57", "#C9A9A6"],
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None) if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=False, linecolor="#CDBFA5", ticks="outside", tickcolor="#CDBFA5", title=xtitle,
                     automargin=True, title_standoff=8)
    fig.update_yaxes(gridcolor="#EFE7D7", zeroline=False, title=ytitle, automargin=True, title_standoff=8)
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})

