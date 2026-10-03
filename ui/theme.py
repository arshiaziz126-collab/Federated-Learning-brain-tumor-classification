"""Visual identity: 'Medical Journal Pro'.

Cormorant Garamond (italic) for page and section titles only; Source Sans 3 for all interface text;
Source Code Pro for codes and numbers. Ivory ground, oxblood sidebar and accent, restrained borders.
The colour constants keep their historical names (FOREST, TEAL, ...) so every page and chart picks up
the palette without further changes.
"""
from __future__ import annotations

import re

import streamlit as st

FOREST = "#5B1A24"   # oxblood - primary accent, headings
TEAL = "#7A2A35"     # lighter claret - secondary accent
CREAM = "#F8F5EF"    # ivory ground
INK = "#211D1A"
MUTED = "#5C554D"
LINE = "#E4DCCD"
SAND = "#F2ECE1"
MINT = "#F6EEEC"     # blush - info backgrounds
SAGE = "#B08D57"     # antique gold
GOLD = "#8A6A2E"
SLATE = "#5E6670"
SIDEBAR = "#4A1620"

CLASS_COLORS = [FOREST, GOLD, SLATE, "#2C5E3A", SAGE, "#C9A9A6", "#7C6A4A", "#3E3A36"]

DISPLAY = "'Cormorant Garamond', Georgia, serif"
UI = "'Source Sans 3', 'Segoe UI', system-ui, sans-serif"
MONO = "'Source Code Pro', Consolas, monospace"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;1,500;1,600&family=Source+Sans+3:wght@400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap');
:root {{
  --forest:{FOREST}; --teal:{TEAL}; --cream:{CREAM}; --ink:{INK}; --muted:{MUTED}; --line:{LINE};
  --sand:{SAND}; --mint:{MINT}; --gold:{GOLD}; --card:#FFFFFF; --soft:#FBF8F3;
  --okbg:#EAF3EC; --ok:#2C5E3A; --warnbg:#F8EEDC; --warn:#6E4A0F; --warnline:#E8D6AE;
}}
.stApp {{ background: var(--cream); color: var(--ink); font-family: {UI}; font-size: 15px; }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {{ font-family: {UI}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stDeployButton"], [data-testid="stAppDeployButton"] {{ display: none !important; }}
/* keep the button that re-opens a collapsed sidebar visible and on-theme */
[data-testid="stToolbar"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {{ visibility: visible !important; opacity: 1 !important; }}
[data-testid="stExpandSidebarButton"] {{ display: inline-flex !important; color: var(--forest) !important; background: #FFFFFF; border: 1px solid var(--line); border-radius: 4px; }}
.block-container {{ max-width: 1200px; padding: 2rem 2rem 4rem; }}
h1, h2, h3 {{ font-family: {DISPLAY}; font-style: italic; font-weight: 600; color: var(--forest); letter-spacing: 0; }}
code, pre, .mono {{ font-family: {MONO}; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {SIDEBAR}; border-right: 0; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{ border-radius: 4px; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {{ color: #E2CDC8; font-family: {UI}; font-size: 14px; font-weight: 500; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,.08); }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(255,255,255,.14); }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: #FFFFFF; font-weight: 600; }}
.nf-side-note {{ color: #E2CDC8; font-size: 12.5px; line-height: 1.5; padding: .4rem .6rem; }}
.nf-side-note b {{ color: #FFFFFF; font-weight: 600; }}

/* page header + status bar */
.nf-statusbar {{ display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; font-size: 13px; color: var(--muted); margin-bottom: .9rem; }}
.nf-statusbar .crumb b {{ color: var(--ink); font-weight: 600; }}
.nf-statusbar .right {{ margin-left: auto; display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }}
.nf-pagehead {{ margin: 0 0 1.4rem; padding-bottom: .8rem; border-bottom: 1px solid var(--line); }}
.nf-pagehead h1 {{ font-size: 2.4rem; line-height: 1.1; margin: 0 0 .3rem; padding: 0; }}
.nf-pagehead p {{ color: var(--muted); font-size: 15px; margin: 0; max-width: 72ch; }}
.nf-h {{ font-family: {DISPLAY}; font-style: italic; font-weight: 600; font-size: 1.55rem; color: var(--forest); margin: 1.8rem 0 .2rem; }}
.nf-hcap {{ color: var(--muted); font-size: 14px; margin: 0 0 .9rem; max-width: 80ch; }}
.nf-eyebrow, .nf-label {{ font-size: 12px; font-weight: 600; color: var(--muted); letter-spacing: .02em; }}

/* hero */
.nf-hero {{ background: {SIDEBAR}; color: #fff; border-radius: 4px; padding: 2.2rem 2.4rem 2rem; margin-bottom: 1.4rem; }}
.nf-hero h1 {{ color: #fff; font-size: 2.6rem; line-height: 1.1; margin: 0 0 .5rem; padding: 0; }}
.nf-hero .sub {{ color: #E8D8D3; font-size: 16px; margin: 0 0 1.2rem; max-width: 60ch; }}
.nf-hero .pledge {{ display: inline-block; border: 1px solid rgba(255,255,255,.35); border-radius: 999px; padding: .3rem .9rem; font-size: 12.5px; font-weight: 600; color: #fff; margin-bottom: 1.4rem; }}

/* tiles / cards */
.nf-tile {{ min-height: 108px; background: var(--card); border: 1px solid var(--line); border-radius: 4px; padding: 1rem 1.1rem; height: 100%; }}
.nf-tile .l {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
.nf-tile .v {{ color: var(--ink); font-family: {UI}; font-size: 1.75rem; font-weight: 700; line-height: 1.2; margin: .25rem 0 .1rem; font-variant-numeric: tabular-nums; }}
.nf-tile .c {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
.nf-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 4px; padding: 1.2rem 1.35rem; height: 100%; }}
.nf-card h4 {{ margin: 0 0 .15rem; font-family: {DISPLAY}; font-style: italic; font-weight: 600; font-size: 1.35rem; color: var(--ink); }}
.nf-card .sub {{ color: var(--muted); font-size: 13.5px; margin-bottom: .8rem; }}
.nf-card p {{ font-size: 14.5px; line-height: 1.6; margin: .2rem 0 .5rem; }}
.nf-kv {{ display: flex; justify-content: space-between; gap: 1rem; padding: .45rem 0; border-top: 1px solid var(--line); font-size: 14px; }}
.nf-kv:first-of-type {{ border-top: 0; }}
.nf-kv .k {{ color: var(--muted); }}
.nf-kv .val {{ color: var(--ink); font-weight: 600; text-align: right; font-family: {MONO}; font-size: 13px; }}
.nf-callout {{ border-radius: 4px; padding: .7rem .9rem; font-size: 14px; line-height: 1.5; margin: .55rem 0; border: 1px solid var(--line); background: var(--soft); color: var(--ink); }}
.nf-callout.info {{ background: var(--mint); border-color: #EAD8D3; color: var(--forest); }}
.nf-callout.warn {{ background: var(--warnbg); border-color: var(--warnline); color: var(--warn); }}
.nf-callout.strong {{ background: var(--forest); border-color: var(--forest); color: #fff; font-weight: 500; }}
.nf-badge, .nf-chip {{ display: inline-block; border-radius: 999px; padding: .15rem .65rem; font-size: 12px; font-weight: 600; white-space: nowrap; background: var(--mint); color: var(--forest); border: 1px solid #EAD8D3; }}
.nf-badge.off, .nf-chip.off {{ background: #F1ECE2; color: var(--muted); border-color: var(--line); }}
.nf-badge.warn, .nf-chip.warn {{ background: var(--warnbg); color: var(--warn); border-color: var(--warnline); }}
.nf-chip.ok {{ background: var(--okbg); color: var(--ok); border-color: #CFE3D5; }}

/* stacked class bar */
.nf-stack {{ display: flex; height: 8px; border-radius: 99px; overflow: hidden; background: var(--sand); margin: .35rem 0 .5rem; }}
.nf-stack span {{ display: block; height: 100%; }}
.nf-legend {{ display: flex; flex-wrap: wrap; gap: .25rem .9rem; font-size: 13px; color: var(--muted); }}
.nf-legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }}

/* flow diagram */
.nf-flow {{ display: flex; align-items: stretch; flex-wrap: wrap; row-gap: 12px; }}
.nf-flow .col {{ display: flex; flex-direction: column; gap: 8px; justify-content: center; }}
.nf-node {{ border: 1px solid var(--line); background: var(--card); border-radius: 4px; padding: .5rem .8rem; min-width: 112px; font-size: 13.5px; color: var(--ink); }}
.nf-node b {{ display: block; color: var(--ink); font-weight: 600; font-size: 14px; }}
.nf-node span {{ display: block; color: var(--muted); font-size: 12.5px; margin-top: 1px; }}
.nf-node.solid {{ background: var(--forest); border-color: var(--forest); }}
.nf-node.solid b {{ color: #fff; }} .nf-node.solid span {{ color: #E8D8D3; }}
.nf-link {{ align-self: center; width: 26px; height: 1px; background: var(--muted); position: relative; margin: 0 6px; flex: none; }}
.nf-link::after {{ content: ""; position: absolute; right: -2px; top: -4px; border-left: 7px solid var(--muted); border-top: 4.5px solid transparent; border-bottom: 4.5px solid transparent; }}
.on-dark .nf-node {{ background: rgba(255,255,255,.06); border-color: rgba(255,255,255,.25); color: #fff; }}
.on-dark .nf-node b {{ color: #fff; }} .on-dark .nf-node span {{ color: #E8D8D3; }}
.on-dark .nf-node.solid {{ background: #fff; border-color: #fff; }}
.on-dark .nf-node.solid b {{ color: var(--forest); }} .on-dark .nf-node.solid span {{ color: var(--muted); }}
.on-dark .nf-link {{ background: #E8D8D3; }} .on-dark .nf-link::after {{ border-left-color: #E8D8D3; }}

/* probability rows and result */
.nf-prob {{ display: grid; grid-template-columns: 120px 1fr 64px; align-items: center; gap: .7rem; padding: .45rem 0; border-top: 1px solid var(--line); font-size: 14px; }}
.nf-prob .bar {{ height: 6px; background: var(--soft); border: 1px solid var(--line); border-radius: 99px; overflow: hidden; }}
.nf-prob .bar i {{ display: block; height: 100%; }}
.nf-prob em {{ font-style: normal; text-align: right; font-family: {MONO}; font-size: 13px; color: var(--ink); }}
.nf-result {{ font-family: {DISPLAY}; font-style: italic; font-size: 2.5rem; font-weight: 600; color: var(--ink); line-height: 1.05; margin: .2rem 0 .15rem; }}

/* privacy boundary diagram */
.nf-priv {{ display: grid; grid-template-columns: 1fr 1fr 34px 1fr 34px 1.05fr; column-gap: 0; row-gap: 10px; align-items: stretch; }}
.nf-priv .hd {{ font-size: 12px; font-weight: 600; color: var(--muted); padding: 0 .2rem .1rem; }}
.nf-priv .cell {{ border-radius: 4px; padding: .55rem .75rem; font-size: 13.5px; border: 1px solid var(--line); background: var(--card); }}
.nf-priv .cell b {{ display: block; font-weight: 600; color: var(--ink); }}
.nf-priv .cell span {{ display: block; color: var(--muted); font-size: 12.5px; }}
.nf-priv .local {{ background: var(--mint); border: 1px dashed var(--teal); }}
.nf-priv .server {{ background: var(--forest); border-color: var(--forest); display: flex; flex-direction: column; justify-content: center; }}
.nf-priv .server b {{ color: #fff; }} .nf-priv .server span {{ color: #E8D8D3; }}
.nf-priv .arrow {{ align-self: center; height: 1px; background: var(--muted); position: relative; }}
.nf-priv .arrow::after {{ content: ""; position: absolute; right: -2px; top: -4px; border-left: 7px solid var(--muted); border-top: 4.5px solid transparent; border-bottom: 4.5px solid transparent; }}
@media (max-width: 760px) {{ .nf-priv {{ grid-template-columns: 1fr; }} .nf-priv .arrow {{ display: none; }} .nf-priv > * {{ grid-column: 1 !important; grid-row: auto !important; }} }}

/* tables */
.nf-table {{ width: 100%; border-collapse: collapse; font-size: 14px; background: var(--card); border: 1px solid var(--line); border-radius: 4px; }}
.nf-table th {{ text-align: left; font-size: 12px; font-weight: 600; color: var(--muted); padding: .6rem .8rem; border-bottom: 1px solid var(--line); background: var(--soft); }}
.nf-table td {{ padding: .5rem .8rem; border-top: 1px solid var(--line); vertical-align: top; }}
.nf-table td.num, .nf-table th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.nf-table td.code {{ font-family: {MONO}; font-size: 13px; }}
.nf-cm td.cm {{ text-align: center; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 13.5px; }}
.nf-wrap {{ overflow-x: auto; }}
.nf-code {{ margin: 0; padding: .9rem 1rem; background: #2A1F1C; color: #EFE6DA; border-radius: 4px; font-family: {MONO}; font-size: 12.5px; line-height: 1.55; overflow-x: auto; white-space: pre; }}

/* native widgets */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--card); border-color: var(--line) !important; border-radius: 4px; }}
[data-testid="stPlotlyChart"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 4px; padding: .6rem .6rem .2rem; }}
[data-testid="stFileUploader"] section {{ background: var(--card); border: 1px dashed #CDBFA5; border-radius: 4px; }}
[data-testid="stImage"] img {{ border-radius: 4px; }}
[data-testid="stImageCaption"], [data-testid="caption"] {{ font-size: 13px; color: var(--muted); }}
.stButton > button, .stDownloadButton > button {{ border-radius: 4px; border: 1px solid var(--line); color: var(--ink); background: var(--card); font-family: {UI}; font-size: 14px; font-weight: 600; min-height: 40px; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--forest); color: var(--forest); }}
.stDownloadButton.primary > button, .nf-primary .stDownloadButton > button {{ background: var(--forest); color: #fff; border-color: var(--forest); }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ font-family: {UI}; font-size: 14px; font-weight: 600; color: var(--muted); }}
.stTabs [aria-selected="true"] {{ color: var(--forest); }}
[data-testid="stExpander"] summary p {{ font-size: 14px; font-weight: 600; color: var(--ink); }}
a {{ color: var(--forest); }}
:focus-visible {{ outline: 2px solid var(--forest); outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def compact(html: str) -> str:
    """Collapse newlines/indentation so Markdown never mistakes HTML for a code block."""
    return re.sub(r"\s*\n\s*", " ", html).strip()


def html(s: str) -> None:
    st.markdown(compact(s), unsafe_allow_html=True)
