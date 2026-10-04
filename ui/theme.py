"""Visual identity: 'Olive and Gold - Soft colour'.

Cormorant Garamond (italic) for page and section titles only; Source Sans 3 for all interface text;
Source Code Pro for codes and numbers. Soft green-cream ground, sage sidebar, olive primary, gold highlights.
The colour constants keep their historical names (FOREST, TEAL, ...) so every page and chart picks up
the palette without further changes.
"""
from __future__ import annotations

import re

import streamlit as st

# ---- historical names (used by pages and charts) ----
FOREST = "#2E4220"   # deep olive - headings, strong text accents
TEAL = "#6F8F52"     # mid olive - secondary accent
CREAM = "#F1F3E0"    # page background
INK = "#26331C"      # body text
MUTED = "#4E5C3F"    # secondary text
LINE = "#CBD6A8"     # borders and dividers
SAND = "#E4EAD0"     # track / neutral fills
MINT = "#E7EED2"     # green card / info background
SAGE = "#D6A93E"     # gold accent
GOLD = "#A8821F"     # darker gold for text and charts
SLATE = "#6B7466"
SIDEBAR = "#E1E8CB"

# ---- new names for the soft colour palette ----
GREEN = "#5F8443"        # primary green: active menu, buttons, solid fills
HERO = "#D4E0AE"         # hero banner
HERO_LINE = "#B5C888"    # hero border
BOX = "#F6EFCB"          # flow boxes
GOLD_CARD = "#FBF1D0"    # gold-tinted card
GOLD_PILL = "#F6E3A3"    # gold pill background
GOLD_TEXT = "#443206"    # text on gold

CLASS_COLORS = [GREEN, SAGE, SLATE, "#2F5E2A", GOLD, HERO_LINE, "#8A7A4A", "#3E4A33"]

DISPLAY = "'Cormorant Garamond', Georgia, serif"
UI = "'Source Sans 3', 'Segoe UI', system-ui, sans-serif"
MONO = "'Source Code Pro', Consolas, monospace"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;1,500;1,600&family=Source+Sans+3:wght@400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap');
:root {{
  --forest:{FOREST}; --teal:{TEAL}; --cream:{CREAM}; --ink:{INK}; --muted:{MUTED}; --line:{LINE};
  --sand:{SAND}; --mint:{MINT}; --gold:{GOLD}; --card:{GOLD_CARD}; --soft:#EEF2DA;
  --primary:{GREEN}; --hero:{HERO}; --heroline:{HERO_LINE}; --box:{BOX};
  --goldfill:{SAGE}; --goldpill:{GOLD_PILL}; --goldtext:{GOLD_TEXT};
  --okbg:#E3EDD0; --ok:#2F5E2A; --warnbg:{GOLD_PILL}; --warn:{GOLD_TEXT}; --warnline:{SAGE};
}}
.stApp {{ background: var(--cream); color: var(--ink); font-family: {UI}; font-size: 15px; }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {{ font-family: {UI}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stDeployButton"], [data-testid="stAppDeployButton"] {{ display: none !important; }}
/* keep the button that re-opens a collapsed sidebar visible and on-theme */
[data-testid="stToolbar"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {{ visibility: visible !important; opacity: 1 !important; }}
[data-testid="stExpandSidebarButton"] {{ display: inline-flex !important; color: var(--forest) !important; background: var(--card); border: 1px solid var(--line); border-radius: 4px; }}
.block-container {{ max-width: 1200px; padding: 2rem 2rem 4rem; }}
h1, h2, h3 {{ font-family: {DISPLAY}; font-style: italic; font-weight: 600; color: var(--forest); letter-spacing: 0; }}
code, pre, .mono {{ font-family: {MONO}; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {SIDEBAR}; border-right: 1px solid var(--line); }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{ border-radius: 6px; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {{ color: var(--forest); font-family: {UI}; font-size: 14px; font-weight: 500; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{ background: rgba(46,66,32,.08); }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: var(--primary); }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: #FFFFFF; font-weight: 600; }}
.nf-side-note {{ color: var(--muted); font-size: 12.5px; line-height: 1.5; padding: .4rem .6rem; }}
.nf-side-note b {{ color: var(--forest); font-weight: 600; }}

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

/* hero (light olive banner) */
.nf-hero {{ background: var(--hero); color: var(--ink); border: 1px solid var(--heroline); border-radius: 12px; padding: 2.2rem 2.4rem 2rem; margin-bottom: 1.4rem; }}
.nf-hero h1 {{ color: var(--forest); font-size: 2.6rem; line-height: 1.1; margin: 0 0 .5rem; padding: 0; }}
.nf-hero .sub {{ color: var(--muted); font-size: 16px; margin: 0 0 1.2rem; max-width: 60ch; }}
.nf-hero .pledge {{ display: inline-block; border: 1px solid var(--goldfill); background: var(--goldpill); border-radius: 999px; padding: .3rem .9rem; font-size: 12.5px; font-weight: 600; color: var(--goldtext); margin-bottom: 1.4rem; }}

/* tiles / cards (tiles alternate gold and green across columns) */
.nf-tile {{ min-height: 108px; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.1rem; height: 100%; }}
[data-testid="stColumn"]:nth-child(even) .nf-tile, [data-testid="column"]:nth-child(even) .nf-tile {{ background: var(--mint); }}
.nf-tile .l {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
.nf-tile .v {{ color: var(--forest); font-family: {UI}; font-size: 1.75rem; font-weight: 700; line-height: 1.2; margin: .25rem 0 .1rem; font-variant-numeric: tabular-nums; }}
.nf-tile .c {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
.nf-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1.2rem 1.35rem; height: 100%; }}
.nf-card h4 {{ margin: 0 0 .15rem; font-family: {DISPLAY}; font-style: italic; font-weight: 600; font-size: 1.35rem; color: var(--forest); }}
.nf-card .sub {{ color: var(--muted); font-size: 13.5px; margin-bottom: .8rem; }}
.nf-card p {{ font-size: 14.5px; line-height: 1.6; margin: .2rem 0 .5rem; }}
.nf-kv {{ display: flex; justify-content: space-between; gap: 1rem; padding: .45rem 0; border-top: 1px solid var(--line); font-size: 14px; }}
.nf-kv:first-of-type {{ border-top: 0; }}
.nf-kv .k {{ color: var(--muted); }}
.nf-kv .val {{ color: var(--ink); font-weight: 600; text-align: right; font-family: {MONO}; font-size: 13px; }}
.nf-callout {{ border-radius: 8px; padding: .7rem .9rem; font-size: 14px; line-height: 1.5; margin: .55rem 0; border: 1px solid var(--line); background: var(--soft); color: var(--ink); }}
.nf-callout.info {{ background: var(--mint); border-color: var(--heroline); color: var(--forest); }}
.nf-callout.warn {{ background: var(--warnbg); border-color: var(--warnline); color: var(--warn); }}
.nf-callout.strong {{ background: var(--primary); border-color: var(--primary); color: #fff; font-weight: 500; }}
.nf-badge, .nf-chip {{ display: inline-block; border-radius: 999px; padding: .15rem .65rem; font-size: 12px; font-weight: 600; white-space: nowrap; background: var(--mint); color: var(--forest); border: 1px solid var(--heroline); }}
.nf-badge.off, .nf-chip.off {{ background: #E9ECD6; color: var(--muted); border-color: var(--line); }}
.nf-badge.warn, .nf-chip.warn {{ background: var(--warnbg); color: var(--warn); border-color: var(--warnline); }}
.nf-chip.ok {{ background: var(--okbg); color: var(--ok); border-color: #B9D2A6; }}

/* stacked class bar */
.nf-stack {{ display: flex; height: 8px; border-radius: 99px; overflow: hidden; background: var(--sand); margin: .35rem 0 .5rem; }}
.nf-stack span {{ display: block; height: 100%; }}
.nf-legend {{ display: flex; flex-wrap: wrap; gap: .25rem .9rem; font-size: 13px; color: var(--muted); }}
.nf-legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }}

/* flow diagram */
.nf-flow {{ display: flex; align-items: stretch; flex-wrap: wrap; row-gap: 12px; }}
.nf-flow .col {{ display: flex; flex-direction: column; gap: 8px; justify-content: center; }}
.nf-node {{ border: 1px solid var(--line); background: var(--card); border-radius: 6px; padding: .5rem .8rem; min-width: 112px; font-size: 13.5px; color: var(--ink); }}
.nf-node b {{ display: block; color: var(--ink); font-weight: 600; font-size: 14px; }}
.nf-node span {{ display: block; color: var(--muted); font-size: 12.5px; margin-top: 1px; }}
.nf-node.solid {{ background: var(--primary); border-color: var(--primary); }}
.nf-node.solid b {{ color: #fff; }} .nf-node.solid span {{ color: #E3EBD0; }}
.nf-link {{ align-self: center; width: 26px; height: 1px; background: var(--muted); position: relative; margin: 0 6px; flex: none; }}
.nf-link::after {{ content: ""; position: absolute; right: -2px; top: -4px; border-left: 7px solid var(--muted); border-top: 4.5px solid transparent; border-bottom: 4.5px solid transparent; }}
/* "on-dark" class is kept for the hero; the hero is light now, so these are light styles */
.on-dark .nf-node {{ background: var(--box); border-color: var(--heroline); color: var(--forest); }}
.on-dark .nf-node b {{ color: var(--forest); }} .on-dark .nf-node span {{ color: var(--muted); }}
.on-dark .nf-node.solid {{ background: var(--goldfill); border-color: var(--goldfill); }}
.on-dark .nf-node.solid b {{ color: var(--goldtext); }} .on-dark .nf-node.solid span {{ color: var(--goldtext); }}
.on-dark .nf-link {{ background: var(--forest); }} .on-dark .nf-link::after {{ border-left-color: var(--forest); }}

/* probability rows and result */
.nf-prob {{ display: grid; grid-template-columns: 120px 1fr 64px; align-items: center; gap: .7rem; padding: .45rem 0; border-top: 1px solid var(--line); font-size: 14px; }}
.nf-prob .bar {{ height: 6px; background: var(--soft); border: 1px solid var(--line); border-radius: 99px; overflow: hidden; }}
.nf-prob .bar i {{ display: block; height: 100%; }}
.nf-prob em {{ font-style: normal; text-align: right; font-family: {MONO}; font-size: 13px; color: var(--ink); }}
.nf-result {{ font-family: {DISPLAY}; font-style: italic; font-size: 2.5rem; font-weight: 600; color: var(--forest); line-height: 1.05; margin: .2rem 0 .15rem; }}

/* privacy boundary diagram */
.nf-priv {{ display: grid; grid-template-columns: 1fr 1fr 34px 1fr 34px 1.05fr; column-gap: 0; row-gap: 10px; align-items: stretch; }}
.nf-priv .hd {{ font-size: 12px; font-weight: 600; color: var(--muted); padding: 0 .2rem .1rem; }}
.nf-priv .cell {{ border-radius: 6px; padding: .55rem .75rem; font-size: 13.5px; border: 1px solid var(--line); background: var(--card); }}
.nf-priv .cell b {{ display: block; font-weight: 600; color: var(--ink); }}
.nf-priv .cell span {{ display: block; color: var(--muted); font-size: 12.5px; }}
.nf-priv .local {{ background: var(--mint); border: 1px dashed var(--teal); }}
.nf-priv .server {{ background: var(--primary); border-color: var(--primary); display: flex; flex-direction: column; justify-content: center; }}
.nf-priv .server b {{ color: #fff; }} .nf-priv .server span {{ color: #E3EBD0; }}
.nf-priv .arrow {{ align-self: center; height: 1px; background: var(--muted); position: relative; }}
.nf-priv .arrow::after {{ content: ""; position: absolute; right: -2px; top: -4px; border-left: 7px solid var(--muted); border-top: 4.5px solid transparent; border-bottom: 4.5px solid transparent; }}
@media (max-width: 760px) {{ .nf-priv {{ grid-template-columns: 1fr; }} .nf-priv .arrow {{ display: none; }} .nf-priv > * {{ grid-column: 1 !important; grid-row: auto !important; }} }}

/* tables */
.nf-table {{ width: 100%; border-collapse: collapse; font-size: 14px; background: var(--card); border: 1px solid var(--line); border-radius: 6px; }}
.nf-table th {{ text-align: left; font-size: 12px; font-weight: 600; color: var(--muted); padding: .6rem .8rem; border-bottom: 1px solid var(--line); background: var(--soft); }}
.nf-table td {{ padding: .5rem .8rem; border-top: 1px solid var(--line); vertical-align: top; }}
.nf-table td.num, .nf-table th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.nf-table td.code {{ font-family: {MONO}; font-size: 13px; }}
.nf-cm td.cm {{ text-align: center; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 13.5px; }}
.nf-wrap {{ overflow-x: auto; }}
.nf-code {{ margin: 0; padding: .9rem 1rem; background: var(--mint); color: var(--forest); border: 1px solid var(--line); border-radius: 6px; font-family: {MONO}; font-size: 12.5px; line-height: 1.55; overflow-x: auto; white-space: pre; }}

/* native widgets */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--card); border-color: var(--line) !important; border-radius: 8px; }}
[data-testid="stPlotlyChart"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: .6rem .6rem .2rem; }}
[data-testid="stFileUploader"] section {{ background: var(--card); border: 1px dashed var(--heroline); border-radius: 8px; }}
[data-testid="stImage"] img {{ border-radius: 6px; }}
[data-testid="stImageCaption"], [data-testid="caption"] {{ font-size: 13px; color: var(--muted); }}
.stButton > button, .stDownloadButton > button {{ border-radius: 6px; border: 1px solid var(--line); color: var(--ink); background: var(--card); font-family: {UI}; font-size: 14px; font-weight: 600; min-height: 40px; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--primary); color: var(--forest); }}
.stDownloadButton.primary > button, .nf-primary .stDownloadButton > button {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ font-family: {UI}; font-size: 14px; font-weight: 600; color: var(--muted); }}
.stTabs [aria-selected="true"] {{ color: var(--forest); }}
[data-testid="stExpander"] summary p {{ font-size: 14px; font-weight: 600; color: var(--ink); }}
a {{ color: var(--primary); }}
:focus-visible {{ outline: 2px solid var(--primary); outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def compact(html: str) -> str:
    """Collapse newlines/indentation so Markdown never mistakes HTML for a code block."""
    return re.sub(r"\s*\n\s*", " ", html).strip()


def html(s: str) -> None:
    st.markdown(compact(s), unsafe_allow_html=True)
