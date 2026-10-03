"""Printable screening report (open in a browser, then Print -> Save as PDF)."""
from __future__ import annotations

from datetime import datetime
from html import escape

import config


def build_report_html(*, result: dict, model_kind: str, pseudonym: str, file_name: str, model_info: dict) -> str:
    probs = "".join(
        f"<tr><td>{escape(n)}</td><td class='num'>{p * 100:.1f}%</td>"
        f"<td><div class='bar'><i style='width:{p * 100:.1f}%'></i></div></td></tr>"
        for n, p in result["probabilities"].items())
    warn = ("<p class='warn'><b>Low confidence.</b> Treat this result as uncertain.</p>"
            if result["low_confidence"] else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroFed AI - Screening report</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,600&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
<style>
 body {{ font-family:'Source Sans 3','Segoe UI',sans-serif; color:#211D1A; background:#F8F5EF; margin:0; font-size:15px; line-height:1.55; }}
 .page {{ max-width:820px; margin:2rem auto; background:#fff; padding:2.4rem 2.8rem; border:1px solid #E4DCCD; }}
 .mast {{ border-bottom:2px solid #5B1A24; padding-bottom:.6rem; display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; }}
 .mast h1 {{ font-family:'Cormorant Garamond',Georgia,serif; font-style:italic; font-weight:600; font-size:2.2rem; color:#5B1A24; margin:0; }}
 .mast .meta {{ font-size:.8rem; color:#5C554D; text-align:right; }}
 h2 {{ font-family:'Cormorant Garamond',Georgia,serif; font-style:italic; font-weight:600; color:#5B1A24; font-size:1.4rem; margin:1.5rem 0 .5rem; }}
 .result {{ font-family:'Cormorant Garamond',Georgia,serif; font-size:2rem; font-weight:600; font-style:italic; color:#211D1A; }}
 table {{ width:100%; border-collapse:collapse; }}
 th {{ font-size:.8rem; color:#5C554D; text-align:left; border-bottom:1px solid #5B1A24; padding:.35rem .4rem; }}
 td {{ border-bottom:1px solid #EFE7D7; padding:.35rem .4rem; }}
 .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .bar {{ background:#F2ECE1; height:6px; border-radius:99px; overflow:hidden; }} .bar i {{ display:block; height:6px; background:#5B1A24; }}
 .small {{ font-size:.9rem; color:#5C554D; }}
 .warn {{ background:#F8EEDC; border:1px solid #E8D6AE; padding:.5rem .8rem; }}
 .disc {{ margin-top:2rem; border-top:1px solid #E4DCCD; padding-top:.6rem; font-size:.85rem; color:#5C554D; }}
 @media print {{ body {{ background:#fff; }} .page {{ border:0; margin:0; }} }}
 @media (max-width:640px) {{ .page {{ padding:1.4rem; }} }}
</style></head><body><div class="page">
 <div class="mast"><h1>NeuroFed AI</h1>
  <div class="meta">AI-assisted MRI screening report<br>{datetime.now().strftime('%d %B %Y, %H:%M')}</div></div>
 <h2>Case</h2>
 <table>
  <tr><td>Pseudonym</td><td>{escape(pseudonym or 'anonymous')}</td></tr>
  <tr><td>Image</td><td>{escape(file_name)} ({result['original_size'][0]} &times; {result['original_size'][1]} px)</td></tr>
  <tr><td>Model</td><td>{escape(model_kind.capitalize())} CNN &middot; {escape(str(model_info.get('algorithm', 'FedAvg')))} &middot; trained {escape(str(model_info.get('trained_at', '-'))[:10])}</td></tr>
 </table>
 <h2>Finding</h2>
 <div class="result">{escape(result['predicted_class'])}</div>
 <p class="small">Probability {result['confidence'] * 100:.1f}% (uncalibrated softmax). Runner-up: {escape(result['runner_up'] or '-')}.</p>
 {warn}
 <table><tr><th>Class</th><th class='num'>Probability</th><th></th></tr>{probs}</table>
 <div class="disc">{escape(config.MEDICAL_DISCLAIMER)} Status: preliminary. Must be reviewed by a qualified clinician.</div>
</div></body></html>"""
