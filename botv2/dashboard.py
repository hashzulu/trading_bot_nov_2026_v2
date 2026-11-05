"""Single-page dashboard: report + charts, styled, auto-refreshing."""
from __future__ import annotations

import base64
import html
import json
import os
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import html as html_lib
import markdown as md_lib

from .config import Config

try:
    from zoneinfo import ZoneInfo
    CET = ZoneInfo("Europe/Berlin")
except Exception:                                   # pragma: no cover - no tzdata
    CET = timezone.utc

CSS = """
:root { color-scheme: dark;
        --bg: #0B0D10; --sidebar: #0D1014; --card: #111419; --elevated: #171B21;
        --border: #252B33; --text: #F2F4F7; --muted: #8D96A4;
        --green: #28C76F; --red: #EA5455; --yellow: #FFBA33;
        --ink: var(--text); --line: var(--border); --red-soft: #ff9a9a; }
* { box-sizing: border-box; }
html { scrollbar-color: #2a3646 #0a0e14; }
body { margin: 0; color: var(--text); font: 14px/1.6 Inter, -apple-system, BlinkMacSystemFont,
       "SF Pro Text", system-ui, sans-serif; background: var(--bg); }
.wrap { max-width: none; margin: 0 0 0 230px; padding: 30px 34px 56px; }
header { position: sticky; top: 0; z-index: 40; display: flex; align-items: center;
         gap: 14px; flex-wrap: wrap; padding: 16px 2px 14px;
         background: rgba(11,13,16,.94);
         backdrop-filter: blur(14px); border-bottom: 1px solid var(--border); }
h1 { font-size: 22px; margin: 0; letter-spacing: -.2px; font-weight: 700;
     color: var(--text); }
.live { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
        box-shadow: 0 0 0 0 rgba(74,222,128,.6); animation: pulse 2s infinite; }
@keyframes pulse { 70% { box-shadow: 0 0 0 9px rgba(74,222,128,0); }
                   100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); } }
.meta { color: var(--muted); font-size: 12px; letter-spacing: .2px; }
.pill { margin-left: auto; font-size: 11.5px; color: #a9e7ff; background: rgba(34,120,170,.14);
        border: 1px solid rgba(90,200,255,.28); padding: 6px 13px; border-radius: 999px;
        letter-spacing: .3px; white-space: nowrap; }
.grid { display: grid; grid-template-columns: minmax(440px, .9fr) minmax(620px, 1.3fr);
        gap: 22px; align-items: start; padding-top: 20px; }
.report { background: linear-gradient(180deg, rgba(20,27,38,.86), rgba(12,17,24,.92));
          border: 1px solid var(--line); border-radius: 20px; padding: 20px 22px;
          max-height: calc(100vh - 120px); overflow: auto;
          box-shadow: 0 24px 60px rgba(0,0,0,.35); }
.report::-webkit-scrollbar { width: 10px; }
.report::-webkit-scrollbar-thumb { background: #223042; border-radius: 8px; }
.report > h1 { display: none; }               /* title already in the sticky header */
.report h2 { font-size: 11.5px; color: var(--red); margin: 26px 0 10px; padding-bottom: 6px;
             border-bottom: 1px solid rgba(255,107,107,.18); letter-spacing: 1.4px;
             text-transform: uppercase; font-weight: 800; }
.report h2::before { content: ""; display: inline-block; width: 14px; height: 3px;
                     background: var(--red); border-radius: 3px; margin-right: 9px;
                     vertical-align: middle; }
.report h3 { font-size: 14px; color: var(--red-soft); margin: 22px 0 8px; font-weight: 700; }
.report h3.rec { color: var(--green); font-size: 15px; }
.report p { margin: 7px 0; }
.report strong { color: #fff; font-weight: 650; }
.report em { color: var(--muted); }
.report a { color: #7dd3fc; }
.report ul { margin: 8px 0 12px; padding-left: 18px; }
.report li { margin: 5px 0; }
.report li::marker { color: rgba(255,107,107,.65); }
.report table { border-collapse: separate; border-spacing: 0; width: 100%; margin: 10px 0 16px;
                font-size: 12.5px; border: 1px solid var(--line); border-radius: 12px;
                overflow: hidden; }
.report th, .report td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--line); }
.report th { color: #aab8c9; font-weight: 700; background: rgba(255,255,255,.035);
             font-size: 11px; letter-spacing: .6px; text-transform: uppercase; }
.report tr:last-child td { border-bottom: none; }
.report tr:hover td { background: rgba(255,255,255,.025); }
.report code { background: rgba(125,211,252,.1); color: #bae6fd; padding: 1px 6px;
               border-radius: 5px; font-size: 12px; }
.report hr { border: none; border-top: 1px solid var(--line); margin: 22px 0; }
p.sec { margin: 18px 0 8px; }
p.sec strong { display: inline-block; color: var(--red); font-size: 10.5px; font-weight: 800;
               letter-spacing: 1.5px; text-transform: uppercase; padding: 5px 10px;
               border-radius: 999px; background: rgba(255,107,107,.1);
               border: 1px solid rgba(255,107,107,.22); }
.charts { display: flex; flex-direction: column; gap: 18px; }
.interactive-panel { background: linear-gradient(180deg, rgba(20,27,38,.86), rgba(12,17,24,.92));
                     border: 1px solid var(--line); border-radius: 20px; padding: 14px;
                     box-shadow: 0 24px 60px rgba(0,0,0,.3); }
.chart-toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.chart-toolbar-title { font-size: 13px; font-weight: 800; color: #ffd7d7; letter-spacing: .2px; }
.chart-controls { margin-left: auto; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.chart-tab { font-size: 10px; text-transform: uppercase; background: rgba(255,255,255,.05);
             border: 1px solid rgba(255,255,255,.12); padding: 4px 9px; border-radius: 999px;
             color: #93a1b5; cursor: pointer; font-weight: 600; letter-spacing: .6px; }
.chart-tab:hover { color: #fff; border-color: rgba(255,255,255,.22); }
.chart-tab.active { background: rgba(255,107,107,.14); border-color: rgba(255,107,107,.35);
                    color: #ffd7d7; font-weight: 800; }
.chart-select { background: #0f141b; color: #e8eef6; border: 1px solid rgba(255,255,255,.15);
                border-radius: 8px; padding: 4px 8px; font-size: 11px; }
.chart-symbol { font-size: 11px; color: #93a1b5; font-weight: 700; letter-spacing: .4px; }
.lw-stack { display: flex; flex-direction: column; gap: 8px; }
.lw-price-wrap { position: relative; height: 560px; }
.lw-chart { position: absolute; inset: 0; }
.lw-title { position: absolute; left: 8px; top: 6px; z-index: 3; font-size: 12px;
            font-weight: 700; color: #e6ebf2; background: rgba(15,20,27,.78);
            border: 1px solid #2a3140; border-radius: 6px; padding: 3px 7px;
            pointer-events: none; }
.lw-overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.lw-summary { position: absolute; right: 12px; bottom: 8px; font-size: 10px; color: #c5cae9;
              background: #0f141b; border: 1px solid #2a3140; border-radius: 8px;
              padding: 4px 8px; max-width: 70%; pointer-events: none; }
.lw-tooltip { position: absolute; display: none; z-index: 5; pointer-events: none;
              background: #0f141b; border: 1px solid #2a3140; border-radius: 6px;
              padding: 3px 6px; font-size: 10px; color: #e8eef6; white-space: nowrap;
              box-shadow: 0 4px 12px rgba(0,0,0,.4); }
.lw-rsi-wrap { position: relative; height: 160px; }
.lw-rsi { position: absolute; inset: 0; }
.lw-rsi-value { position: absolute; right: 8px; top: 4px; font-size: 10px; color: #ab47bc;
                background: #0f141b; border: 1px solid #2a3140; border-radius: 6px;
                padding: 2px 6px; pointer-events: none; }
.chart-hint { color: #5f6c7e; font-size: 10.5px; padding: 6px 2px 0; line-height: 1.45; }
.chart-missing { color: #93a1b5; padding: 20px; }
.sym { background: linear-gradient(180deg, rgba(20,27,38,.86), rgba(12,17,24,.92));
       border: 1px solid var(--line); border-radius: 20px; padding: 14px 14px 10px;
       box-shadow: 0 24px 60px rgba(0,0,0,.3); transition: transform .18s ease, border-color .18s ease; }
.sym:hover { transform: translateY(-2px); border-color: rgba(255,255,255,.14); }
.sym h3 { margin: 0 0 10px; font-size: 14px; color: #ffd7d7; display: flex; gap: 8px;
          align-items: center; font-weight: 800; letter-spacing: .2px; }
.sym h3 span { color: var(--muted); font-weight: 500; font-size: 11.5px; }
.sym.rec { border-color: rgba(74,222,128,.45);
           box-shadow: 0 0 0 1px rgba(74,222,128,.25) inset, 0 24px 60px rgba(0,0,0,.3); }
.sym.rec h3 { color: var(--green); }
.levels { margin: -4px 0 10px; font-size: 11.5px; color: #a7f3c4; letter-spacing: .2px; }
.levels b { color: #eafff2; font-weight: 700; }
.badge { font-style: normal; color: #04240f; background: linear-gradient(180deg, #7ef2a5, #3ecf74);
         border-radius: 999px; padding: 4px 11px; font-size: 10.5px; font-weight: 800;
         letter-spacing: .8px; text-transform: uppercase; margin-left: 6px;
         box-shadow: 0 0 18px rgba(74,222,128,.35); }
.shots { display: grid; grid-template-columns: 1fr; gap: 12px; }
 .tracking-section { background: linear-gradient(180deg, rgba(20,27,38,.92), rgba(12,17,24,.95)); border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 16px 20px; margin: 18px 0; overflow-x: auto; }
 .tracking-section h3 { font-size: 14px; color: #ffd7d7; margin: 0 0 10px; font-weight: 800; letter-spacing: .2px; }
 .tracking-table { width: 100%; min-width: 860px; border-collapse: collapse; font-size: 11.5px; }
 .tracking-table th { color: #aab8c9; font-weight: 700; background: rgba(255,255,255,.04); text-align: left; padding: 6px 8px; font-size: 10px; text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid rgba(255,255,255,.08); }
 .tracking-table td { padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,.06); color: #e8eef6; }
 .tracking-table .hit-tp { color: #7ef2a5; }
 .tracking-table .hit-sl { color: #ff8787; }
 .badge-status { font-size: 9.5px; padding: 2px 7px; border-radius: 999px; font-weight: 700; letter-spacing: .3px; text-transform: uppercase; }
 .badge-rec { background: rgba(251,191,36,.15); color: #fcd34d; border: 1px solid rgba(251,191,36,.3); }
 .badge-open { background: rgba(74,222,128,.12); color: #4ade80; border: 1px solid rgba(74,222,128,.3); }
 .badge-tp { background: rgba(74,222,128,.18); color: #7ef2a5; border: 1px solid rgba(74,222,128,.4); }
 .badge-sl { background: rgba(255,107,107,.12); color: #ff8787; border: 1px solid rgba(255,107,107,.3); }
 .badge-manual { background: rgba(148,163,184,.12); color: #94a3b8; border: 1px solid rgba(148,163,184,.25); }
 .track-row { cursor: pointer; }
 .track-row:hover td { background: rgba(255,255,255,.035); }
 .mark-btn { display: inline-block; font-size: 9.5px; padding: 3px 7px; border-radius: 999px;
             text-decoration: none; font-weight: 700; margin: 1px 3px 1px 0; border: 1px solid; }
 .mark-tp { color: #7ef2a5; border-color: rgba(74,222,128,.4); background: rgba(74,222,128,.1); }
 .mark-tp:hover { background: rgba(74,222,128,.2); }
 .mark-sl { color: #ff8787; border-color: rgba(255,107,107,.4); background: rgba(255,107,107,.1); }
 .mark-sl:hover { background: rgba(255,107,107,.2); }
 .trade-detail-source { display: none; }
 #tdetail { position: fixed; inset: 0; z-index: 105; display: none; overflow: auto;
            background: rgba(3,6,10,.95); padding: 34px 18px; }
 #tdetail.open { display: block; }
 #tdetail .tdetail-card { max-width: 1120px; margin: 0 auto; position: relative;
            background: linear-gradient(180deg, rgba(20,27,38,.98), rgba(12,17,24,.99));
            border: 1px solid rgba(255,255,255,.12); border-radius: 20px; padding: 24px 28px 30px;
            box-shadow: 0 24px 80px rgba(0,0,0,.6); }
 #tdetail .close { position: sticky; top: 0; float: right; color: #c7d2df; font-size: 28px;
            line-height: 1; cursor: pointer; z-index: 2; }
 #tdetail .detail-title { font-size: 19px; font-weight: 800; color: #ffd7d7; letter-spacing: -.2px; }
 #tdetail .detail-meta { color: #93a1b5; font-size: 12px; margin: 4px 0 16px; letter-spacing: .2px; }
 #tdetail .detail-actions { margin: -6px 0 10px; }
 #tdetail .detail-report { max-height: none; overflow: visible; background: transparent;
            border: none; padding: 0; box-shadow: none; }
 #tdetail h3 { margin-top: 20px; }
 #tdetail .report > h1 { display: block; }
 .tab-btn { width: 100%; text-align: left; font-size: 12.5px; background: transparent;
            border: 1px solid transparent; padding: 9px 11px; border-radius: 8px;
            color: var(--muted); font-weight: 600; letter-spacing: .1px; cursor: pointer;
            display: flex; align-items: center; gap: 9px;
            transition: background .18s ease, color .18s ease, border-color .18s ease; }
 .tab-btn:hover { color: var(--text); background: var(--elevated); border-color: var(--border); }
 .tab-active { background: rgba(40,199,111,.1); border-color: rgba(40,199,111,.28); color: var(--text); font-weight: 700; }
 .tab-btn::before { content: ""; width: 3px; height: 14px; border-radius: 2px;
                    background: transparent; flex: 0 0 auto; }
 .tab-active::before { background: var(--green); }
 .main-view { display: block; }
 .tracking-view { display: none; }
 .demo-view { display: none; }
 .demo-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
              gap: 18px; padding-top: 20px; max-width: 1500px; }
 .demo-card { background: var(--card); border: 1px solid var(--border);
              border-radius: 14px; padding: 14px; }
 .demo-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
              margin-bottom: 8px; }
 .demo-head h3 { margin: 0; font-size: 14px; color: #ffd7d7; font-weight: 800; }
 .demo-head h3 span { color: var(--muted); font-size: 11px; font-weight: 600; }
 .demo-settings { color: #93a1b5; font-size: 10.5px; letter-spacing: .2px; }
 .demo-note { color: #93a1b5; font-size: 11px; margin: 14px 2px 0; line-height: 1.5; }
 .demo-section-title { color: #ffd7d7; font-size: 15px; font-weight: 800;
                       margin: 26px 0 2px; letter-spacing: .2px; }
figure { margin: 0; }
figure img { width: 100%; display: block; border-radius: 12px; border: 1px solid var(--line);
             background: #0d1117; cursor: zoom-in; transition: filter .15s ease, transform .15s ease; }
figure img:hover { filter: brightness(1.1) saturate(1.05); }
figcaption { color: var(--muted); font-size: 11px; padding: 6px 2px 0; }
figcaption a { color: #7dd3fc; }
#offline { display: none; position: fixed; top: 0; left: 0; right: 0; z-index: 60;
           background: linear-gradient(90deg, #8c1b1b, #b91c1c); color: #fff;
           text-align: center; padding: 10px; font-size: 12.5px; letter-spacing: .2px; }
#offline.show { display: block; }
#lb { position: fixed; inset: 0; background: rgba(3,6,10,.95); display: none;
      align-items: center; justify-content: center; flex-direction: column; z-index: 50; }
#lb.open { display: flex; }
#lb img { max-width: 96vw; max-height: 90vh; border-radius: 12px; border: 1px solid #223042;
          box-shadow: 0 18px 70px rgba(0,0,0,.7); cursor: zoom-out; }
#lb .cap { color: #b6c4d6; font-size: 13px; padding: 12px 4px 0; letter-spacing: .3px; }
#lb .hint { color: #5f6c7e; font-size: 11px; padding-top: 3px; }
#lb .close { position: absolute; top: 18px; right: 24px; color: #c7d2df; font-size: 28px;
             cursor: pointer; line-height: 1; }
/* ---- left panel: navigation, summary card, ticks ---- */
.toc { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 18px; padding-bottom: 14px;
       border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 20;
       background: linear-gradient(180deg, #10161f 78%, rgba(16,22,31,0)); padding-top: 4px; }
.toc a { font-size: 10.5px; letter-spacing: .7px; text-transform: uppercase; color: #b9c6d8;
         background: rgba(255,255,255,.05); border: 1px solid var(--line); border-radius: 999px;
         padding: 4px 10px; text-decoration: none; transition: background .15s ease, color .15s ease; }
.toc a:hover { background: rgba(255,107,107,.14); color: #fff; border-color: rgba(255,107,107,.35); }
.report h2 { scroll-margin-top: 58px; }
.report h3 { scroll-margin-top: 58px; }
.summary { background: linear-gradient(135deg, rgba(255,107,107,.10), rgba(255,107,107,.02));
           border: 1px solid rgba(255,107,107,.22); border-radius: 14px; padding: 12px 14px 6px;
           margin: 6px 0 18px; }
.summary ul { margin: 4px 0 8px; padding-left: 16px; }
.summary li { font-size: 14px; }
p.sec-setup strong { color: #ff8787; background: rgba(255,107,107,.1); border-color: rgba(255,107,107,.25); }
p.sec-watching strong { color: #7dd3fc; background: rgba(56,189,248,.1); border-color: rgba(56,189,248,.25); }
p.sec-thesis strong { color: #fcd34d; background: rgba(251,191,36,.1); border-color: rgba(251,191,36,.25); }
p.sec-notes strong { color: #cbd5e1; background: rgba(148,163,184,.1); border-color: rgba(148,163,184,.22); }
p.sec-compliance strong { color: #4ade80; background: rgba(74,222,128,.1); border-color: rgba(74,222,128,.25); }
p.sec-levels strong, p.sec-what strong { color: #c084fc; background: rgba(192,132,252,.1);
                                          border-color: rgba(192,132,252,.25); }
.checks { columns: 2; column-gap: 18px; margin: 8px 0 14px; }
.checks li { break-inside: avoid; }
.report thead th { position: sticky; top: 0; z-index: 2; backdrop-filter: blur(6px); }
.report h3 { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
             padding: 10px 12px; border-radius: 12px; background: rgba(255,255,255,.03);
             border: 1px solid var(--line); }
.report h3.rec { background: rgba(74,222,128,.08); border-color: rgba(74,222,128,.25); }
@media (max-width: 1120px) { .grid { grid-template-columns: 1fr; } .report { max-height: none; }
                             .checks { columns: 1; } }

/* ================= dashboard design refresh ================= */
.sidebar { position: fixed; inset: 0 auto 0 0; width: 230px; z-index: 70;
           background: var(--sidebar); border-right: 1px solid var(--border);
           padding: 22px 14px; display: flex; flex-direction: column; gap: 22px; }
.brand { display: flex; align-items: center; gap: 10px; color: var(--text);
         font-size: 14px; font-weight: 700; letter-spacing: .2px; padding: 0 8px; }
.brand-dot { width: 9px; height: 9px; border-radius: 3px; background: var(--green);
             box-shadow: 0 0 14px rgba(40,199,111,.55); }
.side-nav { display: flex; flex-direction: column; gap: 4px; }
.nav-ico { display: inline-flex; align-items: center; justify-content: center;
           width: 14px; height: 14px; flex: 0 0 auto; opacity: .9; }
.nav-ico svg { width: 14px; height: 14px; display: block; }
.side-foot { margin-top: auto; color: var(--muted); font-size: 10.5px;
             line-height: 1.55; padding: 0 10px; }
.meta { color: var(--muted); font-size: 11.5px; }
.pill { margin-left: auto; font-size: 11px; color: var(--muted); background: var(--elevated);
        border: 1px solid var(--border); padding: 6px 12px; border-radius: 999px; }
.live { background: var(--green); box-shadow: 0 0 0 0 rgba(40,199,111,.6); }
.grid { gap: 16px; padding-top: 18px; }
#main-interactive-slot { margin-top: 12px; }
.report, .interactive-panel, .sym, .tracking-section {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  box-shadow: none; }
.report { padding: 18px 20px; max-height: none; overflow: visible; }
.report h2 { color: var(--muted); font-size: 11px; letter-spacing: 1.2px;
             border-bottom: 1px solid var(--border); font-weight: 700; }
.report h2::before { background: var(--green); width: 12px; height: 3px; }
.report h3 { color: var(--text); font-size: 14px; font-weight: 600;
             background: transparent; border: 1px solid var(--border);
             border-radius: 8px; padding: 9px 12px; }
.report h3.rec { color: var(--green); background: rgba(40,199,111,.07);
                 border-color: rgba(40,199,111,.25); }
.report h3.watch { color: #fcd34d; background: rgba(251,191,36,.10);
                   border-color: rgba(251,191,36,.45);
                   box-shadow: inset 3px 0 0 #fbbf24; }
.report table { border: 1px solid var(--border); border-radius: 8px; font-size: 12.5px; }
.report th { background: var(--elevated); color: var(--muted); font-size: 10px;
             letter-spacing: .7px; }
.report th, .report td { border-bottom: 1px solid var(--border); }
.report tr:hover td { background: rgba(255,255,255,.025); }
.summary { background: var(--elevated); border: 1px solid var(--border); border-radius: 8px; }
.toc a:hover { background: rgba(40,199,111,.12); color: var(--text);
               border-color: rgba(40,199,111,.3); }
.tracking-section { padding: 18px 20px; }
.tracking-section h3 { font-size: 15px; color: var(--text); font-weight: 700;
                       margin: 0 0 14px; letter-spacing: 0; }
.tracking-table { min-width: 920px; font-size: 12px; }
.tracking-table th { color: var(--muted); background: var(--elevated); font-size: 10px;
                     letter-spacing: .7px; padding: 8px 10px;
                     border-bottom: 1px solid var(--border); }
.tracking-table td { padding: 9px 10px; border-bottom: 1px solid rgba(255,255,255,.05);
                     color: var(--text); }
.tracking-table th.num, .tracking-table td.num { text-align: right;
                     font-variant-numeric: tabular-nums; }
.tracking-table th.nowrap, .tracking-table td.nowrap { white-space: nowrap; }
.tracking-table td.result-cell { min-width: 104px; }
.tracking-table td.result-cell .mark-btn { display: inline-block; vertical-align: middle; }
.tracking-table tbody tr { transition: background .15s ease; }
.tracking-table tbody tr:hover td { background: rgba(255,255,255,.03); }
.tracking-table tr.hit-tp td { color: var(--green); }
.tracking-table tr.hit-sl td { color: var(--red); }
.badge-status { font-size: 9.5px; padding: 3px 8px; border-radius: 999px;
                font-weight: 700; letter-spacing: .4px; text-transform: uppercase;
                white-space: nowrap; }
.badge-rec { background: rgba(255,186,51,.12); color: var(--yellow);
             border: 1px solid rgba(255,186,51,.3); }
.badge-open { background: rgba(40,199,111,.12); color: var(--green);
              border: 1px solid rgba(40,199,111,.3); }
.badge-tp { background: rgba(40,199,111,.16); color: var(--green);
            border: 1px solid rgba(40,199,111,.4); }
.badge-sl { background: rgba(234,84,85,.12); color: var(--red);
            border: 1px solid rgba(234,84,85,.35); }
.badge-manual { background: rgba(141,150,164,.12); color: var(--muted);
                border: 1px solid rgba(141,150,164,.25); }
.mark-btn { font-size: 10px; padding: 4px 9px; border-radius: 6px; font-weight: 600;
            letter-spacing: .2px; }
.mark-tp { color: #04140a; background: var(--green); border-color: var(--green); }
.mark-tp:hover { background: #23b463; border-color: #23b463; }
.mark-sl { color: var(--red); background: transparent; border-color: rgba(234,84,85,.45); }
.mark-sl:hover { background: rgba(234,84,85,.12); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
            gap: 14px; margin: 20px 0 16px; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
       padding: 14px 16px; min-height: 88px; display: flex; flex-direction: column;
       justify-content: space-between; gap: 5px; }
.kpi-label { font-size: 10px; text-transform: uppercase; letter-spacing: .55px;
             color: var(--muted); font-weight: 600; white-space: nowrap; }
.kpi-value { font-size: 22px; line-height: 1.2; font-weight: 700; color: var(--text);
             font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 10.5px; color: var(--muted); }
.pos { color: var(--green) !important; }
.neg { color: var(--red) !important; }
.muted { color: var(--muted) !important; }
.kpi .sep { color: var(--muted); margin: 0 5px; font-weight: 500; }
.chart-tab { background: var(--elevated); border: 1px solid var(--border);
             color: var(--muted); border-radius: 6px; }
.chart-tab:hover { color: var(--text); border-color: rgba(255,255,255,.2); }
.chart-tab.active { background: rgba(40,199,111,.12);
                    border-color: rgba(40,199,111,.35); color: var(--green); }
.chart-select { background: var(--elevated); border: 1px solid var(--border);
                color: var(--text); }
.chart-toolbar-title { color: var(--text); }
.lw-overlay { z-index: 6; }
.lw-title { z-index: 7; }
.lw-summary { z-index: 7; }
.lw-tooltip { z-index: 8; }
.lw-rsi-value { z-index: 4; }
#interactive-panel:fullscreen { width: 100vw; height: 100vh; padding: 12px;
  border-radius: 0; background: var(--bg); display: flex; flex-direction: column; }
#interactive-panel:fullscreen .lw-stack { flex: 1; min-height: 0; }
#interactive-panel:fullscreen .lw-price-wrap { flex: 1; height: auto; min-height: 0; }
#interactive-panel:fullscreen .lw-rsi-wrap { flex: 0 0 180px; height: 180px; }
#interactive-panel:fullscreen .chart-hint { display: none; }
#interactive-panel.chart-maximized { position: fixed; inset: 0; z-index: 120;
  width: 100vw; height: 100vh; padding: 12px; border-radius: 0;
  background: var(--bg); display: flex; flex-direction: column; }
#interactive-panel.chart-maximized .lw-stack { flex: 1; min-height: 0; }
#interactive-panel.chart-maximized .lw-price-wrap { flex: 1; height: auto; min-height: 0; }
#interactive-panel.chart-maximized .lw-rsi-wrap { flex: 0 0 180px; height: 180px; }
#interactive-panel.chart-maximized .chart-hint { display: none; }
body.chart-maximized-body { overflow: hidden; }
#tdetail .tdetail-card { background: var(--card); border: 1px solid var(--border);
                         border-radius: 10px; }
#tdetail .detail-title { color: var(--text); }
#tdetail .detail-meta { color: var(--muted); }
.trade-chart-grid { display: grid; grid-template-columns: 1fr; gap: 14px; margin: 10px 0 18px; }
.trade-chart { margin: 0; }
.trade-chart img { width: 100%; display: block; border: 1px solid var(--border);
                   border-radius: 8px; background: #0d1117; cursor: zoom-in; }
.trade-chart figcaption { color: var(--muted); font-size: 11px; padding-top: 4px; }
#lb { z-index: 130; }
#offline { z-index: 80; }
@media (max-width: 900px) {
  .sidebar { position: static; width: 100%; flex-direction: row; align-items: center;
             overflow-x: auto; padding: 10px 14px; border-right: none;
             border-bottom: 1px solid var(--border); gap: 14px; }
  .side-nav { flex-direction: row; gap: 4px; }
  .side-nav .tab-btn { width: auto; white-space: nowrap; }
  .side-foot { display: none; }
  .wrap { margin-left: 0; padding: 18px 18px 40px; }
}
@media (max-width: 720px) {
  .wrap { padding: 14px 12px 32px; }
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .kpi-value { font-size: 19px; }
  .report { padding: 14px; }
}
@media (max-width: 520px) {
  .kpi-grid { grid-template-columns: 1fr; }
  h1 { font-size: 19px; }
}
"""




def _img_tag(path: Path, caption: str, root: Path | None = None) -> str:
    try:
        data = base64.b64encode(path.read_bytes()).decode()
        cap = html.escape(caption)
        src = f"data:image/png;base64,{data}"
        served = f"/charts/{path.name}"
        if root is not None:
            try:
                served = "/" + path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                pass
        return (f'<figure>'
                f'<img src="{src}" alt="{cap}" data-caption="{cap}" '
                f'onclick="openLb(this)">'
                f'<figcaption>{cap} · click to expand · '
                f'<a class="orig" href="{served}" target="_blank">open png</a>'
                f'</figcaption></figure>')
    except OSError:
        return f'<figure><figcaption>{html.escape(caption)} (missing)</figcaption></figure>'


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip().replace(" UTC", "")
    if not text or text == "—":
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _cet_time(value: object) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return "—" if value in (None, "") else str(value)
    return parsed.astimezone(CET).strftime("%Y-%m-%d %H:%M %Z")


RANGE_BARS = {"1h": 24}


def _add_chart_ranges(payload: dict) -> dict:
    """Draw the recent high/low range on the main interactive chart only."""
    for symbol in {str(key).split("|", 1)[0] for key in payload}:
        for tf, bars in RANGE_BARS.items():
            entry = payload.get(f"{symbol}|{tf}")
            if not isinstance(entry, dict):
                continue
            candles = entry.get("candles") or []
            if len(candles) <= bars:
                continue
            window = candles[-bars:]
            low = min(float(c["low"]) for c in window)
            high = max(float(c["high"]) for c in window)
            digits = 2 if high >= 10 else 5
            bands = entry.get("bands")
            if not isinstance(bands, list):
                bands = []
            entry["bands"] = bands + [{
                "y0": low, "y1": high,
                "from_time": int(window[0]["time"]),
                "to_time": int(window[-1]["time"]),
                "color": "#FFBA33", "opacity": 0.06,
                "label": f"{tf} range {low:.{digits}f}–{high:.{digits}f}",
            }]
    return payload


def _tracking_kpis(trades: list[dict]) -> str:
    """KPI grid above the tracking table (realised P&L + side splits)."""
    closed_statuses = {"hit_tp", "hit_sl", "closed_manually"}

    def pnl_of(trade: dict) -> float | None:
        if not (trade.get("is_engine") or trade.get("pnl_is_usd")):
            return None
        value = trade.get("pnl")
        return float(value) if isinstance(value, (int, float)) else None

    def side(trade: dict, letter: str) -> bool:
        return str(trade.get("direction", "")).upper().startswith(letter)

    def money(value: float, signed: bool = False, decimals: int = 2) -> str:
        sign = ("+" if value > 0 else "-") if signed and value else ""
        return f"{sign}${abs(value):,.{decimals}f}"

    def win_rate(subset: list[dict]) -> tuple[str, int, int]:
        scored = [t for t in subset if pnl_of(t) is not None]
        if not scored:
            return "—", 0, 0
        won = sum(1 for t in scored if (pnl_of(t) or 0.0) > 0)
        return f"{100 * won / len(scored):.1f}%", won, len(scored)

    closed = [t for t in trades if str(t.get("status")) in closed_statuses]
    scored = [t for t in closed if pnl_of(t) is not None]
    wins = [t for t in scored if (pnl_of(t) or 0.0) > 0]
    losses = [t for t in scored if (pnl_of(t) or 0.0) < 0]
    net = sum(pnl_of(t) or 0.0 for t in scored)
    gross_profit = sum(pnl_of(t) or 0.0 for t in wins)
    gross_loss = abs(sum(pnl_of(t) or 0.0 for t in losses))
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    if not scored:
        pf_text, pf_class = "—", "muted"
    elif gross_loss == 0:
        pf_text, pf_class = ("∞", "pos") if gross_profit > 0 else ("—", "muted")
    else:
        pf = gross_profit / gross_loss
        pf_text, pf_class = f"{pf:.2f}", ("pos" if pf >= 1 else "neg")

    overall_wr, overall_wins, overall_closed = win_rate(closed)
    overall_class = ("pos" if overall_closed and overall_wins / overall_closed >= 0.5
                     else "neg" if overall_closed else "muted")

    shorts = [t for t in trades if side(t, "S")]
    longs = [t for t in trades if side(t, "L")]
    short_wr, short_wins, short_closed = win_rate(shorts)
    long_wr, long_wins, long_closed = win_rate(longs)
    short_class = ("pos" if short_closed and short_wins / short_closed >= 0.5
                   else "neg" if short_closed else "muted")
    long_class = ("pos" if long_closed and long_wins / long_closed >= 0.5
                  else "neg" if long_closed else "muted")

    days = set()
    for trade in trades:
        parsed = _parse_timestamp(trade.get("_opened_at"))
        if parsed is not None:
            days.add(parsed.astimezone(CET).date())

    open_count = len(trades) - len(closed)
    net_class = "pos" if net > 0 else "neg" if net < 0 else "muted"

    def card(label: str, value: str, sub: str = "", cls: str = "muted") -> str:
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {cls}">{value}</div>{sub_html}</div>')

    avg_value = (f'<span class="pos">{money(avg_win, decimals=0)}</span>'
                 f'<span class="sep">/</span>'
                 f'<span class="neg">{money(avg_loss, decimals=0)}</span>')
    return (
        '<div class="kpi-grid">'
        + card("Net P&L", money(net, signed=True), f"{len(scored)} resolved", net_class)
        + card("Total Trades", str(len(trades)),
               f"{len(closed)} closed · {open_count} open")
        + card("Win Rate", overall_wr,
               f"{overall_wins}W / {overall_closed - overall_wins}L", overall_class)
        + card("Profit Factor", pf_text,
               "gross profit / gross loss", pf_class)
        + card("Avg Profit / Avg Loss", avg_value, "avg win / avg loss")
        + card("Short Trades", str(len(shorts)), f"{short_closed} closed")
        + card("Short Win Rate", short_wr,
               f"{short_wins}W / {short_closed - short_wins}L", short_class)
        + card("Long Trades", str(len(longs)), f"{long_closed} closed")
        + card("Long Win Rate", long_wr,
               f"{long_wins}W / {long_closed - long_wins}L", long_class)
        + card("Total Days Traded", str(len(days)), "days traded")
        + "</div>"
    )


def _dom_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def _report_html(markdown_text: str, id_prefix: str = "",
                include_toc: bool = True) -> str:
    # charts are shown in the right column; drop image/link lines from the text
    cleaned = "\n".join(
        line for line in markdown_text.splitlines()
        if not line.startswith("![") and not line.startswith("**") or "]](" not in line
    )
    cleaned = re.sub(r"^## Charts.*?(?=^---)", "", cleaned, flags=re.S | re.M)
    html = md_lib.markdown(cleaned, extensions=["tables", "sane_lists"])

    # a paragraph that is nothing but a bold label becomes a styled section label
    def label(match: re.Match) -> str:
        text = match.group(1)
        slug = re.sub(r"[^a-z]+", "-", text.lower()).strip("-")
        return (f'<p class="sec sec-{slug}"><strong>{text}</strong></p>')

    html = re.sub(r"<p><strong>([^<]{1,40})</strong></p>", label, html)

    # TL;DR becomes a highlight card
    html = re.sub(r"(<h2>TL;DR</h2>\s*)(<ul>.*?</ul>)",
                  r"\1<div class=\"summary\">\2</div>", html, flags=re.S)

    # compliance ticks read better in two columns
    html = re.sub(r"(<p class=\"sec sec-compliance\"><strong>COMPLIANCE</strong></p>\s*)(<ul>.*?</ul>)",
                  r"\1<div class=\"checks\">\2</div>", html, flags=re.S)

    # ids + a small jump bar for the long report pane
    seen: dict[str, int] = {}
    toc: list[tuple[int, str, str]] = []

    def add_id(match: re.Match) -> str:
        level, text = match.group(1), match.group(2)
        base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
        full_anchor = f"{id_prefix}{anchor}"
        plain = re.sub(r"<[^>]+>", "", text)
        if level == "3":                      # keep the jump bar compact: symbol (+ ● if recommended)
            symbol = plain.split("—")[0].strip()
            plain = symbol + (" ●" if "TRADE RECOMMENDED" in plain.upper() else "")
        if level in ("2", "3"):
            toc.append((int(level), plain, full_anchor))
        return f'<h{level} id="{full_anchor}">{text}</h{level}>'

    html = re.sub(r"<h([23])>(.*?)</h\1>", add_id, html, flags=re.S)
    # recommendation headings go green, watchlist headings amber
    html = re.sub(r'<h3 id="([^"]+)">([^<]*TRADE RECOMMENDED[^<]*)</h3>',
                  r'<h3 id="\1" class="rec">\2</h3>', html)
    html = re.sub(r'<h3 id="([^"]+)">([^<]*no trade yet[^<]*)</h3>',
                  r'<h3 id="\1" class="watch">\2</h3>', html)
    if include_toc and toc:
        items = "".join(f'<a href="#{anchor}">{html_lib.escape(name)}</a>'
                        for _, name, anchor in toc)
        html = f'<nav class="toc">{items}</nav>' + html
    return html


def _chart_cards(symbols: list[str], setups: dict[str, dict], charts_dir: Path,
                 root: Path | None = None, slot_id: str | None = None) -> tuple[str, bool]:
    """PNG chart cards, optionally hosting the interactive chart panel."""
    cards: list[str] = []
    slot_added = False
    for symbol in dict.fromkeys(symbols):
        shot_map: dict[str, str] = {}
        for tf in ("1m", "5m", "1h", "4h"):
            path = charts_dir / f"{symbol}_{tf}.png"
            if path.exists():
                shot_map[tf] = _img_tag(path, f"{symbol} {tf}", root=root)
        if shot_map:
            setup = setups.get(symbol)
            if setup:
                badge = (f'<em class="badge">{setup.get("direction_label")} · '
                         f'{setup.get("grade")} · {setup.get("rr", 0):.1f}R</em>')
                card_class = "sym rec"
                digits = 2 if (setup.get("entry") or 0) >= 10 else 5
                levels = (f'<div class="levels">entry <b>{setup.get("entry"):.{digits}f}</b>'
                          f' · SL <b>{setup.get("sl"):.{digits}f}</b>'
                          f' · TP <b>{setup.get("tp"):.{digits}f}</b>'
                          f' · {setup.get("rr", 0):.1f}R</div>')
            else:
                badge, card_class, levels = "", "sym", ""
            slot = ""
            if slot_id and not slot_added:
                slot = f'<div id="{slot_id}" class="interactive-slot"></div>'
                slot_added = True
            top = shot_map.get("1m", "")
            rest = "".join(shot_map[tf] for tf in ("5m", "1h", "4h") if tf in shot_map)
            top_html = f'<div class="shots">{top}</div>' if top else ""
            rest_html = f'<div class="shots">{rest}</div>' if rest else ""
            cards.append(
                f'<section class="{card_class}"><h3>{html.escape(symbol)} '
                f'<span>1m · 5m · 1h · 4h</span>{badge}</h3>{levels}'
                f'{top_html}{slot}{rest_html}</section>')
    return "".join(cards), slot_added


def _demo_section(meta: dict, out_dir: Path) -> str:
    """Swing-detection screenshots for the Demo tab."""
    items = meta.get("timeframes", []) if isinstance(meta, dict) else []
    cards: list[str] = []
    for item in items:
        path = out_dir / "demo" / str(item.get("file", ""))
        if not path.exists():
            continue
        params = item.get("params") or {}
        caption = (f"{meta.get('symbol', '')} {item.get('tf', '')} — "
                   f"{item.get('highs', 0)} highs / {item.get('lows', 0)} lows")
        settings = (f"n={params.get('n', '—')} · prominence ≥{params.get('prom', '—')} ATR"
                    f" · isolation >{params.get('iso_band', '—')} ATR"
                    f" · windows {params.get('prom_window', '—')}/{params.get('iso_window', '—')}")
        bos = ""
        if item.get("tf") == "1m" and (item.get("bos_up") or item.get("bos_down")):
            bos = f" · BOS {item.get('bos_up', 0)}↑/{item.get('bos_down', 0)}↓"
        if item.get("tf") == "1m" and (item.get("choch_up") or item.get("choch_down")):
            bos += (f" · CHoCH {item.get('choch_up', 0)}↑/"
                    f"{item.get('choch_down', 0)}↓")
        cards.append(
            '<section class="demo-card">'
            f'<div class="demo-head"><h3>{html.escape(str(item.get("tf", "")))} '
            f'<span>{item.get("highs", 0)}H / {item.get("lows", 0)}L{html.escape(bos)}'
            f'</span></h3>'
            f'<div class="demo-settings">{html.escape(settings)}</div></div>'
            + _img_tag(path, caption, root=out_dir)
            + '</section>')
    range_cards: list[str] = []
    for item in meta.get("ranges", []) if isinstance(meta, dict) else []:
        path = out_dir / "demo" / str(item.get("file", ""))
        if not path.exists():
            continue
        params = item.get("params") or {}
        caption = (f"{meta.get('symbol', '')} {item.get('tf', '')} — "
                   f"{item.get('boxes', 0)} box(es)")
        settings = (f"5m box · {params.get('min_bars', '—')}–{params.get('max_bars', '—')} bars"
                    f" · height ≤{params.get('k_height', '—')} ATR"
                    f" · ER ≤{params.get('er_max', '—')}"
                    f" · {params.get('min_touch', '—')} touches"
                    f" · {params.get('tol_atr', '—')} ATR tol"
                    f" · break {params.get('break_atr', '—')} ATR"
                    f" · sweep {params.get('sweep_tol_atr', '—')} ATR / "
                    f"{params.get('reclaim_bars', '—')} bars")
        detail = (f"{item.get('boxes', 0)} box · {item.get('breakout', 0)} break"
                  f" · {item.get('sweep', 0)} sweep")
        range_cards.append(
            '<section class="demo-card">'
            f'<div class="demo-head"><h3>{html.escape(str(item.get("tf", "range")))} '
            f'<span>{html.escape(detail)}</span></h3>'
            f'<div class="demo-settings">{html.escape(settings)}</div></div>'
            + _img_tag(path, caption, root=out_dir)
            + '</section>')
    if not cards and not range_cards:
        return ('<p class="chart-missing">No demo screenshots yet — run '
                '<b>./run.sh demo</b> to create the demo CSVs, then refresh.</p>')
    source = html.escape(str(meta.get("source", "")))
    kind = html.escape(str(meta.get("source_kind", "source")))
    parts = []
    if cards:
        parts.append('<div class="demo-grid">' + "".join(cards) + '</div>')
    if range_cards:
        parts.append('<h3 class="demo-section-title">Range detection</h3>')
        parts.append('<div class="demo-grid">' + "".join(range_cards) + '</div>')
    parts.append(
        '<p class="demo-note">Screenshots are generated from the '
        f'<b>{kind}</b> <b>{source}</b> with the same pivot / prominence / '
        'isolation and range detectors used by the live engine. Markers: ▼ swing high · '
        '▲ swing low. Only un-swept levels are shown; a level price has traded '
        'through is stale and is removed with its line. 1m BOS dots mark every '
        'break; text labels show the latest three per direction: '
        '<b>↑ above LH</b> = bearish-context break; <b>↓ below HL</b> = '
        'bullish-context break. Purple diamonds are extreme CHoCH: '
        '<b>↑</b> final LL→high→higher-low, <b>↓</b> final HH→low→lower-high. '
        'Range images show the 5m box, then the resolved BREAK or SWEEP; if the '
        'latest bars have no range, the image goes back to the most recent one.</p>')
    return "".join(parts)


def _tracking_parts(config: Config, plans_dir: Path,
                    manual_files: list[Path]) -> tuple[str, str, list[dict]]:
    """KPI strip + tracking table built from the tracked plans."""
    # ---- tracking section data ----
    tracking_data: list = []
    # Try the manual tracking files in order; the first one that exists wins.
    for tp in manual_files:
        if tp.exists():
            try:
                payload = json.loads(tp.read_text(encoding="utf-8"))
                tracking_data = payload.get("active_trades", []) + payload.get("completed_trades", [])
                break
            except (json.JSONDecodeError, KeyError):
                tracking_data = []

    # Engine-owned plans: the recommendation as it was first logged (frozen levels).
    plan_rows: list[dict] = []
    plans_path = plans_dir / "active_plans.json"
    if plans_path.exists():
        try:
            plans = json.loads(plans_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            plans = {}
        items = plans.items() if isinstance(plans, dict) else []
        for plan_key, plan in items:
            status = plan.get("status", "recommended")
            rr = plan.get("rr")
            pnl = plan.get("pnl_points") if status in ("hit_tp", "hit_sl") else None
            if pnl is None and status in ("hit_tp", "hit_sl"):
                try:
                    entry = float(plan["entry"])
                    sl = float(plan["sl"])
                    tp = float(plan["tp"])
                    direction = 1 if int(plan.get("direction", 1)) > 0 else -1
                    price = tp if status == "hit_tp" else sl
                    pnl = direction * (price - entry)
                except (KeyError, TypeError, ValueError):
                    pnl = None
            pnl_usd = plan.get("pnl_usd") if status in ("hit_tp", "hit_sl") else None
            if pnl_usd is None and status in ("hit_tp", "hit_sl") and isinstance(pnl, (int, float)):
                try:
                    risk = abs(float(plan["entry"]) - float(plan["sl"]))
                    risk_usd = float(plan.get("risk_usd", 300.0) or 300.0)
                    pnl_usd = (pnl / risk * risk_usd) if risk else None
                except (KeyError, TypeError, ValueError):
                    pnl_usd = None
            display_rr = rr
            if status in ("hit_tp", "hit_sl"):
                display_rr = plan.get("pnl_r")
                if not isinstance(display_rr, (int, float)):
                    try:
                        risk = abs(float(plan["entry"]) - float(plan["sl"]))
                        display_rr = (pnl / risk) if isinstance(pnl, (int, float)) and risk else None
                    except (KeyError, TypeError, ValueError):
                        display_rr = None
            elif status == "expired":
                display_rr = None
            trigger_level = plan.get("trigger_level")
            plan_rows.append({
                "key": plan_key,
                "detail_id": "detail-" + _dom_id(plan_key),
                "snapshot": plan.get("snapshot") or {},
                "is_engine": True,
                "symbol": plan.get("symbol", "—"),
                "direction": plan.get("direction_label", "—"),
                "entry": plan.get("entry"),
                "sl": plan.get("sl"),
                "tp": plan.get("tp"),
                "status": status,
                "_opened_at": plan.get("created"),
                "open_time": _cet_time(plan.get("opened_at") or plan.get("created")),
                "close_time": _cet_time(plan.get("closed_at")) if plan.get("closed_at") else "—",
                "pnl": pnl_usd,
                "pnl_is_usd": True,
                "rr": display_rr,
                "trigger": (f"{plan.get('trigger_kind', 'trigger')} at {trigger_level:.5f}"
                            if isinstance(trigger_level, (int, float)) else "—"),
                "sig": plan.get("sig", ""),
                "charts": plan.get("charts") or {},
            })
    for row in tracking_data:
        row.setdefault("is_engine", False)
        row["_opened_at"] = row.get("open_time")
        row["open_time"] = _cet_time(row.get("open_time"))
        row["close_time"] = (_cet_time(row.get("close_time"))
                             if row.get("close_time") not in (None, "", "—") else "—")
    # Manual rows win when the same signature exists there (the user maintains the result).
    manual_sigs = {row.get("sig") for row in tracking_data if row.get("sig")}
    merged = tracking_data + [row for row in plan_rows if row.get("sig") not in manual_sigs]
    merged.sort(key=lambda row: _parse_timestamp(row.get("_opened_at"))
                or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    # Build tracking HTML rows
    tracking_rows = ""
    detail_blocks = ""

    def _cell(value: object) -> object:
        return "—" if value is None or value == "" else value

    def _px(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:.2f}" if number >= 10 else f"{number:.5f}"

    def _snapshot_html(text: str) -> str:
        rendered = _report_html(text or "", id_prefix="snap-")
        return re.sub(r'<nav class="toc">.*?</nav>', '', rendered, flags=re.S)

    for t in merged:
        status = t.get("status", "unknown")
        # manual rows are not auto-updated by the engine; show a stale recommendation as expired
        if status == "recommended" and not t.get("is_engine"):
            opened = _parse_timestamp(t.get("_opened_at"))
            if opened is not None:
                age_hours = (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0
                if age_hours > config.active_plan_max_age_hours:
                    status = "expired"
        row_class = ""
        badge = f'<span class="badge-status badge-manual">{status.upper()}</span>'
        if status == "recommended":
            row_class = ""; badge = '<span class="badge-status badge-rec">RECOMMENDED</span>'
        elif status == "open":
            row_class = ""; badge = '<span class="badge-status badge-open">OPEN</span>'
        elif status == "hit_tp":
            row_class = "hit-tp"; badge = '<span class="badge-status badge-tp">HIT TP</span>'
        elif status == "hit_sl":
            row_class = "hit-sl"; badge = '<span class="badge-status badge-sl">HIT SL</span>'
        elif status == "expired":
            badge = '<span class="badge-status badge-manual">EXPIRED</span>'
        pnl = t.get("pnl")
        if isinstance(pnl, (int, float)):
            pnl_display = (f"{'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}"
                           if t.get("pnl_is_usd") else f"{pnl:.2f}")
        else:
            pnl_display = _cell(pnl)
        rr = t.get("rr")
        rr_display = f"{rr:.2f}" if isinstance(rr, (int, float)) else _cell(rr)
        detail_id = t.get("detail_id")
        row_attrs = ""
        if t.get("is_engine") and detail_id:
            row_attrs = (f' class="track-row {row_class}" style="cursor:pointer;" '
                         f'onclick="showTradeDetail(\'{detail_id}\')"')
        else:
            row_attrs = f' class="{row_class}"'
        if t.get("is_engine") and status in ("recommended", "open"):
            key = urllib.parse.quote(str(t.get("key", "")), safe="")
            actions = (
                f'<a class="mark-btn mark-tp" href="/api/mark?key={key}&status=hit_tp" '
                f'onclick="event.stopPropagation(); return confirm(\'Mark this trade as HIT TP?\')">HIT TP</a>'
                f'<a class="mark-btn mark-sl" href="/api/mark?key={key}&status=hit_sl" '
                f'onclick="event.stopPropagation(); return confirm(\'Mark this trade as HIT SL?\')">HIT SL</a>'
            )
        elif t.get("is_engine"):
            actions = '<span style="color:#5f6c7e;font-size:10px;">closed</span>'
        else:
            actions = "—"
        tracking_rows += (
            f'<tr{row_attrs}><td><b>{_cell(t.get("symbol"))}</b> {_cell(t.get("direction"))}</td>'
            f'<td class="num">{_px(t.get("entry"))}</td><td class="num">{_px(t.get("sl"))}</td>'
            f'<td class="num">{_px(t.get("tp"))}</td>'
            f'<td class="nowrap">{badge}</td><td>{_cell(t.get("open_time"))}</td>'
            f'<td>{_cell(t.get("close_time"))}</td>'
            f'<td class="num">{pnl_display}</td><td class="num">{rr_display}R</td>'
            f'<td style="font-size:10px;color:#93a1b5;">{_cell(t.get("trigger"))}</td>'
            f'<td class="nowrap result-cell">{actions}</td></tr>')
        if t.get("is_engine") and detail_id:
            snapshot = t.get("snapshot") or {}
            charts = t.get("charts") or {}
            chart_html = ""
            if charts:
                items = []
                for tf in ("1m", "5m", "1h"):
                    src = charts.get(tf)
                    if src:
                        caption = f"{t.get('symbol', '')} {tf} · entry / SL / TP"
                        items.append(
                            f'<figure class="trade-chart"><img src="{html.escape(str(src))}" '
                            f'alt="{html.escape(caption)}" data-caption="{html.escape(caption)}" '
                            f'onclick="openLb(this)">'
                            f'<figcaption>{html.escape(caption)}</figcaption></figure>')
                if items:
                    chart_html = ('<h3>Trade charts</h3>'
                                  '<div class="trade-chart-grid">' + "".join(items) + '</div>')
            body = ""
            if snapshot:
                body = (
                    '<h3>TL;DR</h3>' + _snapshot_html(snapshot.get("tldr", ""))
                    + '<h3>Market context</h3>' + _snapshot_html(snapshot.get("market_context", ""))
                    + '<h3>Watchlist / Setup</h3>' + _snapshot_html(snapshot.get("watchlist", ""))
                )
            else:
                body = ('<p style="color:#93a1b5;">No report snapshot stored for this '
                        'recommendation.</p>')
            body = chart_html + body
            detail_blocks += (
                f'<div id="{detail_id}" class="trade-detail-source">'
                f'<div class="detail-title">{_cell(t.get("symbol"))} {_cell(t.get("direction"))} '
                f'· {status.upper()}</div>'
                f'<div class="detail-meta">entry {_px(t.get("entry"))} · '
                f'SL {_px(t.get("sl"))} · TP {_px(t.get("tp"))} · '
                f'{_cell(rr_display)}R'
                + (f' · PNL {pnl_display}' if isinstance(pnl, (int, float)) else "")
                + f' · recommended {_cell(t.get("open_time"))}'
                + (f' · closed {_cell(t.get("close_time"))}'
                   if t.get("close_time") not in (None, "", "—") else "")
                + f'</div><div class="detail-actions">{actions}</div>'
                + '<div class="report detail-report">' + body + '</div></div>')

    if not tracking_rows:
        tracking_rows = ('<tr><td colspan="11" style="color:#93a1b5;padding:14px 8px;">'
                         'No tracked trades yet — new recommendations appear here automatically '
                         'with their levels frozen at first alert.</td></tr>')

    tracking_html = (
        '<section class="tracking-section"><h3>📋 Recommended Trade Tracking</h3>'
        '<table class="tracking-table"><thead><tr>'
        '<th>Symbol / Side</th><th class="num">Entry</th><th class="num">SL</th>'
        '<th class="num">TP</th><th class="nowrap">Status</th>'
        '<th>Open (CET)</th><th>Close (CET)</th><th class="num">PNL</th>'
        '<th class="num">R:R</th><th>Trigger / Note</th>'
        '<th class="nowrap">Result</th>'
        '</tr></thead><tbody>'
        + tracking_rows
        + '</tbody></table>'
        + detail_blocks
        + '<p style="font-size:10.5px;color:#93a1b5;margin-top:8px;letter-spacing:.2px;">'
        'Click any trade to open its full report snapshot (TL;DR, market context, watchlist/setup). '
        'Open/close times are CET (Europe/Berlin). Mark a trade HIT TP / HIT SL to record the result; '
        f'P&L is shown in USD at the fixed ${config.typical_risk_usd:,.0f} risk per trade. '
        'Levels stay frozen at first alert.'
        '</p></section>')

    tracking_kpi_html = _tracking_kpis(merged)
    return tracking_kpi_html, tracking_html, merged


def build_dashboard(config: Config, out_dir: Path | None = None) -> Path | None:
    out_dir = out_dir or config.out_dir
    md_path = out_dir / "latest.md"
    js_path = out_dir / "latest.json"
    if not md_path.exists():
        return None
    markdown_text = md_path.read_text(encoding="utf-8")

    def _load_payload(path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    interactive = _add_chart_ranges(_load_payload(out_dir / "interactive_charts.json"))
    chart_symbols = sorted({str(key).split("|", 1)[0] for key in interactive})
    chart_default_symbol = chart_symbols[0] if chart_symbols else ""
    chart_tfs = [str(key).split("|", 1)[1] for key in interactive
                 if str(key).startswith(chart_default_symbol + "|")]
    chart_default_tf = "1m" if "1m" in chart_tfs else (chart_tfs[0] if chart_tfs else "")
    main_interactive_json = json.dumps(interactive, default=str)
    symbol_options = "".join(
        f'<option value="{html.escape(symbol)}">{html.escape(symbol)}</option>'
        for symbol in chart_symbols)
    if len(chart_symbols) > 1:
        symbol_control = (f'<select id="chart-symbol" class="chart-select" '
                          f'onchange="showChart()">{symbol_options}</select>')
    elif chart_symbols:
        symbol_control = f'<span class="chart-symbol">{html.escape(chart_default_symbol)}</span>'
    else:
        symbol_control = ""
    tf_buttons = "".join(
        f'<button class="chart-tab{" active" if tf == chart_default_tf else ""}" '
        f'data-tf="{tf}" onclick="showChart(\'{tf}\')">{tf}</button>'
        for tf in ("4h", "1h", "5m", "1m")
    )
    if interactive:
        chart_panel = (
            '<div class="interactive-panel" id="interactive-panel">'
            '<div class="chart-toolbar">'
            '<div class="chart-toolbar-title">Interactive chart</div>'
            f'<div class="chart-controls">{symbol_control}{tf_buttons}'
            '<button class="chart-tab chart-refresh" onclick="refreshChart()" '
            'title="Reload chart data">⟳</button>'
            '<button class="chart-tab chart-fullscreen" onclick="toggleChartFullscreen()" '
            'title="Fullscreen">⛶</button></div>'
            '</div>'
            '<div class="lw-stack">'
            '<div class="lw-price-wrap">'
            '<div id="lw-title" class="lw-title"></div>'
            '<div id="lw-chart" class="lw-chart"></div>'
            '<svg id="lw-overlay" class="lw-overlay"></svg>'
            '<div id="lw-summary" class="lw-summary"></div>'
            '<div id="lw-tooltip" class="lw-tooltip"></div>'
            '</div>'
            '<div class="lw-rsi-wrap">'
            '<div id="lw-rsi" class="lw-rsi"></div>'
            '<div id="lw-rsi-value" class="lw-rsi-value">—</div>'
            '</div>'
            '</div>'
            '<div id="lw-legend" class="chart-hint"></div>'
            '</div>'
        )
    else:
        chart_panel = ('<p class="chart-missing">Interactive chart unavailable — '
                       'run <b>./run.sh start</b> to generate it.</p>')

    symbols: list[str] = []
    setups: dict[str, dict] = {}
    if js_path.exists():
        try:
            payload = json.loads(js_path.read_text(encoding="utf-8"))
            symbols = list(payload.get("symbols", {}))
            setups = {sym: node["setup"] for sym, node in payload.get("symbols", {}).items()
                      if node.get("setup")}
        except (json.JSONDecodeError, KeyError, TypeError):
            symbols = []
    if not symbols:
        symbols = re.findall(r"^### (\w+)", markdown_text, flags=re.M)

    chart_cards, slot_added = _chart_cards(symbols, setups, config.chart_dir, root=out_dir,
                                           slot_id="main-interactive-slot")
    marker = '<div id="main-interactive-slot" class="interactive-slot"></div>'
    if slot_added and marker in chart_cards:
        chart_cards = chart_cards.replace(marker, chart_panel)
    elif interactive:
        chart_cards += chart_panel

    generated = datetime.now().strftime("%a %d %b %Y · %H:%M:%S")
    stamp = ""
    for line in markdown_text.splitlines():
        if line.startswith("**Generated:**"):
            stamp = line.replace("**Generated:**", "").strip().rstrip("  ")
            break
    interval = max(5, config.dashboard_refresh_seconds)

    tracking_kpi_html, tracking_html, merged = _tracking_parts(
        config, out_dir,
        [Path("data/trades_active.json"), out_dir / "tracking.json",
         config.out_dir / ".." / "data" / "trades_active.json"])

    from .demo import ensure_demo_charts
    try:
        demo_meta = ensure_demo_charts(config, out_dir)
    except Exception:
        demo_meta = {}
    demo_html = _demo_section(demo_meta, out_dir)

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<!-- auto-refresh handled by JS so an expanded chart is not closed -->
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2016%2016'%3E%3Crect%20width='16'%20height='16'%20rx='3'%20fill='%230B0D10'/%3E%3Cpath%20d='M3%2012.5h10'%20stroke='%2328C76F'%20stroke-width='1.3'%20stroke-linecap='round'/%3E%3Cpath%20d='M5%2010.5V7M8%2010.5V4.5M11%2010.5V6'%20stroke='%2328C76F'%20stroke-width='1.4'%20stroke-linecap='round'/%3E%3C/svg%3E">
<title>Trading read v2</title><style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script></head>
<body><div id="offline">dashboard server unreachable — restart it with <b>./run.sh start</b></div>
<aside class="sidebar">
  <div class="brand"><span class="brand-dot"></span>Trading Journal</div>
  <nav class="side-nav">
    <button onclick="showTab('main')" id="tab-main" class="tab-btn tab-active">
      <span class="nav-ico" aria-hidden="true">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round">
          <path d="M2.5 13.5h11"/>
          <path d="M4.5 11V7.5"/>
          <path d="M7.2 11V5"/>
          <path d="M9.9 11V6.5"/>
          <path d="M12.6 11V3"/>
        </svg>
      </span>
      Report + Charts
    </button>
    <button onclick="showTab('tracking')" id="tab-tracking" class="tab-btn">Trade Tracking</button>
    <button onclick="showTab('demo')" id="tab-demo" class="tab-btn">Demo</button>
  </nav>
  <div class="side-foot">Auto-refresh {interval}s<br>Cycle {config.active_trade_interval_seconds if setups else config.watch_interval_seconds}s{' · trade active' if setups else ''}</div>
</aside>
<div class="wrap">
  <div id="main" class="main-view">
    <header id="header">
      <div class="live"></div>
      <h1>Trading read v2</h1>
      <div class="meta">report {html.escape(stamp or generated)} · refreshed {generated}
        · {len(symbols)} instrument(s)</div>
      <div class="pill">page refresh {interval}s ·
        cycle {config.active_trade_interval_seconds if setups else config.watch_interval_seconds}s
        {'· trade active' if setups else ''}</div>
    </header>
    <div class="grid">
      <article id="report" class="report">{_report_html(markdown_text)}</article>
      <div id="charts" class="charts">{chart_cards or '<p>No charts yet.</p>'}</div>
    </div>
  </div>

  <div id="tracking" class="tracking-view">
    <header id="tracking-header">
      <div class="live"></div>
      <h1>Trade tracking</h1>
      <div class="meta">{len(merged)} trade(s) · engine + manual</div>
      <div class="pill">source: active_plans.json + trades_active.json</div>
    </header>
    {tracking_kpi_html}
    {tracking_html}
  </div>

  <div id="demo" class="demo-view">
    <header id="demo-header">
      <div class="live"></div>
      <h1>Swing detection demo</h1>
      <div class="meta">{html.escape(str(demo_meta.get("symbol", "XAUUSD")))} · 1H / 5M / 1M
        · {html.escape(str(demo_meta.get("source_kind", "source")))}:
        {html.escape(str(demo_meta.get("source", "—")))}</div>
      <div class="pill">strict pivot · prominence · isolation</div>
    </header>
    {demo_html}
  </div>
</div>

<div id="lb" onclick="closeLb(event)">
  <span class="close" onclick="closeLb(event)">×</span>
  <img id="lbimg" alt="" style="cursor: zoom-in; transition: transform 0.2s ease;" onclick="toggleZoom(event)">
  <div class="cap" id="lbcap"></div>
  <div class="hint">click image to zoom around that spot · click again to zoom further · click outside or press Esc to close</div>
</div>

<div id="tdetail" onclick="closeTradeDetail(event)">
  <div class="tdetail-card" onclick="event.stopPropagation()">
    <span class="close" onclick="closeTradeDetail({{target:{{id:'tdetail'}}}})">×</span>
    <div id="tdetail-body"></div>
  </div>
</div>

<script>
const LW_DATA_MAIN = {main_interactive_json};
let LW_DATA = LW_DATA_MAIN;
let lwSource = 'interactive_charts.json';
const LW_DEFAULT = {{ symbol: {json.dumps(chart_default_symbol)}, tf: {json.dumps(chart_default_tf)} }};
let lwSymbol = LW_DEFAULT.symbol;
let lwTf = (function () {{
  try {{ return sessionStorage.getItem('botv2-lw-tf') || LW_DEFAULT.tf; }}
  catch (e) {{ return LW_DEFAULT.tf; }}
}})();
let lwChart = null, lwCandle = null, lwRsiChart = null, lwRsiSeries = null;
let lwPriceLines = [];
let lwRsiGuides = [];
let lwOverlayData = null;
let lwMarkerData = [];
let lwRsiByTime = new Map();
let lwCurrent = null;
let lwReady = false;
let lwSyncing = false;

function lwInit() {{
  if (lwReady) return;
  if (!window.LightweightCharts) {{
    document.getElementById('lw-chart').innerHTML =
      '<p style="color:#93a1b5;padding:20px;">Lightweight Charts could not load — check your internet connection.</p>';
    return;
  }}
  const priceEl = document.getElementById('lw-chart');
  const rsiEl = document.getElementById('lw-rsi');
  lwChart = LightweightCharts.createChart(priceEl, {{
    width: priceEl.clientWidth, height: priceEl.clientHeight, autoSize: true,
    layout: {{ background: {{ color: '#0f141b' }}, textColor: '#d1d4dc', fontSize: 11 }},
    grid: {{ vertLines: {{ color: '#1a1d2b' }}, horzLines: {{ color: '#1a1d2b' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    rightPriceScale: {{ borderColor: '#23263a', scaleMargins: {{ top: 0.08, bottom: 0.08 }}, minimumWidth: 84 }},
    timeScale: {{ borderColor: '#23263a', timeVisible: true, secondsVisible: false, rightOffset: 4, visible: false }},
    handleScale: {{ axisPressedMouseMove: true, axisDoubleClickReset: true, mouseWheel: true, pinch: true }},
    handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true }},
  }});
  lwCandle = lwChart.addCandlestickSeries({{
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  }});
  lwRsiChart = LightweightCharts.createChart(rsiEl, {{
    width: rsiEl.clientWidth, height: rsiEl.clientHeight, autoSize: true,
    layout: {{ background: {{ color: '#0f141b' }}, textColor: '#8890a8', fontSize: 10 }},
    grid: {{ vertLines: {{ color: '#1a1d2b' }}, horzLines: {{ color: '#1a1d2b' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    rightPriceScale: {{ borderColor: '#23263a', scaleMargins: {{ top: 0.1, bottom: 0.1 }}, minimumWidth: 84 }},
    timeScale: {{ borderColor: '#23263a', timeVisible: true, secondsVisible: false }},
    handleScale: {{ axisPressedMouseMove: true, axisDoubleClickReset: true, mouseWheel: true, pinch: true }},
    handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true }},
  }});
  lwRsiSeries = lwRsiChart.addLineSeries({{
    color: '#ab47bc', lineWidth: 2, priceLineVisible: false,
    lastValueVisible: false, crosshairMarkerVisible: true,
    autoscaleInfoProvider: () => ({{ priceRange: {{ minValue: 0, maxValue: 100 }} }}),
  }});
  lwChart.timeScale().subscribeVisibleLogicalRangeChange(function (range) {{
    if (!range || lwSyncing) return;
    lwSyncing = true;
    lwRsiChart.timeScale().setVisibleLogicalRange(range);
    lwSyncing = false;
    lwDrawOverlay();
  }});
  lwRsiChart.timeScale().subscribeVisibleLogicalRangeChange(function (range) {{
    if (!range || lwSyncing) return;
    lwSyncing = true;
    lwChart.timeScale().setVisibleLogicalRange(range);
    lwSyncing = false;
    lwDrawOverlay();
  }});
  lwChart.subscribeCrosshairMove(function (param) {{
    if (!param || param.time === undefined) {{
      lwShowRsi(null);
      lwHideTooltip();
      return;
    }}
    lwShowRsi(lwRsiByTime.get(param.time));
    lwUpdateTooltip(param);
  }});
  const ro = new ResizeObserver(function () {{
    lwChart.applyOptions({{ width: priceEl.clientWidth, height: priceEl.clientHeight }});
    lwRsiChart.applyOptions({{ width: rsiEl.clientWidth, height: rsiEl.clientHeight }});
    lwDrawOverlay();
  }});
  ro.observe(priceEl);
  ro.observe(rsiEl);
  priceEl.addEventListener('dblclick', function (event) {{
    const rect = priceEl.getBoundingClientRect();
    const scaleWidth = lwChart.priceScale('right').width();
    if (event.clientX >= rect.right - scaleWidth) {{
      lwChart.priceScale('right').applyOptions({{ autoScale: true }});
    }}
  }});
  rsiEl.addEventListener('dblclick', function (event) {{
    const rect = rsiEl.getBoundingClientRect();
    const scaleWidth = lwRsiChart.priceScale('right').width();
    if (event.clientX >= rect.right - scaleWidth) {{
      lwRsiChart.priceScale('right').applyOptions({{ autoScale: true }});
    }}
  }});
  lwReady = true;
}}

function lwShowRsi(value) {{
  const el = document.getElementById('lw-rsi-value');
  if (value === null || value === undefined) {{
    el.textContent = lwCurrent ? lwCurrent.rsi_label : '—';
    return;
  }}
  el.textContent = 'RSI 14 ' + Number(value).toFixed(1);
}}

function lwHideTooltip() {{
  const tip = document.getElementById('lw-tooltip');
  if (tip) tip.style.display = 'none';
}}

function lwUpdateTooltip(param) {{
  const tip = document.getElementById('lw-tooltip');
  const chartEl = document.getElementById('lw-chart');
  if (!tip || !chartEl || !param || param.point === undefined || param.time === undefined) {{
    if (tip) tip.style.display = 'none';
    return;
  }}
  if (!lwMarkerData.length) {{
    tip.style.display = 'none';
    return;
  }}
  const candidates = lwMarkerData.filter(function (m) {{ return m.time === param.time; }});
  if (!candidates.length) {{
    tip.style.display = 'none';
    return;
  }}
  let best = null;
  let bestDist = 30;
  for (const marker of candidates) {{
    const y = lwCandle.priceToCoordinate(marker.price);
    if (y === null) continue;
    const dist = Math.abs(y - param.point.y);
    if (dist < bestDist) {{
      best = marker;
      bestDist = dist;
    }}
  }}
  if (!best) {{
    tip.style.display = 'none';
    return;
  }}
  const label = best.tooltip || best.text || '';
  if (!label) {{
    tip.style.display = 'none';
    return;
  }}
  tip.textContent = label;
  tip.style.display = 'block';
  const maxX = Math.max(4, chartEl.clientWidth - 180);
  const x = Math.min(param.point.x + 12, maxX);
  const y = Math.max(4, param.point.y - 20);
  tip.style.left = Math.max(4, x) + 'px';
  tip.style.top = y + 'px';
}}

function lwUpdateRsiGuides(data) {{
  for (const line of lwRsiGuides) {{
    try {{ lwRsiSeries.removePriceLine(line); }} catch (e) {{}}
  }}
  lwRsiGuides = [];
  const levels = [70, 30];
  if (data && Number.isFinite(data.rsi_floor)) levels.push(data.rsi_floor);
  if (data && Number.isFinite(data.rsi_ceiling)) levels.push(data.rsi_ceiling);
  for (const level of levels) {{
    lwRsiGuides.push(lwRsiSeries.createPriceLine({{
      price: level,
      color: level >= 70 ? 'rgba(239,83,80,0.45)' : 'rgba(38,166,154,0.45)',
      lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: false, title: '',
    }}));
  }}
}}

function lwDrawOverlay() {{
  const svg = document.getElementById('lw-overlay');
  const chartEl = document.getElementById('lw-chart');
  if (!svg || !chartEl || !lwReady) return;
  const width = chartEl.clientWidth;
  const height = chartEl.clientHeight;
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const data = lwOverlayData;
  if (!data) return;
  const ts = lwChart.timeScale();
  const ns = 'http://www.w3.org/2000/svg';
  const coord = function (time, price) {{
    const x = ts.timeToCoordinate(time);
    const y = lwCandle.priceToCoordinate(price);
    return (x === null || y === null) ? null : {{ x: x, y: y }};
  }};
  for (const win of (data.windows || [])) {{
    const x1 = ts.timeToCoordinate(win.from_time);
    const x2 = ts.timeToCoordinate(win.to_time);
    if (x1 === null || x2 === null) continue;
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('x', Math.min(x1, x2));
    rect.setAttribute('y', 0);
    rect.setAttribute('width', Math.abs(x2 - x1));
    rect.setAttribute('height', height);
    rect.setAttribute('fill', win.color);
    rect.setAttribute('fill-opacity', win.opacity || 0.07);
    svg.appendChild(rect);
    if (win.label) {{
      const text = document.createElementNS(ns, 'text');
      text.setAttribute('x', Math.max(4, Math.min(width - 4, Math.min(x1, x2) + 3)));
      text.setAttribute('y', 12);
      text.setAttribute('fill', win.color);
      text.setAttribute('font-size', '8');
      text.textContent = win.label;
      svg.appendChild(text);
    }}
  }}
  for (const band of (data.bands || [])) {{
    const a = coord(band.from_time, band.y1);
    const b = coord(band.to_time, band.y0);
    if (!a || !b) continue;
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('x', Math.min(a.x, b.x));
    rect.setAttribute('y', Math.min(a.y, b.y));
    rect.setAttribute('width', Math.abs(b.x - a.x));
    rect.setAttribute('height', Math.abs(b.y - a.y));
    rect.setAttribute('fill', band.color);
    rect.setAttribute('fill-opacity', band.opacity || 0.13);
    svg.appendChild(rect);
    if (band.label) {{
      const text = document.createElementNS(ns, 'text');
      text.setAttribute('x', Math.max(4, Math.min(width - 4, Math.min(a.x, b.x) + 4)));
      text.setAttribute('y', (a.y + b.y) / 2);
      text.setAttribute('fill', band.color);
      text.setAttribute('font-size', '8');
      text.textContent = band.label;
      svg.appendChild(text);
    }}
  }}
  for (const line of (data.polylines || [])) {{
    const pts = [];
    for (const p of (line.points || [])) {{
      const c = coord(p.time, p.price);
      if (!c) {{ pts.length = 0; break; }}
      pts.push(c.x + ',' + c.y);
    }}
    if (!pts.length) continue;
    const poly = document.createElementNS(ns, 'polyline');
    poly.setAttribute('points', pts.join(' '));
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', line.color);
    poly.setAttribute('stroke-width', line.width || 1);
    if (line.style === 2) poly.setAttribute('stroke-dasharray', '4 3');
    svg.appendChild(poly);
  }}
  for (const line of (data.price_lines || [])) {{
    if (!line.label) continue;
    const y = lwCandle.priceToCoordinate(line.price);
    if (y === null) continue;
    const right = line.label_side === 'right';
    const text = document.createElementNS(ns, 'text');
    text.setAttribute('x', right ? width - 6 : 4);
    text.setAttribute('y', y - 4);
    text.setAttribute('text-anchor', right ? 'end' : 'start');
    text.setAttribute('fill', line.label_color || line.color || '#e8eef6');
    text.setAttribute('font-size', '8');
    text.setAttribute('font-weight', '600');
    text.setAttribute('paint-order', 'stroke');
    text.setAttribute('stroke', '#0b0d10');
    text.setAttribute('stroke-width', '3');
    text.setAttribute('stroke-linejoin', 'round');
    text.textContent = line.label;
    svg.appendChild(text);
  }}
}}

function lwSaveView() {{
  if (!lwReady || !lwChart) return;
  try {{
    const range = lwChart.timeScale().getVisibleLogicalRange();
    if (range) sessionStorage.setItem('botv2-lw-view-' + lwSource + '-' + lwTf,
      JSON.stringify({{ from: range.from, to: range.to }}));
  }} catch (e) {{}}
}}
window.addEventListener('beforeunload', lwSaveView);

function showChart(tf, preserveView) {{
  if (tf && tf !== lwTf) lwSaveView();
  if (tf) {{
    lwTf = tf;
    try {{ sessionStorage.setItem('botv2-lw-tf', lwTf); }} catch (e) {{}}
  }}
  const sel = document.getElementById('chart-symbol');
  if (sel) lwSymbol = sel.value;
  if (!LW_DATA[lwSymbol + '|' + lwTf]) {{
    lwSymbol = LW_DEFAULT.symbol;
    lwTf = LW_DEFAULT.tf;
  }}
  document.querySelectorAll('.chart-tab').forEach(function (b) {{
    if (b.dataset.tf) b.classList.toggle('active', b.dataset.tf === lwTf);
  }});
  const data = LW_DATA[lwSymbol + '|' + lwTf];
  if (!data) return;
  lwInit();
  if (!lwReady) return;
  let view = preserveView ? lwChart.timeScale().getVisibleLogicalRange() : null;
  if (!view) {{
    try {{
      const saved = JSON.parse(sessionStorage.getItem(
        'botv2-lw-view-' + lwSource + '-' + lwTf) || 'null');
      if (saved && Number.isFinite(saved.from) && Number.isFinite(saved.to)) view = saved;
    }} catch (e) {{}}
  }}
  for (const line of lwPriceLines) {{
    try {{ lwCandle.removePriceLine(line); }} catch (e) {{}}
  }}
  lwPriceLines = [];
  lwCandle.setData(data.candles);
  lwRsiSeries.setData(data.rsi);
  lwRsiByTime = new Map((data.rsi || []).map(function (p) {{ return [p.time, p.value]; }}));
  lwCurrent = data;
  for (const pl of (data.price_lines || [])) {{
    lwPriceLines.push(lwCandle.createPriceLine({{
      price: pl.price, color: pl.color, lineWidth: pl.width || 1,
      lineStyle: pl.style || 0, axisLabelVisible: pl.axis_label !== false,
      title: pl.title || '',
    }}));
  }}
  lwMarkerData = data.markers || [];
  lwCandle.setMarkers(lwMarkerData.map(function (m) {{
    return {{ time: m.time, position: m.position, shape: m.shape,
              color: m.color, text: m.text }};
  }}));
  lwOverlayData = data;
  document.getElementById('lw-title').textContent = data.title || '';
  document.getElementById('lw-summary').textContent = data.summary || '';
  document.getElementById('lw-legend').textContent = data.legend || '';
  lwShowRsi(null);
  lwUpdateRsiGuides(data);
  if (view) {{
    lwChart.timeScale().setVisibleLogicalRange(view);
    lwRsiChart.timeScale().setVisibleLogicalRange(view);
  }} else {{
    lwChart.timeScale().fitContent();
    lwRsiChart.timeScale().fitContent();
  }}
  lwDrawOverlay();
}}

function ensureChart() {{
  if (!lwReady) {{
    lwInit();
    if (lwReady) showChart(lwTf, false);
    return;
  }}
  const priceEl = document.getElementById('lw-chart');
  const rsiEl = document.getElementById('lw-rsi');
  if (priceEl && rsiEl) {{
    lwChart.applyOptions({{ width: priceEl.clientWidth, height: priceEl.clientHeight }});
    lwRsiChart.applyOptions({{ width: rsiEl.clientWidth, height: rsiEl.clientHeight }});
  }}
  lwDrawOverlay();
}}

async function refreshChart(preserveView) {{
  try {{
    const res = await fetch(lwSource + '?t=' + Date.now(), {{ cache: 'no-store' }});
    if (!res.ok) throw new Error('http ' + res.status);
    const data = await res.json();
    Object.keys(LW_DATA).forEach(function (key) {{ delete LW_DATA[key]; }});
    Object.assign(LW_DATA, data);
    showChart(lwTf, preserveView !== false);
  }} catch (err) {{
    // keep the current chart if the refresh fails
  }}
}}
function lwAfterFullscreen() {{
  setTimeout(function () {{
    ensureChart();
    lwDrawOverlay();
  }}, 60);
}}
function lwMaximizePanel(panel) {{
  panel.classList.add('chart-maximized');
  document.body.classList.add('chart-maximized-body');
  lwAfterFullscreen();
}}
function toggleChartFullscreen() {{
  const panel = document.getElementById('interactive-panel');
  if (!panel) return;
  if (document.fullscreenElement) {{
    document.exitFullscreen();
    return;
  }}
  if (panel.classList.contains('chart-maximized')) {{
    panel.classList.remove('chart-maximized');
    document.body.classList.remove('chart-maximized-body');
    lwAfterFullscreen();
    return;
  }}
  if (panel.requestFullscreen) {{
    panel.requestFullscreen().then(lwAfterFullscreen).catch(function () {{
      lwMaximizePanel(panel);
    }});
  }} else {{
    lwMaximizePanel(panel);
  }}
}}
document.addEventListener('fullscreenchange', function () {{
  const btn = document.querySelector('.chart-fullscreen');
  if (btn) btn.title = document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen';
  if (!document.fullscreenElement) lwAfterFullscreen();
}});
function showTab(name) {{
  const main = document.getElementById("main");
  const tracking = document.getElementById("tracking");
  const demo = document.getElementById("demo");
  if (main) main.style.display = name === "main" ? "block" : "none";
  if (tracking) tracking.style.display = name === "tracking" ? "block" : "none";
  if (demo) demo.style.display = name === "demo" ? "block" : "none";
  document.getElementById("tab-main").classList.toggle("tab-active", name === "main");
  document.getElementById("tab-tracking").classList.toggle("tab-active", name === "tracking");
  document.getElementById("tab-demo").classList.toggle("tab-active", name === "demo");
  try {{ localStorage.setItem("botv2-view", name); }} catch (e) {{}}
  window.scrollTo(0, 0);
  const panel = document.getElementById('interactive-panel');
  if (name === "main" && panel) {{
    const slot = document.getElementById('main-interactive-slot');
    if (slot && panel.parentElement !== slot) {{
      lwSaveView();
      slot.appendChild(panel);
    }}
    setTimeout(function () {{
      ensureChart();
      if (lwReady) showChart(lwTf, true);
    }}, 0);
  }}
}}
// keep the selected view across the auto-refresh reload
try {{
  const saved = localStorage.getItem("botv2-view");
  if (saved === "main" || saved === "tracking" || saved === "demo") showTab(saved);
  else showTab("main");
}} catch (e) {{ showTab("main"); }}
let lbZoom = 1;
let lbBaseRect = null;
function openLb(img) {{
  const lb = document.getElementById('lb');
  const lbimg = document.getElementById('lbimg');
  lbimg.src = img.src;
  lbimg.dataset.caption = img.dataset.caption || '';
  document.getElementById('lbcap').textContent = img.dataset.caption || '';
  lbZoom = 1;
  lbBaseRect = null;
  lbimg.style.transformOrigin = 'center center';
  lbimg.style.transform = 'scale(1)';
  lbimg.style.cursor = 'zoom-in';
  lbimg.dataset.zoomed = 'false';
  lb.classList.add('open');
}}
function closeLb(event) {{
  const lb = document.getElementById('lb');
  const lbimg = document.getElementById('lbimg');
  // close only when clicking outside the image (on the overlay)
  if (event && event.target && event.target.id !== 'lbimg') {{
    lbZoom = 1;
    lbBaseRect = null;
    lbimg.style.transformOrigin = 'center center';
    lbimg.style.transform = 'scale(1)';
    lbimg.dataset.zoomed = 'false';
    lbimg.style.cursor = 'zoom-in';
    lb.classList.remove('open');
  }}
}}
function toggleZoom(event) {{
  const lbimg = document.getElementById('lbimg');
  if (!lbimg) return;
  if (lbZoom <= 1) {{
    lbBaseRect = lbimg.getBoundingClientRect();
    lbZoom = 2;
  }} else {{
    lbZoom = Math.min(lbZoom + 1, 5);
  }}
  const rect = lbBaseRect || lbimg.getBoundingClientRect();
  let x = 50, y = 50;
  if (event && rect.width && rect.height) {{
    x = ((event.clientX - rect.left) / rect.width) * 100;
    y = ((event.clientY - rect.top) / rect.height) * 100;
    x = Math.max(0, Math.min(100, x));
    y = Math.max(0, Math.min(100, y));
  }}
  lbimg.style.transformOrigin = x.toFixed(2) + '% ' + y.toFixed(2) + '%';
  lbimg.style.transform = 'scale(' + lbZoom + ')';
  lbimg.style.cursor = 'zoom-in';
  lbimg.dataset.zoomed = 'true';
}}
function showTradeDetail(id) {{
  const source = document.getElementById(id);
  const modal = document.getElementById('tdetail');
  if (!source || !modal) return;
  document.getElementById('tdetail-body').innerHTML = source.innerHTML;
  modal.classList.add('open');
  modal.scrollTop = 0;
}}
function closeTradeDetail(event) {{
  const modal = document.getElementById('tdetail');
  if (!modal) return;
  if (event && event.target && event.target.id !== 'tdetail') return;
  modal.classList.remove('open');
}}
document.addEventListener('keydown', function (e) {{
  if (e.key === 'Escape') {{
    closeLb({{target: {{id: 'overlay'}}}});
    closeTradeDetail({{target: {{id: 'tdetail'}}}});
    const panel = document.getElementById('interactive-panel');
    if (panel && panel.classList.contains('chart-maximized')) {{
      panel.classList.remove('chart-maximized');
      document.body.classList.remove('chart-maximized-body');
      lwAfterFullscreen();
    }}
  }}
}});
// refresh only when a chart or a trade detail is open; show a banner if the server is down
setInterval(async function () {{
  if (document.fullscreenElement) return;
  const panel = document.getElementById('interactive-panel');
  if (panel && panel.classList.contains('chart-maximized')) return;
  if (document.getElementById('lb').classList.contains('open')) return;
  if (document.getElementById('tdetail').classList.contains('open')) return;
  try {{
    const response = await fetch('dashboard.html', {{ cache: 'no-store' }});
    if (!response.ok) throw new Error('http ' + response.status);
    document.getElementById('offline').classList.remove('show');
    const fresh = new URL(location.href);
    fresh.searchParams.set('t', Date.now());
    location.replace(fresh.toString());
  }} catch (err) {{
    document.getElementById('offline').classList.add('show');
  }}
}}, {interval * 1000});
// keep the interactive chart on the same fresh 1-min data as the PNG
setInterval(function () {{
  const view = localStorage.getItem('botv2-view');
  if (view !== 'main') return;
  if (document.getElementById('tdetail').classList.contains('open')) return;
  refreshChart();
}}, 30000);
// 1m candle follows the 1s MT4 feed; 1m history still comes from XAUUSD_1.csv
setInterval(async function () {{
  if (localStorage.getItem('botv2-view') !== 'main') return;
  if (lwTf !== '1m' || !lwReady) return;
  if (document.getElementById('tdetail').classList.contains('open')) return;
  try {{
    const res = await fetch('interactive_1m_live.json?t=' + Date.now(), {{ cache: 'no-store' }});
    if (!res.ok) return;
    const data = await res.json();
    if (!data || !data.key || !data.candle) return;
    const entry = LW_DATA[data.key];
    if (!entry || !entry.candles) return;
    const arr = entry.candles;
    const last = arr[arr.length - 1];
    if (last && last.time === data.candle.time) arr[arr.length - 1] = data.candle;
    else if (!last || data.candle.time > last.time) arr.push(data.candle);
    if (lwSymbol === data.key.split('|')[0]) lwCandle.update(data.candle);
  }} catch (e) {{}}
}}, 1000);
</script>
</body></html>"""
    path = out_dir / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# local HTTP serving (avoids Chrome's macOS "Desktop folder" permission issue)
# --------------------------------------------------------------------------- #

import socket
import subprocess
import sys
import time


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pid_path(out_dir: Path) -> Path:
    return out_dir / ".dashboard_server.pid"


def start_server(out_dir: Path, preferred_port: int = 8766) -> tuple[str, int]:
    """Serve ``out_dir`` on 127.0.0.1 and return (base_url, port)."""
    pid_file = _pid_path(out_dir)
    # reuse a server we already started
    if pid_file.exists():
        try:
            old_port = int(pid_file.read_text().split()[0])
            if _port_open(old_port):
                return f"http://127.0.0.1:{old_port}", old_port
        except (OSError, ValueError, IndexError):
            pass

    for port in range(preferred_port, preferred_port + 10):
        if _port_open(port):
            continue
        log = out_dir / "dashboard_server.log"
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        with log.open("ab") as handle:
            proc = subprocess.Popen(
                [sys.executable, "-m", "botv2.serve",
                 "--port", str(port), "--directory", str(out_dir)],
                stdout=handle, stderr=handle,
                start_new_session=True, env=env,
            )
        deadline = time.time() + 5
        while time.time() < deadline:
            if _port_open(port):
                pid_file.write_text(f"{port} {proc.pid}")
                return f"http://127.0.0.1:{port}", port
            time.sleep(0.1)
        proc.terminate()
    raise RuntimeError("could not start local dashboard server")


def stop_server(out_dir: Path) -> bool:
    pid_file = _pid_path(out_dir)
    if not pid_file.exists():
        return False
    try:
        parts = pid_file.read_text().split()
        pid = int(parts[1])
        subprocess.run(["kill", str(pid)], check=False)
    except (OSError, ValueError, IndexError):
        pass
    pid_file.unlink(missing_ok=True)
    return True


def ensure_server(out_dir: Path, preferred_port: int = 8766) -> str:
    """Return the dashboard URL, restarting the local server if it died."""
    pid_file = _pid_path(out_dir)
    if pid_file.exists():
        try:
            port = int(pid_file.read_text().split()[0])
            if _port_open(port):
                return f"http://127.0.0.1:{port}/dashboard.html"
        except (OSError, ValueError, IndexError):
            pass
        pid_file.unlink(missing_ok=True)
    base_url, _ = start_server(out_dir, preferred_port)
    return f"{base_url}/dashboard.html"