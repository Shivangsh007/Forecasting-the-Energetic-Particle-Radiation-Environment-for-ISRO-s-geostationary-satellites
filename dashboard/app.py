"""
SolarSentinel - Streamlit + Plotly dashboard ("Asset Command").

Command-center restyle: near-black dense HUD, centered forecast hero flanked by
compact panels, plus a Model Performance panel that reads real numbers from
evaluate/model_performance_panel.json (never hardcoded). Data shown, hazard logic,
and how models are called are UNCHANGED — this is a visual/layout pass only.

Run:  streamlit run dashboard/app.py
"""

import json
import os
import sys
import threading
import time as _time
from datetime import timezone, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(ROOT))                 # so `import realtime_daemon` resolves
from snapshot import build_payload, LIVE_OUT, load_context  # noqa: E402
from charts import forecast_figure, map_figure, sparkline_figure  # noqa: E402
from hazard import HORIZON_LABELS  # noqa: E402

PANEL_JSON = ROOT / "evaluate" / "model_performance_panel.json"
IST = timezone(timedelta(hours=5, minutes=30))
FC_CONFIG = {"displayModeBar": True, "displaylogo": False,
             "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

st.set_page_config(page_title="SolarSentinel — Asset Command", layout="wide",
                   page_icon="🛰️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=Roboto+Mono:wght@400;500&display=swap');
:root { --bg:#04070e; --panel:#070b16; --edge:rgba(200,212,240,0.20);
        --ink:#dbe3f4; --muted:#7c8aa8; --num:#eef2fb;
        --mono:'Roboto Mono',ui-monospace,'DejaVu Sans Mono',monospace;
        --cond:'Barlow Condensed','Arial Narrow',sans-serif; }
.stApp { background:var(--bg); }
.block-container { padding:0.55rem 1rem 1rem 1rem; max-width:100%; }
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none!important; }
.stButton>button, .stDownloadButton>button { border-radius:0!important; border:1px solid var(--edge)!important;
  background:var(--panel)!important; color:var(--ink)!important; font-family:var(--mono)!important;
  padding:2px 10px!important; min-height:0!important; line-height:1.35!important; }
div[data-testid="stHorizontalBlock"] { align-items:center; }
[data-testid="stRadio"] div[role="radiogroup"] { gap:0.6rem; }
[data-testid="stRadio"] label p { font-family:var(--mono); font-size:0.78rem; color:var(--ink); }
h1,h2,h3,h4,h5 { font-family:var(--cond)!important; letter-spacing:0.06em; }
.panel { background:var(--panel); border:1px solid var(--edge); border-radius:0;
         padding:8px 11px; margin-bottom:7px; }
.lbl { font-family:var(--cond); text-transform:uppercase; letter-spacing:0.11em;
       font-size:0.62rem; color:var(--muted); }
.val { font-family:var(--mono); font-size:1.45rem; font-weight:500; color:var(--num);
       line-height:1.12; }
.unit { font-size:0.62rem; color:var(--muted); }
.banner { border-radius:0; padding:11px 15px; margin-bottom:8px; border-left-width:4px;
          border-left-style:solid; }
/* Streamlit bordered containers -> sharp panels */
[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--edge)!important;
    border-radius:0!important; background:var(--panel); }
[data-testid="stVerticalBlockBorderWrapper"] > div { padding:6px 8px; }
/* model-performance table */
table.mp { width:100%; border-collapse:collapse; font-family:var(--mono); }
table.mp th, table.mp td { border:1px solid var(--edge); padding:4px 9px; text-align:right;
    font-size:0.78rem; color:var(--num); }
table.mp th { font-family:var(--cond); text-transform:uppercase; letter-spacing:0.08em;
    color:var(--muted); font-weight:600; font-size:0.7rem; }
table.mp td.rl { text-align:left; font-family:var(--cond); text-transform:uppercase;
    letter-spacing:0.05em; color:var(--muted); font-size:0.72rem; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def _ctx():
    return load_context()


@st.cache_resource
def _live_daemon():
    """Start the NOAA poll loop ONCE per app process, shared across all visitors.

    On Streamlit Community Cloud every visitor hits the SAME process; st.cache_resource
    runs this body exactly once and hands the same object to every session, so the
    background poll thread launches a single time (not per visitor). It reuses
    realtime_daemon.poll() -> the SAME assemble_features()/payload_from_frame()
    pipeline as training and replay (no second feature path). The thread only writes
    data/live/latest.json (atomically); it never touches Streamlit, so there is no
    ScriptRunContext issue. Set SOLARSENTINEL_INPROC_DAEMON=0 to disable (e.g. when
    running an external `python realtime_daemon.py --loop` locally instead).
    """
    if os.environ.get("SOLARSENTINEL_INPROC_DAEMON", "1") != "1":
        return {"enabled": False, "reason": "disabled via SOLARSENTINEL_INPROC_DAEMON=0"}
    from realtime_daemon import poll, POLL_SECONDS

    def _run():
        while True:
            try:
                poll()
            except Exception as e:                # never die on a single poll
                print(f"[inproc-daemon] {type(e).__name__}: {e}", flush=True)
            _time.sleep(POLL_SECONDS)

    th = threading.Thread(target=_run, name="solarsentinel-daemon", daemon=True)
    th.start()
    return {"enabled": True, "thread_name": th.name,
            "started_at": pd.Timestamp.utcnow().isoformat()}


def fmt_time(iso, tz_ist):
    t = pd.Timestamp(iso)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.astimezone(IST) if tz_ist else t.astimezone(timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M ") + ("IST" if tz_ist else "UTC")


def telem_html(payload):
    cards = [("Global >2 MeV Flux", "flux_2MeV", "{:,.0f}"),
             ("Solar Wind Speed", "flow_speed", "{:,.0f}"),
             ("IMF Bz (GSM)", "BZ_GSM", "{:+.1f}"),
             ("SYM-H", "SYM_H", "{:+.0f}")]
    out = ""
    for lbl, key, fmt in cards:
        tm = payload["telemetry"][key]
        v, d = tm["value"], tm["delta"]
        vtxt = fmt.format(v) if v is not None else "—"
        dtxt = ""
        if d is not None:
            arrow = "▲" if d > 0 else "▼" if d < 0 else "▬"
            dtxt = f'<span class="lbl">{arrow} {abs(d):.1f} {tm["unit"]}/h · 1h</span>'
        out += (f'<div class="panel"><div class="lbl">{lbl}</div>'
                f'<div class="val">{vtxt} <span class="unit">{tm["unit"]}</span></div>{dtxt}</div>')
    return out


def model_perf_html(pd_):
    ph, ctx = pd_["per_horizon"], pd_["context"]
    hs = ["30min", "6h", "12h"]
    lab = {"30min": "+30 MIN", "6h": "+6 H", "12h": "+12 H"}
    rows = [("MAE (pfu)", lambda d: f"{d['mae_pfu']:,.0f}"),
            ("RMSE (pfu)", lambda d: f"{d['rmse_pfu']:,.0f}"),
            ("R²", lambda d: f"{d['r2_raw']:.3f}"),
            ("Pearson r", lambda d: f"{d['pearson_raw']:.3f}"),
            ("Skill vs persist", lambda d: f"{d['skill_vs_persistence_pct']:+.1f}%"),
            ("POD @1k pfu", lambda d: f"{d['pod']:.3f}"),
            ("FAR @1k pfu", lambda d: f"{d['far']:.3f}"),
            ("HSS @1k pfu", lambda d: f"{d['hss']:.3f}")]
    th = "".join(f"<th>{lab[h]}</th>" for h in hs)
    body = ""
    for name, fn in rows:
        tds = "".join(f"<td>{fn(ph[h])}</td>" for h in hs)
        body += f"<tr><td class='rl'>{name}</td>{tds}</tr>"
    return f'<table class="mp"><tr><th style="text-align:left">METRIC</th>{th}</tr>{body}</table>'


# -------------------------------------------------------- header + controls --- #
_ctx()
_live_daemon()          # start the shared background poll loop once per app process
st.markdown("""<div style="display:flex;align-items:baseline;gap:12px;
   border-bottom:1px solid var(--edge);padding-bottom:5px;margin-bottom:7px">
   <span style="font-family:var(--cond);font-size:1.5rem;letter-spacing:0.2em;
   color:var(--num);font-weight:600">SOLARSENTINEL</span>
   <span class="lbl">Energetic particle radiation forecast · GEO &gt;2 MeV electrons</span>
   </div>""", unsafe_allow_html=True)

# thin inline control strip (no walled-off sidebar block)
cc = st.columns([1.25, 2.4, 0.7, 0.42, 0.8], gap="small")
mode = cc[0].radio("src", ["Live", "Replay"], horizontal=True, label_visibility="collapsed")
day = tmv = None
if mode == "Replay":
    di = _ctx()["df"].index
    dd = pd.Timestamp("2025-10-06 03:00")
    rc = cc[1].columns(2)
    day = rc[0].date_input("d", value=dd.date(), min_value=di.min().date(),
                           max_value=di.max().date(), label_visibility="collapsed")
    tmv = rc[1].time_input("t", value=dd.time(), label_visibility="collapsed")
tz_ist = cc[2].toggle("IST", value=True)
if cc[3].button("↻", help="Refresh data"):
    st.rerun()
report_slot = cc[4].empty()

# payload (mode-driven)
if mode == "Replay":
    payload = build_payload(f"{day} {tmv}")
elif Path(LIVE_OUT).exists():
    payload = json.loads(Path(LIVE_OUT).read_text())
else:
    payload = build_payload(None, source="latest_available")
status = payload.get("status", "ok")

# --- error guard: a daemon 'error' payload has no telemetry/satellites --- #
if status == "error" or "satellites" not in payload:
    st.markdown(f"""<div style="border:1px solid #ef4444;border-left:3px solid #ef4444;
      background:var(--panel);padding:8px 13px"><span style="font-family:var(--cond);
      text-transform:uppercase;letter-spacing:0.1em;color:#ef4444;font-weight:600">
      [■] Live feed unavailable</span><span style="color:#c4cde0;margin-left:10px;font-size:0.85rem">
      {payload.get('error','No current data.')} · last check {payload.get('checked_at','?')}</span>
      </div>""", unsafe_allow_html=True)
    st.stop()

hz = payload["hazard"]
report = (f"SolarSentinel Hazard Bulletin\nValid: {payload['valid_time']} UTC\n"
          f"Status: {hz['level']} ({hz['title']})\n{hz['message']}\n\nForecast:\n" +
          "\n".join(f"  {HORIZON_LABELS[h]}: {payload['forecast'][h]['flux']:,.0f} pfu "
                    f"(skill: {payload['forecast'][h]['skill']})" for h in
                    ["30min", "6h", "12h"]))
report_slot.download_button("⤓ Report", report,
                            file_name=f"bulletin_{payload['valid_time'][:16].replace(':','')}.txt")

# thin stale / degraded strip (same logic, HUD styling)
if status in ("stale", "degraded"):
    if status == "stale":
        scol, stitle = "#ef4444", "STALE DATA"
        smsg = (f"Live feed check failed ({payload.get('stale_reason','feed error')}); showing "
                f"LAST GOOD data — NOT current. Last valid {fmt_time(payload['valid_time'], tz_ist)}.")
    else:
        scol, stitle = "#eab308", "DEGRADED FEED"
        down = [f for f, ok in payload.get("feeds_ok", {}).items() if not ok]
        smsg = (f"Driver feed(s) down: {', '.join(down) or 'n/a'}; forecast on reduced inputs, "
                f"longer-horizon skill lower.")
    st.markdown(f"""<div style="border:1px solid var(--edge);border-left:2px solid {scol};
      background:var(--panel);padding:5px 13px;margin-bottom:6px">
      <span style="font-family:var(--cond);text-transform:uppercase;letter-spacing:0.11em;
      color:{scol};font-weight:600;font-size:0.76rem">⚠ {stitle}</span>
      <span style="color:#c4cde0;font-size:0.82rem;margin-left:8px">{smsg}</span></div>""",
                unsafe_allow_html=True)

# hazard status — thin HUD strip (bracketed indicator + mono, restrained accent)
st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;border:1px solid var(--edge);
  border-left:2px solid {hz['color']};background:var(--panel);padding:7px 13px;margin-bottom:8px">
  <span style="color:{hz['color']};font-family:var(--mono);font-size:0.72rem">[■]</span>
  <span style="font-family:var(--cond);text-transform:uppercase;letter-spacing:0.14em;
   color:{hz['color']};font-weight:600;font-size:0.83rem;white-space:nowrap">{hz['level']} · {hz['title']}</span>
  <span style="color:#c4cde0;font-size:0.83rem;flex:1;min-width:0">{hz['message']}</span>
  <span class="lbl" style="white-space:nowrap">VALID {fmt_time(payload['valid_time'], tz_ist)} · {payload['source'].upper()}</span>
  </div>""", unsafe_allow_html=True)

# --------------------------------------------------- centered hero + flanks --- #
lcol, ccol, rcol = st.columns([1.05, 2.35, 1.2], gap="small")

with lcol:
    st.markdown(telem_html(payload), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="lbl">Global flux · last 24 h</div>', unsafe_allow_html=True)
        st.plotly_chart(sparkline_figure(payload["observed"]), width="stretch",
                        config={"displayModeBar": False})

with ccol:
    with st.container(border=True):
        st.markdown('<div class="lbl">Observed → forecast · >2 MeV flux (pfu, log scale)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(forecast_figure(payload), width="stretch", config=FC_CONFIG)
    st.markdown('<div class="lbl">Forecasts are discrete (3 trained horizons); markers not '
                'interpolated. Marker size/opacity ∝ validated skill vs persistence — '
                '+30 min ties persistence (HSS≈0), +6 h/+12 h add skill.</div>',
                unsafe_allow_html=True)

with rcol:
    st.markdown('<div class="lbl">Asset select</div>', unsafe_allow_html=True)
    sel = st.selectbox("sat", [s["name"] for s in payload["satellites"]],
                       label_visibility="collapsed")
    sat = next(s for s in payload["satellites"] if s["name"] == sel)
    st.markdown(f"""<div class="panel"><div class="lbl">{sat['name']} · {sat['lon_e']:.0f}°E</div>
      <div style="color:#c9d4f0;font-size:0.78rem;margin:2px 0">{sat['function']}</div>
      <div class="lbl">Local >2 MeV flux</div>
      <div class="val" style="font-size:1.1rem">{sat['local_flux']:,.0f} <span class="unit">pfu</span></div>
      <div style="margin-top:3px"><span class="lbl">Status</span>
      <b style="color:{hz['color']};font-family:var(--mono)"> {hz['level'].upper()}</b></div>
      </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="panel"><div class="lbl">Bulletin</div>
      <div style="color:#c9d4f0;font-size:0.76rem;line-height:1.35">{hz['message']}</div></div>""",
                unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="lbl">GEO assets · selected highlighted</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(map_figure(payload, sel), width="stretch",
                        config={"displayModeBar": False})

# --------------------------------------------------- model performance panel --- #
st.markdown('<div style="border-top:1px solid var(--edge);margin:8px 0 6px 0"></div>',
            unsafe_allow_html=True)
if PANEL_JSON.exists():
    pdj = json.loads(PANEL_JSON.read_text())
    ctx = pdj["context"]
    g = ctx["grasp_cross_longitude_2017_out_of_time"]
    n = pdj["per_horizon"]
    mp1, mp2 = st.columns([1.5, 1], gap="small")
    with mp1:
        st.markdown('<div class="lbl" style="font-size:0.72rem">Model performance · '
                    'test set · per horizon (not blended)</div>', unsafe_allow_html=True)
        st.markdown(model_perf_html(pdj), unsafe_allow_html=True)
        note = n["30min"].get("note")
        if note:
            st.markdown(f'<div class="lbl" style="margin-top:5px;color:#c9a86a">⚠ +30 MIN: '
                        f'{note}</div>', unsafe_allow_html=True)
    with mp2:
        st.markdown(f"""<div class="panel" style="border-left:4px solid var(--edge)">
          <div class="lbl">Cross-longitude validation · strongest differentiator</div>
          <div style="color:#c9d4f0;font-size:0.76rem;margin:3px 0 5px 0">Independent ISRO
          GSAT-19/GRASP (Indian longitude), <b>2017 out-of-time</b> — never in training:</div>
          <table class="mp" style="font-size:0.74rem"><tr><th style="text-align:left">HORIZON</th>
          <th>PEARSON r</th><th>HSS</th><th>PERSIST</th></tr>
          <tr><td class="rl">+6 H</td><td>{g['6h']['pearson_r_log']:.2f}</td>
          <td>{g['6h']['hss_1000pfu']:.2f}</td><td>{g['6h']['hss_persist']:.2f}</td></tr>
          <tr><td class="rl">+12 H</td><td>{g['12h']['pearson_r_log']:.2f}</td>
          <td>{g['12h']['hss_1000pfu']:.2f}</td><td>{g['12h']['hss_persist']:.2f}</td></tr></table>
          <div class="lbl" style="margin-top:6px">{ctx['model_revision']} · trained
          {ctx['last_trained_utc'][:10]} · test n={n['30min']['n_test']:,}/{n['6h']['n_test']:,}/{n['12h']['n_test']:,}</div>
          </div>""", unsafe_allow_html=True)
else:
    st.caption("model_performance_panel.json not found — run "
               "`python evaluate/build_model_performance_panel.py`.")

# ---------------------------------------------------------------- footer --- #
_notes = []
if payload.get("flux_calibration", {}).get("applied"):
    fcb = payload["flux_calibration"]
    _notes.append(f"Flux calibrated SWPC→NCEI {fcb['raw_swpc_pfu']:.0f}→"
                  f"{fcb['calibrated_pfu']:.0f} pfu (provisional)")
_notes.append(payload["note_local_flux"])
st.markdown('<div class="lbl" style="margin-top:12px;border-top:1px solid var(--edge);'
            'padding-top:8px">' + ' · '.join(_notes) + '</div>', unsafe_allow_html=True)
