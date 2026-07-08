"""
SolarSentinel - dashboard Plotly figures (pure functions -> testable outside Streamlit).

Command-center restyle (visual only — no data/logic change): near-black surfaces,
thin recessive grid, glowing near-white observed line, grayscale everywhere with
colour reserved for hazard STATUS (Green/Yellow/Red) only.

Forecast chart honesty rules (unchanged, carried from Section 6):
  - Only 3 discrete horizons -> DISCRETE markers, never an interpolated curve.
  - Marker opacity/size encode VALIDATED skill over persistence (+30 min ties
    persistence, rendered faint + tagged "= persistence"); +6 h/+12 h add skill.
  - The persistence no-skill baseline is drawn explicitly for comparison.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hazard import (SKILL_STYLE, THRESH_EVENT, THRESH_SEVERE,  # noqa: E402
                    LEVELS, HORIZON_LABELS)

# command-center palette (grayscale + status only)
BG = "#04070e"            # near-black surface
GRIDC = "rgba(150,165,195,0.10)"   # recessive grid
INK = "#dbe3f4"           # primary ink
MUTED = "#7c8aa8"         # labels / secondary
OBS = "#cfe0ff"           # observed line (near-white, cool)
OBS_GLOW = "rgba(150,196,255,0.16)"
PERSIST = "rgba(150,162,190,0.55)"
MONO = "ui-monospace, 'Roboto Mono', 'DejaVu Sans Mono', monospace"
ORDER = ["30min", "6h", "12h"]


def _flux_color(flux):
    if flux >= THRESH_SEVERE:
        return LEVELS["Red"]["color"]
    if flux >= THRESH_EVENT:
        return LEVELS["Yellow"]["color"]
    return LEVELS["Green"]["color"]


def forecast_figure(payload):
    t0 = pd.Timestamp(payload["valid_time"])
    obs = payload["observed"]
    fc = payload["forecast"]
    cur = payload["telemetry"]["flux_2MeV"]["value"] or 0.0
    ox = pd.to_datetime(obs["time"])
    fig = go.Figure()

    # observed history — glow underlay + thin bright line
    fig.add_trace(go.Scatter(x=ox, y=obs["flux"], mode="lines",
                             line=dict(color=OBS_GLOW, width=7), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(
        x=ox, y=obs["flux"], mode="lines", line=dict(color=OBS, width=1.8),
        name="Observed >2 MeV flux",
        hovertemplate="%{x|%d %b %H:%M}Z<br>%{y:,.0f} pfu<extra>observed</extra>"))

    # persistence no-skill baseline (flat at current flux)
    last = pd.Timestamp(fc["12h"]["valid_time"])
    fig.add_trace(go.Scatter(
        x=[t0, last], y=[cur, cur], mode="lines",
        line=dict(color=PERSIST, width=1.3, dash="dot"),
        name="Persistence (no-skill baseline)"))

    # discrete forecast: whiskers + dashed connector + skill-weighted markers
    xs, ys = [t0], [cur]
    for h in ORDER:
        f = fc[h]
        tt = pd.Timestamp(f["valid_time"])
        style = SKILL_STYLE[f["skill"]]
        fig.add_trace(go.Scatter(
            x=[tt, tt], y=[f["lo"], f["hi"]], mode="lines",
            line=dict(color=_flux_color(f["flux"]), width=2), opacity=style["opacity"],
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[tt], y=[f["flux"]], mode="markers",
            marker=dict(size=style["size"], color=_flux_color(f["flux"]),
                        line=dict(color="#04070e", width=1.4), opacity=style["opacity"]),
            name=f"{HORIZON_LABELS[h]} forecast",
            hovertemplate=(f"<b>{HORIZON_LABELS[h]}</b><br>%{{y:,.0f}} pfu<br>"
                           f"skill: {f['skill']} (HSS gain {f['hss_gain']:+.3f})"
                           f"<extra></extra>")))
        if f.get("actual") is not None:      # replay verification
            fig.add_trace(go.Scatter(
                x=[tt], y=[f["actual"]], mode="markers",
                marker=dict(size=9, symbol="circle-open", color=INK, line=dict(width=2)),
                name="Observed (verification)", showlegend=(h == "30min"), hoverinfo="y"))
        xs.append(tt); ys.append(f["flux"])
        if f["skill"] == "none":
            fig.add_annotation(x=tt, y=f["flux"], text="= persistence", showarrow=True,
                               arrowcolor=MUTED, ax=0, ay=-32,
                               font=dict(color=MUTED, size=10, family=MONO))

    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="rgba(160,172,199,0.7)", width=1.4, dash="dash"),
                             name="Forecast (discrete)", hoverinfo="skip"))

    for thr, lab, col in [(THRESH_EVENT, "1000 pfu event", LEVELS["Yellow"]["color"]),
                          (THRESH_SEVERE, "10000 pfu critical", LEVELS["Red"]["color"])]:
        fig.add_hline(y=thr, line=dict(color=col, width=1, dash="dash"),
                      annotation_text=lab, annotation_position="right",
                      annotation_font=dict(color=col, size=9, family=MONO))
    fig.add_vline(x=t0.timestamp() * 1000, line=dict(color=INK, width=1.1),
                  annotation_text="NOW", annotation_position="top",
                  annotation_font=dict(color=INK, size=9, family=MONO))

    fig.update_yaxes(type="log", title=None, gridcolor=GRIDC, zeroline=False,
                     tickfont=dict(family=MONO, size=10, color=MUTED),
                     linecolor=GRIDC, ticksuffix="  pfu")
    fig.update_xaxes(title=None, gridcolor=GRIDC, linecolor=GRIDC,
                     tickfont=dict(family=MONO, size=10, color=MUTED))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=INK, family=MONO),
        height=520, margin=dict(l=8, r=8, t=26, b=8),
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color=MUTED, family=MONO)),
        hoverlabel=dict(bgcolor="#0a0f1c", font=dict(family=MONO, color=INK)))
    return fig


def sparkline_figure(observed, color=OBS, height=64):
    """Compact recent-flux sparkline (no axes/labels) for a side HUD panel."""
    x = pd.to_datetime(observed["time"])
    y = observed["flux"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=OBS_GLOW, width=5),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=1.3),
                             hovertemplate="%{y:,.0f} pfu<extra></extra>", showlegend=False))
    fig.update_yaxes(type="log", visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=0, r=0, t=2, b=0),
                      hoverlabel=dict(bgcolor="#0a0f1c", font=dict(family=MONO)))
    return fig


def map_figure(payload, selected_name):
    lons, lats, colors, sizes, texts, lines = [], [], [], [], [], []
    for i, s in enumerate(payload["satellites"]):
        is_sel = s["name"] == selected_name
        lons.append(s["lon_e"])
        lats.append(0.0 + (i - 3) * 1.1)          # display jitter only (real GEO lat~0)
        colors.append(LEVELS[s["status"]]["color"] if is_sel else "rgba(220,228,245,0.72)")
        sizes.append(15 if is_sel else 6)
        lines.append(2.0 if is_sel else 0.0)
        texts.append(f"{s['name']} ({s['lon_e']:.0f}E) - {s['status']}")
    fig = go.Figure(go.Scattergeo(
        lon=lons, lat=lats, text=texts, mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.95,
                    line=dict(color="#e8eefc", width=lines)),
        hovertemplate="%{text}<extra></extra>"))
    fig.update_geos(projection_type="natural earth", bgcolor=BG,
                    showland=True, landcolor=BG, showocean=True, oceancolor=BG,
                    lakecolor=BG, showcountries=True,
                    coastlinecolor="rgba(210,222,248,0.42)", coastlinewidth=0.6,
                    countrycolor="rgba(210,222,248,0.22)", countrywidth=0.5,
                    lataxis=dict(showgrid=True, gridcolor=GRIDC, gridwidth=0.5),
                    lonaxis=dict(showgrid=True, gridcolor=GRIDC, gridwidth=0.5),
                    center=dict(lon=60, lat=0), lataxis_range=[-55, 55],
                    lonaxis_range=[-110, 150])
    fig.update_layout(paper_bgcolor=BG, height=250, margin=dict(l=0, r=0, t=0, b=0),
                      font=dict(color=INK, family=MONO),
                      hoverlabel=dict(bgcolor="#0a0f1c", font=dict(family=MONO)))
    return fig
