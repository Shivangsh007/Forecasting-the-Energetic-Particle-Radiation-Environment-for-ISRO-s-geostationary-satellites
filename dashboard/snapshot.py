"""
SolarSentinel - dashboard data payload builder.

Produces the JSON payload the dashboard renders. For Section 7 it builds the
payload for any timestamp from the historical feature matrix + trained models
(so we can drive the UI with REAL data, incl. historical storms). The Section 8
daemon will emit the SAME schema from live NOAA feeds -> the dashboard code does
not change between replay and live.

CLI:  python dashboard/snapshot.py [--time "YYYY-MM-DD HH:MM"] [--out path.json]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "features" / "features_master.parquet"
# Deployment fallback: the full 8-year training matrix (~200 MB) is git-ignored and
# absent from a fresh cloud checkout. features_replay_sample.parquet is a small,
# committed recent slice (identical columns / same code path) so the deployed app's
# load_context + Replay still work. Local dev uses the full matrix if present.
FEAT_SAMPLE = ROOT / "features" / "features_replay_sample.parquet"
MANIFEST = ROOT / "features" / "feature_manifest.json"
MODEL_DIR = ROOT / "models"
METRICS5 = MODEL_DIR / "metrics_section5.json"
METRICS6 = ROOT / "evaluate" / "metrics_summary.json"
LIVE_OUT = ROOT / "data" / "live" / "latest.json"

import sys
sys.path.insert(0, str(ROOT / "dashboard"))
from hazard import classify_hazard, skill_from_hss_gain, SATELLITES  # noqa: E402

HORIZONS = {"30min": 6, "6h": 72, "12h": 144}      # steps @ 5-min
TELEMETRY = {"flux_2MeV": "pfu", "flow_speed": "km/s", "BZ_GSM": "nT", "SYM_H": "nT"}


def inv_log(x):
    return float(np.clip(10.0 ** x - 1.0, 0.0, None))


_CTX = None


def load_context():
    global _CTX
    if _CTX is not None:
        return _CTX
    feat_path = FEAT if FEAT.exists() else FEAT_SAMPLE
    df = pd.read_parquet(feat_path).sort_index()
    manifest = json.loads(MANIFEST.read_text())
    models = {h: xgb.XGBRegressor() for h in HORIZONS}
    for h, m in models.items():
        m.load_model(MODEL_DIR / f"xgb_{h}.json")
    m5 = {r["horizon"]: r for r in json.loads(METRICS5.read_text())}
    m6 = json.loads(METRICS6.read_text())
    rmse_log = {h: m5[h]["xgb_rmse_log"] for h in HORIZONS}
    hss_gain = {h: (m6[h]["threshold"]["xgb"]["HSS"]
                    - m6[h]["threshold"]["persist"]["HSS"]) for h in HORIZONS}
    _CTX = dict(df=df, manifest=manifest, models=models,
                rmse_log=rmse_log, hss_gain=hss_gain)
    return _CTX


def _delta(df, t, col, steps=12):
    """value at t minus value ~1h earlier (steps*5 min)."""
    try:
        now = float(df.at[t, col])
    except KeyError:
        return None, None
    pos = df.index.get_loc(t)
    prev = float(df.iloc[pos - steps][col]) if pos - steps >= 0 else np.nan
    d = None if np.isnan(prev) else now - prev
    return (None if np.isnan(now) else now), d


def payload_from_frame(df, t, source, status="ok", extra=None):
    """Build the dashboard payload from ANY assembled 5-min frame at time `t`.

    Single payload code path shared by historical replay (features_master) and the
    live daemon (live grid) -> forecast/telemetry/hazard are computed identically.
    `df` must carry the raw columns + all feature columns + log_flux (i.e. the
    output of assemble_features / the training matrix).
    """
    ctx = load_context()
    manifest = ctx["manifest"]

    telem = {}
    for col, unit in TELEMETRY.items():
        val, d = _delta(df, t, col)
        telem[col] = {"value": val, "delta": d, "unit": unit}
    current_flux = telem["flux_2MeV"]["value"] or 0.0

    hist = df.loc[t - pd.Timedelta(hours=24):t, "flux_2MeV"].dropna()
    observed = {"time": [ts.isoformat() for ts in hist.index],
                "flux": [float(v) for v in hist.values]}

    forecast = {}
    pos = df.index.get_loc(t)
    for h, steps in HORIZONS.items():
        cols = manifest["horizon_feature_columns"][h]
        pred_log = float(ctx["models"][h].predict(df.loc[[t], cols])[0])
        rmse = ctx["rmse_log"][h]
        gain = ctx["hss_gain"][h]
        fut = None                                  # future truth (replay only)
        if pos + steps < len(df):
            fv = df.iloc[pos + steps]["flux_2MeV"]
            fut = float(fv) if pd.notna(fv) else None
        forecast[h] = {
            "lead_min": steps * 5,
            "valid_time": (t + pd.Timedelta(minutes=steps * 5)).isoformat(),
            "flux": inv_log(pred_log),
            "lo": inv_log(pred_log - rmse), "hi": inv_log(pred_log + rmse),
            "persist": current_flux, "rmse_log": rmse, "hss_gain": gain,
            "skill": skill_from_hss_gain(gain), "actual": fut,
        }

    hazard = classify_hazard(current_flux, {h: forecast[h]["flux"] for h in HORIZONS})
    sats = [{**s, "local_flux": current_flux, "status": hazard["level"]}
            for s in SATELLITES]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid_time": t.isoformat(),
        "source": source,
        "status": status,
        "telemetry": telem,
        "observed": observed,
        "forecast": forecast,
        "hazard": hazard,
        "satellites": sats,
        "note_local_flux": ("Per-satellite local flux uses the global GOES-derived "
                            ">2 MeV flux; longitude-resolved flux awaits GRASP/GSAT-19."),
    }
    if extra:
        payload.update(extra)
    return payload


def build_payload(valid_time=None, source="historical_replay"):
    """Historical/replay payload from the training feature matrix."""
    ctx = load_context()
    df = ctx["df"]
    if valid_time is None:
        t = df.index.max()
    else:
        t = pd.Timestamp(valid_time)
        if t not in df.index:
            t = df.index[df.index.get_indexer([t], method="nearest")[0]]
    return payload_from_frame(df, t, source, status="replay")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default=None, help='e.g. "2025-10-06 03:00"')
    ap.add_argument("--out", default=str(LIVE_OUT))
    args = ap.parse_args()
    payload = build_payload(args.time)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    hz = payload["hazard"]
    print(f"[snapshot] valid_time={payload['valid_time']}  "
          f"hazard={hz['level']} ({hz['title']})  peak={hz['peak_flux']:,.0f} pfu")
    print(f"[snapshot] wrote {out}")


if __name__ == "__main__":
    main()
