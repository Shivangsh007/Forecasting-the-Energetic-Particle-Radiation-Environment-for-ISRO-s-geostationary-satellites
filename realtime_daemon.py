"""
SolarSentinel - Section 8: real-time inference daemon.

Polls NOAA SWPC public JSON feeds, builds features through the SAME
assemble_features() as training (no second feature path), runs the 3 trained
models via the SAME payload_from_frame() as the dashboard replay, and writes
data/live/latest.json for the Streamlit UI to read. The UI never calls models
inline; it only reads this file.

Feeds (verified working 2026-07; the brief's products/solar-wind/*-1-day.json
URLs now 404 -> replaced with the current RTSW endpoints):
  - IMF Bz            : json/rtsw/rtsw_mag_1m.json      (bz_gsm)
  - solar wind        : json/rtsw/rtsw_wind_1m.json     (proton_speed, proton_density)
  - >2 MeV electrons  : json/goes/primary/integral-electrons-1-day.json (flux, >=2 MeV)

Live-data limitations (flagged, NOT worked around with a divergent pipeline):
  - SYM_H is now sourced live from the Kyoto Dst feed (products/kyoto-dst.json):
    hourly Dst is the same physical ring-current disturbance as the 1-min SYM_H
    used in training, on the same nT scale. It fills the SYM_H raw column and the
    SAME assemble_features() derives all SYM_H features. Offline ablation shows this
    recovers ~100% of the SYM_H-attributable +6h/+12h skill (the top +6h/+12h driver
    SYM_H_mean_24h is a 24h average that hourly Dst reproduces almost exactly). See
    PROGRESS.md R10. AE_INDEX still has no SWPC real-time equivalent -> stays NaN
    (only ~2-3% of model gain, so the residual live gap is small). +30 min is
    flux_now-dominated and was never affected.
  - 27-day Carrington + the longest SYM_H lags (48-60 h) need a long history -> the
    daemon accumulates a persistent buffer (data/live/history.parquet); those stay
    NaN until it matures (~40 d). The high-importance 24h-window SYM_H features
    populate immediately from the 7-day Dst feed.
  - SWPC integral flux vs the NCEI L2 training flux may carry a calibration offset;
    worth a cross-check over an overlapping period.

Run once (cron/systemd-friendly):  python realtime_daemon.py
Continuous:                         python realtime_daemon.py --loop
"""

import argparse
import json
import os
import ssl
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "dashboard"))
from build_features import assemble_features        # noqa: E402  same code as training
from snapshot import payload_from_frame, load_context  # noqa: E402  same as dashboard

LIVE_DIR = ROOT / "data" / "live"
LATEST = LIVE_DIR / "latest.json"
BUFFER = LIVE_DIR / "history.parquet"
CALIB_PATH = LIVE_DIR / "flux_calibration.json"    # SWPC->NCEI scale (fit_flux_calibration.py)


def load_calibration():
    if CALIB_PATH.exists():
        try:
            return json.loads(CALIB_PATH.read_text())
        except Exception:                            # noqa: BLE001
            return None
    return None

FEEDS = {
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "wind": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    "electrons": "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json",
    "dst": "https://services.swpc.noaa.gov/products/kyoto-dst.json",  # SYM_H proxy (R10)
}
RAW_COLS = ["BZ_GSM", "flow_speed", "proton_density", "AE_INDEX", "SYM_H", "flux_2MeV"]
POLL_SECONDS = 300
STALE_MIN = 15                # data older than this -> flag stale
BUFFER_DAYS = 40
SSL_CTX = ssl.create_default_context()


def log(m):
    print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {m}", flush=True)


def utcnow():
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def _atomic_write_text(path, text):
    """Write text atomically (temp file in same dir + os.replace). A concurrent
    reader (the shared-cloud dashboard reads latest.json every rerun) therefore
    never observes a half-written file. os.replace is atomic on Windows + POSIX."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_parquet(df, path):
    """Atomic parquet write (temp file + os.replace) for the shared history buffer."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# Fetch + parse (each parser returns a 5-min DataFrame or None on any failure)
# --------------------------------------------------------------------------- #
def fetch_json(url, timeout=25):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SolarSentinel/1.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
            return json.load(r)
    except Exception as e:                            # noqa: BLE001 - any failure = None
        log(f"[fetch] FAIL {url.split('/')[-1]}: {type(e).__name__}: {e}")
        return None


def _to_naive_utc(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def _resample5(df):
    df = df.copy()
    df.index = df.index.floor("5min")
    return df.groupby(level=0).mean()


def parse_generic(data, mapping):
    """`mapping`: {source_key: our_col}. Returns 5-min DataFrame or None."""
    try:
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data)
        if "time_tag" not in df.columns or not set(mapping).issubset(df.columns):
            log(f"[parse] unexpected columns: {list(df.columns)[:8]}")
            return None
        out = pd.DataFrame(index=_to_naive_utc(df["time_tag"]))
        for src, col in mapping.items():
            out[col] = pd.to_numeric(df[src], errors="coerce").values
        out = out.dropna(how="all")
        return _resample5(out) if len(out) else None
    except Exception as e:                            # noqa: BLE001
        log(f"[parse] FAIL: {type(e).__name__}: {e}")
        return None


def parse_electrons(data):
    try:
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data)
        if not {"time_tag", "flux", "energy"}.issubset(df.columns):
            return None
        df = df[df["energy"] == ">=2 MeV"]
        if df.empty:
            log("[parse] no >=2 MeV rows in electron feed")
            return None
        out = pd.DataFrame({"flux_2MeV": pd.to_numeric(df["flux"], errors="coerce").values},
                           index=_to_naive_utc(df["time_tag"])).dropna()
        return _resample5(out) if len(out) else None
    except Exception as e:                            # noqa: BLE001
        log(f"[parse] electrons FAIL: {type(e).__name__}: {e}")
        return None


# --------------------------------------------------------------------------- #
def load_buffer():
    if BUFFER.exists():
        try:
            return pd.read_parquet(BUFFER)
        except Exception as e:                        # noqa: BLE001
            log(f"[buffer] unreadable ({e}); starting fresh")
    return None


def update_buffer(new):
    buf = load_buffer()
    combined = new if buf is None else new.combine_first(buf)
    combined = combined.sort_index()
    combined = combined.loc[combined.index.max() - pd.Timedelta(days=BUFFER_DAYS):]
    for c in RAW_COLS:                                # guarantee all raw cols exist
        if c not in combined.columns:
            combined[c] = np.nan
    try:
        _atomic_write_parquet(combined, BUFFER)
    except Exception as e:                            # noqa: BLE001
        log(f"[buffer] write failed: {e}")
    return combined


def write_stale(reason):
    """Never show old numbers as current: mark the last payload stale, or emit error."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    if LATEST.exists():
        try:
            prev = json.loads(LATEST.read_text())
            prev["status"] = "stale"
            prev["stale_reason"] = reason
            prev["checked_at"] = now
            _atomic_write_text(LATEST, json.dumps(prev, indent=2))
            log(f"[write] STALE (kept prior data, flagged): {reason}")
            return
        except Exception:                             # noqa: BLE001
            pass
    _atomic_write_text(LATEST, json.dumps({
        "status": "error", "checked_at": now, "error": reason,
        "hazard": {"level": "Yellow", "color": "#eab308", "title": "DATA UNAVAILABLE",
                   "message": f"Live feed unavailable: {reason}. No current forecast."},
    }, indent=2))
    log(f"[write] ERROR payload: {reason}")


# --------------------------------------------------------------------------- #
def poll():
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    raw = {k: fetch_json(u) for k, u in FEEDS.items()}
    mag = parse_generic(raw["mag"], {"bz_gsm": "BZ_GSM"})
    wind = parse_generic(raw["wind"], {"proton_speed": "flow_speed",
                                       "proton_density": "proton_density"})
    elec = parse_electrons(raw["electrons"])
    dst = parse_generic(raw["dst"], {"dst": "SYM_H"})   # Kyoto Dst -> SYM_H proxy (R10)
    feeds_ok = {"mag": mag is not None, "wind": wind is not None,
                "electrons": elec is not None, "dst": dst is not None}
    log(f"[poll] feeds_ok={feeds_ok}")

    # electron flux is the target + autoregressive anchor; without it, no forecast
    if elec is None:
        write_stale("GOES >2 MeV electron feed unavailable/malformed")
        return

    new = elec
    for part in (mag, wind):
        if part is not None:
            new = new.join(part, how="outer")
    new["AE_INDEX"] = np.nan          # no SWPC real-time source
    new["SYM_H"] = np.nan             # no SWPC real-time source

    combined = update_buffer(new)                    # buffer keeps RAW SWPC flux
    full = pd.date_range(combined.index.min(), combined.index.max(), freq="5min")
    grid = combined.reindex(full)
    grid.index.name = "Time"

    # SYM_H from real-time Kyoto Dst (hourly ring-current index; the same physical
    # quantity and nT scale as the 1-min SYM_H used in training). Forward-fill the
    # hourly value onto the 5-min grid within a 2 h tolerance -> populates the
    # SYM_H-derived features (mean/min/std/roc + short lags) that were previously
    # all-NaN live, recovering most of the offline +6h/+12h skill. AE_INDEX still
    # has no SWPC real-time source. See PROGRESS.md R10. Dst spans ~7 days, so the
    # 27-day Carrington + longest SYM_H lags still mature via the buffer.
    if dst is not None:
        grid["SYM_H"] = dst["SYM_H"].reindex(grid.index, method="ffill",
                                             tolerance=pd.Timedelta("120min"))

    # calibrate SWPC live flux -> NCEI (training) scale so flux_now matches training
    calib = load_calibration()
    raw_flux = grid["flux_2MeV"].copy()
    if calib:
        f = grid["flux_2MeV"]
        grid["flux_2MeV"] = np.clip(
            10 ** (calib["a"] * np.log10(f + 1) + calib["b"]) - 1, 0, None)

    grid = assemble_features(grid, load_context()["manifest"])   # SAME feature code

    flux_valid = grid["flux_2MeV"].dropna()
    if flux_valid.empty:
        write_stale("no valid >2 MeV flux samples after parsing")
        return
    now_t = flux_valid.index.max()
    data_age = (utcnow() - now_t).total_seconds() / 60.0

    missing = ["AE_INDEX"]                # no SWPC real-time source
    if dst is None:
        missing.append("SYM_H")           # Dst proxy unavailable this poll
    missing += [f for f, ok in feeds_ok.items() if not ok]
    if data_age > STALE_MIN:
        status = "stale"
    elif not (feeds_ok["mag"] and feeds_ok["wind"]):
        status = "degraded"
    else:
        status = "ok"

    extra = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "data_age_min": round(data_age, 1),
        "feeds_ok": feeds_ok,
        "missing_inputs": missing,
        "feed_note": ("SYM_H sourced from real-time Kyoto Dst (hourly ring-current "
                      "proxy for the 1-min SYM_H); AE_INDEX has no SWPC real-time "
                      "source (NaN live). +30 min unaffected (flux_now-dominated)."),
        "flux_calibration": {
            "applied": bool(calib),
            "raw_swpc_pfu": round(float(raw_flux.get(now_t, float("nan"))), 1),
            "calibrated_pfu": round(float(grid.at[now_t, "flux_2MeV"]), 1),
            "provisional": bool(calib.get("provisional")) if calib else None,
            "note": (("SWPC->NCEI(training) scale; " + calib["mechanism"])
                     if calib else "no calibration file; SWPC flux used raw"),
        },
    }
    payload = payload_from_frame(grid, now_t, "live_daemon",
                                 status=status, extra=extra)
    _atomic_write_text(LATEST, json.dumps(payload, indent=2))
    hz = payload["hazard"]
    log(f"[write] status={status} valid={now_t} age={data_age:.1f}min "
        f"flux={payload['telemetry']['flux_2MeV']['value']:.1f}pfu "
        f"hazard={hz['level']} -> {LATEST}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="poll every 5 min forever")
    args = ap.parse_args()
    if not args.loop:
        poll()
        return
    log(f"[daemon] loop mode, every {POLL_SECONDS}s")
    while True:
        try:
            poll()
        except Exception as e:                        # noqa: BLE001 - never die on one poll
            log(f"[poll] UNHANDLED {type(e).__name__}: {e}")
            write_stale(f"unhandled daemon error: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
