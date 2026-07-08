"""
SolarSentinel - Live flux calibration: SWPC real-time -> NCEI L2 (training) scale.

The models were trained on NCEI L2 GOES flux (AvgIntElectronFlux, max across the 5
telescopes); the live daemon reads SWPC's near-real-time integral-electron feed.
Those are the SAME physical channel but different products, so `flux_now` (which
dominates short-horizon predictions) can be biased live unless calibrated.

MECHANISM (expected, not mysterious): SWPC serves near-real-time PROVISIONAL data;
NCEI L2 is the REPROCESSED science-quality final archive. Real-time-vs-final
product drift is a well-understood phenomenon.

Fits log10(NCEI+1) = a*log10(SWPC+1) + b over an overlap window, restricted to
SWPC >= FIT_FLOOR pfu so the fit reflects the operationally-relevant range and is
NOT distorted by the low-flux max-of-telescopes floor artifact (where the naive
whole-range fit undershoots the storm range). Saves params to
data/live/flux_calibration.json for realtime_daemon.py to apply.

CAVEAT: fit from a single overlap week/event -> PROVISIONAL. Re-run when a future
storm provides a second independent overlap window; do not assume permanent.

Run:  python evaluate/fit_flux_calibration.py
"""

import json
import re
import ssl
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "live" / "flux_calibration.json"
SWPC_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json"
NCEI_BASE = ("https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
             "goes/goes19/l2/data/mpsh-l2-avg5m_science")
FIT_FLOOR = 200.0                 # pfu; exclude low-flux telescope-averaging floor
CTX = ssl.create_default_context()


def log(m):
    print(m, flush=True)


def fetch_swpc():
    with urllib.request.urlopen(SWPC_URL, timeout=30, context=CTX) as r:
        d = [x for x in json.load(r) if x.get("energy") == ">=2 MeV"]
    t = pd.to_datetime([x["time_tag"] for x in d], utc=True).tz_convert(None)
    s = pd.Series([float(x["flux"]) for x in d], index=t, name="swpc").sort_index()
    s.index = s.index.floor("5min")
    return s.groupby(level=0).mean()


def fetch_ncei_day(d):
    ds_str = f"{d.year}{d.month:02d}{d.day:02d}"
    folder = f"{NCEI_BASE}/{d.year}/{d.month:02d}/"
    try:
        with urllib.request.urlopen(folder, timeout=30, context=CTX) as r:
            html = r.read().decode()
        m = re.search(rf"sci_mpsh-l2-avg5m_g19_d{ds_str}_v[\d\-]+\.nc", html)
        if not m:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
        urllib.request.urlretrieve(folder + m.group(0), tmp)
        ds = xr.open_dataset(tmp)
        flux = ds["AvgIntElectronFlux"].max(dim="telescopes").values   # SAME as training
        s = pd.Series(flux, index=pd.to_datetime(ds["time"].values), name="ncei")
        ds.close()
        s[s < 0] = np.nan
        return s
    except Exception as e:                            # noqa: BLE001
        log(f"  NCEI {ds_str} fail: {e}")
        return None


def main():
    swpc = fetch_swpc()
    days = pd.date_range(swpc.index.min().normalize(), swpc.index.max().normalize(), freq="D")
    ncei = pd.concat([p for d in days if (p := fetch_ncei_day(d)) is not None]).sort_index()
    ncei.index = ncei.index.floor("5min")
    ncei = ncei.groupby(level=0).mean()

    df = pd.concat([swpc, ncei], axis=1).dropna()
    log(f"overlap: {len(df)} matched 5-min points, {df.index.min()} -> {df.index.max()}")
    log(f"flux range pfu: SWPC {df.swpc.min():.0f}-{df.swpc.max():.0f}  "
        f"NCEI {df.ncei.min():.0f}-{df.ncei.max():.0f}")

    fit = df[df.swpc >= FIT_FLOOR]
    xs, ys = np.log10(fit.swpc + 1), np.log10(fit.ncei + 1)
    a, b = (float(v) for v in np.polyfit(xs, ys, 1))
    r2 = float(1 - np.sum((ys - (a * xs + b)) ** 2) / np.sum((ys - ys.mean()) ** 2))

    def cal(s):
        return np.clip(10 ** (a * np.log10(s + 1) + b) - 1, 0, None)

    log(f"\nCALIBRATION (fit on SWPC>={FIT_FLOOR:.0f}): "
        f"log10(NCEI+1) = {a:.4f}*log10(SWPC+1) + {b:.4f}  R2={r2:.4f}  n={len(fit)}")
    log("before/after by flux bin (logRMSE vs NCEI; ratio = NCEI/prediction):")
    for lo, hi, name in [(200, 1000, "moderate"), (1000, 5000, "STORM/operational"),
                         (5000, 1e9, "extreme")]:
        g = df[(df.swpc >= lo) & (df.swpc < hi)]
        if not len(g):
            continue
        lrb = np.sqrt(np.mean((np.log10(g.swpc + 1) - np.log10(g.ncei + 1)) ** 2))
        lra = np.sqrt(np.mean((np.log10(cal(g.swpc) + 1) - np.log10(g.ncei + 1)) ** 2))
        log(f"  [{lo:>5}-{hi:>6}) {name:>17} n={len(g):>4}: "
            f"logRMSE {lrb:.4f} -> {lra:.4f}  | median ratio "
            f"{(g.ncei/g.swpc).median():.3f} -> {(g.ncei/cal(g.swpc)).median():.3f}")

    payload = {
        "form": "log10(ncei_flux + 1) = a*log10(swpc_flux + 1) + b",
        "apply": "ncei_equiv = clip(10**(a*log10(swpc+1) + b) - 1, 0, None)",
        "a": a, "b": b, "r2": r2, "n_fit": int(len(fit)),
        "fit_floor_pfu": FIT_FLOOR,
        "overlap_window": [str(df.index.min()), str(df.index.max())],
        "n_overlap": int(len(df)),
        "provisional": True,
        "caveat": ("Fit from a single overlap week/event; treat as PROVISIONAL and "
                   "re-validate when a future storm gives a second independent overlap."),
        "mechanism": ("SWPC = near-real-time provisional; NCEI L2 = reprocessed "
                      "science-quality final. Real-time-vs-final drift is expected."),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "satellite": "GOES-19 (both products)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
