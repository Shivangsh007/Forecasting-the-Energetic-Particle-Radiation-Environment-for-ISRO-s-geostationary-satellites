"""
SolarSentinel - GRASP<->GOES longitude scale calibration (#2).

Characterises the scale relationship between GRASP (GSAT-19, Indian sector, 48E
slot — confirmed empirically by the +8 h GRASP-GOES lag-correlation peak, see
PROGRESS R5) and GOES (US longitude, the training target) >2 MeV electron flux, so the
cross-longitude validation can be reported in a common (GOES/training) scale
rather than only via scale-invariant correlation (addresses GRASP caveat [A],
the ~2x median offset).

Standard: fit on one time subset, VALIDATE on a held-out subset, report held-out
numbers. Overlap where BOTH products exist at the same timestamps = 2018
(GRASP 2018 + GOES from the training master). Fit on early 2018, hold out late 2018.

Direction: log10(GOES+1) = a*log10(GRASP+1) + b  -> maps GRASP obs to GOES-equiv.

CAVEATS (attached): GRASP (48E) and GOES (-75.2E) at the same UTC are ~8.2 h apart
in LOCAL time, so this relationship conflates instrument/scale AND local-time /
longitude diurnal differences -> residual scatter is partly physical, not error.
Also a single ~8-month overlap; treat as provisional.

Run:  python evaluate/fit_grasp_goes_calibration.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRASP = ROOT / "data" / "grasp" / "grasp_master.parquet"
GRASP_META = ROOT / "data" / "grasp" / "grasp_metadata.csv"
MASTER = ROOT / "training_data_master_2018_2025.csv"
OUT = ROOT / "data" / "grasp" / "grasp_goes_calibration.json"
SPLIT = "2018-06-01"                                # fit < SPLIT ; validate >= SPLIT


def log(m):
    print(m, flush=True)


def rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    grasp = pd.read_parquet(GRASP)["electron_flux"].rename("grasp")
    goes = pd.read_csv(MASTER, usecols=["Time", "flux_2MeV"],
                       parse_dates=["Time"]).set_index("Time")["flux_2MeV"].rename("goes")
    df = pd.concat([grasp, goes], axis=1).dropna()
    df = df[(df.index >= "2018-01-01") & (df.index < "2019-01-01")]
    log(f"GRASP-GOES 2018 overlap: {len(df):,} matched 5-min points, "
        f"{df.index.min()} -> {df.index.max()}")
    log(f"raw medians: GRASP {df.grasp.median():.0f}  GOES {df.goes.median():.0f}  "
        f"(GOES/GRASP median ratio {(df.goes/df.grasp).median():.2f})")

    fit = df[df.index < SPLIT]
    hold = df[df.index >= SPLIT]
    log(f"fit (<{SPLIT}): {len(fit):,} pts | held-out (>={SPLIT}): {len(hold):,} pts")

    xs, ys = np.log10(fit.grasp + 1), np.log10(fit.goes + 1)
    a, b = (float(v) for v in np.polyfit(xs, ys, 1))
    r2_fit = float(1 - np.sum((ys - (a * xs + b)) ** 2) / np.sum((ys - ys.mean()) ** 2))
    log(f"\nFIT  log10(GOES+1) = {a:.4f}*log10(GRASP+1) + {b:.4f}   R2(fit)={r2_fit:.4f}")

    def to_goes(g):
        return np.clip(10 ** (a * np.log10(g + 1) + b) - 1, 0, None)

    # ---- held-out evaluation (the number that matters) ----
    hg = np.log10(hold.grasp + 1).to_numpy()
    hgo = np.log10(hold.goes + 1).to_numpy()
    cal = a * hg + b                                # calibrated GRASP -> GOES-equiv (log)

    def block(mask, name):
        gg, gc, gt = hg[mask], cal[mask], hgo[mask]
        r_before = rmse(gg, gt)                     # raw GRASP-log vs GOES-log
        r_after = rmse(gc, gt)                      # calibrated vs GOES-log
        ratio_before = np.median((10 ** gt - 1) / (10 ** gg - 1 + 1e-9))
        ratio_after = np.median((10 ** gt - 1) / (10 ** gc - 1 + 1e-9))
        return {"n": int(mask.sum()), "pearson_log": pear(gg, gt),
                "logRMSE_before": r_before, "logRMSE_after": r_after,
                "median_GOES/GRASP_before": float(ratio_before),
                "median_GOES/calib_after": float(ratio_after)}

    storm_dates = set(pd.read_csv(GRASP_META, parse_dates=["date"])
                      .query("electron_activity == 'High'")["date"].dt.normalize())
    storm = hold.index.normalize().isin(storm_dates)
    res = {"overall": block(np.ones(len(hold), bool), "overall"),
           "storm_High": block(storm, "storm"),
           "quiet": block(~storm, "quiet")}

    log("\nHELD-OUT (>= {}) results:".format(SPLIT))
    log(f"{'subset':>11} {'n':>7} | {'pearson_log':>11} | "
        f"{'logRMSE before->after':>22} | {'GOES/GRASP ratio before->after':>30}")
    for k, m in res.items():
        log(f"{k:>11} {m['n']:>7,} | {m['pearson_log']:>11.3f} | "
            f"{m['logRMSE_before']:>9.4f} -> {m['logRMSE_after']:<9.4f} | "
            f"{m['median_GOES/GRASP_before']:>13.2f} -> {m['median_GOES/calib_after']:<13.2f}")

    # averaging-scale check: separate the scale offset from magnetic-local-time phase
    lg = lambda s: np.log10(s + 1)                                    # noqa: E731
    ms = {}
    for freq, name in [("5min", "5min"), ("1h", "hourly"), ("D", "daily")]:
        if freq == "5min":
            aa, bb = lg(hold.grasp), lg(hold.goes)
        else:
            rr = hold[["grasp", "goes"]].resample(freq).mean().dropna()
            aa, bb = lg(rr.grasp), lg(rr.goes)
        ms[name] = {"n": int(len(aa)), "pearson_log": float(np.corrcoef(aa, bb)[0, 1])}
    log("\naveraging-scale held-out pearson_log (isolates scale from MLT phase): "
        + ", ".join(f"{k} {v['pearson_log']:.3f}(n={v['n']})" for k, v in ms.items()))

    payload = {
        "form": "log10(goes_flux + 1) = a*log10(grasp_flux + 1) + b",
        "averaging_scale_pearson_holdout": ms,
        "apply": "goes_equiv = clip(10**(a*log10(grasp+1) + b) - 1, 0, None)",
        "a": a, "b": b, "r2_fit": r2_fit,
        "fit_window": ["2018-01-01", SPLIT], "holdout_window": [SPLIT, "2019-01-01"],
        "n_fit": int(len(fit)), "n_holdout": int(len(hold)),
        "holdout_results": res,
        "provisional": True,
        "caveat": ("Single ~8-month 2018 overlap; GRASP(India)/GOES(US) at same UTC differ "
                   "in LOCAL time (~8.2 h; GSAT-19 at 48E) so residual scatter is partly physical "
                   "longitude/diurnal difference, not instrument error. Provisional."),
        "purpose": ("Report cross-longitude validation in GOES/training scale + characterise "
                    "the Indian-longitude flux offset; NOT used in the live daemon."),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
