"""
SolarSentinel - Section 6: GRASP / GSAT-19 cross-longitude validation (REAL DATA).

Rewritten for the actual GRASP delivery format (tab-separated daily .txt + .xml;
the brief's FITS/PDS4 assumption was wrong). Run grasp_consolidate.py first.

Two windows, reported side by side (do NOT collapse into one):
  * 2017_out_of_time : Jul-Dec 2017 - OUTSIDE the model's training time-range
      (2017 OMNI from omni_2017.csv was never seen in training) -> the strongest,
      genuinely out-of-sample cross-longitude test.
  * 2018_in_train    : Jan-Aug 2018 - inside the train time-range (OMNI drivers
      seen in training) -> tests longitude generalisation only.

Both use the SAME assemble_features(); GRASP flux = target + autoregressive anchor;
global OMNI drivers (Bz, speed, density, AE, SYM-H). Metrics = Section 6:
RMSE + Pearson (log & raw), XGB vs persistence, overall/storm/quiet
(storm = GRASP Electron_Activity_level == High).

CAVEATS (kept attached):
  [A] GRASP energy channel UNCONFIRMED as >2 MeV (no instrument doc; ~2x median
      scale offset) -> correlation (scale-invariant) is the headline.
  [B] AE_INDEX carries 99999 fills in the 2018+ training data (unreplaced there);
      2017 OMNI has no AE gaps. Processed by the identical puller method for
      consistency; SYM-H (the important driver) is clean in both.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "dashboard"))
from build_features import assemble_features        # noqa: E402  same feature code
from snapshot import load_context                    # noqa: E402  models + manifest

GRASP_MASTER = ROOT / "data" / "grasp" / "grasp_master.parquet"
GRASP_META = ROOT / "data" / "grasp" / "grasp_metadata.csv"
MASTER_CSV = ROOT / "training_data_master_2018_2025.csv"     # OMNI drivers 2018+
OMNI_2017 = ROOT / "omni_2017.csv"                           # OMNI drivers 2017 (unseen)
OUT_JSON = ROOT / "evaluate" / "grasp_metrics.json"

OMNI_COLS = ["BZ_GSM", "flow_speed", "proton_density", "AE_INDEX", "SYM_H"]
ALERT_PFU = 1000.0

WINDOWS = {
    "2017_out_of_time": ("2017-07-01", "2018-01-01"),   # genuinely unseen
    "2018_in_train": ("2018-01-01", "2019-01-01"),      # longitude-only
}


def log(m):
    print(m, flush=True)


def rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.any() else np.nan


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan


def skill(obs, fc, thr):
    o, f = obs >= thr, fc >= thr
    a = int((o & f).sum()); b = int((~o & f).sum())
    c = int((o & ~f).sum()); d = int((~o & ~f).sum())
    pod = a / (a + c) if a + c else np.nan
    far = b / (a + b) if a + b else np.nan
    den = (a + c) * (c + d) + (a + b) * (b + d)
    hss = 2 * (a * d - b * c) / den if den else np.nan
    return {"hits": a, "false_alarms": b, "misses": c, "n_events_obs": a + c,
            "POD": pod, "FAR": far, "HSS": hss}


def build_grid(ctx):
    if not GRASP_MASTER.exists():
        raise SystemExit("Missing grasp_master.parquet - run grasp_consolidate.py first.")
    grasp = pd.read_parquet(GRASP_MASTER).sort_index()
    full = pd.date_range(grasp.index.min(), grasp.index.max(), freq="5min")
    g = pd.DataFrame(index=full)
    g.index.name = "Time"
    g["flux_2MeV"] = grasp["electron_flux"].reindex(full)      # GRASP = target + anchor

    # OMNI drivers: 2018+ from master (seen in training), 2017 from omni_2017 (unseen)
    om = pd.read_csv(MASTER_CSV, parse_dates=["Time"],
                     usecols=["Time"] + OMNI_COLS).set_index("Time")
    if OMNI_2017.exists():
        o17 = pd.read_csv(OMNI_2017, parse_dates=["Time"]).set_index("Time")
        if o17.index.tz is not None:
            o17.index = o17.index.tz_localize(None)
        om = pd.concat([o17[OMNI_COLS], om[OMNI_COLS]])
    om = om[~om.index.duplicated(keep="last")].sort_index()
    for c in OMNI_COLS:
        g[c] = om[c].reindex(full)

    g = assemble_features(g, ctx["manifest"])                  # SAME code path

    meta = pd.read_csv(GRASP_META, parse_dates=["date"])
    high = set(meta.loc[meta["electron_activity"] == "High", "date"].dt.normalize())
    storm = pd.Series(g.index.normalize().isin(high), index=g.index)
    return g, storm


def eval_subset(y, px, pl, mask, tag):
    yv, xv, lv = y[mask], px[mask], pl[mask]
    fx = np.clip(10**xv - 1, 0, None); fp = np.clip(10**lv - 1, 0, None); ft = 10**yv - 1
    return {
        "subset": tag, "n": int(mask.sum()),
        "xgb_rmse_log": rmse(xv, yv), "persist_rmse_log": rmse(lv, yv),
        "xgb_r_log": pearson(xv, yv), "persist_r_log": pearson(lv, yv),
        "xgb_rmse_raw": rmse(fx, ft), "persist_rmse_raw": rmse(fp, ft),
        "xgb_r_raw": pearson(fx, ft),
        "xgb_bias_raw": float(np.nanmean(fx - ft)),
        "obs_mean_raw": float(np.nanmean(ft)), "pred_mean_raw": float(np.nanmean(fx)),
    }


def evaluate_window(g, storm, ctx, lo, hi):
    in_win = (g.index >= pd.Timestamp(lo)) & (g.index < pd.Timestamp(hi))
    out = {}
    for hname, hsteps in ctx["manifest"]["horizons_steps"].items():
        cols = ctx["manifest"]["horizon_feature_columns"][hname]
        y = g["log_flux"].shift(-hsteps)
        persist = g["log_flux"]
        valid = in_win & y.notna() & g["flux_2MeV"].notna()
        px = pd.Series(np.nan, index=g.index)
        px.loc[valid] = ctx["models"][hname].predict(g.loc[valid, cols])
        yv, pxv, plv = y.to_numpy(), px.to_numpy(), persist.to_numpy()
        vmask = valid.to_numpy()
        st = (valid & storm).to_numpy()
        qt = (valid & ~storm).to_numpy()
        out[hname] = {
            "n_days": int(g.index[vmask].normalize().nunique()),
            "overall": eval_subset(yv, pxv, plv, vmask, "overall"),
            "storm": eval_subset(yv, pxv, plv, st, "storm(High)"),
            "quiet": eval_subset(yv, pxv, plv, qt, "quiet"),
            "hss_1000pfu": {
                "xgb": skill(np.clip(10**yv[vmask]-1, 0, None),
                             np.clip(10**pxv[vmask]-1, 0, None), ALERT_PFU),
                "persist": skill(np.clip(10**yv[vmask]-1, 0, None),
                                 np.clip(10**plv[vmask]-1, 0, None), ALERT_PFU),
            },
        }
    return out


def report(all_results):
    log("\n" + "=" * 96)
    log("GRASP / GSAT-19 CROSS-LONGITUDE VALIDATION  (real ISRO data)")
    log("=" * 96)
    log("CAVEAT [A]: >2 MeV channel-equivalence ASSUMED (unconfirmed; ~2x scale offset -> "
        "raw RMSE\n            confounded, correlation is the headline).\n")

    for win, results in all_results.items():
        tag = ("OUT-OF-TIME (genuinely unseen)" if "2017" in win
               else "IN-TRAINING-TIME (longitude generalisation only)")
        d0 = results["30min"]["n_days"]
        log(f"----- WINDOW: {win}  [{tag}]  ({d0} days) -----")
        log(f"{'horizon':>7} {'subset':>11} {'n':>7} | {'xgb_r_log':>9} {'per_r_log':>9} | "
            f"{'xgb_rmse_log':>12} {'per_rmse_log':>12} | {'xgb HSS@1k':>10} {'per HSS':>8}")
        for h, r in results.items():
            hss = r["hss_1000pfu"]
            for s in ["overall", "storm", "quiet"]:
                m = r[s]
                extra = (f"{hss['xgb']['HSS']:>10.3f} {hss['persist']['HSS']:>8.3f}"
                         if s == "overall" else f"{'':>10} {'':>8}")
                log(f"{h:>7} {m['subset']:>11} {m['n']:>7,} | "
                    f"{m['xgb_r_log']:>9.3f} {m['persist_r_log']:>9.3f} | "
                    f"{m['xgb_rmse_log']:>12.4f} {m['persist_rmse_log']:>12.4f} | {extra}")
        log("")

    # focused side-by-side: correlation (r_log), XGB vs persist, both windows
    log("=" * 96)
    log("SIDE-BY-SIDE  -  Pearson r_log (XGB / persist),  the headline metric")
    log("=" * 96)
    w17 = all_results.get("2017_out_of_time", {})
    w18 = all_results.get("2018_in_train", {})
    log(f"{'horizon':>7} {'subset':>11} | {'2017 OUT-OF-TIME':>22} | {'2018 in-train':>22}")
    log("-" * 74)
    for h in w17:
        for s in ["overall", "storm", "quiet"]:
            a = w17[h][s]; b = w18[h][s]
            log(f"{h:>7} {a['subset']:>11} | "
                f"{a['xgb_r_log']:>9.3f} /{a['persist_r_log']:>8.3f}   | "
                f"{b['xgb_r_log']:>9.3f} /{b['persist_r_log']:>8.3f}")


def main():
    ctx = load_context()
    g, storm = build_grid(ctx)
    all_results = {win: evaluate_window(g, storm, ctx, lo, hi)
                   for win, (lo, hi) in WINDOWS.items()}
    report(all_results)
    OUT_JSON.write_text(json.dumps(all_results, indent=2, default=float))
    log(f"\n[grasp] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
