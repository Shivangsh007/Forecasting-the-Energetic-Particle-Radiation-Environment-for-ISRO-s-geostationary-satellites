"""
SolarSentinel - Section 6: evaluation.

Consumes the frozen Section-5 test-set predictions
(models/test_predictions_{h}.parquet) and the trained models
(models/xgb_{h}.json). No retraining.

Reports, per horizon (+30min / +6h / +12h):
  - RMSE + Pearson, log and raw scale, XGBoost vs persistence
  - storm-specific (SYM-H < -50 nT) vs quiet breakdown
  - HSS / POD / FAR at the 1000 pfu operational alert threshold
  - feature importance + a flux_now-dominance check
  - a focused +30 min raw-RMSE diagnosis (storm-spike vs broad, with HSS/POD/FAR)

Outputs to evaluate/:
  metrics_regression.csv, metrics_threshold.csv,
  feature_importance_{h}.csv, feature_importance_{h}.png,
  metrics_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FEAT_DIR = ROOT / "features"
OUT_DIR = ROOT / "evaluate"

HORIZONS = ["30min", "6h", "12h"]
STORM_SYMH = -50.0            # nT, geomagnetic storm threshold
ALERT_PFU = 1000.0           # NOAA >2 MeV electron event threshold


def log(m):
    print(m, flush=True)


def rmse(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2))) if m.any() else np.nan


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def contingency(obs_flux, fcst_flux, thr):
    """2x2 event table + HSS/POD/FAR at threshold `thr`."""
    obs = obs_flux >= thr
    fc = fcst_flux >= thr
    a = int(np.sum(obs & fc))          # hits
    b = int(np.sum(~obs & fc))         # false alarms
    c = int(np.sum(obs & ~fc))         # misses
    d = int(np.sum(~obs & ~fc))        # correct negatives
    pod = a / (a + c) if (a + c) else np.nan          # detection rate
    far = b / (a + b) if (a + b) else np.nan          # false-alarm RATIO
    denom = (a + c) * (c + d) + (a + b) * (b + d)     # Heidke
    hss = (2 * (a * d - b * c) / denom) if denom else np.nan
    return {"hits": a, "false_alarms": b, "misses": c, "corr_neg": d,
            "POD": pod, "FAR": far, "HSS": hss,
            "n_events_obs": a + c, "n_events_fcst": a + b}


def reg_metrics(d, mask=None):
    """RMSE/Pearson (log & raw) for xgb and persistence on rows selected by mask."""
    s = d if mask is None else d[mask]
    out = {"n": int(len(s))}
    for who, lg, rw in [("xgb", "xgb_log", "flux_xgb"),
                        ("persist", "persist_log", "flux_persist")]:
        out[f"{who}_rmse_log"] = rmse(s[lg].to_numpy(), s["y_log_true"].to_numpy())
        out[f"{who}_rmse_raw"] = rmse(s[rw].to_numpy(), s["flux_true"].to_numpy())
        out[f"{who}_r_log"] = pearson(s[lg].to_numpy(), s["y_log_true"].to_numpy())
        out[f"{who}_r_raw"] = pearson(s[rw].to_numpy(), s["flux_true"].to_numpy())
    return out


# --------------------------------------------------------------------------- #
# Feature importance + dominance check
# --------------------------------------------------------------------------- #
def feature_importance(hname, manifest):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"xgb_{hname}.json")
    booster = model.get_booster()
    gain = booster.get_score(importance_type="total_gain")   # {feat: total gain}

    cols = manifest["horizon_feature_columns"][hname]
    imp = pd.Series({c: gain.get(c, 0.0) for c in cols}).sort_values(ascending=False)
    total = imp.sum()
    share = (imp / total) if total > 0 else imp

    ar = set(manifest["feature_groups"]["flux_autoregressive"])
    coup = set(manifest["feature_groups"]["coupling_derived"])
    groups = {
        "flux_now": float(share.get("flux_now", 0.0)),
        "flux_history": float(sum(share[c] for c in cols if c in ar and c != "flux_now")),
        "coupling": float(sum(share[c] for c in cols if c in coup)),
        "all_drivers": float(sum(share[c] for c in cols if c not in ar)),
    }
    groups["all_flux_autoregressive"] = groups["flux_now"] + groups["flux_history"]
    top_drivers = [(c, float(share[c])) for c in share.index if c not in ar][:6]
    return imp, share, groups, top_drivers


def plot_importance(hname, share):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = share.head(20)[::-1]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top.index, 100 * top.values, color="#1f77b4")
        ax.set_xlabel("share of total gain (%)")
        ax.set_title(f"Feature importance - {hname} (top 20)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"feature_importance_{hname}.png", dpi=110)
        plt.close(fig)
    except Exception as e:                      # noqa: BLE001
        log(f"  [plot] skipped ({type(e).__name__}: {e})")


# --------------------------------------------------------------------------- #
def diagnose_30min(d):
    """Storm-vs-quiet decomposition of the +30 min raw-RMSE regression."""
    storm = d["SYM_H"] < STORM_SYMH
    quiet = ~storm
    ft = d["flux_true"].to_numpy()
    fx = d["flux_xgb"].to_numpy()
    fp = d["flux_persist"].to_numpy()
    sse_x = (fx - ft) ** 2
    sse_p = (fp - ft) ** 2

    def block(m):
        return {
            "n": int(m.sum()),
            "frac_rows": float(m.mean()),
            "xgb_rmse_raw": rmse(fx[m], ft[m]),
            "persist_rmse_raw": rmse(fp[m], ft[m]),
            "xgb_sse": float(sse_x[m].sum()),
            "persist_sse": float(sse_p[m].sum()),
            "xgb_bias_raw": float(np.mean(fx[m] - ft[m])),   # <0 = underpredict
        }
    res = {"storm": block(storm.to_numpy()), "quiet": block(quiet.to_numpy())}

    gap_total = float(sse_x.sum() - sse_p.sum())             # xgb worse if >0
    gap_storm = res["storm"]["xgb_sse"] - res["storm"]["persist_sse"]
    res["sse_gap_total"] = gap_total
    res["sse_gap_from_storm_pct"] = float(100 * gap_storm / gap_total) if gap_total else np.nan

    # concentration: how much of xgb's total raw error sits in the worst 1% rows
    order = np.argsort(-sse_x)
    k = max(1, len(sse_x) // 100)
    res["xgb_sse_worst1pct_share"] = float(sse_x[order[:k]].sum() / sse_x.sum())
    res["worst1pct_median_symh"] = float(np.median(d["SYM_H"].to_numpy()[order[:k]]))
    res["worst1pct_frac_storm"] = float(np.mean(
        d["SYM_H"].to_numpy()[order[:k]] < STORM_SYMH))
    return res


# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(exist_ok=True)
    manifest = json.loads((FEAT_DIR / "feature_manifest.json").read_text())

    preds = {h: pd.read_parquet(MODEL_DIR / f"test_predictions_{h}.parquet")
             for h in HORIZONS}

    reg_rows, thr_rows, summary = [], [], {}

    for h in HORIZONS:
        d = preds[h]
        storm = d["SYM_H"] < STORM_SYMH
        subsets = {"overall": None, "storm": storm.to_numpy(),
                   "quiet": (~storm).to_numpy()}
        summary[h] = {"regression": {}, "threshold": {}}

        for sname, mask in subsets.items():
            m = reg_metrics(d, mask)
            m.update({"horizon": h, "subset": sname})
            reg_rows.append(m)
            summary[h]["regression"][sname] = m

        for who, col in [("xgb", "flux_xgb"), ("persist", "flux_persist")]:
            ct = contingency(d["flux_true"].to_numpy(), d[col].to_numpy(), ALERT_PFU)
            ct.update({"horizon": h, "model": who, "threshold_pfu": ALERT_PFU})
            thr_rows.append(ct)
            summary[h]["threshold"][who] = ct

        imp, share, groups, top_drivers = feature_importance(h, manifest)
        share.rename("gain_share").to_csv(OUT_DIR / f"feature_importance_{h}.csv")
        plot_importance(h, share)
        summary[h]["importance_groups"] = groups
        summary[h]["top_drivers"] = top_drivers

    diag = diagnose_30min(preds["30min"])
    summary["diagnosis_30min"] = diag

    pd.DataFrame(reg_rows).to_csv(OUT_DIR / "metrics_regression.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(OUT_DIR / "metrics_threshold.csv", index=False)
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2,
                                                             default=float))

    _report(summary, diag)


def _report(summary, diag):
    log("\n" + "=" * 84)
    log("SECTION 6 CHECKPOINT  -  full metrics (TEST set)")
    log("=" * 84)

    log("\n(1) REGRESSION  -  RMSE & Pearson r,  XGBoost vs persistence")
    log(f"{'horizon':>7} {'subset':>7} {'n':>8} | "
        f"{'xgb_rmse_log':>12} {'per_rmse_log':>12} | "
        f"{'xgb_r_log':>10} | {'xgb_rmse_raw':>12} {'per_rmse_raw':>12} | {'xgb_r_raw':>9}")
    log("-" * 100)
    for h in HORIZONS:
        for s in ["overall", "storm", "quiet"]:
            m = summary[h]["regression"][s]
            log(f"{h:>7} {s:>7} {m['n']:>8,} | "
                f"{m['xgb_rmse_log']:>12.4f} {m['persist_rmse_log']:>12.4f} | "
                f"{m['xgb_r_log']:>10.3f} | "
                f"{m['xgb_rmse_raw']:>12.1f} {m['persist_rmse_raw']:>12.1f} | "
                f"{m['xgb_r_raw']:>9.3f}")

    log(f"\n(2) OPERATIONAL ALERT SKILL  @ {ALERT_PFU:.0f} pfu  (HSS / POD / FAR)")
    log(f"{'horizon':>7} {'model':>8} | {'obs_ev':>7} {'hits':>6} {'miss':>6} "
        f"{'F.alarm':>8} | {'HSS':>7} {'POD':>7} {'FAR':>7}")
    log("-" * 74)
    for h in HORIZONS:
        for who in ["xgb", "persist"]:
            c = summary[h]["threshold"][who]
            log(f"{h:>7} {who:>8} | {c['n_events_obs']:>7,} {c['hits']:>6,} "
                f"{c['misses']:>6,} {c['false_alarms']:>8,} | "
                f"{c['HSS']:>7.3f} {c['POD']:>7.3f} {c['FAR']:>7.3f}")

    log("\n(3) FEATURE IMPORTANCE  -  flux_now dominance check (share of total gain)")
    log(f"{'horizon':>7} | {'flux_now':>9} {'flux_hist':>10} {'ALL_flux':>9} "
        f"{'drivers':>8} {'coupling':>9} | top driver features")
    log("-" * 100)
    for h in HORIZONS:
        g = summary[h]["importance_groups"]
        td = ", ".join(f"{c}({100*s:.1f}%)" for c, s in summary[h]["top_drivers"][:3])
        log(f"{h:>7} | {100*g['flux_now']:>8.1f}% {100*g['flux_history']:>9.1f}% "
            f"{100*g['all_flux_autoregressive']:>8.1f}% {100*g['all_drivers']:>7.1f}% "
            f"{100*g['coupling']:>8.1f}% | {td}")

    log("\n(4) +30 MIN RAW-RMSE DIAGNOSIS  (storm vs quiet)")
    s, q = diag["storm"], diag["quiet"]
    log(f"  storm rows (SYM-H<-50): n={s['n']:,} ({100*s['frac_rows']:.2f}% of test)"
        f"  xgb_rmse_raw={s['xgb_rmse_raw']:.1f}  persist_rmse_raw={s['persist_rmse_raw']:.1f}"
        f"  xgb_bias={s['xgb_bias_raw']:+.1f}")
    log(f"  quiet rows            : n={q['n']:,} ({100*q['frac_rows']:.2f}% of test)"
        f"  xgb_rmse_raw={q['xgb_rmse_raw']:.1f}  persist_rmse_raw={q['persist_rmse_raw']:.1f}"
        f"  xgb_bias={q['xgb_bias_raw']:+.1f}")
    log(f"  raw-SSE gap (xgb - persist) total = {diag['sse_gap_total']:.3e}; "
        f"{diag['sse_gap_from_storm_pct']:.1f}% of it comes from storm rows")
    log(f"  worst 1% of rows hold {100*diag['xgb_sse_worst1pct_share']:.1f}% of xgb raw error;"
        f" {100*diag['worst1pct_frac_storm']:.0f}% of those are storm rows "
        f"(median SYM-H={diag['worst1pct_median_symh']:.0f} nT)")


if __name__ == "__main__":
    main()
