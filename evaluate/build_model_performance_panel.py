"""
SolarSentinel - assemble the Model Performance panel data (dashboard reads this).

Reads ONLY real saved artifacts (metrics_section5.json, metrics_summary.json,
grasp_metrics.json, test_predictions_{h}.parquet). Computes MAE + R2 (raw scale)
fresh from the frozen test predictions (cheap, no retrain). Writes
evaluate/model_performance_panel.json so the UI never recomputes or hardcodes -
re-run this after any retrain and the panel stays true.

Honesty: per-horizon (never blended). The +30 min operational-skill caveat is
derived from the ACTUAL HSS gap vs persistence, not assumed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
EVAL = ROOT / "evaluate"
OUT = EVAL / "model_performance_panel.json"
HORIZONS = ["30min", "6h", "12h"]
HSS_TIE_EPS = 0.02          # below this HSS gain = "no operational improvement"


def main():
    m5 = {r["horizon"]: r for r in json.loads((MODELS / "metrics_section5.json").read_text())}
    ms = json.loads((EVAL / "metrics_summary.json").read_text())
    grasp = json.loads((EVAL / "grasp_metrics.json").read_text())

    per_horizon = {}
    for h in HORIZONS:
        pred = pd.read_parquet(MODELS / f"test_predictions_{h}.parquet")
        yt = pred["flux_true"].to_numpy(dtype="float64")
        yp = pred["flux_xgb"].to_numpy(dtype="float64")
        mae = float(np.mean(np.abs(yp - yt)))                        # derived, raw pfu
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot)                              # derived, raw scale

        reg = ms[h]["regression"]["overall"]
        thr_x = ms[h]["threshold"]["xgb"]
        thr_p = ms[h]["threshold"]["persist"]
        hss_gain = thr_x["HSS"] - thr_p["HSS"]

        note = None
        if hss_gain < HSS_TIE_EPS:
            note = ("No significant improvement over persistence at this horizon: "
                    f"operational HSS {thr_x['HSS']:.3f} ties persistence "
                    f"{thr_p['HSS']:.3f} (the +{m5[h]['improvement_log_pct']:.1f}% is "
                    "log-scale RMSE only, not an alerting gain).")

        per_horizon[h] = {
            "mae_pfu": round(mae, 1),
            "rmse_pfu": round(reg["xgb_rmse_raw"], 1),
            "r2_raw": round(r2, 3),
            "pearson_raw": round(reg["xgb_r_raw"], 3),
            "skill_vs_persistence_pct": round(m5[h]["improvement_log_pct"], 1),
            "pod": round(thr_x["POD"], 3),
            "far": round(thr_x["FAR"], 3),
            "hss": round(thr_x["HSS"], 3),
            "hss_persist": round(thr_p["HSS"], 3),
            "hss_gain": round(hss_gain, 3),
            "n_test": int(m5[h]["n_test"]),
            "note": note,
        }

    # Training/validation/test calendar periods, derived from the real feature
    # matrix + the same split fractions as train_common (70/15/15 chronological),
    # so the dashboard can state the training range without hardcoding it.
    # Builder runs offline where features_master.parquet exists; the values are
    # embedded in the JSON for the deployed app.
    training = None
    feat = ROOT / "features" / "features_master.parquet"
    if feat.exists():
        idx = pd.read_parquet(feat, columns=["log_flux"]).index
        n = len(idx)
        t_train_end, t_val_end = idx[int(n * 0.70)], idx[int(n * 0.85)]
        training = {
            "data_start": str(idx.min())[:16],
            "train_end": str(t_train_end)[:16],
            "val_end": str(t_val_end)[:16],
            "data_end": str(idx.max())[:16],
            "split": "chronological 70/15/15 (never shuffled)",
            "source": "GOES >2 MeV integral electrons (NCEI L2) + OMNI HRO2, 5-min cadence",
        }

    g17 = grasp["2017_out_of_time"]
    context = {
        "model_revision": ("post-Stage-A R7-R11 (83 features: +short-flux-trend, "
                           "+rate-of-change/pressure-jump; +12h storm-weighted; "
                           "+30min/+6h depth-tuned)"),
        "last_trained_utc": datetime.fromtimestamp(
            (MODELS / "xgb_6h.json").stat().st_mtime, tz=timezone.utc).isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training": training,
        "grasp_cross_longitude_2017_out_of_time": {
            "description": ("Independent ISRO GSAT-19/GRASP data, Indian longitude, "
                            "Jul-Dec 2017 - OUTSIDE the training time window."),
            "6h": {"pearson_r_log": round(g17["6h"]["overall"]["xgb_r_log"], 3),
                   "hss_1000pfu": round(g17["6h"]["hss_1000pfu"]["xgb"]["HSS"], 3),
                   "hss_persist": round(g17["6h"]["hss_1000pfu"]["persist"]["HSS"], 3)},
            "12h": {"pearson_r_log": round(g17["12h"]["overall"]["xgb_r_log"], 3),
                    "hss_1000pfu": round(g17["12h"]["hss_1000pfu"]["xgb"]["HSS"], 3),
                    "hss_persist": round(g17["12h"]["hss_1000pfu"]["persist"]["HSS"], 3)},
        },
    }

    panel = {"per_horizon": per_horizon, "context": context,
             "threshold_pfu": 1000, "scale_note": "MAE/RMSE in pfu (raw); R2/Pearson raw scale."}
    OUT.write_text(json.dumps(panel, indent=2))
    print(json.dumps(panel, indent=2))
    print(f"\n[panel] wrote {OUT}")


if __name__ == "__main__":
    main()
