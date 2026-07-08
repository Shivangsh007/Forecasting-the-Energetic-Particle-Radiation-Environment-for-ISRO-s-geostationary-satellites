"""
Step 4 (PROGRESS.md R9) - storm-focused sample-weighting experiment.

Retrains each horizon upweighting geomagnetic-storm rows (SYM_H < -50 nT) by a
few candidate factors and compares, against the UNWEIGHTED model (canonical Step
3 predictions), BOTH overall metrics AND storm-subset metrics -- so the genuine
tradeoff (does storm skill improve at the cost of quiet-period skill?) is on the
table, not just the flattering side.

Control  (weight 1.0) : models/test_predictions_{h}.parquet  (already on disk)
Variants (weight k)   : trained here, saved to *_sw{k}.parquet (no clobber)

Does NOT overwrite the canonical models. Adoption decision is made from the
printed tradeoff and, if adopted, applied separately.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import train_one, load_context, MODEL_DIR, STORM_SYMH  # noqa: E402

HORIZONS = ["30min", "6h", "12h"]
WEIGHTS = [3.0, 5.0]              # candidates; 1.0 control read from canonical files
ALERT_PFU = 1000.0
OUT = Path(__file__).resolve().parent.parent / "evaluate" / "storm_weight_experiment.json"


def rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.any() else np.nan


def hss(obs_flux, fc_flux, thr=ALERT_PFU):
    o, f = obs_flux >= thr, fc_flux >= thr
    a = int((o & f).sum()); b = int((~o & f).sum())
    c = int((o & ~f).sum()); d = int((~o & ~f).sum())
    den = (a + c) * (c + d) + (a + b) * (b + d)
    return (2 * (a * d - b * c) / den) if den else np.nan


def metrics_from_preds(df):
    storm = (df["SYM_H"] < STORM_SYMH).to_numpy()
    quiet = ~storm
    out = {}
    for tag, m in [("overall", np.ones(len(df), bool)),
                   ("storm", storm), ("quiet", quiet)]:
        s = df[m]
        out[tag] = {
            "n": int(m.sum()),
            "xgb_rmse_log": rmse(s["xgb_log"].to_numpy(), s["y_log_true"].to_numpy()),
            "persist_rmse_log": rmse(s["persist_log"].to_numpy(), s["y_log_true"].to_numpy()),
            "xgb_rmse_raw": rmse(s["flux_xgb"].to_numpy(), s["flux_true"].to_numpy()),
            "persist_rmse_raw": rmse(s["flux_persist"].to_numpy(), s["flux_true"].to_numpy()),
            "xgb_hss": hss(s["flux_true"].to_numpy(), s["flux_xgb"].to_numpy()),
            "persist_hss": hss(s["flux_true"].to_numpy(), s["flux_persist"].to_numpy()),
        }
        p, x = out[tag]["persist_rmse_log"], out[tag]["xgb_rmse_log"]
        out[tag]["skill_pct"] = round(100 * (p - x) / p, 2) if p else np.nan
    return out


def main():
    ctx = load_context()
    results = {}

    # ---- train the upweighted variants ----
    for w in WEIGHTS:
        suffix = f"_sw{w}"
        for h in HORIZONS:
            print(f"[train] horizon={h} storm_weight={w}")
            train_one(h, ctx=ctx, save=True, storm_weight=w, save_suffix=suffix)

    # ---- gather metrics: control (1.0) + variants ----
    for h in HORIZONS:
        results[h] = {}
        ctrl = pd.read_parquet(MODEL_DIR / f"test_predictions_{h}.parquet")
        results[h]["1.0"] = metrics_from_preds(ctrl)
        for w in WEIGHTS:
            p = pd.read_parquet(MODEL_DIR / f"test_predictions_{h}_sw{w}.parquet")
            results[h][str(w)] = metrics_from_preds(p)

    OUT.write_text(json.dumps(results, indent=2, default=float))

    # ---- report ----
    print("\n" + "=" * 92)
    print("STORM SAMPLE-WEIGHTING TRADEOFF  (TEST set)  -  storm = SYM_H < -50 nT")
    print("=" * 92)
    for h in HORIZONS:
        print(f"\n----- {h} -----")
        print(f"{'weight':>7} | {'OVERALL':>26} | {'STORM subset':>32} | {'QUIET subset':>20}")
        print(f"{'':>7} | {'logRMSE  skill%   HSS':>26} | "
              f"{'logRMSE  rawRMSE   HSS':>32} | {'logRMSE   HSS':>20}")
        print("-" * 92)
        for wkey in ["1.0"] + [str(w) for w in WEIGHTS]:
            o = results[h][wkey]["overall"]
            s = results[h][wkey]["storm"]
            q = results[h][wkey]["quiet"]
            tag = "  (ctrl)" if wkey == "1.0" else ""
            print(f"{wkey:>7} | {o['xgb_rmse_log']:>7.4f} {o['skill_pct']:>6.1f}% "
                  f"{o['xgb_hss']:>7.3f} | "
                  f"{s['xgb_rmse_log']:>7.4f} {s['xgb_rmse_raw']:>8.1f} {s['xgb_hss']:>7.3f} | "
                  f"{q['xgb_rmse_log']:>7.4f} {q['xgb_hss']:>7.3f}{tag}")
        # persistence storm reference (weight-independent)
        sp = results[h]["1.0"]["storm"]
        print(f"{'persist':>7} | {'':>26} | "
              f"{sp['persist_rmse_log']:>7.4f} {sp['persist_rmse_raw']:>8.1f} "
              f"{sp['persist_hss']:>7.3f} |")
    print(f"\n[experiment] wrote {OUT}")


if __name__ == "__main__":
    main()
