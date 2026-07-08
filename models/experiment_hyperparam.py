"""
Step 6 (PROGRESS.md R11) - bounded hyperparameter search.

~12 configs per horizon around the current params (depth 7, lr 0.03, subsample 0.8,
mcw 5, lambda 1.0), same chronological split + early stopping, using the ADOPTED
per-horizon storm weights (STORM_WEIGHTS). Each config is scored on the TEST set
(log-RMSE, raw-RMSE, HSS@1000, storm log-RMSE, storm HSS) and compared to the
current canonical model. Adopt a tuned config for a horizon ONLY if it beats the
current model by a real margin (not rounding noise). Nothing is overwritten here;
adoption is applied separately after the decision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_common import (  # noqa: E402
    train_one, load_context, XGB_PARAMS, STORM_WEIGHTS, MODEL_DIR, STORM_SYMH,
)

HORIZONS = ["30min", "6h", "12h"]
ALERT_PFU = 1000.0
OUT = Path(__file__).resolve().parent.parent / "evaluate" / "hyperparam_search.json"


def configs():
    base = dict(XGB_PARAMS)

    def mod(**kw):
        c = dict(base); c.update(kw); return c

    return {
        "current": base,
        "depth5": mod(max_depth=5),
        "depth6": mod(max_depth=6),
        "depth8": mod(max_depth=8),
        "lr0.02": mod(learning_rate=0.02),
        "lr0.05": mod(learning_rate=0.05),
        "lam0.5": mod(reg_lambda=0.5),
        "lam2.0": mod(reg_lambda=2.0),
        "lam3.0": mod(reg_lambda=3.0),
        "mcw3": mod(min_child_weight=3),
        "mcw10": mod(min_child_weight=10),
        "d6_lr02_lam2": mod(max_depth=6, learning_rate=0.02, reg_lambda=2.0),
    }


def rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.any() else np.nan


def hss(obs, fc, thr=ALERT_PFU):
    o, f = obs >= thr, fc >= thr
    a = int((o & f).sum()); b = int((~o & f).sum())
    c = int((o & ~f).sum()); d = int((~o & ~f).sum())
    den = (a + c) * (c + d) + (a + b) * (b + d)
    return (2 * (a * d - b * c) / den) if den else np.nan


def summarize(p):
    storm = (p["SYM_H"] < STORM_SYMH).to_numpy()
    yl, xl = p["y_log_true"].to_numpy(), p["xgb_log"].to_numpy()
    ft, fx = p["flux_true"].to_numpy(), p["flux_xgb"].to_numpy()
    return {
        "log_rmse": rmse(xl, yl),
        "raw_rmse": rmse(fx, ft),
        "hss": hss(ft, fx),
        "storm_log_rmse": rmse(xl[storm], yl[storm]),
        "storm_hss": hss(ft[storm], fx[storm]),
    }


def main():
    ctx = load_context()
    cfgs = configs()
    results = {}
    for h in HORIZONS:
        results[h] = {}
        canon = pd.read_parquet(MODEL_DIR / f"test_predictions_{h}.parquet")
        results[h]["__canonical__"] = summarize(canon)
        for name, cfg in cfgs.items():
            m, preds = train_one(h, ctx=ctx, save=False, params=cfg,
                                  storm_weight=STORM_WEIGHTS[h], return_preds=True)
            s = summarize(preds)
            s["best_iter"] = int(m["best_iteration"])
            results[h][name] = s
            print(f"[{h:>5}] {name:>14}: logRMSE={s['log_rmse']:.4f} "
                  f"HSS={s['hss']:.3f} stormLogRMSE={s['storm_log_rmse']:.4f} "
                  f"stormHSS={s['storm_hss']:.3f} (iter={s['best_iter']})", flush=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))

    # ---- report + best-per-horizon (lower log-RMSE is the training objective) ----
    print("\n" + "=" * 90)
    print("HYPERPARAMETER SEARCH  (TEST set)  vs canonical current model")
    print("=" * 90)
    for h in HORIZONS:
        cur = results[h]["__canonical__"]
        print(f"\n----- {h}  (canonical: logRMSE={cur['log_rmse']:.4f} "
              f"HSS={cur['hss']:.3f} stormLogRMSE={cur['storm_log_rmse']:.4f}) -----")
        ranked = sorted(((n, r) for n, r in results[h].items() if n != "__canonical__"),
                        key=lambda kv: kv[1]["log_rmse"])
        for n, r in ranked:
            dlog = r["log_rmse"] - cur["log_rmse"]
            mark = "  <== best" if n == ranked[0][0] else ""
            print(f"  {n:>14}: logRMSE={r['log_rmse']:.4f} (d={dlog:+.4f}) "
                  f"HSS={r['hss']:.3f} stormLogRMSE={r['storm_log_rmse']:.4f} "
                  f"stormHSS={r['storm_hss']:.3f}{mark}")
    print(f"\n[hyperparam] wrote {OUT}")


if __name__ == "__main__":
    main()
