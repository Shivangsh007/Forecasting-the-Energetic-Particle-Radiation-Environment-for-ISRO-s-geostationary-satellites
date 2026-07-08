"""
SolarSentinel - Section 5: shared modelling logic.

One independent XGBoost regressor per horizon (+30min / +6h / +12h) on the
log10(flux+1) target, plus the mandatory persistence baseline
("flux(t+h) = flux(t)"). Chronological 70/15/15 split, no shuffling.

The thin per-horizon entry points (train_30min.py / train_6h.py / train_12h.py)
and the consolidated runner (train_all.py) all call into here so there is a
single training/evaluation code path.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
FEAT_PATH = ROOT / "features" / "features_master.parquet"
MANIFEST_PATH = ROOT / "features" / "feature_manifest.json"
MODEL_DIR = ROOT / "models"

# Chronological split fractions (test = 1 - TRAIN - VAL).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# XGBoost hyper-parameters (shared across horizons; early stopping on val RMSE).
XGB_PARAMS = dict(
    n_estimators=3000,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=-1,
    random_state=42,
)
EARLY_STOPPING_ROUNDS = 60

# Geomagnetic storm threshold (nT) for optional storm-focused sample weighting.
STORM_SYMH = -50.0

# Per-horizon storm sample weight (Step 4, PROGRESS.md R9). Upweighting SYM_H<-50
# rows helps ONLY +12h (win-win: better storm AND overall skill, quiet not degraded);
# it is neutral/harmful at +30min/+6h, so those stay unweighted. 1.0 = off.
STORM_WEIGHTS = {"30min": 1.0, "6h": 1.0, "12h": 3.0}

# Per-horizon hyperparameter overrides on top of XGB_PARAMS (Step 6, PROGRESS.md R11).
# The bounded grid found shallower trees (depth 5) generalise slightly better at the
# two shorter near-persistence horizons (best log-RMSE + best HSS/storm-HSS, no
# downside); +12h keeps depth 7 because every shallower/tuned config there traded
# away the Step-4 storm-HSS advantage. Empty dict = XGB_PARAMS unchanged.
PER_HORIZON_PARAMS = {"30min": {"max_depth": 5}, "6h": {"max_depth": 5}, "12h": {}}


# --------------------------------------------------------------------------- #
def log(msg):
    print(msg, flush=True)


def inv_log(x):
    """Invert the target transform log10(flux+1) -> raw pfu (clip >= 0)."""
    return np.clip(np.power(10.0, x) - 1.0, 0.0, None)


def rmse(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2)))


def load_context():
    """Load feature matrix + manifest and compute shared split boundaries."""
    df = pd.read_parquet(FEAT_PATH).sort_index()
    manifest = json.loads(MANIFEST_PATH.read_text())

    idx = df.index
    n = len(idx)
    t_train_end = idx[int(n * TRAIN_FRAC)]
    t_val_end = idx[int(n * (TRAIN_FRAC + VAL_FRAC))]
    return {
        "df": df,
        "manifest": manifest,
        "t_train_end": t_train_end,
        "t_val_end": t_val_end,
        "n_labelled": n,
    }


def train_one(hname, ctx=None, save=True, storm_weight=1.0, save_suffix="",
              params=None, return_preds=False):
    """Train + evaluate a single horizon. Returns a metrics dict.

    storm_weight : if != 1.0, upweight training/val rows with SYM_H < STORM_SYMH
                   by this factor (Step 4 storm-focused experiment). Default 1.0
                   reproduces the standard unweighted model exactly.
    save_suffix  : appended to the saved model / predictions filenames so an
                   experiment can persist variants without clobbering the
                   canonical xgb_{h}.json / test_predictions_{h}.parquet.
    params       : XGBoost param dict; defaults to the shared XGB_PARAMS (Step 6
                   hyperparameter search passes overrides here).
    """
    if params is None:
        params = XGB_PARAMS
    if ctx is None:
        ctx = load_context()
    df = ctx["df"]
    man = ctx["manifest"]
    t_train_end = ctx["t_train_end"]
    t_val_end = ctx["t_val_end"]

    hsteps = man["horizons_steps"][hname]
    ycol = man["horizon_target_cols"][hname]
    feat_cols = man["horizon_feature_columns"][hname]

    # rows with a valid future target for this horizon
    sub = df[df[ycol].notna()]
    t = sub.index
    tr = t < t_train_end
    va = (t >= t_train_end) & (t < t_val_end)
    te = t >= t_val_end

    X = sub[feat_cols]
    y = sub[ycol].to_numpy(dtype="float64")
    persist = sub["log_flux"].to_numpy(dtype="float64")   # flux(t) as log

    n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())
    log(f"\n[{hname}] horizon = {hsteps} steps ({hsteps*5} min), "
        f"{len(feat_cols)} features")
    log(f"[{hname}] rows  train={n_tr:,}  val={n_va:,}  test={n_te:,}")
    log(f"[{hname}] train {t[tr].min()} -> {t[tr].max()}")
    log(f"[{hname}] val   {t[va].min()} -> {t[va].max()}")
    log(f"[{hname}] test  {t[te].min()} -> {t[te].max()}")

    # ---- XGBoost ---- #
    t0 = time.time()
    model = xgb.XGBRegressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                             **params)
    fit_kwargs = dict(eval_set=[(X[va], y[va])], verbose=False)
    if storm_weight and storm_weight != 1.0:
        # upweight geomagnetic-storm rows; NaN SYM_H -> weight 1.0 (never storm).
        w = np.where(sub["SYM_H"].to_numpy(dtype="float64") < STORM_SYMH,
                     float(storm_weight), 1.0)
        fit_kwargs["sample_weight"] = w[tr]
        fit_kwargs["sample_weight_eval_set"] = [w[va]]
    model.fit(X[tr], y[tr], **fit_kwargs)
    best_it = model.best_iteration
    fit_s = time.time() - t0

    pred_te = model.predict(X[te])
    pred_va = model.predict(X[va])

    # ---- metrics on TEST set ---- #
    y_te = y[te]
    p_te = persist[te]
    m = {
        "horizon": hname,
        "horizon_steps": hsteps,
        "n_features": len(feat_cols),
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "best_iteration": int(best_it),
        "fit_seconds": round(fit_s, 1),
        # log scale
        "persist_rmse_log": rmse(p_te, y_te),
        "xgb_rmse_log": rmse(pred_te, y_te),
        # raw scale (pfu)
        "persist_rmse_raw": rmse(inv_log(p_te), inv_log(y_te)),
        "xgb_rmse_raw": rmse(inv_log(pred_te), inv_log(y_te)),
        # val-set XGB (for early-stopping sanity)
        "xgb_rmse_log_val": rmse(pred_va, y[va]),
    }
    m["improvement_log_pct"] = round(
        100 * (m["persist_rmse_log"] - m["xgb_rmse_log"]) / m["persist_rmse_log"], 1)
    m["beats_persistence"] = m["xgb_rmse_log"] < m["persist_rmse_log"]

    log(f"[{hname}] fit {fit_s:.1f}s  best_iter={best_it}  "
        f"| persist RMSE(log)={m['persist_rmse_log']:.4f}  "
        f"xgb RMSE(log)={m['xgb_rmse_log']:.4f}  "
        f"({m['improvement_log_pct']:+.1f}%)")

    m["storm_weight"] = float(storm_weight)
    preds_df = None
    if save or return_preds:
        # test-set predictions for Section 6 (avoids retraining) / in-memory reuse
        out = pd.DataFrame({
            "y_log_true": y_te,
            "xgb_log": pred_te,
            "persist_log": p_te,
            "SYM_H": sub["SYM_H"].to_numpy()[te],
            "flux_true": inv_log(y_te),
            "flux_xgb": inv_log(pred_te),
            "flux_persist": inv_log(p_te),
        }, index=t[te])
        out.index.name = "Time"
        preds_df = out
        if save:
            MODEL_DIR.mkdir(exist_ok=True)
            model.save_model(MODEL_DIR / f"xgb_{hname}{save_suffix}.json")
            out.to_parquet(MODEL_DIR / f"test_predictions_{hname}{save_suffix}.parquet")

    if return_preds:
        return m, preds_df
    return m


def print_checkpoint(results):
    log("\n" + "=" * 78)
    log("SECTION 5 CHECKPOINT  -  persistence baseline vs XGBoost (TEST set)")
    log("=" * 78)
    log(f"{'horizon':>8} | {'train':>9} {'val':>8} {'test':>8} | "
        f"{'persist':>9} {'xgb':>9} {'impr%':>7} | {'persist':>10} {'xgb':>10}")
    log(f"{'':>8} | {'rows':>9} {'rows':>8} {'rows':>8} | "
        f"{'RMSE-log':>9} {'RMSE-log':>9} {'':>7} | {'RMSE-raw':>10} {'RMSE-raw':>10}")
    log("-" * 78)
    for m in results:
        flag = "" if m["beats_persistence"] else "  <-- FAILS baseline!"
        log(f"{m['horizon']:>8} | {m['n_train']:>9,} {m['n_val']:>8,} "
            f"{m['n_test']:>8,} | {m['persist_rmse_log']:>9.4f} "
            f"{m['xgb_rmse_log']:>9.4f} {m['improvement_log_pct']:>6.1f}% | "
            f"{m['persist_rmse_raw']:>10.1f} {m['xgb_rmse_raw']:>10.1f}{flag}")
    log("-" * 78)
    log("RMSE-log is the training-scale metric (log10(flux+1)); RMSE-raw is in pfu "
        "and\nis dominated by rare storm spikes -> see Section 6 for storm-specific "
        "and\nthreshold (HSS/POD/FAR) metrics.")
    all_beat = all(m["beats_persistence"] for m in results)
    if all_beat:
        log("\nAll horizons beat their persistence baseline.")
    else:
        fails = [m["horizon"] for m in results if not m["beats_persistence"]]
        log(f"\nWARNING: horizon(s) failing baseline: {fails} -- needs investigation.")
