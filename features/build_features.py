"""
SolarSentinel - Section 4: Feature engineering.

Builds the model-ready feature matrix from `training_data_master_2018_2025.csv`.

Pipeline (see SolarSentinel brief, Section 4):
  1. Reindex the 5-min series onto a complete regular grid so every `.shift(N)`
     is exactly N*5 minutes in real time (gaps become honest NaNs).
  2. Target transform:  log_flux = log10(flux_2MeV + 1)  [invert: 10**x - 1].
  3. Coupling / derived terms (Bz-only rectified-southward proxy; By unavailable).
  4. Rolling-window statistics (mean/std + targeted extremes).
  5. 27.27-day Carrington-rotation recurrence lag on BOTH inputs and the target.
  6. Lag-correlation study: per horizon (+30min / +6h / +12h), for each driver,
     scan feature lags 0-72h and select the strongest-correlated lags.

Outputs (written to features/):
  - features_master.parquet        model-ready matrix (labelled rows only)
  - feature_manifest.json          transforms + per-horizon feature column lists
  - lag_correlation_study.csv      full CCF curves (for the presentation)
  - lag_correlation_selected.csv   selected top-N lags per horizon/variable
  - lag_correlation_plot.png       CCF plot (best-effort)

Nothing here touches / regenerates the master CSV.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
SRC_CSV = ROOT / "training_data_master_2018_2025.csv"
OUT_DIR = ROOT / "features"

CADENCE_MIN = 5
STEPS_PER_HOUR = 60 // CADENCE_MIN          # 12
STEPS_PER_DAY = 24 * STEPS_PER_HOUR         # 288

INPUT_VARS = ["BZ_GSM", "flow_speed", "proton_density", "AE_INDEX", "SYM_H"]
TARGET_RAW = "flux_2MeV"

# Forecast horizons (Section 5) -- EXACTLY these three, in 5-min steps.
HORIZONS = {"30min": 6, "6h": 72, "12h": 144}

# 27.27-day Carrington rotation, in 5-min steps.
CARRINGTON_DAYS = 27.27
CARR_STEPS = int(round(CARRINGTON_DAYS * STEPS_PER_DAY))   # 7854

# Rolling windows.
ROLL_MEAN_HOURS = [1, 6, 24]
ROLL_STD_HOURS = [6, 24]

# Lag-correlation study.
LAG_MAX_HOURS = 72
TOP_N_LAGS = 3               # strongest lags kept per (horizon, variable)
MIN_PAIRS = 5000             # min overlapping obs to trust a correlation
# Solar-wind CCFs plateau over tens of hours; require selected lags to be
# genuinely distinct rather than 3 adjacent points on the same plateau.
SEP_SELECT_STEPS = 3 * STEPS_PER_HOUR    # min 3h between per-horizon picks
SEP_MAT_STEPS = 6 * STEPS_PER_HOUR       # min 6h between materialized columns
MAX_LAGS_PER_VAR = 3                     # cap materialized lags per variable

# Fine 5-min resolution to 2h, then 30-min resolution out to 72h.
_fine = list(range(0, 2 * STEPS_PER_HOUR + 1))
_coarse = list(range(2 * STEPS_PER_HOUR, LAG_MAX_HOURS * STEPS_PER_HOUR + 1,
                     STEPS_PER_HOUR // 2))
LAG_GRID = sorted(set(_fine + _coarse))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def log(msg):
    print(msg, flush=True)


def shift_np(a: np.ndarray, k: int) -> np.ndarray:
    """Shift array forward by k (past value at t-k); fills leading k with NaN."""
    if k == 0:
        return a
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def masked_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r over pairwise-finite entries; NaN if too few pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < MIN_PAIRS:
        return np.nan
    xv = x[m]
    yv = y[m]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = np.sqrt((xv * xv).sum() * (yv * yv).sum())
    if denom <= 0:
        return np.nan
    return float((xv * yv).sum() / denom)


# --------------------------------------------------------------------------- #
# Load + reindex to a complete 5-min grid
# --------------------------------------------------------------------------- #
def load_grid() -> pd.DataFrame:
    log(f"[load] reading {SRC_CSV.name} ...")
    df = pd.read_csv(SRC_CSV, parse_dates=["Time"])
    n_obs = len(df)
    df = df.sort_values("Time").set_index("Time")

    # OMNI fill cleanup missed for AE_INDEX/SYM_H when the master was built:
    # AE_INDEX == 99999 for all of 2020 (12.3% of rows). Treat as missing so
    # rolling/lag/Carrington features are not corrupted. See PROGRESS.md R3.
    for _c in ("AE_INDEX", "SYM_H"):
        df[_c] = df[_c].replace(99999, np.nan)

    full = pd.date_range(df.index.min(), df.index.max(),
                         freq=f"{CADENCE_MIN}min")
    df = df.reindex(full)
    df.index.name = "Time"

    log(f"[load] observed rows            : {n_obs:,}")
    log(f"[load] full 5-min grid rows      : {len(df):,}")
    log(f"[load] gap slots inserted (NaN)  : {len(df) - n_obs:,}")
    return df, n_obs


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #
def add_target(df):
    df["log_flux"] = np.log10(df[TARGET_RAW].astype("float64") + 1.0)
    return ["log_flux"]


def add_coupling(df):
    """Bz-only rectified-southward coupling proxies + dynamic pressure.

    Full Newell coupling needs the IMF clock angle (By / |B|), unavailable in
    this dataset -- see PROGRESS.md 'Known limitations'. These proxies capture
    the dominant southward-IMF driver.
    """
    bz = df["BZ_GSM"].astype("float64")
    v = df["flow_speed"].astype("float64")
    n = df["proton_density"].astype("float64")

    bs = (-bz).clip(lower=0.0)                         # southward-Bz magnitude
    df["ec_newell_proxy"] = np.power(v, 4.0 / 3.0) * np.power(bs, 2.0 / 3.0)
    df["vbs_rect"] = v * bs                            # rectified dawn-dusk E ~ v*Bs
    df["p_dyn"] = 1.6726e-6 * n * v * v                # dynamic pressure (nPa)
    return ["ec_newell_proxy", "vbs_rect", "p_dyn"]


def add_rolling(df):
    cols = []
    for var in INPUT_VARS:
        s = df[var]
        for h in ROLL_MEAN_HOURS:
            w = h * STEPS_PER_HOUR
            name = f"{var}_mean_{h}h"
            df[name] = s.rolling(w, min_periods=max(1, w // 2)).mean()
            cols.append(name)
        for h in ROLL_STD_HOURS:
            w = h * STEPS_PER_HOUR
            name = f"{var}_std_{h}h"
            df[name] = s.rolling(w, min_periods=max(2, w // 2)).std()
            cols.append(name)

    # Targeted physical extremes over 24h.
    w = 24 * STEPS_PER_HOUR
    mp = max(1, w // 2)
    extremes = {
        "AE_INDEX_max_24h": df["AE_INDEX"].rolling(w, min_periods=mp).max(),
        "SYM_H_min_24h": df["SYM_H"].rolling(w, min_periods=mp).min(),
        "BZ_GSM_min_24h": df["BZ_GSM"].rolling(w, min_periods=mp).min(),
        "flow_speed_max_24h": df["flow_speed"].rolling(w, min_periods=mp).max(),
    }
    for name, ser in extremes.items():
        df[name] = ser
        cols.append(name)
    return cols


def add_carrington(df):
    """Value one Carrington rotation (27.27 d) earlier -- inputs AND target."""
    cols = []
    for var in INPUT_VARS:
        name = f"{var}_carr"
        df[name] = df[var].shift(CARR_STEPS)
        cols.append(name)
    df["log_flux_carr"] = df["log_flux"].shift(CARR_STEPS)   # target recurrence
    cols.append("log_flux_carr")
    return cols


def assemble_features(df, manifest):
    """Compute the full feature set on a prepared 5-min grid, reusing the SAME
    per-feature code as training and the manifest's locked lag set.

    Used by out-of-sample inference paths (GRASP validation, live daemon) so
    features are NEVER computed by a second, divergent code path. `df` must be a
    regular 5-min DatetimeIndex frame carrying the raw columns
    (BZ_GSM, flow_speed, proton_density, AE_INDEX, SYM_H, flux_2MeV).
    Returns df with all feature columns added.
    """
    add_target(df)
    add_coupling(df)
    add_rate_of_change(df)
    add_rolling(df)
    add_carrington(df)
    add_flux_autoregressive(df)
    for name in manifest["feature_groups"]["lags_materialized"]:
        var, steps = name.rsplit("_lag", 1)      # e.g. "proton_density_lag66"
        df[name] = df[var].shift(int(steps))
    return df


def add_rate_of_change(df):
    """Rate-of-change (derivative) + pressure-jump features (Step 3, PROGRESS.md R8).

    Rolling mean/std capture the LEVEL of a driver; these capture whether it is
    INTENSIFYING right now -- the storm-onset signal, complementary to level and
    aimed at +6h/+12h lead. Requires p_dyn, so add_coupling must run first. The
    p_dyn JUMP (max-minus-min over a short window) is a known precursor of sudden
    storm commencement (SSC). BZ_GSM/p_dyn ROC are live-computable; SYM_H/AE_INDEX
    ROC are NaN live (as their raw values are) but drive offline +6h/+12h skill.
    """
    cols = []
    for var in ["BZ_GSM", "SYM_H", "AE_INDEX", "p_dyn"]:
        s = df[var].astype("float64")
        for h in [1, 3, 6]:
            w = h * STEPS_PER_HOUR
            name = f"{var}_roc_{h}h"
            df[name] = s - s.shift(w)              # change over the last h hours
            cols.append(name)
    p = df["p_dyn"].astype("float64")
    for h in [1, 3]:
        w = h * STEPS_PER_HOUR
        mp = max(2, w // 2)
        name = f"p_dyn_jump_{h}h"
        df[name] = (p.rolling(w, min_periods=mp).max()
                    - p.rolling(w, min_periods=mp).min())
        cols.append(name)
    return cols


def add_flux_autoregressive(df):
    """Autoregressive features on the target's own recent history (log scale).

    flux(t) and recent flux are the dominant near-future predictors and are what
    let the model beat persistence -- a deliberate persistence+correction hybrid
    (see PROGRESS.md 'Decision revision, Section 5'). All are known at forecast
    time from the live GOES electron feed, so there is no leakage and the
    Section 8 daemon can compute them identically.
    """
    lf = df["log_flux"]
    cols = []
    df["flux_now"] = lf                                    # current log-flux
    cols.append("flux_now")
    for k, nm in [(12, "1h"), (36, "3h"), (72, "6h"), (144, "12h")]:
        name = f"flux_lag_{nm}"
        df[name] = lf.shift(k)
        cols.append(name)
    for h in [1, 6, 24]:
        w = h * STEPS_PER_HOUR
        name = f"flux_mean_{h}h"
        df[name] = lf.rolling(w, min_periods=max(1, w // 2)).mean()
        cols.append(name)
    for h in [6, 24]:
        w = h * STEPS_PER_HOUR
        name = f"flux_std_{h}h"
        df[name] = lf.rolling(w, min_periods=max(2, w // 2)).std()
        cols.append(name)
    df["flux_trend_6h"] = lf - lf.shift(6 * STEPS_PER_HOUR)   # recent rise/decay
    cols.append("flux_trend_6h")

    # Short-horizon flux trend + volatility (Step 2, PROGRESS.md R7). The 15-min
    # flux trend correlates strongly with what persistence gets wrong at +30 min;
    # these test whether that closes the +30 min gap or confirms a nowcast ceiling.
    # 15 min = 3 steps, 30 min = 6 steps @ 5-min cadence. All live-computable from
    # the GOES electron feed (unlike AE/SYM-H), so they also help the live +30 min.
    for k, nm in [(3, "15min"), (6, "30min")]:
        tname = f"flux_trend_{nm}"
        df[tname] = lf - lf.shift(k)
        cols.append(tname)
        sname = f"flux_std_{nm}"
        df[sname] = lf.rolling(k, min_periods=2).std()
        cols.append(sname)
    return cols


# --------------------------------------------------------------------------- #
# Lag-correlation study (per horizon)
# --------------------------------------------------------------------------- #
def _select_minsep(lags, corrs, n, sep):
    """Greedily take up to n strongest-|r| lags, each >= `sep` steps apart."""
    order = np.argsort(-np.abs(np.nan_to_num(corrs, nan=0.0)))
    chosen = []
    for i in order:
        if not np.isfinite(corrs[i]):
            continue
        L = int(lags[i])
        if all(abs(L - c) >= sep for c, _ in chosen):
            chosen.append((L, float(corrs[i])))
        if len(chosen) >= n:
            break
    return chosen                            # list of (lag_steps, corr)


def lag_study(df):
    """For each horizon & driver, corr(driver(t-L), log_flux(t+h)) over L in 0-72h.

    Selection uses minimum-separation peak-picking so the chosen lags are
    genuinely distinct rather than adjacent points on a broad CCF plateau.
    """
    log_flux = df["log_flux"].to_numpy(dtype="float64")
    lag_hours = np.array(LAG_GRID) / STEPS_PER_HOUR

    rows = []                                # full CCF curves
    selected = {h: {} for h in HORIZONS}     # horizon -> var -> [(lag, corr)]

    for hname, hsteps in HORIZONS.items():
        yf = np.concatenate([log_flux[hsteps:], np.full(hsteps, np.nan)])
        for var in INPUT_VARS:
            x = df[var].to_numpy(dtype="float64")
            corrs = np.array([masked_pearson(shift_np(x, L), yf)
                              for L in LAG_GRID])
            for L, lh, c in zip(LAG_GRID, lag_hours, corrs):
                rows.append((hname, var, L, round(float(lh), 3), c))

            picks = _select_minsep(LAG_GRID, corrs, TOP_N_LAGS, SEP_SELECT_STEPS)
            selected[hname][var] = picks
            top_str = ", ".join(f"{L}({L/STEPS_PER_HOUR:.2f}h,r={c:+.3f})"
                                for L, c in picks)
            log(f"    [{hname:>5} | {var:<14}] top{TOP_N_LAGS} lags: {top_str}")

    study_df = pd.DataFrame(rows, columns=["horizon", "variable", "lag_steps",
                                           "lag_hours", "pearson_r"])
    return study_df, selected


def materialize_lags(df, selected):
    """Materialize a de-duplicated union of selected lags.

    Pools every per-horizon pick per variable, then keeps up to
    MAX_LAGS_PER_VAR representatives that are >= SEP_MAT_STEPS apart (strongest
    |r| first). Also maps each horizon's picks to the representative column they
    fall on, so per-horizon feature lists stay consistent with what's built.
    Returns (created_cols, reps_per_var, horizon_lag_cols).
    """
    # pool lag -> best |r| across horizons
    pool = {v: {} for v in INPUT_VARS}
    for hname in HORIZONS:
        for var in INPUT_VARS:
            for L, c in selected[hname][var]:
                pool[var][L] = max(pool[var].get(L, 0.0), abs(c))

    reps = {v: [] for v in INPUT_VARS}       # var -> [lag_steps]
    for var in INPUT_VARS:
        for L in sorted(pool[var], key=lambda k: -pool[var][k]):
            if all(abs(L - r) >= SEP_MAT_STEPS for r in reps[var]):
                reps[var].append(L)
            if len(reps[var]) >= MAX_LAGS_PER_VAR:
                break
        reps[var].sort()

    created = []
    for var in INPUT_VARS:
        for L in reps[var]:
            if L == 0:
                continue                     # L==0 already present as raw current
            name = f"{var}_lag{L}"
            df[name] = df[var].shift(L)
            created.append(name)

    # per-horizon: map each pick to its nearest representative column
    horizon_lag_cols = {h: [] for h in HORIZONS}
    for hname in HORIZONS:
        for var in INPUT_VARS:
            names = []
            for L, _ in selected[hname][var]:
                if not reps[var]:
                    continue
                r = min(reps[var], key=lambda rr: abs(rr - L))
                if r != 0:
                    nm = f"{var}_lag{r}"
                    if nm not in names:
                        names.append(nm)
            horizon_lag_cols[hname].extend(names)

    return created, reps, horizon_lag_cols


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_study(study_df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(INPUT_VARS), 1, figsize=(9, 12),
                                 sharex=True)
        for ax, var in zip(axes, INPUT_VARS):
            for hname in HORIZONS:
                sub = study_df[(study_df.variable == var) &
                               (study_df.horizon == hname)]
                ax.plot(sub.lag_hours, sub.pearson_r, label=hname, lw=1.2)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel(var, fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].legend(title="horizon", fontsize=8)
        axes[-1].set_xlabel("feature lag (hours)")
        fig.suptitle("Lag-correlation: corr(driver(t-L), log_flux(t+h))")
        fig.tight_layout()
        out = OUT_DIR / "lag_correlation_plot.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        log(f"[plot] wrote {out.name}")
    except Exception as e:                    # noqa: BLE001 - best-effort plot
        log(f"[plot] skipped ({type(e).__name__}: {e})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    df, n_obs = load_grid()

    log("[feat] target transform (log10(flux+1)) ...")
    target_cols = add_target(df)

    log("[feat] coupling / derived terms ...")
    coupling_cols = add_coupling(df)

    log("[feat] rate-of-change + pressure-jump terms ...")
    roc_cols = add_rate_of_change(df)

    log("[feat] rolling-window statistics ...")
    rolling_cols = add_rolling(df)

    log(f"[feat] Carrington lag ({CARRINGTON_DAYS} d = {CARR_STEPS} steps) ...")
    carr_cols = add_carrington(df)

    log("[feat] autoregressive flux history ...")
    flux_ar_cols = add_flux_autoregressive(df)

    log("[study] lag-correlation study (per horizon) ...")
    study_df, selected = lag_study(df)
    lag_cols, reps, horizon_lag_cols = materialize_lags(df, selected)

    # Horizon target columns (log scale), computed on the grid.
    ytargets = {}
    for hname, hsteps in HORIZONS.items():
        col = f"y_log_{hname}"
        df[col] = df["log_flux"].shift(-hsteps)
        ytargets[hname] = col

    raw_cols = list(INPUT_VARS)
    feature_cols = (raw_cols + coupling_cols + roc_cols + rolling_cols + carr_cols
                    + flux_ar_cols + lag_cols)

    # ---- per-horizon feature column lists (shared feats + horizon lags) ---- #
    shared = (raw_cols + coupling_cols + roc_cols + rolling_cols + carr_cols
              + flux_ar_cols)
    horizon_feature_columns = {}
    for hname in HORIZONS:
        cols = list(shared)
        for nm in horizon_lag_cols[hname]:
            if nm not in cols:
                cols.append(nm)
        horizon_feature_columns[hname] = cols

    # ---- restrict to labelled rows (target present) ---- #
    labelled = df[df["log_flux"].notna()].copy()

    # downcast features to float32 to keep the parquet small
    keep = feature_cols + target_cols + [TARGET_RAW] + list(ytargets.values())
    out = labelled[keep].astype("float32")

    out_path = OUT_DIR / "features_master.parquet"
    out.to_parquet(out_path)

    study_df.to_csv(OUT_DIR / "lag_correlation_study.csv", index=False)
    sel_rows = [(h, v, i, L, round(L / STEPS_PER_HOUR, 3), round(c, 4))
                for h in HORIZONS for v in INPUT_VARS
                for i, (L, c) in enumerate(selected[h][v])]
    pd.DataFrame(sel_rows, columns=["horizon", "variable", "rank", "lag_steps",
                                    "lag_hours", "pearson_r"]).to_csv(
        OUT_DIR / "lag_correlation_selected.csv", index=False)

    selected_json = {
        h: {v: [{"lag_steps": L, "lag_hours": round(L / STEPS_PER_HOUR, 3),
                 "r": round(c, 4)} for L, c in selected[h][v]]
            for v in INPUT_VARS}
        for h in HORIZONS
    }

    manifest = {
        "meta": {
            "source_csv": SRC_CSV.name,
            "cadence_min": CADENCE_MIN,
            "carrington_days": CARRINGTON_DAYS,
            "carrington_steps": CARR_STEPS,
            "lag_grid_max_hours": LAG_MAX_HOURS,
            "top_n_lags": TOP_N_LAGS,
            "known_limitation": "Coupling is a Bz-only rectified-southward proxy; "
                                "full Newell coupling needs By/|B| (not in dataset).",
        },
        "target": {
            "raw_col": TARGET_RAW,
            "transform": "log10(x + 1)",
            "inverse": "10**x - 1",
            "log_col": "log_flux",
        },
        "horizons_steps": HORIZONS,
        "horizon_target_cols": ytargets,
        "feature_groups": {
            "raw_current": raw_cols,
            "coupling_derived": coupling_cols,
            "rate_of_change": roc_cols,
            "rolling": rolling_cols,
            "carrington": carr_cols,
            "flux_autoregressive": flux_ar_cols,
            "lags_materialized": lag_cols,
        },
        "materialized_lags_per_var": reps,
        "selected_lags_per_horizon": selected_json,
        "horizon_feature_columns": horizon_feature_columns,
        "n_features_total": len(feature_cols),
    }
    with open(OUT_DIR / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    plot_study(study_df)

    # ------------------------------------------------------------------ #
    # Checkpoint report
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 68)
    log("SECTION 4 CHECKPOINT")
    log("=" * 68)
    log(f"feature groups:")
    log(f"  raw current        : {len(raw_cols):>3}  {raw_cols}")
    log(f"  coupling/derived   : {len(coupling_cols):>3}  {coupling_cols}")
    log(f"  rate-of-change     : {len(roc_cols):>3}  {roc_cols}")
    log(f"  rolling            : {len(rolling_cols):>3}")
    log(f"  carrington         : {len(carr_cols):>3}  {carr_cols}")
    log(f"  flux autoregressive: {len(flux_ar_cols):>3}  {flux_ar_cols}")
    log(f"  lags (materialized): {len(lag_cols):>3}")
    log(f"  -----------------------")
    log(f"  TOTAL FEATURES     : {len(feature_cols):>3}")
    log("")
    log(f"per-horizon feature count (shared + horizon-specific lags):")
    for hname in HORIZONS:
        log(f"  {hname:>5}: {len(horizon_feature_columns[hname])} features")
    log("")
    log(f"output matrix shape  : {out.shape}  "
        f"(rows x [{len(feature_cols)} feat + {len(target_cols)+1} target + "
        f"{len(ytargets)} horizon-target])")
    log(f"rows retained        : {len(out):,} / {n_obs:,} original "
        f"({100*len(out)/n_obs:.2f}%)  -- dropped {n_obs-len(out):,}")
    log(f"  (all labelled rows kept; feature NaNs from limited history are "
        f"retained -- XGBoost handles missing values natively)")
    log("")
    log(f"full feature list ({len(feature_cols)}):")
    for c in feature_cols:
        log(f"    {c}")
    log("")
    log(f"artifacts written to {OUT_DIR}:")
    for p in ["features_master.parquet", "feature_manifest.json",
              "lag_correlation_study.csv", "lag_correlation_selected.csv"]:
        log(f"    {p}")
    log(f"\n[done] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
