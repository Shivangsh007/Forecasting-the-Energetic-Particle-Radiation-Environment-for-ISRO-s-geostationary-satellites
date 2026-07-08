# SolarSentinel

ISRO Bharatiya Antariksh Hackathon (BAH) 2026 entry. Forecasts >2 MeV relativistic
electron flux at GEO at **+30 min / +6 h / +12 h** to warn satellite operators of
deep-dielectric-charging / SEU hazard. One independent XGBoost regressor per horizon.

Full build narrative, phase checkpoints, and every revision's before/after numbers
live in `PROGRESS.md` — read it for history; this file is the quick-orientation map.

## Locked decisions (do not relitigate without flagging)

- **Target/transform:** `flux_2MeV` (pfu) → `log10(flux + 1)`; invert `10**x - 1` downstream.
- **Split:** chronological 70/15/15, never shuffled (autocorrelation leakage).
- **Coupling:** Bz-only rectified-southward proxy (no By/|B| in dataset — documented limitation).
- **Carrington 27.27-day lag** applied to both inputs and target flux.
- **Missing values:** left as NaN — XGBoost handles natively, never imputed.
- **Framing:** "persistence + physics-correction hybrid" — `flux_now` anchors the level,
  driver features supply the correction. Persistence is the mandatory baseline to beat.
- **Never** touch/regenerate `training_data_master_2018_2025.csv`.
- **Never** apply the GRASP↔GOES (#2) calibration as a correction — documented finding only.
- **Calibration #1 (SWPC→NCEI)** applies ONLY in the live daemon path, never in training/validation.
- **Dashboard never hardcodes metrics** — always reads `evaluate/model_performance_panel.json`
  (built by `evaluate/build_model_performance_panel.py` from real saved artifacts).

## Pipeline (in build order)

| Stage | Script | Output |
|---|---|---|
| Fetch | `get_omni.py`, `get_goes.py` | `datasets/*.csv` (raw OMNI/GOES); AE_INDEX/SYM_H sentinel `99999`→NaN on load |
| Merge | `merged_omni_goes.py` | `training_data_master_2018_2025.csv` (DO NOT regenerate) |
| Features | `features/build_features.py` (`assemble_features()` — **single shared code path**, also used by daemon + GRASP validation) | `features/features_master.parquet`, `feature_manifest.json` |
| Train | `models/train_{30min,6h,12h}.py` (shared params in `train_common.py`), or `train_all.py` | `models/xgb_{h}.json`, `test_predictions_{h}.parquet`, `metrics_section5.json` |
| Evaluate | `evaluate/evaluate.py` | `metrics_summary.json`, `metrics_regression/threshold.csv`, `feature_importance_{h}.png/csv` |
| GRASP validation | `evaluate/grasp_consolidate.py` → `evaluate/grasp_validation.py` | `data/grasp/grasp_master.parquet`, `evaluate/grasp_metrics.json` |
| Calibration #1 (live) | `evaluate/fit_flux_calibration.py` | `data/live/flux_calibration.json` (SWPC→NCEI, applied in daemon only) |
| Calibration #2 (finding only) | `evaluate/fit_grasp_goes_calibration.py` | `data/grasp/grasp_goes_calibration.json` (GRASP↔GOES MLT scale — never applied) |
| Performance panel | `evaluate/build_model_performance_panel.py` | `evaluate/model_performance_panel.json` (re-run after any retrain) |
| Real-time daemon | `realtime_daemon.py` | `data/live/latest.json`, `data/live/history.parquet` |
| Bulletin | `bulletin_generator.py` | text bulletin from `latest.json` |
| Dashboard | `dashboard/app.py` (+ `charts.py`, `hazard.py`, `snapshot.py`) | Streamlit UI |

XGBoost params (`models/train_common.py`): `n_estimators=3000, max_depth=7, lr=0.03,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0`,
early stopping 60 rounds on val RMSE. 65 features post-audit R3.

## Dashboard architecture

Dark "command-center" aesthetic: grayscale everywhere, color reserved for hazard
STATUS (Green/Yellow/Red) only. See the `dataviz` skill for the palette/validation method.

- `dashboard/app.py` — layout: header → inline control strip (Live/Replay, IST toggle,
  ↻ refresh, report download — no sidebar) → thin 1px HUD hazard strip → 3-col
  (telemetry+sparkline | forecast hero | asset/bulletin/map) → full-width Model
  Performance panel.
- `dashboard/charts.py` — pure/testable Plotly figures: `forecast_figure` (discrete
  3-horizon markers, never interpolated; skill-weighted opacity/size; persistence
  baseline drawn explicitly), `sparkline_figure`, `map_figure`.
- `dashboard/hazard.py` — thresholds (`THRESH_EVENT`=1000 pfu, `THRESH_SEVERE`=10000 pfu),
  `LEVELS`, `SKILL_STYLE`, `HORIZON_LABELS`.
- `dashboard/snapshot.py` — replay-mode sample snapshot fallback.

## Known limitations (state openly, don't hide)

- Bz-only coupling proxy (no By/|B| in dataset).
- +30 min: XGBoost beats persistence on log-RMSE but ties/slightly trails on raw-pfu
  RMSE at storm spikes — panel auto-flags this via `HSS_TIE_EPS` note, don't overclaim.
- Live daemon: SYM_H now sourced from real-time Kyoto Dst (`products/kyoto-dst.json`, hourly
  ring-current proxy) → offline ablation shows it recovers ~100% of the SYM_H-attributable
  +6h/+12h skill (R10). AE_INDEX still has no SWPC real-time source → NaN (only ~2-3% of gain).
  27-day Carrington + longest SYM_H lags still mature over the buffer. +30min unaffected.
- GRASP↔GOES weak instantaneous correlation is confirmed MLT longitude physics
  (GSAT-19 48°E vs GOES 75°W, ~8h apart), not a bug — never "fix" this with calibration #2.

## Data layout

- `datasets/` — raw fetched OMNI/GOES CSVs + intermediate merge artifacts.
- `data/grasp/` — consolidated real ISRO GSAT-19/GRASP validation data (`raw/` = per-day
  tab-separated .txt+.xml from `GRASP_data.zip`; `grasp_master.parquet` = consolidated).
- `data/live/` — daemon runtime state: `latest.json` (current payload), `history.parquet`
  (accumulated feed history for lag features), `flux_calibration.json` (calibration #1).
- `omni_2017.csv` (repo root) — 2017 OMNI pull for the GRASP out-of-time validation window.

## Environment

Windows 11, PowerShell primary (Bash tool also available). Python 3.10.
`requirements.txt`: numpy>=2.2, pandas>=2.3, pyarrow>=24.0, xgboost>=3.2,
matplotlib>=3.10, streamlit>=1.58, plotly>=6.8, xarray>=2025.6, netCDF4>=1.7,
hapiclient>=0.2.9. Watch cp1252 console encoding — keep new scripts ASCII-safe
(no ●, —, Δ, ∩ in print/logging).


