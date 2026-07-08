# SolarSentinel — Build Progress Log

**Team:** SolarSentinel · **Hackathon:** BAH 2026 (ISRO)
**Problem:** Forecast >2 MeV relativistic electron flux at GEO at +30 min / +6 h / +12 h.

---

## Locked decisions (confirmed with reviewer before build)

- **Target:** `flux_2MeV` (pfu). **Transform:** `log10(flux + 1)`; invert with `10**x - 1`
  everywhere downstream. Chosen over noise-floor clipping — stable at zero, nothing arbitrary to defend.
- **Horizons:** exactly three — +30 min (6 steps), +6 h (72 steps), +12 h (144 steps). One
  independent XGBoost regressor per horizon.
- **Split (Section 5):** chronological 70/15/15, **not** shuffled (autocorrelation would leak).
- **Coupling:** Bz-only rectified-southward proxy (see Known Limitations).
- **Carrington 27.27-day recurrence lag:** applied to BOTH inputs and the target flux (option b) —
  the target's own recurrence is the core differentiator claim.
- **Missing input values:** NOT dropped/imputed — XGBoost handles NaN natively.
- **GRASP/GSAT-19 validation:** blocked on PRADAN access; build standalone, do not run on fake data.

---

## Decision revisions (documented, not quiet changes)

### R1 — Section 4/5: added autoregressive flux features → persistence+correction hybrid

**What changed:** the originally-approved Section 4 feature set was driver-only (solar-wind
+ geomagnetic drivers, plus a 27-day Carrington lag of the target). We added an 11-feature
autoregressive flux block (`flux_now`, flux lags 1/3/6/12 h, rolling flux mean 1/6/24 h +
std 6/24 h, 6 h flux trend). Feature total 54 → 65 (still in the 50-65 band).

**Original driver-only result (test set, RMSE-log vs persistence):**
30min −111.9% · 6h −15.7% · 12h +7.5% — i.e. FAILED the mandatory baseline at 2/3 horizons.

**Diagnosis:** without any recent-flux feature the model could not anchor to the current
flux level (which persistence gets for free); its RMSE was a flat ~0.50 at every horizon —
the signature of predicting a driver-implied level while blind to the current state.
Confirmed by a controlled scratch experiment (adding flux history flipped all three horizons
positive) before any pipeline change.

**Result after revision (test set, RMSE-log vs persistence):**
30min **+14.0%** · 6h **+22.5%** · 12h **+28.5%** — all beat baseline; skill grows with horizon.

**Design framing (presentation talking point, NOT a weakness):** this is a deliberate
**persistence + physics-correction hybrid**. `flux_now` anchors the current level; the
solar-wind/coupling/geomagnetic features supply the *correction* that lets the forecast
depart from persistence to catch storm onset and decay. `flux_now` is a real-time GOES
measurement available at forecast time — no leakage, and the Section 8 daemon computes it
from the same live feed. This is the standard, correct operational design; a driver-only
model that loses to persistence at short lead would not be operationally useful.

**Open question carried to Section 6:** report per-horizon feature importance and explicitly
judge whether the driver features (Bz, coupling, AE, SYM-H, etc.) carry real weight or
whether `flux_now`/flux-history dominate so heavily the model is barely persistence-plus-noise.
This decides the "does it actually catch storm onsets" claim — flag either way, don't just
report the RMSE win.

**Nuance to keep honest:** at +30 min, XGBoost beats persistence on RMSE-log (+14.0%) but is
slightly worse on RMSE-raw (569.7 vs 469.1 pfu) — absolute-pfu error is dominated by storm
spikes the 30-min-ahead flux barely moves on, where persistence is near-optimal. At 6 h/12 h
XGBoost wins on both scales.

---

### R2 — Section 8: live feed reality (URL fix + AE/SYM-H unavailable), NO pipeline divergence

**Finding 1 — brief's feed URLs are dead.** `products/solar-wind/mag-1-day.json` and
`plasma-1-day.json` (and 2h/6h variants) now return HTTP 404. Current working sources are
the RTSW feeds: `json/rtsw/rtsw_mag_1m.json` (`bz_gsm`) and `json/rtsw/rtsw_wind_1m.json`
(`proton_speed`, `proton_density`). The GOES `integral-electrons-1-day.json` feed works
(GOES-19, `>=2 MeV`, 5-min). This is a data-SOURCE URL correction, not a feature change.

**Finding 2 — no SWPC real-time AE_INDEX / SYM_H.** These come from OMNI/ground magnetometer
networks, not SWPC JSON. **Resolution (explicit, no second code path):** the live grid runs
through the SAME `assemble_features()`; `AE_INDEX`/`SYM_H` columns are present but all-NaN, and
XGBoost consumes them as missing (identical to training-time instrument gaps). Consequence,
flagged in the payload (`missing_inputs`, `feed_note`) and bulletin: +6h/+12h lean on SYM-H/AE
drivers (Section-6 top features) so LIVE long-horizon skill is below the offline test numbers;
+30 min (98.5% flux_now, which IS live) is unaffected. 27-day Carrington + long lags also need
history the 1-day feeds lack → the daemon accumulates `data/live/history.parquet`; those stay
NaN until it matures. **No divergence from the training feature pipeline.**

**Finding 3 — possible flux calibration offset.** SWPC integral >2 MeV flux vs the NCEI L2
`AvgIntElectronFlux` used in training may differ in processing/units; worth a cross-check over
an overlapping period. Live values (~70-190 pfu, quiet) look scale-consistent so far.

---

### R3 — AE_INDEX 99999 fill contamination: quantified, isolated to 2020, zero result impact

**Scope (measured):** AE_INDEX == 99999 (exact) in 92,288 rows = 12.32% of the master — but
**100% of it is in calendar year 2020** (every other year 0.0%), i.e. OMNI AE was entirely
fill for 2020 and the puller never replaced it. 2020 is wholly inside the TRAIN split.

**Feature distortion:** for 2020 rows `AE_INDEX_max_24h` = 99999 (meaningless that year) and
`AE_INDEX_mean_24h` inflated up to ~5e4; outside 2020 the features are identical (no fills).

**Model reliance:** AE-derived features = 0.24% / 1.92% / 3.07% of gain (30min/6h/12h) — small.

**Impact on reported results = ZERO (proven):** running the FROZEN models on cleaned-vs-
contaminated AE across the test set changes predictions by 0.00000 (mean & max), RMSE-log
identical — because test (2024–25) and both GRASP windows (2017, 2018) contain no fills and no
evaluated feature reaches back to 2020. So every Section 5/6 and GRASP number stands as-is.

**Fix applied:** `.replace(99999, np.nan)` for AE_INDEX (+SYM_H, defensive) added to
`get_omni.py` (future pulls) AND `features/build_features.py` load.

**RESOLVED via Option A (regenerated + retrained all 3 horizons).** Confirmed clean, numbers
not asserted:
- Only structural feature change: AE lags `lag22/lag108` (1.8h/9h — contamination artifacts in
  the CCF) → `lag720/lag858` (60h/71.5h — physically sensible substorm→acceleration lag). All
  other 63 features byte-identical; still 65 total. Per-horizon counts shifted 61/63/61 →
  **61/63/60** (12h's AE picks now map to one materialized lag column instead of two).
- Section 5 test ΔRMSE-log (before→after): 30min +0.0008, 6h −0.0025, 12h −0.0066 (MAX 0.0066,
  all < 0.01). Improvement% 14.0→13.7 / 22.5→23.1 / 28.5→29.7 — 6h/12h marginally BETTER.
  Persistence unchanged (model-independent). No horizon regressed past baseline.
- GRASP both windows re-run on retrained models: every correlation moved ≤0.004; pattern and
  the 2017≈2018 anti-overfit finding intact.
Contamination confirmed inconsequential to results; pipeline now clean and code↔artifacts in sync.

### R4 — Live flux calibration: SWPC real-time → NCEI L2 (training) scale

Models trained on NCEI L2 flux; daemon reads SWPC real-time. `flux_now` dominates short-horizon
predictions, so a scale mismatch would bias live forecasts. `evaluate/fit_flux_calibration.py`
pulls an overlap window (SWPC 7-day feed ∩ NCEI archive) and fits; daemon applies it (raw SWPC
kept in the buffer, calibration applied at feature-build; `flux_calibration` block in latest.json).

**Discrepancy measured (1,809 pts, 2026-06-29→07-05, GOES-19, flux 4–5,687 pfu incl. a storm):**
- Operationally negligible where it counts: **0 / 1,826 disagreements at the 1000 pfu alert
  threshold**; 200–1000 pfu products agree exactly (ratio 1.00).
- One real systematic effect: **NCEI ~30% higher than SWPC in the 1000–5000 pfu storm range**
  (ratio 1.30). Plus a very-low-flux (<50) ~5× floor from NCEI's max-of-5-telescopes — an
  artifact, operationally irrelevant.
- Naive whole-range log-linear fit is INAPPROPRIATE (its intercept, driven by the low-flux
  floor, makes it *undershoot* the storm range). Fixed by fitting on SWPC ≥ 200 pfu:
  **log10(NCEI+1) = 1.071·log10(SWPC+1) − 0.156, R²=0.990**.
- Before→after: storm range (1000–5000) logRMSE **0.114 → 0.067**, offset **30% → 8%**;
  200–1000 untouched (0.0337→0.0346). >5000 (n=13) overshoots ~28% — extrapolation, conservative
  (errs toward Critical), tiny sample. Live demo: raw 1,161 → calibrated 1,335 pfu (elevated event).

**Caveats (documented, not blockers):**
- PROVISIONAL — fit from a single overlap week/event; re-run `fit_flux_calibration.py` when a
  future storm gives a second independent overlap; do not assume permanent.
- Mechanism is expected, not mysterious: SWPC = near-real-time PROVISIONAL; NCEI L2 = REPROCESSED
  science-quality final. Real-time-vs-final product drift is a normal, well-understood phenomenon.

### R5 — GRASP↔GOES longitude scale calibration (#2): weak instantaneously, strong at daily scale (MLT)

`evaluate/fit_grasp_goes_calibration.py`. Purpose: characterise the Indian-longitude (GRASP)
vs US-longitude (GOES/training) flux scale so cross-longitude results can be discussed in a
common scale (GRASP caveat [A], the ~2x offset). Fit on early 2018, VALIDATE on held-out late
2018 (62,124 overlapping 2018 points total; fit 38,595 / held-out 23,529).

**Fit:** log10(GOES+1) = 0.605·log10(GRASP+1) + 1.273 — but **R²(fit) = 0.38 (weak)**.
**Held-out:** overall pearson_log 0.637, logRMSE 0.77→0.61, median GOES/GRASP 3.09→0.84;
**storm(High) pearson_log only 0.209** (ratio 4.11→2.70 — calibration can't reconcile storms).

**Why weak — and it's PHYSICS, not error (key finding):** held-out pearson_log by averaging
scale = **5-min 0.637 → hourly 0.673 → daily 0.868**. GRASP (GSAT-19, **48°E** slot — confirmed
empirically by the +8 h lag peak below; an earlier "~83°E/~10.5 h" assumption was corrected in
the audit) and GOES (~75°W) are **~8.2 h**
apart in magnetic LOCAL TIME, and GEO >2 MeV flux is strongly MLT-dependent, so *instantaneous*
same-UTC fluxes diverge; at DAILY scale (shared storm enhancements dominate) they correlate
strongly (0.87) with a ~1.9× offset. So the ~2× scale offset is real and well-defined at daily
scale, but a clean *instantaneous* GRASP→GOES conversion is not achievable.

**Conclusions:** (1) scale-invariant CORRELATION remains the correct way to report the
cross-longitude validation — caveat [A] now EXPLAINED (MLT + real-time offset), not just noted.
(2) This *reinforces* the forecasting approach: the model generalises to Indian longitude by
using GRASP's OWN recent flux + global drivers, NOT by assuming India = US × constant (which
fails instantaneously, r=0.21 in storms). (3) NOT used in the live daemon — it's a
characterisation/reporting artifact. Caveats: single 2018 overlap (provisional); sub-daily
scatter is physical MLT, not instrument error.

**MLT framing POSITIVELY CONFIRMED (lag + outlier diagnostics, requested before finalising):**
- Timestamp parsing verified: parsed txt times match the XML's stated UTC observation windows
  exactly across 5 files spanning 2017–2018 (parsed 00:05..24:00 vs XML 00:00..00:00). No DOY bug.
- Lag cross-correlation (±13 h): smooth diurnal-shaped curve, trough −4.5 h → zero-lag 0.62 →
  **peak +8.0 h (0.785)** → declining — NOT a sharp fixed-offset spike. Peak at +8.0 h equals the
  GRASP–GOES local-time separation ((~45–48°E − (−75.2°E))/15 ≈ 8 h), i.e. the MLT diurnal phase
  offset. A timestamp bug (e.g. 5.5 h IST) would give a sharp spike recovering the full ~0.87
  correlation; this is a broad 24 h modulation peaking at the longitude offset instead.
- Storm r=0.21 is broad/systematic, NOT outlier-driven: Spearman 0.235 ≈ Pearson; dropping top-5%
  residuals → only 0.255; per-High-day Pearson median −0.36 (within-day ANTI-correlation from the
  ~8 h ≈ ⅓-day MLT phase), while daily means correlate 0.87. Coherent MLT picture, alignment sound.

---

### R6 — UI: command-center restyle + Model Performance panel (visual/layout only)

Data, hazard logic, model calls, and all validated numbers UNCHANGED. New:
`evaluate/build_model_performance_panel.py` → `evaluate/model_performance_panel.json` (per-horizon
MAE + R² computed fresh from `test_predictions_*`; RMSE/Pearson/skill/POD/FAR/HSS from saved
artifacts; GRASP 2017 out-of-time headline; revision/date/test-size). Dashboard reads that JSON —
never hardcoded, stays true after any retrain.
Restyled `dashboard/app.py` + `dashboard/charts.py`: near-black surfaces, 1px edges, sharp corners,
mono numerals with uppercase labels, grayscale + hazard-status colour only. Layout = centered
forecast hero flanked by dense HUD panels (telemetry+sparkline left; asset/bulletin/map right);
secondary wireframe map; full-width Model Performance table (column per horizon) with the +30 min
"ties persistence" note surfaced from the real HSS gap, and the GRASP differentiator given visible
placement. Plotly modebar (zoom/pan/download) preserved on the hero plot.
Verified: headless AppTest (live + replay→CRITICAL, panel renders real numbers, no exception);
real `streamlit run` + DevTools screenshot confirms the layout, modebar, and panel numbers.

**Rev 2 (reviewer fixes):** (1) dissolved the sidebar "Asset Command" block — Live/Replay
source, IST toggle, icon-only ↻ refresh, and ⤓ Report are now a thin inline control strip at the
top; sidebar hidden; calibration/local-flux notes moved to a footer. (2) Hazard banner restyled
from a filled colour toast to a thin 1px HUD strip (`[■]` bracket indicator + mono-uppercase level
in the restrained status colour + right-aligned VALID/SRC; message in muted ink). Right-side
satellite/asset/map panel unchanged. Re-verified via AppTest + DevTools capture.

---

### R7 — Step 2: short flux-trend features (+30 min experiment) — modest gain, ceiling confirmed

**Question (audit-deferred):** the +30 min horizon ties persistence on operational skill.
Short flux trends we never built (15-min/30-min) were hypothesised to close that gap. Test
whether they do, or confirm a genuine nowcast ceiling.

**Change:** added 4 features to `add_flux_autoregressive()` (so they flow through the SAME
`assemble_features()` used by daemon + GRASP): `flux_trend_15min`, `flux_trend_30min`
(= log_flux(t) − log_flux(t−3/−6 steps)) and `flux_std_15min`, `flux_std_30min` (rolling std,
3/6 steps). Feature total **65 → 69** (flux_autoregressive 11 → 15); per-horizon 61/63/60 →
**65/67/64**. Lag structure byte-identical (lag study unaffected). All 4 are live-computable
from the GOES electron feed (unlike AE/SYM-H) — a bonus for live +30 min.

**Result (TEST set, retrained all 3 horizons):**
| metric | 30min | 6h | 12h |
|---|---|---|---|
| log-RMSE skill vs persist | 13.7% → **14.4%** | 23.1% → 22.9% | 29.7% → 29.0% |
| raw-RMSE (pfu), xgb | 567.7 → **548.4** | 1331.8 → 1371.9 | 1500.5 → 1530.6 |
| HSS@1000 (xgb / persist) | 0.894 → 0.897 / 0.896 | 0.697 → 0.702 / 0.619 | 0.663 → 0.663 / 0.491 |
| 30min STORM raw-RMSE (xgb / persist) | 519.7 → **503.6** / 550.1 | — | — |

**Verdict (both outcomes recorded, per the honesty rule):**
- The short trends give a **real but modest +30 min gain** on log-RMSE (+0.7 pp) and raw-RMSE
  (567.7 → 548.4 pfu), concentrated in **storm rows** (503.6 < persistence 550.1). The new
  features are genuinely used (flux_trend_30min 0.18%, flux_trend_15min 0.12% of gain).
- BUT +30 min **HSS stays tied to persistence** (0.897 vs 0.896) — the operational-alert
  ceiling at 30-min lead is **genuine**: at that lead persistence's level-tracking is near-
  optimal at the 1000 pfu threshold, and no short-trend feature changes the hit/miss table.
- 6h/12h moved within retrain variance (log-RMSE% ~flat, HSS +0.005/+0.000). Not the target
  of this step; adopted the retrained models as the new baseline going into Step 3.

**Verification (all run, not asserted):** horizon-target timestamp alignment 0 mismatches /
12,000 sampled (max |diff| 2.3e-7); train↔daemon feature identity across all **69** columns,
worst rel diff **5.96e-08**; GRASP both windows re-run — every correlation moved ≤0.02, the
2017-out-of-time ≈ 2018-in-train anti-overfit finding intact (6h 0.850≈0.843, 12h 0.831≈0.834,
12h-storm 0.689≈0.697), +30 min GRASP r ticked up (2017 0.937→0.943, 2018 0.906→0.924).

---

### R8 — Step 3: rate-of-change + pressure-jump features (storm-onset focus)

**Goal:** rolling mean/std capture a driver's LEVEL; add features for whether a driver is
INTENSIFYING right now (storm onset) — targeted at +6h/+12h and storm-specific skill.

**Change:** new `add_rate_of_change()` block (runs after `add_coupling`, in the SAME
`assemble_features()`): **12 rate-of-change** features `{BZ_GSM,SYM_H,AE_INDEX,p_dyn}_roc_{1,3,6}h`
(= value(t) − value(t−window)) + **2 dynamic-pressure JUMP** features `p_dyn_jump_{1,3}h`
(rolling max−min; a known sudden-storm-commencement precursor). New manifest group
`rate_of_change`. Feature total **69 → 83**; per-horizon 65/67/64 → **79/81/78**. Lag structure
byte-identical. BZ_GSM/p_dyn ROC are live-computable; SYM_H/AE ROC are NaN live (like their
raw values) but drive offline skill.

**Result (TEST set):** overall metrics ~flat (storms are only 7% of test, so they barely move
the overall numbers) — log-RMSE skill 14.6% / 23.0% / 28.8%, HSS@1000 0.898 / 0.702 / 0.663.
The gain shows up where it was aimed — the **storm subset (SYM-H < −50)**:
| storm-subset metric | 30min | 6h | 12h |
|---|---|---|---|
| storm log-RMSE (R3 → Step2 → **Step3**) | 0.303→0.290→**0.289** | 0.386→0.384→**0.381** | 0.450→0.449→**0.443** |
| storm raw-RMSE (pfu), xgb | 519.7→503.6→**483.4** | 862→839→**832** | — (log improved; raw noisy) |
- **Driver share of total gain rose**: 6h 19.0% → **20.0%**, 12h 30.7% → **32.3%** — the physics
  features carry more weight now, confirming the "catches storm onset via physics" claim.
- New features genuinely used: **p_dyn_jump_3h** is the strongest storm-onset feature
  (0.29% at 6h, 0.19% at 12h — SSC precursor validated); SYM_H_roc_3h and AE_INDEX_roc_1h
  enter the +30 min top-driver list. All 14 have non-zero gain.

**Verdict:** storm-onset features **improve storm-specific accuracy at all three horizons**
and raise physical-driver importance, while overall/quiet metrics stay flat (no regression).
This is the intended, honest outcome — adopted as the new baseline going into Step 4.

**Verification (all run):** horizon-target alignment 0 mismatches / 12,000; train↔daemon
feature identity across all **83** columns, worst rel diff **5.96e-08** (roc features computed
identically in the daemon/GRASP path); GRASP both windows re-run — all r moved ≤0.01, the
2017≈2018 anti-overfit finding intact (6h 0.851≈0.849, 12h 0.831≈0.833, 12h-storm 0.687≈0.693),
+30 min GRASP r 0.943→0.947, 6h-storm(2017) 0.724→0.730.

---

### R9 — Step 4: storm sample-weighting — a PER-HORIZON decision (adopt +12h only)

**Experiment** (`models/experiment_storm_weight.py`, `evaluate/storm_weight_experiment.json`):
retrained each horizon upweighting SYM_H < −50 rows by k ∈ {3, 5}× vs the unweighted Step-3
control, reporting **overall AND storm AND quiet** metrics (the honest tradeoff, not just the
flattering subset). `train_one()` gained a `sample_weight` hook (NaN SYM_H → weight 1.0).

**Tradeoff (TEST set), storm = SYM-H < −50:**
| horizon | control (unweighted) | k=3 | k=5 | reading |
|---|---|---|---|---|
| 30min storm HSS / raw-RMSE | **0.747 / 483** | 0.735 / 509 | 0.729 / 534 | weighting **hurts** — strictly worse |
| 6h storm HSS / overall HSS | **0.625 / 0.702** | 0.618 / 0.700 | 0.621 / 0.694 | neutral-to-negative — control best |
| 12h storm HSS / storm raw / overall HSS | 0.644 / 1119 / 0.663 | **0.660 / 1035 / 0.665** | 0.660 / 961 / 0.657 | **win-win at k=3** |

**Decision (per-horizon, justified by the numbers + the independent-model-per-horizon design):**
**adopt `storm_weight=3.0` for +12h only; keep +30 min and +6h unweighted.**
- +30 min is flux_now-dominated (98% AR gain) — upweighting storms only injects noise, so it
  degrades the storm subset. +6h: control already leads on both overall and storm HSS.
- +12h at k=3 is a genuine **win-win**: storm HSS 0.644 → **0.660**, storm log-RMSE 0.4428 →
  **0.4309**, storm raw-RMSE 1119 → **1035** pfu, AND overall skill 28.8% → **29.7%** /
  HSS 0.663 → **0.665**, with **quiet not degraded** (quiet log-RMSE 0.3816 → 0.3773).
  Physically sensible: at 12 h lead the model leans most on storm-onset drivers (32.6% gain).

**Made reproducible, not a one-off:** `STORM_WEIGHTS = {30min:1.0, 6h:1.0, 12h:3.0}` in
`train_common.py`, applied by `train_all.py`. Canonical retrain confirms determinism (+30 min /
+6h reproduce Step-3 exactly, best_iter 370 / 910) and the adopted +12h (best_iter 518).

**Out-of-sample check (the important one — is it test-set overfit?):** GRASP re-run — adopted
+12h **improves out-of-time**: 2017 r_log 0.831 → **0.837**, 2017 12h-storm 0.687 → **0.697**,
and the anti-overfit signature holds (2017 12h 0.837 ≈ 2018 12h 0.842). +30 min/+6h GRASP
unchanged. The +12h weighting **generalises to Indian longitude + unseen time**, so it is a
real gain, not test-set tuning. (Experiment variant model files were not committed.)

---

### R10 — Step 5: real-time SYM_H via Kyoto Dst — the live +6h/+12h gap is essentially CLOSED

**The open limitation:** live +6h/+12h skill was below the offline numbers because the SWPC
real-time feeds carry no AE_INDEX/SYM_H, so both were NaN live (R2). SYM-H drivers are the top
+6h/+12h features, so this was the biggest live-vs-offline gap.

**Investigation (verified against live SWPC docs 2026-07-07):**
- **`products/kyoto-dst.json`** — hourly Kyoto **Dst** (nT), live/maintained. Dst is the
  hourly ring-current disturbance index; **SYM_H is the same physical quantity at 1-min
  cadence, same nT scale** → a genuine, scale-appropriate real-time proxy for SYM_H.
- `json/planetary_k_index_1m.json` (Kp) exists but is a 0-9 quasi-log planetary index, a
  DIFFERENT quantity/scale from AE_INDEX (nT auroral electrojet). Feeding Kp into the AE column
  would misalign the model's learned nT splits (actively harmful); AE is only ~2-3% of gain →
  **AE deliberately left NaN**, documented, not force-mapped.

**Wiring (same pipeline, no fork):** the daemon fetches Dst, forward-fills the hourly value onto
the 5-min grid (2 h tolerance), and fills the **SYM_H raw column**; the SAME `assemble_features()`
then derives every SYM_H feature. This is a data-SOURCE change (live SYM_H now = Dst instead of
NaN), not a feature-code change — train↔daemon feature identity is unaffected.

**Quantified gap closure — offline SYM_H ablation** (3 variants through the pipeline + adopted
models, TEST set; `scratchpad/step5_symh_validation.py`), skill = log-RMSE vs persistence:
| horizon | true 1-min SYM_H (ideal) | **NaN (prior live)** | **Dst-proxy (now)** | gap recovered |
|---|---|---|---|---|
| +6h | +22.98% | **−14.68%** (BELOW persistence!) | **+22.94%** | **~100%** |
| +12h | +29.73% | +12.26% | **+29.75%** | **~100%** |
Storm-subset log-RMSE tells the same story (6h: true 0.381, NaN 0.558, Dst 0.382). The Dst
proxy recovers essentially all SYM_H-attributable skill because the dominant SYM_H feature
(`SYM_H_mean_24h`) is a 24 h average that hourly Dst reproduces almost exactly.

**Live end-to-end verified (real poll 2026-07-07 16:32Z):** all 4 feeds OK incl. `dst`; payload
`SYM_H = −20.0 nT` (was `null`), `missing_inputs = ["AE_INDEX"]` (SYM_H dropped), status `ok`.

**Status: the live +6h/+12h gap is essentially CLOSED.** The 24 h-window SYM_H features (incl.
the top driver) populate immediately from the 7-day Dst feed; the 27-day Carrington + 48-60 h
SYM_H lags still mature over the buffer's first ~40 days. Residual live gap is only the
AE_INDEX-attributable part (~2-3% of gain). Dst latency ~1 h and hourly cadence are immaterial
to the long-window SYM_H features that carry the importance.

---

### R11 — Step 6: bounded hyperparameter search — small, per-horizon adoptions

**Grid** (`models/experiment_hyperparam.py`, `evaluate/hyperparam_search.json`): 12 configs per
horizon around the current params (depth 7, lr 0.03, mcw 5, λ 1.0) — varying depth {5,6,8},
lr {0.02,0.05}, λ {0.5,2,3}, mcw {3,10}, and one combined config — same chronological split +
early stopping, using the adopted per-horizon storm weights. Scored on the TEST set (log/raw
RMSE, HSS, storm log-RMSE, storm HSS) vs the canonical model.

**Finding:** the current params are already near-optimal — **no config beats current by more than
0.0024 log-RMSE**, all within the ~0.01 materiality band R3 established. But the differences are
deterministic (seed-fixed) and directionally consistent: shallower/more-regularized trees help
the two short near-persistence horizons; +12h needs its depth.

**Decision (per-horizon, adopt only where real + no downside; HSS = operational tiebreaker):**
| horizon | adopted | why |
|---|---|---|
| +30min | **max_depth=5** | log-RMSE 0.2010→**0.1995** (+13.7%→**+15.2%** skill), storm HSS 0.747→**0.757**, HSS flat 0.898 — free gain |
| +6h | **max_depth=5** | best operational gain in the grid: HSS 0.702→**0.710**, storm HSS 0.625→**0.650**, log-RMSE best |
| +12h | **keep** (depth 7, w=3) | every config improving overall log-RMSE **degraded storm HSS** 0.660→0.641, undoing the Step-4 storm gain |

Baked into `train_common.PER_HORIZON_PARAMS = {30min:{depth 5}, 6h:{depth 5}, 12h:{}}`, applied by
`train_all.py`; final canonical retrain reproduces the grid numbers exactly (determinism).

**Overfit guard (GRASP, the important check):** the tuning is chosen on TEST, so GRASP must not
regress. It does not — 2017-out-of-time stable (6h 0.851→**0.848**, 12h 0.837), and the
2017≈2018 anti-overfit signature holds (6h 0.848≈0.841, 12h 0.837≈0.842). The tuning
generalises to Indian longitude + unseen time; it is not test-set overfit.

**Verification (final model set):** train↔daemon feature identity across all 83 columns, worst
rel diff 5.96e-08; horizon-target alignment 0 mismatches / 12,000 sampled.

---

### R12 — Stage C-1: shared-deployment daemon singleton + atomic writes

**Problem:** Streamlit Community Cloud runs ONE process shared by all visitors; the daemon was a
separate external process (impossible on Cloud), and the dashboard only read `latest.json`.

**Fix (reuses the existing pipeline, no fork):**
- `dashboard/app.py` `_live_daemon()` — an `@st.cache_resource` singleton that starts the NOAA
  poll loop in ONE background thread per process (cache_resource runs its body once and hands the
  same object to every session → exactly one thread regardless of visitor count). It calls
  `realtime_daemon.poll()` → the SAME `assemble_features()`/`payload_from_frame()` path (no second
  pipeline). The thread only writes files, never touches Streamlit (no ScriptRunContext issue).
  `SOLARSENTINEL_INPROC_DAEMON=0` disables it for local users running the external daemon.
- `realtime_daemon.py` — writes are now **atomic** (`_atomic_write_text` / `_atomic_write_parquet`
  = temp file + `os.replace`, atomic on Windows + POSIX) for `latest.json` and `history.parquet`,
  so a visitor reading `latest.json` mid-write never sees a truncated file → no per-visitor crash.

**Verified (headless AppTest, `scratchpad/test_singleton.py`):** 5 concurrent sessions each render
with NO exception; `threading.enumerate()` shows **exactly 1** `solarsentinel-daemon` thread after
all 5 (singleton confirmed); the thread's live poll succeeded (all 4 feeds OK). A separate real
daemon poll confirms atomic `latest.json` is valid JSON. Graceful stale/degraded handling and the
sample-snapshot fallback are unchanged from Section 8 (the thread wraps `poll()` in try/except so a
single failed poll never kills the loop).

## Known limitations (state openly in the presentation)

- **Coupling function is Bz-only.** The full Newell coupling function needs the IMF clock angle,
  which requires **By** and total transverse field **|B|** — neither is present in the current
  dataset (only `BZ_GSM`). We use a rectified-southward proxy
  (`v^(4/3)·|Bz_south|^(2/3)` and `v·Bz_south`) that captures the dominant southward-IMF driver.
  Deliberate tradeoff (not re-fetching By), documented rather than hidden.
  **Honesty note (Step 7, measured — presentation talking point):** with By unavailable, these
  two coupling proxies carry **~0 measured feature-importance gain** — `ec_newell_proxy`
  0.007% / 0.012% / 0.024% and `vbs_rect` 0.006% / 0.008% / 0.013% of total gain (30min/6h/12h),
  i.e. essentially zero. They are kept as brief deliverables (the coupling term is part of the
  standard physics story), but we state plainly that the model earns its skill from flux
  autoregression + the SYM-H/speed/density level & rate-of-change drivers, NOT from the coupling
  proxies. The rectified-southward driver signal is instead carried by `BZ_GSM_min_24h`, the
  `BZ_GSM_roc_*` rate-of-change terms, and `BZ_GSM` rolling stats.

---

## Final model state (post-Stage-A, revisions R7–R11) — the current numbers

Everything below in "Phase status" is the historical build narrative; the **superseded-number
notes point here**. This is the authoritative current state after Stage A (Steps 1–7).

**Features:** 83 total (was 65 at R3). Added: 4 short flux trend/vol (R7), 12 rate-of-change +
2 pressure-jump (R8). Per-horizon 79 / 81 / 78. Lag structure unchanged since R3.
**Models:** one XGBoost per horizon; +30min & +6h `max_depth=5` (R11), +12h `max_depth=7` +
storm sample-weight 3× on SYM-H<−50 (R9); all other params per `XGB_PARAMS`.

**Test set (chronological, 2024-11-22 → 2025-12-31), XGB vs persistence:**
| horizon | skill vs persist (log-RMSE) | HSS@1000 (xgb / persist) | storm HSS | storm raw-RMSE (pfu) | raw-RMSE (pfu) |
|---|---|---|---|---|---|
| +30min | **+15.2%** (R3: +13.7%) | 0.898 / 0.896 (ties — nowcast ceiling) | 0.757 | 495 | 532 |
| +6h | +23.1% | **0.710** / 0.619 (R3: 0.697) | **0.650** (R3: 0.625) | 833 | 1347 |
| +12h | +29.7% | 0.665 / 0.491 | **0.660** (R3: 0.644) | **1035** (R3: ~1119) | 1518 |

**GRASP cross-longitude (2017 out-of-time / 2018 in-train), Pearson r_log (headline):**
+30min 0.946/0.933 · +6h 0.848/0.841 · +12h 0.837/0.842 · 12h-storm 0.697/0.691. Anti-overfit
finding intact (2017≈2018); GRASP stable across all of Stage A (no out-of-sample regression).

**Live daemon:** SYM_H now sourced from real-time Kyoto Dst (R10) → offline ablation shows
~100% recovery of the SYM_H-attributable +6h/+12h skill; AE_INDEX still NaN (~2-3% gain).

---

## Phase status

### ✅ Section 4 — Feature engineering  (COMPLETE — approved)

> **SUPERSEDED NUMBERS NOTE (updated 2026-07-07, Stage A):** the counts below describe the
> ORIGINAL driver-only build. Superseded by R1 (+11 AR flux → 65), R3 (AE lags), and Stage A
> R7/R8 (+4 short-trend, +14 rate-of-change/jump). **Current: 83 features, per-horizon
> 79 / 81 / 78 — see the "Final model state" section above for the authoritative numbers.**
> The description of the method (grid reindex, transform, lag study) remains accurate.

Script: `features/build_features.py` (single code path — reused later by the live daemon, Section 8).

**What it does**
1. Reindexes the 749,358 observed 5-min rows onto the complete 841,536-slot grid
   (92,178 gaps → NaN) so every `.shift(N)` is exactly N×5 min in real time.
2. Target transform `log10(flux+1)`.
3. Coupling/derived (3): `ec_newell_proxy`, `vbs_rect`, `p_dyn` (dynamic pressure).
4. Rolling stats (29): mean over 1/6/24 h + std over 6/24 h for all 5 drivers, plus 24 h
   physical extremes (AE max, SYM-H min, Bz min, speed max).
5. Carrington lag (6): each of 5 drivers + target `log_flux`, 7854 steps (27.27 d) earlier.
6. Lag-correlation study: for each horizon and driver, `corr(driver(t−L), log_flux(t+h))`
   over L = 0–72 h; select top-3 **minimum-separation** lags (≥3 h apart per horizon), then
   materialise a de-duplicated union (≥6 h apart, ≤3 per driver → 11 lag features).

**Actual output**
- Total engineered features: **54** (within the 50–65 target).
  - raw current 5 · coupling/derived 3 · rolling 29 · Carrington 6 · lags 11.
- Per-horizon feature counts: 30 min → 50, 6 h → 52, 12 h → 50.
- Output matrix: **749,358 × 59** (54 features + `flux_2MeV` + `log_flux` + 3 horizon targets), float32.
- **Rows retained: 749,358 / 749,358 (100%), 0 dropped.** Feature NaNs from limited history
  (first ~27 d for Carrington, first 24 h for rolling) are kept — XGBoost handles them.
- Target `log10(x+1)` inverts to raw with max error 0.023 pfu (float32 rounding).

**Key physical finding (good presentation point):** all three horizons converge on the *same*
solar-wind response lag — for `flow_speed`, L+h ≈ 48.5 h regardless of horizon (30 min→48 h,
6 h→42.5 h, 12 h→36.5 h). This is the known ~2-day delay between a high-speed stream and the
>2 MeV flux enhancement, and it validates the lag study.

**Method note:** first pass used naive top-3 argmax, which selected near-adjacent collinear lags
on the broad solar-wind CCF plateaus (e.g. lag630/654/660, all within 30 min) and produced 82
features. Replaced with minimum-separation peak selection → distinct lags, 54 features.

**Artifacts** (`features/`): `features_master.parquet`, `feature_manifest.json`,
`lag_correlation_study.csv`, `lag_correlation_selected.csv`, `lag_correlation_plot.png`.

### ✅ Section 5 — Modelling — COMPLETE (awaiting review)

Files: `models/train_common.py` (shared path) + `train_{30min,6h,12h}.py` (per-horizon
entry points) + `train_all.py` (runner/checkpoint). 3 independent XGBoost regressors
(reg:squarederror on log10(flux+1), hist, early stopping on val RMSE) + persistence
baseline. Chronological 70/15/15 split, boundaries computed once on the full labelled
index and applied per horizon → identical calendar periods across horizons.

Split: **train < 2023-08-02 06:00 ≤ val < 2024-11-22 15:30 ≤ test**.

> **SUPERSEDED by Stage A (R7–R11) — see "Final model state" above for current numbers.**
(Values below = the R3 build. Pre-R3 values differed by ≤0.0066 RMSE-log.)
| horizon | train | val | test | best_iter | persist RMSE-log | xgb RMSE-log | impr | persist RMSE-raw | xgb RMSE-raw |
|---|---|---|---|---|---|---|---|---|---|
| 30min | 480,225 | 96,366 | 109,495 | 329 | 0.2353 | 0.2030 | **+13.7%** | 469.1 | 567.7 |
| 6h | 475,272 | 95,511 | 109,201 | 375 | 0.4437 | 0.3414 | **+23.1%** | 1770.3 | 1331.8 |
| 12h | 471,790 | 94,920 | 109,016 | 414 | 0.5427 | 0.3815 | **+29.7%** | 2162.2 | 1500.5 |

All horizons beat persistence on RMSE-log (the training scale). See R1 above for the
+30 min RMSE-raw nuance and the persistence+correction design framing. **Stage A current:
skill +15.2% / +23.1% / +29.7%; xgb RMSE-raw 532 / 1347 / 1518 (see Final model state).**

Artifacts (`models/`): `xgb_{30min,6h,12h}.json`, `test_predictions_{h}.parquet`
(true/xgb/persist in log & raw + SYM-H, for Section 6 without retraining),
`metrics_section5.json`.

### ✅ Section 6 — Evaluation — COMPLETE — approved

> **SUPERSEDED NUMBERS NOTE:** tables below are the PRE-R3 run. **Current (Stage A, R7–R11) —
> see "Final model state" above:** RMSE-log 0.1995/0.3412/0.3813; HSS@1000 XGB **0.898/0.710/0.665**
> vs persist 0.896/0.619/0.491 (skill labels none/moderate/high unchanged); storm HSS
> 0.757/0.650/0.660. The flux_now-dominance story and the +30 min raw-RMSE (nowcast-ceiling)
> conclusions are unchanged; the +6h/+12h storm-specific numbers improved (R8/R9/R11).

`evaluate/evaluate.py` (consumes frozen Section-5 test predictions + models, no retrain) +
`evaluate/grasp_validation.py` (see the real-data GRASP section below).

**Regression (test set), RMSE-log XGB vs persist + Pearson-log:**
| horizon | xgb RMSE-log | persist | xgb r-log | storm xgb/persist RMSE-log |
|---|---|---|---|---|
| 30min | 0.2023 | 0.2353 | 0.947 | 0.2891 / 0.3033 |
| 6h | 0.3438 | 0.4437 | 0.839 | 0.3866 / 0.5060 |
| 12h | 0.3881 | 0.5427 | 0.797 | 0.4499 / 0.5930 |
XGB beats persist on RMSE-log & Pearson at all horizons and on storm-subset at all horizons.

**Operational alert skill @ 1000 pfu (HSS / POD / FAR):**
| horizon | xgb HSS/POD/FAR | persist HSS/POD/FAR |
|---|---|---|
| 30min | 0.895 / 0.899 / 0.058 | 0.896 / 0.922 / 0.078 |
| 6h | 0.694 / 0.728 / 0.193 | 0.619 / 0.713 / 0.287 |
| 12h | 0.655 / 0.715 / 0.238 | 0.491 / 0.616 / 0.383 |
→ **At 30 min XGB ties persistence on HSS (no operational gain).** At 6 h/12 h XGB wins
decisively on every skill score — this is where the model earns its value.

**Feature-importance dominance check (share of total gain):**
| horizon | flux_now | flux_history | ALL flux | drivers | coupling |
|---|---|---|---|---|---|
| 30min | 67.9% | 30.6% | **98.5%** | 1.5% | 0.1% |
| 6h | 44.6% | 36.8% | 81.3% | **18.7%** | 0.3% |
| 12h | 0.2% | 68.2% | 68.4% | **31.6%** | 0.2% |
→ At 30 min the model is ~98.5% flux-autoregression (≈ persistence+correction), drivers
negligible — consistent with the tied HSS. At 6 h/12 h flux_now's dominance collapses
(68%→45%→0.2%) and **drivers carry real weight (18.7% / 31.6%)**; top drivers are
SYM_H_mean_24h, flow_speed_mean_24h, proton_density stats. So the "catches storm onsets via
physics" claim is TRUE and valuable at 6 h/12 h, and honestly NOT at 30 min.

**+30 min raw-RMSE diagnosis (revised from earlier "storm-spike" guess — the guess was wrong):**
- Storm rows (SYM-H<-50, 7.0% of test): XGB raw-RMSE 519.7 **< persist 550.1** — XGB is
  *better* during real geomagnetic storms.
- Quiet rows (93%): XGB raw-RMSE 573.3 **> persist 462.4** — the regression is here.
- SSE decomposition: storm rows contribute **−2.2%** of the (xgb−persist) raw-SSE gap
  (i.e. reduce it). The gap is entirely a quiet-condition effect.
- Concentration: worst 1% of rows hold 66% of XGB raw error, but only **6%** of those are
  SYM-H storms (median SYM-H −23 nT). They are high-flux HIGH-SPEED-STREAM enhancements —
  large flux without a strong SYM-H storm signature, so the SYM-H split labels them "quiet".
- Interpretation: XGB slightly under-predicts peak magnitude of large-flux events
  (bias ≈ −65 to −80 pfu) because at 30-min lead persistence's level-tracking is near-optimal
  and the drivers add ~0 signal (1.5% gain). This does NOT hurt hazard detection (HSS tied).
  Legitimate, well-understood limitation — state plainly; not a hidden weakness.

**GRASP:** *(SUPERSEDED — the FITS-stub description that stood here is obsolete: real GRASP
data arrived in tab-separated .txt format, the script was rewritten, and the validation was
RUN for real on two windows — see the dedicated GRASP section below.)*

Artifacts (`evaluate/`): metrics_regression.csv, metrics_threshold.csv,
feature_importance_{h}.csv/.png, metrics_summary.json.

### 🔬 GRASP / GSAT-19 cross-longitude validation — REAL RESULTS, TWO WINDOWS (PENDING REVIEW)

Data unblocked: real ISRO GRASP data (Jul 2017–Aug 2018) consolidated by
`evaluate/grasp_consolidate.py` (nested zips → 425 unique days, 122,400 rows). Format was
tab-separated .txt + .xml, NOT FITS — `grasp_validation.py` rewritten. Features via the SAME
`assemble_features()`; GRASP flux = target + AR anchor; storm = `Electron_Activity_level`=High.
**2017 OMNI pulled** (`omni_2017.csv`, identical `OMNI_HRO2_5MIN` method) to enable a genuinely
out-of-time test. Two windows reported side by side:
- **2017_out_of_time** (Jul–Dec 2017, 182 days) — OUTSIDE the model's train time-range → the
  strongest, genuinely out-of-sample cross-longitude test.
- **2018_in_train** (Jan–Aug 2018, 243 days) — inside train time → longitude generalisation only.

**Pearson r_log (XGB / persist) — headline (retrained models, R3; pre-R3 differed ≤0.004):**
| horizon · subset | 2017 OUT-OF-TIME | 2018 in-train |
|---|---|---|
| 30min overall | 0.937 / **0.979** | 0.906 / **0.986** |
| 6h overall | **0.850** / 0.710 | **0.845** / 0.764 |
| 12h overall | **0.834** / 0.464 | **0.839** / 0.563 |
| 6h storm | **0.729** / 0.411 | **0.649** / 0.246 |
| 12h storm | **0.685 / −0.085** | **0.693 / −0.297** |

**HSS @ 1000 pfu (XGB / persist):** 2017 → 6h 0.606/0.350, 12h 0.579/0.091 · 2018 → 6h
0.686/0.345, 12h 0.662/0.073.

**KEY FINDING:** the out-of-time 2017 result **essentially equals** the in-train 2018 result
(6h 0.850≈0.845; 12h 0.837≈0.843; 12h-storm 0.689≈0.695). If the model were memorising the
training time-period, 2018 would beat 2017 — it does not. So the skill is genuine
generalisation across **both longitude AND time**, not temporal overfit. This **resolves former
caveat [B]**. Pattern holds in both windows: no skill at +30 min (persistence wins), real skill
at +6h/+12h beating persistence, most dramatically in storms where +12h persistence goes
anti-correlated (r ≈ −0.09 to −0.30) while the model holds r ≈ 0.69.

**Remaining caveat [A]:** GRASP energy channel UNCONFIRMED as >2 MeV (no instrument doc; ~2×
median scale offset) → correlation (scale-invariant) is the honest headline; raw RMSE caveated.
Note: AE_INDEX has 99999 fills in 2018+ training data (unreplaced); 2017 OMNI has none; processed
identically for consistency; SYM-H (the important driver) is clean in both.

Status: strong + defensible, but NOT to be called "validated" in presentation material until
reviewer signs off on framing.

### ✅ Section 7 — Dashboard (Streamlit + Plotly) — COMPLETE (awaiting review)

Files: `dashboard/hazard.py` (shared thresholds/levels/satellites — reused by daemon +
bulletin), `dashboard/snapshot.py` (payload builder + CLI → `data/live/latest.json`),
`dashboard/charts.py` (Plotly figures, pure/testable), `dashboard/app.py` (Streamlit UI).

Dark "Asset Command" theme. Sidebar: data-source (Live / Historical replay date-time),
satellite dropdown + asset card (function/slot/local flux/status), Refresh + Download Report,
UTC/IST toggle. Main: colored hazard banner (names the exact threshold breached), 4 telemetry
cards (flux/speed/Bz/SYM-H + 1 h trend delta), GEO asset map (selected highlighted), forecast
chart. All values come from the payload — nothing hardcoded; models are only ever called via
the shared `snapshot.build_payload` (same builder the Section 8 daemon will use).

**Honesty carried into the UI (per reviewer):** forecast chart draws the persistence no-skill
baseline explicitly; the 3 horizons are DISCRETE markers with asymmetric confidence whiskers
(no smooth interpolation → no false between-horizon precision); marker size/opacity ∝ validated
Section-6 skill — +30 min rendered faint and tagged "= persistence", +6 h/+12 h progressively
prominent. Caption states the skill basis. Replay mode also overlays actual future values as
hollow "verification" markers.

**Verification (driven, not assumed):**
- Hazard banner exercised on 4 REAL test-set timestamps — changes colour correctly:
  2024-12-30 (15 pfu → 🟢 NOMINAL) · 2025-08-15 (2,427 pfu → 🟡 ELEVATED, names 1000 pfu) ·
  2025-10-06 03:00 (18,432 pfu → 🔴 CRITICAL, names 10000 pfu) · 2025-11-12 (SYM-H −253 but
  1,147 pfu → 🟡 ELEVATED, and forecast correctly falls 387→164→216 as the e⁻ enhancement lags).
- Streamlit `AppTest` (headless) runs the real app with NO exception in live mode AND after
  driving the actual UI widgets to the 2025-10-06 date → banner renders CRITICAL.
- Real `streamlit run` server boots: health endpoint HTTP 200, root 200.
- **Bug caught & fixed by the AppTest** (not assumed working): telemetry card crashed on a
  `None` solar-wind value at a gap timestamp (`fmt.format(None)`) → now renders "—".

Limitation (stated in UI): per-satellite "local flux" uses the global GOES-derived flux;
longitude-resolved flux awaits GRASP/GSAT-19.

### ✅ Section 8 — Real-time daemon — COMPLETE (verified against live NOAA)

`realtime_daemon.py`. Polls the 3 working NOAA feeds (see R2), applies the R4 SWPC→NCEI flux
calibration (raw SWPC preserved in the buffer), builds features via the SAME
`assemble_features()`, runs the 3 models via the SAME `payload_from_frame()` as the dashboard,
writes `data/live/latest.json`. Persistent `history.parquet` buffer for long-lag context.
Single-poll (cron) + `--loop` modes.

**Verified with REAL polls (not described — run):**
- Live poll: all 3 feeds OK, status `ok`, valid 2026-07-03 05:45Z, flux 188.7 pfu → Green.
  Real `latest.json` written (telemetry from live feeds; SYM_H `null` + flagged).
- **Failure handling tested by simulation** (not just happy path):
  - all feeds down → status `stale`, keeps last-good data FLAGGED (no crash);
  - malformed electron JSON → `stale`; partial (mag+wind down, electrons up) → `degraded`,
    still forecasts; recovery → `ok`. No path crashes.
- Dashboard shows these honestly: `stale` → red "STALE DATA - NOT current" strip; `degraded`
  → yellow reduced-inputs strip; `error` → "LIVE FEED UNAVAILABLE" page (AppTest-verified,
  no exception on missing telemetry).

### ✅ Section 9 — Bulletin generator + deliverables — COMPLETE

`bulletin_generator.py` (ASCII-safe; a Unicode-glyph crash on cp1252 was caught & fixed).
Reads live `latest.json` or a replay time. **Horizon honesty enforced in text** (matches the
UI): +30 min = "NOWCAST ONLY - ties persistence"; +6h = MODERATE; +12h = HIGH, each with the
HSS gain. Recommendations scale by level (Green routine → Red safing). Verified on 2 REAL
bulletins: live (Green/routine) and 2025-10-06 storm replay (Red/safing).

---

## Audit (2026-07-06) — full-system review, pre-packaging

Verified: all claimed artifacts exist; sentinel scan clean on every column of master/omni_2017/
GRASP (only the known, handled AE 99999); horizon alignment re-verified by timestamp post-retrain
(0 mismatches / 2000 sampled); train↔daemon feature identity PROVEN numerically (65 features ×
749,358 rows, max rel diff 5.96e-08); calibration wiring correct (#1 daemon-only, #2 no readers);
no hardcoded local paths; post-retrain AppTest + live poll pass.
Fixed during audit: created `requirements.txt` (was missing; incl. implicit pyarrow/netCDF4);
`__main__`-guarded get_omni/get_goes module-level fetch loops (root cause of the Jul-6 accidental
omni re-pull); corrected GSAT-19 longitude 83°E→48°E (R5 + calibration script + regenerated JSON,
fit numbers unchanged a=0.6051/b=1.2734); marked superseded numbers in Sections 4/6; updated
checklist. Open items moved to "Open / deferred" below.

## Final deliverables checklist (Section 9 of brief)

- [x] `features/build_features.py` — lag corr, rolling, Carrington, coupling (+ AR flux, R1)
- [x] `models/train_{30min,6h,12h}.py` — 3 XGBoost + persistence baselines (+ train_common/all)
- [x] `evaluate/` — RMSE/Pearson/HSS/POD/FAR + storm breakdown + feature-importance plots
- [x] `evaluate/grasp_validation.py` — **RUN on real ISRO GRASP data**, two windows (2017
  out-of-time + 2018 in-train); see GRASP section (+ `grasp_consolidate.py`, both calibration fitters)
- [x] `realtime_daemon.py` — live NOAA polling + shared feature pipeline + inference
- [x] `dashboard/app.py` — Streamlit dashboard (+ hazard/snapshot/charts modules)
- [x] `bulletin_generator.py` — auto Green/Yellow/Red bulletin + recommendations
- [x] `PROGRESS.md` — running log

**Open / deferred (updated 2026-07-07, after Stage A):**
- ~~GRASP validation blocked~~ → DONE on real data (two windows).
- ~~Flux calibration cross-check~~ → DONE (R4; provisional, re-fit on next storm).
- ~~30-min short-trend feature experiment~~ → DONE (R7): modest gain, nowcast ceiling confirmed.
- ~~Hyperparameter search~~ → DONE (R11): depth-5 adopted at +30min/+6h, small real gains.
- ~~Live +6h/+12h skill below offline (no real-time AE/SYM-H)~~ → **LARGELY CLOSED (R10)**:
  SYM_H now sourced from real-time Kyoto Dst, recovers ~100% of the SYM_H-attributable skill.
- Still open: **AE_INDEX** has no real-time source → NaN live (~2-3% of gain; small) —
  re-confirmed 2026-07-08 (Stage D): Kyoto's digital realtime-AE repo is monthly batches
  with ~1–5 wk latency; SuperMAG SME is a research API, not an operational feed;
  GRASP energy-channel equivalence unconfirmed (caveat [A]); R4 calibration refresh on a second
  storm; the 27-day Carrington + longest SYM_H lags still mature over the live buffer's ~40 days.

**Stage A additions (2026-07-07):** new scripts `models/experiment_storm_weight.py` (R9),
`models/experiment_hyperparam.py` (R11); new artifacts `evaluate/storm_weight_experiment.json`,
`evaluate/hyperparam_search.json`. Config now carries `STORM_WEIGHTS` + `PER_HORIZON_PARAMS` in
`models/train_common.py`. Dashboard Model Performance panel regenerated
(`python evaluate/build_model_performance_panel.py`) against these models.

## Stage C-2 — deployment readiness (2026-07-07)

Prepared for Streamlit Community Cloud (push + deploy are the user's outward-facing steps):
- **Blocker resolved:** `features_master.parquet` (198 MB) exceeds GitHub's 100 MB file limit and
  is needed by `load_context`. Fixed by shipping a **20 MB recent slice**
  `features/features_replay_sample.parquet` (2025-06-01→end, all demo dates, identical columns);
  `dashboard/snapshot.py:load_context` uses the full matrix if present else the slice. The full
  parquet is now git-untracked (`git rm --cached`) + ignored.
- **Repo trimmed:** `.gitignore` now excludes offline-only large data (`datasets/`,
  `data/grasp/raw/`, `GRASP_data.zip`, `omni_2017.csv`, `scratchpad/`). A full `git add -A`
  stages ~59 MB / 51 files, no file >100 MB.
- **Verified deploy-ready:** cloud-mode render test (full parquet forced absent → slice only) —
  app.py renders in BOTH Live and Replay with no exception, Model Performance panel present.
  requirements.txt complete; no secrets; no absolute paths; daemon singleton + atomic writes (R12).
- **Friend-facing note:** `SHARE_NOTE.md` (fill in the URL post-deploy).
- **Remaining (user):** commit + push to the public GitHub repo, then deploy at share.streamlit.io
  pointing at `dashboard/app.py`; free tier sleeps after inactivity (wake button restores it, and
  the daemon thread + live data resume on wake).

## Stage C-3 — deploy fix + responsive/polish pass (2026-07-08)

**Deploy crash fixed (real Streamlit Cloud traceback):** `xgb.XGBRegressor()` in
`snapshot.load_context` raised `ImportError: sklearn needs to be installed` on Cloud — the
xgboost sklearn wrapper requires scikit-learn as a separate package, present locally only
transitively. Added `scikit-learn>=1.5` to `requirements.txt` (commit a9cb855). Lesson noted:
the cloud-mode render test runs in the local env and cannot catch missing-package gaps.

**UI responsive/bugfix pass (visual only — data, hazard logic, model calls unchanged),
verified with real headless-Edge screenshots at 1920/1366/1024/768/430 px, Live AND Replay:**
- **Responsive:** content columns (hero 3-col, model-perf 2-col) stack full-width ≤1200 px via
  media queries on `stHorizontalBlock`/`stColumn` (`:has(stRadio)` exempts the control strip,
  which collapses to natural-width wrapping chips ≤900 px); telemetry cards auto-grid
  (`repeat(auto-fit,minmax(190px,1fr))`); metric tables scroll inside `.mp-wrap`
  (never the page); HUD strips are `flex-wrap` with a `flex:1 1 240px` message.
- **Root-caused overlap bug:** Streamlit 1.58 puts `margin-bottom:-16px` on every
  `stMarkdownContainer` (compensates a trailing `<p>` margin raw-HTML markdown doesn't have) —
  the LAST label in a column bled 16 px onto the next stacked column on narrow screens.
  Fixed by zeroing that margin only on `:last-child` element containers (internal spacing
  untouched). Diagnosed via live computed-style probes, not guesswork.
- **Header/alignment:** title no longer hidden under Streamlit's fixed chrome (header bg
  matched + 3.2 rem top padding); control strip vertically centered
  (`vertical_alignment="center"` + flex); ↻ button was unstyled because 1.58 renamed button
  DOM (`.stButton>button` no longer matches) → all buttons now targeted via
  `[data-testid=stButton]/[stDownloadButton] button`, uniform 2.1 rem height.
- **Report button:** HUD restyle — 1 px border, sharp corners, mono uppercase, no shadow,
  hover brightens border only.
- **Stray red removed via `.streamlit/config.toml`** (`primaryColor #8fa3c8`): default
  Streamlit red radio dot / IST toggle / focus rings are now cool gray-blue. Functional red
  untouched — replay of 2025-10-06 re-screenshotted: CRITICAL strip, red markers, red map dot
  all render exactly as designed. (Stale-feed strip red kept: it is a data-integrity alarm.)
- **Fonts:** `--muted` contrast raised `#7c8aa8→#93a2c2` (~4:1→~5.5:1 on bg), `.lbl`
  0.62→0.7 rem, table cells 0.78→0.8 rem, widget labels 0.8 rem mono, `white-space:nowrap` on
  radio labels/buttons (were wrapping mid-word at 768 px).
- **General polish:** BaseWeb selectbox/date/time inputs sharp-cornered on panel surface in
  mono; plotly modebar moved vertical-right + muted (was overlapping the legend); forecast
  legend anchored to grow upward into a raised top margin (was spilling over the plot on
  narrow widths); removed dead `.banner` CSS.
- **GEO map fixed (follow-up, same day):** natural-earth projection cropped lon range
  [-110,150] with a curved edge → flat **equirectangular, full world [-180,180]/[-90,90]**,
  wireframe styling unchanged; fixed 220 px height + overflow guard on all plotly containers
  (map/panels can never scroll internally). Verified programmatically (panel scrollWidth ==
  clientWidth, no page h-scroll) + screenshots at 1920/1366/768/430 px — whole world edge to
  edge, all GEO slots visible, no internal scroll.
- **Regression:** cloud-mode AppTest re-run post-changes — Live + Replay render, no exception,
  perf panel present.

## Stage D — full data-integrity re-verification of the Stage-A accuracy work (2026-07-08)

**Context:** an accuracy-refinement request re-specified Stage A's items (30-min trend
features, storm ROC/jump features, storm weighting, live-gap closure, hyperparameter grid,
coupling honesty note). All six were found already implemented, adopted, committed (4aaafb4)
and pushed as R7–R11 + the Step-7 note — confirmed against the artifacts, not the log
(metrics_section5.json, feature_manifest.json, train_common.py STORM_WEIGHTS/PER_HORIZON_PARAMS,
panel JSON all match the R7–R11 numbers exactly). **Judgment call: verify, don't re-run** —
the experiments are seed-fixed/deterministic and re-running them would only reproduce the
adopted results while risking artifact drift. This session therefore executed the
verification-and-closure items fresh against the FINAL committed models:

**1. Train↔daemon feature identity — re-PROVEN numerically (not asserted):** rebuilt all
features from the raw master CSV through the daemon/GRASP `assemble_features()` path and
compared against the stored training matrix: **83/83 columns × 749,358 rows, worst relative
diff 5.96e-08 (float32 storage epsilon), 0 NaN-mask mismatches**. Per-horizon counts
79/81/78 confirmed from the manifest.

**2. Horizon-target timestamp alignment:** y_log_{h}(t) vs log_flux(t+h) by TIMESTAMP lookup,
4,000 sampled rows per horizon — **0 mismatches / 12,000, max |diff| = 0.0**.

**3. Committed models reproduce committed metrics — bit-exact:** loading xgb_{h}.json and
predicting the chronological test split reproduces the frozen test_predictions_{h}.parquet
with max |pred−frozen| = 0.0 at every horizon, and every number in metrics_section5.json,
metrics_summary.json, and model_performance_panel.json to float precision. Storm-subset
(SYM-H<−50) metrics recomputed and confirmed: 30min 495 pfu / HSS 0.757, 6h 833 pfu /
HSS 0.650 (persist 0.474), 12h 1035 pfu / HSS 0.660 (persist 0.387).

**4. Dashboard integrity — every displayed number traces to real current data:** code audit
of app.py/charts.py/hazard.py found **no hardcoded metric values** — the only numeric
literals are the documented operational thresholds (1000/10000 pfu), skill-band cutoffs,
and CSS/styling; all metrics flow from model_performance_panel.json and
metrics_section5/metrics_summary via `snapshot.load_context()`. **Single-payload consistency
confirmed:** hazard strip, telemetry cards, sparkline, forecast hero, asset cards, bulletin,
and report download all render the ONE payload dict loaded per rerun (no panel shows a
different moment); checked the live payload programmatically — hazard peak_flux == max(now,
forecasts), satellites' local_flux == telemetry flux, observed series ends at valid_time.
**Live end-to-end re-verified today:** fresh daemon poll 2026-07-08 13:23Z — all 4 feeds OK
(incl. Kyoto Dst→SYM_H), status ok, 273.4 pfu → Green. Headless AppTest: Live renders the
fresh payload with the panel-JSON HSS/GRASP numbers present; Replay of 2025-10-06 renders
CRITICAL; no exceptions.

**5. AE_INDEX live-gap re-investigated (item-4 residual) — still no wireable source, now with
evidence:** Kyoto WDC began publishing *digital* real-time-AE values (Dec 2024) at
`wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir/`, but they are **monthly WDC-format batches
with ~1–5 week publication latency** (measured today: the June 2026 batch appeared
2026-07-08; May appeared Jun 12; no July directory exists) — useless for current-time
features, and terms are non-commercial monitoring only. SuperMAG SME (AE-equivalent) is a
registered research API, not an operational real-time feed. **Decision: AE stays NaN live**,
per R10. Quantified impact unchanged: AE-derived features are 0.24% / 1.92% / 3.07% of
per-horizon gain, i.e. the residual live deficit is ~2–3% of gain at +6h/+12h only
(SYM_H — the driver that matters — is live via Dst, ~100% recovered).

**6. model_performance_panel.json regenerated against the final models:** output
**byte-identical except `generated_utc`** — confirming the committed panel already reflected
the final Stage-A models (it was built 4 min after the last retrain), now re-stamped from a
fresh run.

No model, feature, or hazard-logic changes were made this session (none were warranted by
the verification). Changed files: this log, the re-stamped panel JSON, and the fresh live
payload/buffer from today's poll.
