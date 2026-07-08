"""
SolarSentinel - shared hazard logic + asset config (used by the dashboard AND,
later, the Section 8 daemon and bulletin generator -> single source of truth).
"""

# NOAA >2 MeV electron event threshold = 1000 pfu (deep-dielectric charging risk).
THRESH_EVENT = 1000.0      # pfu -> Elevated
THRESH_SEVERE = 10000.0    # pfu -> Critical

LEVELS = {
    "Green":  {"title": "NOMINAL",  "color": "#22c55e", "bg": "#0f2e1d"},
    "Yellow": {"title": "ELEVATED", "color": "#eab308", "bg": "#2e280a"},
    "Red":    {"title": "CRITICAL", "color": "#ef4444", "bg": "#2e1010"},
}

# Nominal GEO longitudes (deg E). Real GEO latitude ~= 0; positions are nominal slots.
SATELLITES = [
    {"name": "INSAT-3DR", "function": "Meteorology & search-and-rescue", "lon_e": 74.0},
    {"name": "INSAT-3DS", "function": "Meteorology (next-generation)",    "lon_e": 82.0},
    {"name": "GSAT-30",   "function": "C/Ku-band communications",         "lon_e": 83.0},
    {"name": "GSAT-24",   "function": "Ku-band DTH broadcast",            "lon_e": 83.0},
    {"name": "GSAT-6",    "function": "S-band mobile communications",     "lon_e": 83.0},
    {"name": "GSAT-14",   "function": "C/Ku-band communications",         "lon_e": 74.0},
    {"name": "IRNSS-1C",  "function": "NavIC navigation (GEO)",           "lon_e": 83.0},
]

HORIZON_LABELS = {"30min": "+30 min", "6h": "+6 h", "12h": "+12 h"}


def skill_from_hss_gain(gain):
    """Validated skill of XGBoost over persistence (from Section 6 HSS deltas)."""
    if gain is None or gain < 0.02:
        return "none"          # +30 min: ties persistence
    if gain < 0.10:
        return "moderate"      # +6 h
    return "high"              # +12 h


# visual weight for the forecast chart, driven by validated skill (honesty rule)
SKILL_STYLE = {
    "none":     {"opacity": 0.45, "size": 11, "label": "no skill gain vs persistence"},
    "moderate": {"opacity": 0.80, "size": 15, "label": "adds skill over persistence"},
    "high":     {"opacity": 1.00, "size": 18, "label": "adds strong skill over persistence"},
}


def flux_level(flux):
    if flux is None:
        return "Green"
    if flux >= THRESH_SEVERE:
        return "Red"
    if flux >= THRESH_EVENT:
        return "Yellow"
    return "Green"


def classify_hazard(current_flux, forecast_fluxes):
    """Overall hazard = worst of current + the three forecast fluxes.

    forecast_fluxes: {"30min": pfu, "6h": pfu, "12h": pfu}. Returns a dict with
    level/color/title/message naming the specific threshold breached.
    """
    candidates = {"now": current_flux}
    candidates.update({k: v for k, v in forecast_fluxes.items() if v is not None})
    driver = max(candidates, key=candidates.get)
    worst = candidates[driver]
    level = flux_level(worst)
    info = LEVELS[level]

    when = "currently" if driver == "now" else f"in the {HORIZON_LABELS.get(driver, driver)} forecast"
    if level == "Red":
        msg = (f">2 MeV flux reaches {worst:,.0f} pfu {when}, breaching the "
               f"{THRESH_SEVERE:,.0f} pfu CRITICAL threshold (severe deep-charging risk).")
    elif level == "Yellow":
        msg = (f">2 MeV flux reaches {worst:,.0f} pfu {when}, breaching the "
               f"{THRESH_EVENT:,.0f} pfu NOAA event threshold (elevated charging risk).")
    else:
        msg = (f">2 MeV flux stays below the {THRESH_EVENT:,.0f} pfu event threshold "
               f"across all horizons (peak {worst:,.0f} pfu).")

    return {"level": level, "color": info["color"], "bg": info["bg"],
            "title": info["title"], "message": msg,
            "threshold_pfu": (THRESH_SEVERE if level == "Red" else
                              THRESH_EVENT if level == "Yellow" else None),
            "driver_horizon": driver, "peak_flux": float(worst)}
