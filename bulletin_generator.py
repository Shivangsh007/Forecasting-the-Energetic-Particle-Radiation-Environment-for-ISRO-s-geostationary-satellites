"""
SolarSentinel - Section 9: auto-generated hazard bulletin.

Turns a dashboard payload (live data/live/latest.json, or a replay time) into a
plain-text operator bulletin: hazard level, current conditions, per-horizon
forecast, and recommendations. ASCII-only for portability (cp1252 consoles, .txt).

HORIZON HONESTY (matches the dashboard's visual weighting): the text must NOT
imply the same forecasting confidence at +30 min as at +6h/+12h. +30 min ties
persistence (Section 6 HSS), so it is labelled a NOWCAST, not a skillful forecast.

CLI:
  python bulletin_generator.py                             # from live latest.json
  python bulletin_generator.py --time "2025-10-06 03:00"   # historical replay
  python bulletin_generator.py --out bulletin.txt
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LATEST = ROOT / "data" / "live" / "latest.json"
import sys
sys.path.insert(0, str(ROOT / "dashboard"))
from hazard import HORIZON_LABELS  # noqa: E402

# per-horizon confidence language keyed off validated Section-6 skill
SKILL_PHRASE = {
    "none": "NOWCAST ONLY - statistically ties persistence (no skill gain over "
            "'no change'); a current-state indicator, not a lead-time forecast",
    "moderate": "MODERATE confidence - validated skill above persistence",
    "high": "HIGH confidence - strong validated skill above persistence",
}

RECOMMENDATIONS = {
    "Green": ["Routine monitoring. No mitigation required.",
              "Continue nominal operations."],
    "Yellow": ["Heightened monitoring of deep-dielectric-charging-susceptible units.",
               "Defer non-critical high-voltage operations and non-essential maneuvers.",
               "Brief operators; prepare to safe vulnerable subsystems if flux rises."],
    "Red": ["SAFE / power down charging-susceptible subsystems where feasible.",
            "Postpone maneuvers and high-voltage operations.",
            "Increase monitoring cadence; enable anomaly-response procedures.",
            "Notify mission stakeholders of elevated SEU / discharge risk."],
}


def _fmt(v, nd=0):
    return "n/a" if v is None else f"{v:,.{nd}f}"


def build_bulletin(payload):
    L = []
    W = 64
    L.append("=" * W)
    L.append(" SOLARSENTINEL - ENERGETIC ELECTRON HAZARD BULLETIN")
    L.append(" GEO >2 MeV electron flux | BAH 2026 | Team SolarSentinel")
    L.append("=" * W)

    status = payload.get("status", "ok")
    if status in ("stale", "error"):
        L.append(f" ** {status.upper()}: "
                 f"{payload.get('stale_reason') or payload.get('error','')} **")
        L.append(" ** Data below may not be current. **")
        L.append("-" * W)
    elif status == "degraded":
        down = [f for f, ok in payload.get("feeds_ok", {}).items() if not ok]
        L.append(f" ** DEGRADED: driver feed(s) down ({', '.join(down)}); "
                 f"+6h/+12h skill reduced. **")
        L.append("-" * W)

    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L.append(f" Issued : {issued}")
    L.append(f" Valid  : {payload['valid_time'].replace('T', ' ')} "
             f"[source: {payload.get('source', '?')}, status: {status.upper()}]")
    L.append("")

    hz = payload["hazard"]
    L.append(f" HAZARD LEVEL: [*] {hz['title']} ({hz['level'].upper()})")
    L.append(f"   {hz['message']}")
    if hz.get("driver_horizon") in ("now", "30min"):
        L.append("   NOTE: this level reflects current / near-current conditions "
                 "(a nowcast);")
        L.append("         skillful lead-time warning comes from the +6h / +12h forecasts.")
    L.append("")

    t = payload["telemetry"]
    L.append(" CURRENT CONDITIONS")
    L.append(f"   >2 MeV flux   : {_fmt(t['flux_2MeV']['value'])} pfu")
    L.append(f"   Solar wind    : {_fmt(t['flow_speed']['value'])} km/s")
    L.append(f"   IMF Bz        : {_fmt(t['BZ_GSM']['value'], 1)} nT")
    L.append(f"   SYM-H         : {_fmt(t['SYM_H']['value'], 0)} nT")
    L.append("")

    L.append(" FORECAST  (3 trained horizons - confidence DIFFERS by horizon)")
    for h in ["30min", "6h", "12h"]:
        f = payload["forecast"][h]
        L.append(f"   {HORIZON_LABELS[h]:>7} : {_fmt(f['flux']):>7} pfu  "
                 f"(range {_fmt(f['lo'])}-{_fmt(f['hi'])})")
        L.append(f"            {SKILL_PHRASE[f['skill']]} "
                 f"[HSS gain {f['hss_gain']:+.3f}]")
    L.append("")

    L.append(" RECOMMENDATION")
    for r in RECOMMENDATIONS[hz["level"]]:
        L.append(f"   - {r}")
    L.append("")

    L.append(" NOTES")
    if payload.get("missing_inputs"):
        L.append(f"   - Live inputs unavailable: {', '.join(payload['missing_inputs'])} "
                 "(no SWPC real-time source).")
        L.append("     +6h/+12h forecasts run on reduced inputs; +30 min unaffected.")
    L.append("   - Per-satellite flux uses global GOES >2 MeV; longitude-resolved "
             "flux awaits GRASP.")
    L.append("=" * W)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default=None, help="historical replay time")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.time:
        from snapshot import build_payload
        payload = build_payload(args.time)
    else:
        if not LATEST.exists():
            raise SystemExit("No data/live/latest.json - run realtime_daemon.py first.")
        payload = json.loads(LATEST.read_text())

    text = build_bulletin(payload)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\n[bulletin] wrote {args.out}")


if __name__ == "__main__":
    main()
