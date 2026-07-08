# SolarSentinel — what this link is

**Live demo:** <PASTE YOUR streamlit.app URL HERE AFTER DEPLOY>

It's just a website — **no install, no sign-up.** Click the link and it opens.

## What it does
SolarSentinel watches "space weather" — bursts of high-energy electrons from the Sun
that build up static charge on satellites in orbit and can damage them. It reads live
data from NOAA's space-weather satellites and **forecasts the radiation hazard at three
lead times: 30 minutes, 6 hours, and 12 hours ahead**, so satellite operators would get
a heads-up before a dangerous build-up.

## What you'll see
- A **status light** (Green / Yellow / Red) for the current radiation hazard at
  geostationary orbit.
- The **forecast** for +30 min / +6 h / +12 h, next to a "do nothing" baseline so you can
  see where the model actually adds skill.
- Live readings (solar wind, magnetic field, storm index) and a map of GEO satellites.
- A **Replay** mode — pick a past date and watch how it would have called a real storm
  (try the pre-set October 2025 storm — it goes Red).

## How fresh is the data?
It pulls new NOAA data **every ~5 minutes** on its own. If nobody has opened it for a while,
the free hosting puts it to sleep — you'll see a **"Yes, get this app back up!"** button;
click it and it's live again within a few seconds.

## Honesty note (it's a research demo, not an operational system)
At the 30-minute lead the model essentially ties the simple "assume no change" baseline —
the panel says so plainly. Its real edge is at **+6 h and +12 h**, and especially during
storms, which is exactly when a warning matters.
