"""
SolarSentinel - GRASP/GSAT-19 consolidation (Part 1).

Real ISRO GSAT-19 GRASP payload data, Jul 2017 - Aug 2018, delivered as nested
zips (GRASP_data.zip -> {01,02}.zip -> per-day grasp_5_min_avg_DD-MON-YYYY.zip
-> .txt + .xml + .png). NOT FITS/PDS4 (the original brief's assumption was wrong).

Steps:
  1. Extract every daily .txt/.xml into data/grasp/raw/ (ignore .png), de-duping
     the days present in both batches (identical content -> keep one copy).
  2. Parse all daily .txt -> one table Time / electron_flux / proton_flux
     (fractional day-of-year + year-from-filename -> UTC timestamp, 5-min grid).
  3. Parse .xml sidecars -> per-day metadata (activity level, flare/CME, missing%).
  4. Save data/grasp/grasp_master.parquet (+ .csv) and grasp_metadata.csv over the
     FULL 425-day range (no window restriction here).
"""

import io
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_ZIP = ROOT / "GRASP_data.zip"
RAW = ROOT / "data" / "grasp" / "raw"
OUT_DIR = ROOT / "data" / "grasp"


def log(m):
    print(m, flush=True)


def extract():
    """Extract daily .txt/.xml to RAW/, de-duping by filename across batches."""
    RAW.mkdir(parents=True, exist_ok=True)
    seen, dupes, wrote = set(), 0, 0
    with zipfile.ZipFile(SRC_ZIP) as z:
        for batch in z.namelist():
            if not batch.lower().endswith(".zip"):
                continue
            zb = zipfile.ZipFile(io.BytesIO(z.read(batch)))
            for daily in zb.namelist():
                if not daily.lower().endswith(".zip"):
                    continue
                dz = zipfile.ZipFile(io.BytesIO(zb.read(daily)))
                for member in dz.namelist():
                    low = member.lower()
                    if not (low.endswith(".txt") or low.endswith(".xml")):
                        continue                      # ignore .png
                    name = Path(member).name
                    if name in seen:
                        dupes += 1
                        continue
                    seen.add(name)
                    (RAW / name).write_bytes(dz.read(member))
                    wrote += 1
    days = len({n for n in seen if n.lower().endswith(".txt")})
    log(f"[extract] wrote {wrote} files ({days} unique days), "
        f"skipped {dupes} duplicate members")
    return days


def _date_from_name(name):
    # grasp_5_min_avg_13-FEB-2018.txt -> datetime(2018,2,13)
    token = name.replace(".txt", "").replace(".xml", "").split("grasp_5_min_avg_")[-1]
    return datetime.strptime(token.title(), "%d-%b-%Y")


def parse_txt():
    rows = []
    bad = 0
    for f in sorted(RAW.glob("*.txt")):
        try:
            d = _date_from_name(f.name)
            year_base = pd.Timestamp(d.year, 1, 1)
            df = pd.read_csv(f, sep="\t", skiprows=1, header=None,
                             usecols=[0, 1, 2],
                             names=["doy", "electron_flux", "proton_flux"])
            df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["doy"])
            t = year_base + (df["doy"] - 1.0) * pd.Timedelta(days=1)
            df["Time"] = t.dt.round("5min")
            rows.append(df[["Time", "electron_flux", "proton_flux"]])
        except Exception as e:                        # noqa: BLE001
            bad += 1
            log(f"[txt] FAIL {f.name}: {type(e).__name__}: {e}")
    master = pd.concat(rows).dropna(subset=["Time"]).sort_values("Time")
    before = len(master)
    master = master.drop_duplicates("Time").set_index("Time")
    log(f"[txt] parsed {len(rows)} files, {before:,} rows -> "
        f"{len(master):,} unique 5-min timestamps ({bad} bad files)")
    return master


def _xml_text(root, tag):
    el = root.iter(tag)
    for e in el:
        return (e.text or "").strip()
    return None


def parse_xml():
    recs = []
    for f in sorted(RAW.glob("*.xml")):
        try:
            root = ET.fromstring(f.read_text(encoding="utf-8", errors="replace").strip())
            recs.append({
                "date": _xml_text(root, "Date_of_observation"),
                "doy": _xml_text(root, "Day_of_the_year"),
                "electron_activity": _xml_text(root, "Electron_Activity_level"),
                "proton_activity": _xml_text(root, "Proton_Activity_level"),
                "flare": _xml_text(root, "Flare_association"),
                "cme": _xml_text(root, "CME_association"),
                "missing_data": _xml_text(root, "Missing_data"),
                "missing_pct": _xml_text(root, "Percentage_of_missing_data"),
                "units": _xml_text(root, "Units_of_flux"),
            })
        except Exception as e:                        # noqa: BLE001
            log(f"[xml] FAIL {f.name}: {type(e).__name__}: {e}")
    meta = pd.DataFrame(recs)
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
    meta["missing_pct"] = pd.to_numeric(meta["missing_pct"], errors="coerce")
    meta = meta.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    log(f"[xml] parsed {len(meta)} sidecars")
    log(f"[xml] electron_activity values: "
        f"{meta['electron_activity'].value_counts().to_dict()}")
    log(f"[xml] flare=Yes: {(meta['flare']=='Yes').sum()}  "
        f"cme=Yes: {(meta['cme']=='Yes').sum()}  "
        f"missing_data=Yes: {(meta['missing_data']=='Yes').sum()}")
    return meta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extract()
    master = parse_txt()
    meta = parse_xml()

    master.to_parquet(OUT_DIR / "grasp_master.parquet")
    master.to_csv(OUT_DIR / "grasp_master.csv")
    meta.to_csv(OUT_DIR / "grasp_metadata.csv", index=False)

    log("\n=== GRASP master summary ===")
    log(f"span: {master.index.min()} -> {master.index.max()}")
    log(f"rows: {len(master):,}")
    ef = master["electron_flux"]
    log(f"electron_flux: min {ef.min():.3g}  median {ef.median():.3g}  "
        f"mean {ef.mean():.3g}  max {ef.max():.3g}")
    log(f"electron_flux quantiles 50/90/99/max: "
        f"{ef.quantile(0.5):.2f} / {ef.quantile(0.9):.2f} / "
        f"{ef.quantile(0.99):.2f} / {ef.max():.2f}")
    yrs = master.index.year.value_counts().sort_index().to_dict()
    log(f"rows per year: {yrs}")
    overlap = master[master.index >= "2018-01-01"]
    log(f"rows in 2018+ training-overlap window: {len(overlap):,} "
        f"({overlap.index.normalize().nunique()} days)")
    log(f"wrote grasp_master.parquet/.csv, grasp_metadata.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
