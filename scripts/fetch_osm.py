#!/usr/bin/env python3
"""
Fetch real OpenStreetMap street geometry for the two benchmark sites
(Mutrah, Muscat; Al-Haffa, Salalah) via the Overpass API and write the raw
JSON into data/osm/.

The repository already ships these files, so you only need this script to
re-fetch or to add new sites. Requires network access to overpass-api.de.

Usage:
    python scripts/fetch_osm.py
"""
import json, os, time, urllib.request

SITES = {
    # key : (south, west, north, east)  -- bounding box in lat/lon
    "muscat_mutrah": (23.610, 58.558, 23.628, 58.580),   # Mutrah Corniche & Souq, Muscat
    "salalah_haffa": (17.010, 54.075, 17.030, 54.105),   # Al-Haffa Souq & Corniche, Salalah
}
LABELS = {
    "muscat_mutrah": "Mutrah Corniche & Souq, Muscat",
    "salalah_haffa": "Al-Haffa & Corniche, Salalah",
}
ENDPOINT = "https://overpass-api.de/api/interpreter"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "osm")


def fetch(bbox, retries=4):
    q = f'[out:json][timeout:60];(way["highway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););out geom;'
    for attempt in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=q.encode(),
                                         headers={"User-Agent": "earp-gait-benchmark"})
            d = json.load(urllib.request.urlopen(req, timeout=90))
            return [e for e in d.get("elements", []) if e.get("type") == "way" and e.get("geometry")]
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(8)
    raise RuntimeError("Overpass fetch failed after retries")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, bbox in SITES.items():
        print(f"Fetching {key} ...")
        ways = fetch(bbox)
        out = {"bbox": list(bbox), "label": LABELS[key], "ways": ways}
        path = os.path.join(OUT_DIR, f"{key}.json")
        json.dump(out, open(path, "w"))
        print(f"  {len(ways)} ways -> {path}")
        time.sleep(3)


if __name__ == "__main__":
    main()
