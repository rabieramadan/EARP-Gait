# OpenStreetMap data attribution

The files `muscat_mutrah.json` and `salalah_haffa.json` in this directory contain
street geometry derived from **OpenStreetMap**.

- © OpenStreetMap contributors.
- OpenStreetMap data is available under the **Open Database License (ODbL) v1.0**:
  https://www.openstreetmap.org/copyright and https://opendatacommons.org/licenses/odbl/1-0/
- Any produced work or derived database that uses this geometry must attribute
  OpenStreetMap and remain compatible with the ODbL.

The bounding boxes fetched are:
- Mutrah Corniche & Souq, Muscat: 23.610, 58.558, 23.628, 58.580
- Al-Haffa & Corniche, Salalah:   17.010, 54.075, 17.030, 54.105

Re-fetch with `python scripts/fetch_osm.py` (requires network access to
overpass-api.de).

**Note on charging stations.** OpenStreetMap does not provide EV/robot charging
infrastructure for these areas. The docking stations used in the benchmark are
placed *synthetically* at well-connected real intersections (with an enforced
minimum spacing); they are not surveyed charger locations.
