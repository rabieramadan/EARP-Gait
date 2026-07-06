# EARP-Gait
# EARP-Gait

**Anticipatory Energy- and Thermal-Aware Whole-Body Locomotion Planning for Legged Robots in Hot, Crowded Environments**

Code and data accompanying the paper submitted to the *Intelligent Computing* (SPJ/AAAS)
special issue **"Intelligent Computing for Embodied AI and Robotics: Foundations and Platform Technologies."**

EARP-Gait generalizes the data-driven, traffic-aware ant colony optimizer **DDTA-ACO**
(originally for electric-vehicle charge-route planning) to an embodied legged robot that
must cross a dynamic, human-shared space on a limited battery, choosing not only *where* to
go and *when to recharge* but also *which gait to use* — under the thermal stress of a hot
climate. The study is **simulation-only** and grounded in the climate of Oman, with two
scenarios built on **real OpenStreetMap street geometry** (Mutrah, Muscat; Al-Haffa, Salalah).

---

## What this repository contains

```
earp-gait/
├── earp_gait/              # the package
│   ├── model.py            # robot, energy/thermal model, bilayer Environment
│   ├── planner.py          # gait-conditioned surrogate, MMAS engine, EARP-Gait, baselines
│   ├── osm_env.py          # build a bilayer Environment from real OpenStreetMap geometry
│   └── experiments.py      # full experiment harness (E1–E7)
├── scripts/
│   ├── run_experiments.py  # reproduce the synthetic study  -> results/results.json
│   ├── run_oman_real.py    # reproduce the real-graph study -> results/results_oman_real.json
│   ├── fetch_osm.py        # (re)fetch OSM geometry for the two sites
│   └── make_figures.py     # regenerate results-derived figure panels
├── data/osm/               # shipped OpenStreetMap geometry (ODbL; see LICENSE_OSM.md)
├── results/                # JSON result files used in the paper
├── figures/                # the 10 paper figures (300 dpi PNG)
├── docs/
│   ├── manuscript.docx         # the journal-formatted manuscript (Intelligent Computing template)
│   ├── cover_letter.md / .docx # cover letter to the editors
│   ├── mathematical_model.md   # the formal model (33 numbered equations)
│   └── references.md           # the 22-entry reference list with DOIs
├── requirements.txt / pyproject.toml
├── LICENSE (MIT, code)  ·  CITATION.cff
```

## Method in one paragraph

The environment is a **bilayer graph**: an upper topological layer of docking stations and
points of interest over a lower metric layer of road/footway segments carrying length,
slope, roughness, surface type, and a time-varying human-density field. The robot has a
discrete **gait set** (slow, cruise, fast, roll) with gait- and terrain-dependent cost of
transport. A temperature-dependent **energy/thermal model** couples ambient heat to battery
capacity, cooling power, and charging efficiency; a *(segment, gait)* pair fails when the
required cooling power exceeds what the cooling system can sustain. A **gait-conditioned
surrogate** (a data-driven world model, the analogue of DDTA-ACO's CRTE travel-time model)
predicts energy, time, thermal risk, and clearance for each candidate move. A **Max–Min Ant
System** engine jointly constructs the route, per-segment gait, and recharge schedule inside
a receding-horizon loop, with a safety/thermal constraint learner (RPA generalization) and
elite intensification (EAS generalization). The planner minimizes a normalized three-term
objective — battery **energy**, **time**, and a comfort/clearance **penalty** — the same
structure as DDTA-ACO with the monetary charging-expense term replaced by battery energy.

Full equations: [`docs/mathematical_model.md`](docs/mathematical_model.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # optional: install the package
```
Dependencies are only `numpy`, `scipy`, `matplotlib` (the surrogate is a hand-rolled
ridge regression — no scikit-learn needed). Python ≥ 3.9.

## Reproduce the results

```bash
# 1) synthetic study (E1–E7): 5 environments x 24 missions per condition, ~18 min CPU
python scripts/run_experiments.py            # or --quick for a fast smoke test

# 2) real-geometry study (Table 4): 24 missions per site at 44 C
python scripts/run_oman_real.py
```

All randomness is seeded (`RNG_MASTER = 20260706` in `experiments.py`; fixed seeds in the
real-graph script), so runs are deterministic on a given NumPy version. Both scripts were
run end-to-end from this packaged layout: the synthetic run (n_envs=5, n_missions=24,
~20 min) reproduces the main-comparison table exactly (e.g. EARP-Gait 5.29 Wh / 0% thermal
failures vs DDTA-ACO-naive 8.17 Wh / 100% at the stress condition), and the real-graph run
reproduces the real-network table exactly (Muscat *p* = 2.5×10⁻⁷, Salalah *p* = 1.2×10⁻⁷).

## Headline result

At the hot/dense stress condition (44 °C), EARP-Gait keeps **0% thermal-derating failures**
while a naive DDTA-ACO transplant that ignores the thermal coupling fails on **88–100%** of
missions and yields worse overall plan quality — and the effect reproduces on the two
**real Omani street networks** (Wilcoxon *p* < 0.001 at both sites). Energy per mission is
comparable at Muscat and markedly lower for EARP-Gait at Salalah (6.5 vs 11.8 Wh); the
decisive and consistent gap across both sites is thermal safety. As heat rises, EARP-Gait
progressively abandons fast gaits, which is the mechanism behind the safety margin.

**Anticipation vs reaction.** A *thermal-reactive* baseline — a competent planner that
downgrades gait only *after* it observes a segment overheating — also reaches 0% thermal
failures, so the benefit is not merely respecting the thermal limit. EARP-Gait still reaches
a significantly better objective (1.42 vs 1.77, *p* < 0.001) using ~27% less energy
(5.29 vs 7.25 Wh), because anticipating the thermal cost before committing to a route and
gait mix beats throttling after the fact.

## Honest scope of the grounding

This is a **simulation proof-of-concept**, not an empirically-calibrated digital twin. To be
explicit about what is real and what is modeled:

| Component | Status |
|---|---|
| Street/footway geometry, surface types, POIs | **Real** (OpenStreetMap) |
| Ambient-temperature range | **Real** (Omani climate, ~25–48 °C) |
| Pedestrian *motion* / crowd density | **Synthetic** hand-authored field — *not fitted or calibrated to any pedestrian dataset* |
| Terrain slope | **Synthetic** (no matched elevation model) |
| Charging-dock locations | **Synthetic**, placed at real intersections (OSM has no charger data here) |
| Robot / battery / thermal parameters | **Representative model values**, not a specific hardware unit |

No open Omani pedestrian-trajectory dataset exists; fitting the crowd and terrain models to
measured data is stated as future work.

## Data & licenses

- **Code:** MIT (`LICENSE`).
- **OpenStreetMap geometry** in `data/osm/`: © OpenStreetMap contributors, ODbL v1.0 — see
  [`data/osm/LICENSE_OSM.md`](data/osm/LICENSE_OSM.md). Attribution and share-alike apply to
  derived databases.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper if you use the code or benchmark.

## Acknowledgment

The planning backbone generalizes DDTA-ACO (Wang et al., *Intelligent Computing*, 2026,
doi:10.34133/icomputing.0361). See [`docs/references.md`](docs/references.md) for the full list.
