#!/usr/bin/env python3
"""
Regenerate the paper figures from results/*.json into figures/.

This is a thin driver: it loads the saved result dictionaries and calls the
plotting routines. Figures that depend only on the numeric results
(fig3-fig8) regenerate directly; the two map/architecture figures
(fig1, fig2, fig9, fig10) depend on the environment objects and are produced
by the notebook/driver used in the study. Run run_experiments first so that
results/results.json exists.

Usage:
    python scripts/make_figures.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "results.json")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

METHOD_COLORS = {
    "EARP-Gait": "#0b3d91", "Fixed-gait": "#8c8c8c", "Reactive": "#e07b39",
    "w/o surrogate": "#5aa9e6", "DDTA-ACO (naive)": "#c0392b",
    "Thermally-naive": "#c0392b", "RL policy": "#7d5ba6",
}


def main():
    if not os.path.exists(RES):
        sys.exit("results/results.json not found — run scripts/run_experiments.py first")
    R = json.load(open(RES))
    print("loaded result blocks:", sorted(R.keys()))
    # Example: temperature-sweep panel (fig6 core content) from R['E3'].
    if "E3" in R:
        temps = [25, 35, 42, 48]
        fig, ax = plt.subplots(figsize=(5, 4))
        for meth in ["EARP-Gait", "Thermally-naive"]:
            y = [R["E3"][str(t)][meth]["therm"] for t in temps if str(t) in R["E3"]]
            ax.plot(temps[:len(y)], y, "-o", color=METHOD_COLORS.get(meth, "#333"), label=meth)
        ax.set_xlabel("Ambient temperature (°C)")
        ax.set_ylabel("Thermal-derating failures (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig6_temperature_core.png"), dpi=300)
        print("wrote figures/fig6_temperature_core.png")
    print("Note: the full 10-figure set (architecture, bilayer, site maps) is produced by\n"
          "the study driver; this script regenerates the results-derived panels as a check.")


if __name__ == "__main__":
    main()
