#!/usr/bin/env python3
"""
Run the full simulation study (synthetic environments) and write
results/results.json. This reproduces experiments E1-E7 in the paper:
main comparison, ablation, temperature sweep, crowd sweep, robustness,
parameter analysis, and surrogate calibration.

Scale in the paper: 5 environments x 24 missions per condition
(~18 min on a laptop CPU). Use --quick for a fast smoke run.

Usage:
    python scripts/run_experiments.py           # full run (paper scale)
    python scripts/run_experiments.py --quick    # fast smoke run
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earp_gait.experiments import main

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    out = os.path.join(os.path.dirname(__file__), "..", "results")
    main(out_dir=out, quick=quick)
