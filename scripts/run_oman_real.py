#!/usr/bin/env python3
"""
Run the real-geometry comparison (Table 4 in the paper) on the two Omani
OpenStreetMap street networks: EARP-Gait vs a thermally-naive DDTA-ACO
transplant vs a reactive baseline, 24 missions per site at 44 C, with
Wilcoxon significance. Writes results/results_oman_real.json.

Usage:
    python scripts/run_oman_real.py
"""
import os, sys, json
import numpy as np
from scipy.stats import ranksums

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from earp_gait import model
from earp_gait.osm_env import OSMEnvironment
from earp_gait.planner import (Surrogate, Mission, plan_earp, plan_reactive,
                               evaluate_plan, ACOPlanner)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "osm")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "results_oman_real.json")
SITES = ["muscat_mutrah", "salalah_haffa"]


def naive_plan(env, robot, mission, surrogate, seed):
    aco = ACOPlanner(robot, env, surrogate, seed=seed, use_surrogate=True,
                     use_safety=False, use_elite=False, thermal_aware=False, mean_only=False)
    return aco.solve(mission)


def run_site(key, n_missions=24, T_amb=44, seed0=100):
    env = OSMEnvironment(os.path.join(DATA, f"{key}.json"), seed=1, max_nodes=58)
    rob = model.Robot()
    sur = Surrogate().fit(rob, env, np.random.default_rng(1), n=3000)
    rng = np.random.default_rng(seed0)
    N = env.n_nodes
    res = {"EARP-Gait": [], "DDTA-ACO (naive)": [], "Reactive": []}
    D = np.linalg.norm(env.pos[:, None, :] - env.pos[None, :, :], axis=2)
    for k in range(n_missions):
        s = int(rng.integers(N)); far = np.argsort(D[s])[-N // 3:]; d = int(rng.choice(far))
        if s == d:
            continue
        m = Mission(source=s, dest=d, alpha0=0.7, T_amb=T_amb, base_density=0.5,
                    surge=float(rng.uniform(0, 0.2)))
        _, _, r, _ = plan_earp(robot=rob, env=env, mission=m, surrogate=sur, seed=k + 1)
        _, _, r2, _ = naive_plan(env, rob, m, sur, seed=k + 1)
        rrt, rmd, _ = plan_reactive(rob, env, m)
        rr = evaluate_plan(rob, env, m, rrt, rmd, {})
        res["EARP-Gait"].append(r); res["DDTA-ACO (naive)"].append(r2); res["Reactive"].append(rr)
    return env, res


def main():
    out = {"meta": {"T_amb": 44, "base_density": 0.5, "n_missions": 24, "source": "OpenStreetMap",
                     "note": ("real street geometry from OpenStreetMap; pedestrian density is a synthetic "
                              "hand-authored time-varying field (hash-seeded diurnal oscillation with hotspots "
                              "at real POIs), NOT fitted or calibrated to any pedestrian dataset")}}
    for key in SITES:
        env, res = run_site(key)
        site = {"label": env.label, "methods": {}}
        for meth, rs in res.items():
            site["methods"][meth] = {
                "succ": 100 * np.mean([r["feasible"] for r in rs]),
                "energy": float(np.mean([r["energy"] for r in rs])),
                "energy_sd": float(np.std([r["energy"] for r in rs])),
                "time": float(np.mean([r["time"] for r in rs])),
                "fit": float(np.mean([r["fitness"] for r in rs])),
                "therm": 100 * np.mean([r["thermal_fail"] for r in rs]),
            }
        fe = [r["fitness"] for r in res["EARP-Gait"]]
        site["sig"] = {m: float(ranksums(fe, [r["fitness"] for r in res[m]]).pvalue)
                       for m in ["DDTA-ACO (naive)", "Reactive"]}
        out[key] = site
        em = site["methods"]["EARP-Gait"]; nm = site["methods"]["DDTA-ACO (naive)"]
        print(f"{env.label}: EARP succ={em['succ']:.0f}% therm={em['therm']:.0f}% | "
              f"naive succ={nm['succ']:.0f}% therm={nm['therm']:.0f}% | p={site['sig']['DDTA-ACO (naive)']:.1e}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
