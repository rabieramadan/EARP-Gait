# Cover Letter

**To:** The Guest Editors, Special Issue on *Intelligent Computing for Embodied AI and Robotics: Foundations and Platform Technologies*
**Journal:** *Intelligent Computing* (a Science Partner Journal; Zhejiang Lab / AAAS)

**Date:** [DATE]

---

Dear Guest Editors (Prof. Darwin Caldwell, Prof. Fei Chen, Prof. Ming Fang, Prof. Yanhe Zhu, Prof. Jie Zhao, and Prof. Toshio Fukuda),

We are pleased to submit our manuscript, **"EARP-Gait: Anticipatory Energy- and Thermal-Aware Whole-Body Locomotion Planning for Legged Robots in Hot, Crowded Environments,"** for consideration in the special issue on *Intelligent Computing for Embodied AI and Robotics*.

**Motivation and fit with the special issue.** The call asks for work that connects world and predictive models to reasoning, decision-making, and control for embodied agents, validated through simulation infrastructure and evaluated with attention to safety. Our paper contributes exactly this: a data-driven *world model* (a gait-conditioned surrogate that predicts energy, time, thermal risk, and human clearance) embedded inside a receding-horizon decision engine that plans the route, the per-segment locomotion mode, and the recharge schedule of an energy-constrained legged robot. We deliver it as an **open, reproducible simulation platform and benchmark**, and we make **thermal safety a first-class evaluation axis** — a dimension embodied-AI benchmarks rarely test but that is decisive for real deployment in hot climates.

**Core idea and novelty.** We start from a recent advance in a neighboring field — traffic-aware, data-driven ant colony optimization for electric-vehicle charge-route planning (DDTA-ACO; Wang et al., *Intelligent Computing*, 2026) — and identify a structural analogy: an energy-limited robot navigating a dynamic, human-shared space is the embodied counterpart of an electric vehicle in dynamic traffic. We transfer three mechanisms across this analogy — a data-driven travel-cost surrogate, an infeasibility-repair/avoidance learner, and elite intensification — and, critically, we extend the objective from monetary charging expense to **battery energy** and add a **temperature-dependent energy/thermal coupling** with gait choice as a new decision variable. This coupling is the heart of the contribution: it lets the planner *anticipate* thermal overload and trade gait for safety before a failure occurs, rather than reacting to it.

**Key findings.** In a simulation study spanning five procedurally generated environments and two scenarios built on **real OpenStreetMap street geometry** for crowded districts of Muscat and Salalah, Oman, EARP-Gait sustains **zero thermal-derating failures** at a 44 °C stress condition, whereas a thermally-blind transplant of the baseline optimizer fails on 88–100% of missions. The effect is statistically significant (Wilcoxon *p* < 0.001) and reproduces on both real street networks, with EARP-Gait also using substantially less energy at the larger Salalah site. Crucially, against a thermally *reactive* baseline that throttles gait only after detecting overheating — and which therefore also avoids failures — EARP-Gait still attains a significantly better objective using about 27% less energy, isolating the value of anticipation over reaction. An ablation isolates which mechanisms matter; a temperature sweep reveals the underlying behavior — the planner progressively abandons fast gaits as heat rises.

**Scope and integrity.** We are explicit that this is a **simulation proof-of-concept**, not an empirically calibrated digital twin. Street geometry and the ambient-temperature range are real; pedestrian motion, terrain slope, charging-dock placement, and robot/thermal parameters are modeled. We state these boundaries plainly in the manuscript so that the contribution — the planning method and the thermal-coupling mechanism — is not overclaimed. All code, data, the formal mathematical model, and scripts to reproduce every figure and table are released openly.

**Originality.** This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no conflicts of interest. We suggest the manuscript is well matched to the special issue's emphasis on foundation/world models for embodied decision-making, open simulation platforms, and safety-aware evaluation.

Thank you for considering our work. We would be glad to address any questions.

Sincerely,

**[CORRESPONDING AUTHOR NAME]**
on behalf of all authors
[AFFILIATION]
[EMAIL]
