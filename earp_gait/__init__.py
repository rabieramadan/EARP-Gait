"""
EARP-Gait: Anticipatory Energy-Aware Whole-Body Locomotion Planning.

A data-driven, thermal-aware extension of traffic-aware ant colony optimization
(DDTA-ACO) to embodied legged robotics, evaluated in simulation grounded in
Omani climate and real OpenStreetMap street geometry.

Modules
-------
model       : robot, energy/thermal model, and the bilayer environment
planner     : gait-conditioned surrogate, MMAS ant-colony engine, EARP-Gait, baselines
osm_env     : build a bilayer Environment from real OpenStreetMap geometry
social_force: Helbing social-force pedestrian simulator -> per-edge crowd density
experiments : full experiment harness (comparison, ablation, sweeps, calibration)
"""
__version__ = "1.0.0"

from . import model, planner, osm_env, social_force  # noqa: F401
