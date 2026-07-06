"""
EARP-Gait — energy/thermal model and bilayer environment.

Implements the mathematical model from the manuscript:
  - gait-conditioned cost-of-transport  E_loco
  - temperature-coupled thermal term    E_thermal, with COP(T_amb) falloff
  - auxiliary power                      E_aux
  - temperature-dependent capacity       Q(T_amb) and charging factor kappa(T_amb)

All quantities SI unless noted. Energy reported in Wh for readability.
Deterministic given a numpy Generator, so experiments are reproducible.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Robot parameters (loosely based on a ~15 kg quadruped class, e.g. Unitree Go2)
# ----------------------------------------------------------------------
@dataclass
class Robot:
    M: float = 15.0            # mass (kg)
    g: float = 9.81            # gravity
    Q_nom: float = 0.6         # nominal usable battery energy (kWh) -> 600 Wh
    P_aux: float = 40.0        # auxiliary electronics power (W)
    eta_drive: float = 0.70    # drivetrain efficiency (fraction of P_mech that is useful)
    hA: float = 2.5            # lumped heat-transfer coefficient * area (W/K)
    T_target: float = 35.0     # pack/electronics setpoint (C)
    T_max: float = 55.0        # pack thermal limit (C) for derating
    P_cool_max: float = 60.0   # max electrical cooling power the pack can sustain (W)
    P_charger: float = 300.0   # dock charger electrical power (W)

# ----------------------------------------------------------------------
# Locomotion modes: (gait, speed band) -> nominal speed and a base COT.
# Cost of transport is dimensionless: energy per (weight * distance).
# Fast gaits move quicker but cost more per metre and dump more waste heat.
# ----------------------------------------------------------------------
GAITS = ["slow", "cruise", "fast", "roll"]
# base COT on flat smooth ground, nominal speed (m/s)
GAIT_BASE = {
    "slow":   dict(cot=0.45, v=0.4),
    "cruise": dict(cot=0.60, v=0.9),
    "fast":   dict(cot=0.95, v=1.6),
    "roll":   dict(cot=0.25, v=1.2),   # wheeled-rolling: cheap but fails on rough terrain
}

def cot(mode: str, slope: float, roughness: float, surface: str) -> float:
    """Gait/terrain-dependent cost of transport (dimensionless)."""
    base = GAIT_BASE[mode]["cot"]
    # slope penalty (uphill positive slope in radians): linear in tan(slope)
    slope_pen = 3.0 * max(0.0, np.tan(slope))
    # roughness penalty (0..1): rough terrain costs more, roll is worst-hit
    rough_pen = (2.0 if mode == "roll" else 0.8) * roughness
    # soft-surface penalty
    soft_pen = {"pave": 0.0, "tile": 0.0, "grass": 0.25, "sand": 0.6}.get(surface, 0.0)
    if mode == "roll":
        soft_pen *= 2.0
    return base * (1.0 + slope_pen + rough_pen + soft_pen)

def gait_feasible(mode: str, slope: float, roughness: float, surface: str) -> bool:
    """Roll mode is infeasible on steps/rough/soft ground and steep slopes."""
    if mode == "roll":
        if roughness > 0.35 or surface in ("grass", "sand") or abs(slope) > 0.12:
            return False
    if abs(slope) > 0.45:   # too steep for any gait in this model
        return False
    return True

# ----------------------------------------------------------------------
# Temperature couplings — all simple, monotonic, documented forms.
# ----------------------------------------------------------------------
def COP(T_amb: float) -> float:
    """Cooling coefficient of performance: falls as ambient rises.
    ~3.5 at 25C down toward ~1.4 at 48C (linear, floored)."""
    return max(0.8, 3.5 - 0.09 * (T_amb - 25.0))

def Q_capacity(robot: Robot, T_amb: float) -> float:
    """Usable capacity (Wh) shrinks at temperature extremes (inverted-U around ~25C)."""
    wh = robot.Q_nom * 1000.0
    derate = 1.0 - 0.004 * abs(T_amb - 25.0) - 0.003 * max(0.0, T_amb - 40.0)
    return wh * max(0.6, derate)

def kappa_charge(T_amb: float) -> float:
    """Charging efficiency/throttle factor (<=1); BMS throttles above ~38C."""
    return max(0.45, 1.0 - 0.035 * max(0.0, T_amb - 38.0))

# ----------------------------------------------------------------------
# Segment energy & time
# ----------------------------------------------------------------------
def segment_energy_time(robot: Robot, mode: str, length: float, slope: float,
                        roughness: float, surface: str, T_amb: float):
    """Return (E_Wh, dt_s, waste_heat_W, pack_over_limit_bool) for one segment."""
    v = GAIT_BASE[mode]["v"]
    dt = length / v                                   # traversal time (s)
    c = cot(mode, slope, roughness, surface)
    E_loco_J = c * robot.M * robot.g * length         # mechanical energy (J)
    P_mech = E_loco_J / dt                            # mechanical power (W)

    # waste heat + environmental heat gain
    Q_gen = (1.0 - robot.eta_drive) * P_mech + robot.P_aux
    Q_env = robot.hA * (T_amb - robot.T_target)       # >0 when hot
    P_cool = max(0.0, Q_gen + Q_env) / COP(T_amb)
    E_thermal_J = P_cool * dt
    E_aux_J = robot.P_aux * dt

    E_total_J = E_loco_J + E_thermal_J + E_aux_J
    E_Wh = E_total_J / 3600.0

    # thermal-derating flag: the pack overheats when the electrical cooling power
    # REQUIRED to reject (waste heat + environmental gain) exceeds what the cooling
    # system can sustain. Because COP(T_amb) falls with heat and Q_gen grows with
    # faster/harder gaits, high-heat + fast-gait combinations breach this capacity
    # while slow/cruise gaits at the same ambient stay within it.
    P_cool_required = P_cool
    over_limit = P_cool_required > robot.P_cool_max
    return E_Wh, dt, Q_gen, over_limit

# ----------------------------------------------------------------------
# Bilayer environment
# ----------------------------------------------------------------------
@dataclass
class Environment:
    n_nodes: int = 24
    seed: int = 0
    rng: np.random.Generator = field(default=None)
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._build()

    def _build(self):
        rng = self.rng
        n = self.n_nodes
        # node positions on a 100x100 m site
        self.pos = rng.uniform(0, 100, size=(n, 2))
        # designate ~1/6 of nodes as docking stations
        self.docks = set(rng.choice(n, size=max(2, n // 6), replace=False).tolist())
        # build a connected geometric graph (k nearest neighbours)
        self.edges = {}   # (i,j) -> attributes
        k = 4
        for i in range(n):
            d = np.linalg.norm(self.pos - self.pos[i], axis=1)
            nn = np.argsort(d)[1:k + 1]
            for j in nn:
                j = int(j)
                key = (min(i, j), max(i, j))
                if key in self.edges:
                    continue
                length = float(d[j]) + 1e-6
                slope = float(rng.normal(0, 0.06))            # radians
                roughness = float(np.clip(rng.beta(2, 6), 0, 1))
                surface = rng.choice(["pave", "tile", "grass", "sand"], p=[0.5, 0.2, 0.2, 0.1])
                self.edges[key] = dict(length=length, slope=slope,
                                       roughness=roughness, surface=str(surface))
        self.adj = {i: [] for i in range(n)}
        for (i, j) in self.edges:
            self.adj[i].append(j)
            self.adj[j].append(i)

    def edge_attr(self, i, j):
        return self.edges[(min(i, j), max(i, j))]

    def crowd_density(self, i, j, t, base_density=0.3, surge=0.0):
        """Time-varying human density on edge (i,j) in [0, ~1]. Deterministic in (i,j)."""
        key = (min(i, j), max(i, j))
        phase = (hash(key) % 1000) / 1000.0 * 2 * np.pi
        # diurnal-like oscillation + optional surge
        d = base_density * (0.6 + 0.4 * np.sin(0.001 * t + phase)) + surge
        return float(np.clip(d, 0, 1.2))


if __name__ == "__main__":
    # sanity checks: energy rises with slope, heat, and faster gaits
    r = Robot()
    flat = segment_energy_time(r, "cruise", 20, 0.0, 0.1, "pave", 25)[0]
    uphill = segment_energy_time(r, "cruise", 20, 0.15, 0.1, "pave", 25)[0]
    hot = segment_energy_time(r, "cruise", 20, 0.0, 0.1, "pave", 48)[0]
    fastg = segment_energy_time(r, "fast", 20, 0.0, 0.1, "pave", 25)[0]
    assert uphill > flat, "uphill must cost more"
    assert hot > flat, "hot ambient must cost more"
    assert fastg > flat, "fast gait must cost more per segment energy"
    assert COP(48) < COP(25), "COP must fall with heat"
    assert Q_capacity(r, 48) < Q_capacity(r, 25), "capacity must derate in heat"
    print(f"flat={flat:.3f}  uphill={uphill:.3f}  hot={hot:.3f}  fast={fastg:.3f} Wh")
    print(f"COP25={COP(25):.2f} COP48={COP(48):.2f}  "
          f"Q25={Q_capacity(r,25):.0f} Q48={Q_capacity(r,48):.0f} Wh  "
          f"kappa48={kappa_charge(48):.2f}")
    print("all sanity checks passed")
