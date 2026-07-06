"""
Social-force pedestrian simulator (Helbing & Molnar 1995, Phys Rev E 51:4282 [15]).

Pedestrians are simulated as point agents whose acceleration is the sum of:
  (1) a driving force toward a goal at a desired speed,
  (2) repulsive interaction forces from other pedestrians (exponential, Eq. 3 of [15]),
  (3) repulsive forces from walls / obstacles.
We run the crowd on the SAME metric node graph the robot plans on: agents travel
between crowded points of interest, and their positions are aggregated into a
time-varying per-EDGE density that the planner already consumes via
env.crowd_density(). Reciprocal, collision-avoiding motion of the ORCA family [16]
is the same phenomenon this pairwise-repulsion model produces at the density level.

Design contract: this REPLACES the hand-authored density field as the source of
env.crowd_density(); nothing else in the planner or the thermal model changes.
"""
from __future__ import annotations
import numpy as np


# Helbing-Molnar 1995 parameters (their reported values)
V0_MEAN   = 1.34      # desired speed (m/s), Helbing 1995
V0_SD     = 0.26      # speed spread
TAU       = 0.5       # relaxation time (s)
A_PED     = 2.1       # interaction strength (m/s^2) toward the ped-ped force scale
B_PED     = 0.3       # interaction range (m)
A_WALL    = 10.0      # wall repulsion strength
B_WALL    = 0.2       # wall range (m)
PED_RADIUS= 0.3       # pedestrian radius (m)
LAMBDA    = 0.35      # anisotropy: forces from ahead weigh more (Helbing 2000 extension)


class SocialForceCrowd:
    """Runs a Helbing social-force crowd on an Environment's node geometry and
    aggregates agent positions into a per-edge density field."""

    def __init__(self, env, n_agents=120, seed=0, dt=0.3,
                 edge_radius=6.0, density_scale=None):
        """
        env          : Environment or OSMEnvironment (needs .pos, .edges, .crowd_nodes/.docks)
        n_agents     : number of pedestrians
        edge_radius  : metres; agents within this distance of an edge midpoint
                       count toward that edge's crowd density
        density_scale: agents-per-edge -> density normalization; if None, auto.
        """
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.edge_radius = edge_radius
        self.pos = env.pos.astype(float)
        self.N = self.pos.shape[0]

        # crowded points of interest: real POI hotspots if present, else high-degree nodes
        hotspots = getattr(env, "crowd_nodes", None)
        if not hotspots:
            deg = {i: 0 for i in range(self.N)}
            for (i, j) in env.edges:
                deg[i] += 1; deg[j] += 1
            hotspots = sorted(deg, key=deg.get, reverse=True)[: max(3, self.N // 4)]
        self.hotspots = list(hotspots)

        # edge midpoints for density aggregation
        self.edge_keys = list(env.edges.keys())
        self.edge_mid = {k: 0.5 * (self.pos[k[0]] + self.pos[k[1]]) for k in self.edge_keys}

        # spawn agents near hotspots, give each a hotspot goal
        self.na = n_agents
        self.p = np.zeros((n_agents, 2))
        self.v = np.zeros((n_agents, 2))
        self.goal = np.zeros((n_agents, 2))
        self.v0 = np.clip(self.rng.normal(V0_MEAN, V0_SD, n_agents), 0.5, 2.2)
        for a in range(n_agents):
            h = self.pos[self.rng.choice(self.hotspots)]
            self.p[a] = h + self.rng.normal(0, 4.0, 2)
            g = self.pos[self.rng.choice(self.hotspots)]
            self.goal[a] = g + self.rng.normal(0, 3.0, 2)

        # site bounds (for gentle boundary containment)
        self.lo = self.pos.min(0) - 5.0
        self.hi = self.pos.max(0) + 5.0
        self.density_scale = density_scale
        self._t = 0.0

    def _driving_force(self):
        d = self.goal - self.p
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        e = d / dist
        desired = self.v0[:, None] * e
        return (desired - self.v) / TAU

    def _ped_repulsion(self):
        # pairwise exponential repulsion (Helbing-Molnar Eq. 3), with anisotropy
        diff = self.p[:, None, :] - self.p[None, :, :]      # (a,b,2): from b to a
        dist = np.linalg.norm(diff, axis=2)                  # (a,b)
        np.fill_diagonal(dist, np.inf)
        dhat = diff / (dist[:, :, None] + 1e-9)
        mag = A_PED * np.exp((2 * PED_RADIUS - dist) / B_PED)   # (a,b)
        # anisotropy: agents react more to what's in front of them
        vnorm = np.linalg.norm(self.v, axis=1, keepdims=True) + 1e-9
        heading = self.v / vnorm
        cos = -(dhat * heading[:, None, :]).sum(axis=2)         # cos angle to interactant
        w = LAMBDA + (1 - LAMBDA) * 0.5 * (1 + cos)
        f = (mag * w)[:, :, None] * dhat
        return f.sum(axis=1)

    def _boundary_force(self):
        f = np.zeros_like(self.p)
        for dim in (0, 1):
            dlo = self.p[:, dim] - self.lo[dim]
            dhi = self.hi[dim] - self.p[:, dim]
            f[:, dim] += A_WALL * np.exp(-dlo / B_WALL)
            f[:, dim] -= A_WALL * np.exp(-dhi / B_WALL)
        return f

    def step(self, n=1):
        for _ in range(n):
            F = self._driving_force() + self._ped_repulsion() + self._boundary_force()
            self.v += F * self.dt
            sp = np.linalg.norm(self.v, axis=1, keepdims=True)
            cap = 2.5
            self.v = np.where(sp > cap, self.v / sp * cap, self.v)
            self.p += self.v * self.dt
            # re-goal agents that arrived, to keep the crowd alive
            arrived = np.linalg.norm(self.goal - self.p, axis=1) < 2.0
            for a in np.where(arrived)[0]:
                g = self.pos[self.rng.choice(self.hotspots)]
                self.goal[a] = g + self.rng.normal(0, 3.0, 2)
            self._t += self.dt

    def edge_counts(self):
        """Number of agents within edge_radius of each edge midpoint."""
        counts = {}
        P = self.p
        for k, mid in self.edge_mid.items():
            d = np.linalg.norm(P - mid, axis=1)
            counts[k] = int((d < self.edge_radius).sum())
        return counts

    def density_field(self):
        """Per-edge crowd density in ~[0, 1.4], normalized from agent counts."""
        counts = self.edge_counts()
        if self.density_scale is None:
            mx = max(counts.values()) if counts else 1
            self.density_scale = max(1.0, mx / 0.9)   # busiest edge ~ 0.9 density
        return {k: float(np.clip(c / self.density_scale, 0.0, 1.4))
                for k, c in counts.items()}


def build_density_snapshots(env, n_agents=120, seed=0, warmup=40, n_snapshots=12,
                            steps_between=6, dt=0.3, edge_radius=6.0):
    """Run the crowd, return a list of per-edge density dicts sampled over time.
    warmup lets the crowd reach a realistic spatial distribution first."""
    crowd = SocialForceCrowd(env, n_agents=n_agents, seed=seed, dt=dt, edge_radius=edge_radius)
    crowd.step(warmup)
    snaps = []
    for _ in range(n_snapshots):
        snaps.append(crowd.density_field())
        crowd.step(steps_between)
    return snaps, crowd


class SocialForceDensity:
    """Adapter that makes an Environment read its per-edge density from precomputed
    social-force snapshots, matching the env.crowd_density(i,j,t,base,surge) signature.

    time t (seconds) is mapped to a snapshot index; base_density scales the field
    (so the same mild/stress base levels as the hand-authored model still apply),
    surge is added on top."""

    def __init__(self, snapshots, snapshot_period_s=30.0, fallback=0.15):
        self.snaps = snapshots
        self.period = snapshot_period_s
        self.fallback = fallback

    def __call__(self, i, j, t, base_density=0.3, surge=0.0):
        key = (min(i, j), max(i, j))
        idx = int((t / self.period)) % len(self.snaps)
        raw = self.snaps[idx].get(key, self.fallback)
        # base_density acts as a global crowd-level multiplier (mild vs stress),
        # keeping comparability with the hand-authored model's base levels.
        d = (base_density / 0.3) * raw + surge
        return float(np.clip(d, 0.0, 1.4))
