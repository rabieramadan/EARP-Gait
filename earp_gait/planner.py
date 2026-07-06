"""
EARP-Gait planner + baselines.

Planner: MMAS ant-colony backbone with joint (node, locomotion-mode, recharge)
construction, a gait-conditioned surrogate, an RPA-style safety/thermal constraint
learner, and an EAS-style elite intensification operator, in a receding-horizon loop.

Baselines: fixed-gait shortest-energy (Dijkstra), reactive, w/o-surrogate,
and a naive DDTA-ACO transplant (ACO without the embodied thermal/safety extensions).
"""
from __future__ import annotations
import numpy as np
import heapq
from dataclasses import dataclass
try:
    from .model import (Robot, Environment, GAITS, GAIT_BASE, segment_energy_time,
                        gait_feasible, Q_capacity, kappa_charge, cot)
except ImportError:  # allow running as a plain script (python planner.py)
    from model import (Robot, Environment, GAITS, GAIT_BASE, segment_energy_time,
                        gait_feasible, Q_capacity, kappa_charge, cot)

# ----------------------------------------------------------------------
# Mission spec
# ----------------------------------------------------------------------
@dataclass
class Mission:
    source: int
    dest: int
    alpha0: float = 0.7          # initial SoC (fraction)
    T_amb: float = 25.0          # ambient temperature (C)
    base_density: float = 0.3    # crowd level
    surge: float = 0.0           # injected crowd surge
    t0: float = 0.0
    max_hops: int = 40
    clearance_min: float = 0.35  # min acceptable clearance proxy (higher density -> lower clearance)

# fitness weights (energy, time, penalty). The parent paper (DDTA-ACO) uses EQUAL
# weights (omega_e=omega_t=omega_p=1/3, its Table 1); here we deliberately depart
# from that and up-weight time so that fast gaits are genuinely tempting. That is
# what makes thermal-awareness matter in this study: a heat-blind planner takes
# fast gaits to save time and pays with thermal-derating failures. The three-term
# normalized structure is inherited from the parent objective; the weight VALUES
# are our own experimental choice, not the paper's.
W = dict(E=0.30, T=0.45, P=0.25)

def clearance(density):
    """Map crowd density -> clearance proxy in metres (more crowd, less clearance)."""
    return float(np.clip(1.2 - density, 0.05, 1.2))

# ----------------------------------------------------------------------
# Gait-conditioned surrogate.
# Trained on rolled-out segment samples to predict (energy, time) from
# (mode, length, slope, roughness, surface_code, T_amb). Lightweight ridge
# regression on nonlinear features — stands in for the learned world model.
# ----------------------------------------------------------------------
SURF_CODE = {"pave": 0.0, "tile": 0.1, "grass": 0.6, "sand": 1.0}

def _feat(mode, length, slope, roughness, surf, T_amb):
    m = [1.0 if mode == g else 0.0 for g in GAITS]
    return np.array(m + [
        length, length*max(0,np.tan(slope)), length*roughness, length*SURF_CODE[surf],
        length*max(0.0, T_amb-35.0), length, 1.0
    ])

class Surrogate:
    def __init__(self, mean_only=False):
        self.wE = None; self.wT = None; self.resid = 1.0; self.mean_only = mean_only
    def fit(self, robot, env, rng, n=1500):
        X, yE, yT = [], [], []
        keys = list(env.edges.keys())
        for _ in range(n):
            i, j = keys[rng.integers(len(keys))]
            a = env.edge_attr(i, j)
            mode = GAITS[rng.integers(len(GAITS))]
            if not gait_feasible(mode, a["slope"], a["roughness"], a["surface"]):
                continue
            T = rng.uniform(20, 50)
            E, dt, _, _ = segment_energy_time(robot, mode, a["length"], a["slope"],
                                              a["roughness"], a["surface"], T)
            X.append(_feat(mode, a["length"], a["slope"], a["roughness"], a["surface"], T))
            yE.append(E); yT.append(dt)
        X = np.array(X); yE = np.array(yE); yT = np.array(yT)
        lam = 1e-3 * np.eye(X.shape[1])
        self.wE = np.linalg.solve(X.T@X + lam, X.T@yE)
        self.wT = np.linalg.solve(X.T@X + lam, X.T@yT)
        predE = X@self.wE
        self.resid = float(np.std(yE - predE))
        return self
    def predict(self, mode, length, slope, roughness, surf, T_amb):
        f = _feat(mode, length, slope, roughness, surf, T_amb)
        E = float(f@self.wE); dt = float(f@self.wT)
        sigma = 0.0 if self.mean_only else self.resid
        return max(E, 1e-3), max(dt, 1e-3), sigma

# ----------------------------------------------------------------------
# Evaluate a full plan with the TRUE model (ground truth for fitness/feasibility)
# ----------------------------------------------------------------------
def evaluate_plan(robot, env, mission, route, modes, recharges):
    """Return dict with feasibility, energy, time, penalty, fitness, failures."""
    alpha = mission.alpha0
    Qcap = Q_capacity(robot, mission.T_amb)
    E_tot = 0.0; T_tot = 0.0; pen = 0.0
    t = mission.t0
    depleted = False; thermal_fail = False; min_clear = 1.2
    cont_time = 0.0
    for k in range(len(route)-1):
        i, j = route[k], route[k+1]
        a = env.edge_attr(i, j)
        mode = modes[k]
        if not gait_feasible(mode, a["slope"], a["roughness"], a["surface"]):
            pen += 5.0  # infeasible gait choice
        E, dt, qgen, over = segment_energy_time(robot, mode, a["length"], a["slope"],
                                                a["roughness"], a["surface"], mission.T_amb)
        # crowd effect: dense crowds slow the robot and cut clearance
        dens = env.crowd_density(i, j, t, mission.base_density, mission.surge)
        dt *= (1.0 + 0.8*dens)
        cl = clearance(dens); min_clear = min(min_clear, cl)
        if cl < mission.clearance_min:
            pen += 2.0*(mission.clearance_min - cl)
        if over:
            thermal_fail = True; pen += 3.0
        alpha -= E/Qcap
        E_tot += E; T_tot += dt; t += dt; cont_time += dt
        if alpha <= 0.0:
            depleted = True; break
        # recharge at docks
        if j in env.docks and recharges.get(k+1, 0.0) > 0:
            dref = recharges[k+1]
            add = min(dref, 1.0 - alpha)
            # charging time throttled by temperature
            charge_energy = add*Qcap
            tc = charge_energy / (robot.P_charger*kappa_charge(mission.T_amb)) * 3.6  # ~scaled
            T_tot += tc; t += tc; alpha += add; cont_time = 0.0
        # continuous-driving soft limit (4h analogue -> scaled)
        if cont_time > 3600.0:
            pen += 1.0
    feasible = (not depleted) and (route[-1] == mission.dest)
    # normalized fitness (lower is better)
    E_min, T_min, P_min = 5.0, 60.0, 1.0
    fitness = W["E"]*E_tot/E_min + W["T"]*T_tot/T_min + W["P"]*(pen+ (0 if feasible else 10))/P_min
    return dict(feasible=feasible, energy=E_tot, time=T_tot, penalty=pen,
                fitness=fitness, depleted=depleted, thermal_fail=thermal_fail,
                min_clear=min_clear, final_alpha=alpha)

# ----------------------------------------------------------------------
# Shortest-path helper (Dijkstra on a scalar edge cost)
# ----------------------------------------------------------------------
def dijkstra(env, src, dst, cost_fn):
    pq = [(0.0, src, [src])]; seen = set()
    while pq:
        c, u, path = heapq.heappop(pq)
        if u == dst: return path
        if u in seen: continue
        seen.add(u)
        for v in env.adj[u]:
            if v in seen: continue
            heapq.heappush(pq, (c + cost_fn(u, v), v, path+[v]))
    return None

# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def plan_fixed_gait(robot, env, mission, gait="cruise"):
    def cost(u, v):
        a = env.edge_attr(u, v)
        return cot(gait, a["slope"], a["roughness"], a["surface"])*a["length"]
    route = dijkstra(env, mission.source, mission.dest, cost)
    if route is None: return None
    modes = [gait]*(len(route)-1)
    recharges = _greedy_recharge(robot, env, mission, route, modes)
    return route, modes, recharges

def plan_reactive(robot, env, mission):
    """Greedy: at each node pick the neighbour minimizing immediate energy toward dest."""
    route = [mission.source]; modes = []
    cur = mission.source; visited = {cur}
    for _ in range(mission.max_hops):
        if cur == mission.dest: break
        best = None
        for v in env.adj[cur]:
            if v in visited: continue
            a = env.edge_attr(cur, v)
            # progress heuristic toward destination
            prog = (np.linalg.norm(env.pos[cur]-env.pos[mission.dest]) -
                    np.linalg.norm(env.pos[v]-env.pos[mission.dest]))
            e = cot("cruise", a["slope"], a["roughness"], a["surface"])*a["length"]
            score = e - 8.0*prog
            if best is None or score < best[0]:
                best = (score, v)
        if best is None: break
        v = best[1]; route.append(v); modes.append("cruise"); visited.add(v); cur = v
    recharges = _greedy_recharge(robot, env, mission, route, modes)
    return route, modes, recharges

def plan_thermal_reactive(robot, env, mission):
    """Non-strawman baseline: anticipation vs reaction.

    Plans a time-efficient route with the fastest feasible gait per segment
    (like a throughput-seeking planner), but REACTS to thermal overload: it
    simulates execution, and whenever a (segment, mode) pair is observed to
    overheat, it downgrades that segment to the next cooler gait and re-checks,
    repeating until the pair no longer overheats (or the coolest gait is
    reached). This captures a competent controller that throttles gait AFTER
    detecting overheating, in contrast to EARP-Gait, which anticipates the
    thermal cost BEFORE committing. Route is a shortest-energy path so that the
    only difference from EARP-Gait is anticipation, not routing.
    """
    # cooling order: fastest/hottest -> coolest
    order = ["fast", "cruise", "slow", "roll"]

    def base_cost(u, v):
        a = env.edge_attr(u, v)
        return cot("cruise", a["slope"], a["roughness"], a["surface"]) * a["length"]
    route = dijkstra(env, mission.source, mission.dest, base_cost)
    if route is None:
        return None
    modes = []
    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        a = env.edge_attr(i, j)
        # start with the fastest gait that is terrain-feasible on this segment
        chosen = None
        for g in order:
            if gait_feasible(g, a["slope"], a["roughness"], a["surface"]):
                chosen = g
                break
        if chosen is None:
            chosen = "roll"
        # REACT: if this gait is observed to overheat, downgrade to a cooler,
        # still-feasible gait until it no longer overheats.
        start = order.index(chosen) if chosen in order else 1
        for g in order[start:]:
            if not gait_feasible(g, a["slope"], a["roughness"], a["surface"]):
                continue
            over = segment_energy_time(robot, g, a["length"], a["slope"],
                                       a["roughness"], a["surface"], mission.T_amb)[3]
            chosen = g
            if not over:
                break  # reaction succeeded: stop downgrading
        modes.append(chosen)
    recharges = _greedy_recharge(robot, env, mission, route, modes)
    return route, modes, recharges


def _greedy_recharge(robot, env, mission, route, modes, target=0.9):
    """Recharge to `target` whenever passing a dock and SoC below 0.4."""
    recharges = {}
    alpha = mission.alpha0; Qcap = Q_capacity(robot, mission.T_amb)
    for k in range(len(route)-1):
        i, j = route[k], route[k+1]; a = env.edge_attr(i, j)
        E = segment_energy_time(robot, modes[k], a["length"], a["slope"],
                                a["roughness"], a["surface"], mission.T_amb)[0]
        alpha -= E/Qcap
        if j in env.docks and alpha < 0.4:
            recharges[k+1] = max(0.0, target-alpha); alpha = target
    return recharges

# ----------------------------------------------------------------------
# Ant Colony core — EARP-Gait and the naive DDTA-ACO transplant
# ----------------------------------------------------------------------
class ACOPlanner:
    """
    Unified ACO with feature flags so ablations reuse one code path.
      use_surrogate : anticipatory gait-conditioned surrogate guides mode choice
      use_safety    : RPA-style penalty field over (edge,mode) on violations
      use_elite     : EAS-style elite intensification / local mode refinement
      mean_only     : surrogate ignores predicted variance (risk-blind)
      thermal_aware : include predicted thermal cost when scoring modes
    The naive DDTA-ACO transplant = surrogate on, safety/elite off, thermal_aware off.
    """
    def __init__(self, robot, env, surrogate=None, n_ants=20, n_iter=30,
                 rho=0.1, lam=0.8, seed=0,
                 use_surrogate=True, use_safety=True, use_elite=True,
                 mean_only=False, thermal_aware=True, risk_k=1.0):
        self.robot=robot; self.env=env; self.sur=surrogate
        self.n_ants=n_ants; self.n_iter=n_iter; self.rho=rho; self.lam=lam
        self.rng=np.random.default_rng(seed)
        self.use_surrogate=use_surrogate; self.use_safety=use_safety
        self.use_elite=use_elite; self.mean_only=mean_only
        self.thermal_aware=thermal_aware; self.risk_k=risk_k
        self.tau={}; self.safety_pen={}   # pheromone & learned safety penalties
        self.tau0=1.0; self.tmin=0.1; self.tmax=5.0

    def _pher(self, i, j, mode):
        return self.tau.get((min(i,j),max(i,j),mode), self.tau0)

    def _mode_score(self, i, j, a, mission):
        """Heuristic desirability of each feasible mode on edge (i,j)."""
        scores={}
        for m in GAITS:
            if not gait_feasible(m, a["slope"], a["roughness"], a["surface"]):
                continue
            if self.use_surrogate and self.sur is not None:
                if self.thermal_aware:
                    E,dt,sig = self.sur.predict(m, a["length"], a["slope"], a["roughness"],
                                                a["surface"], mission.T_amb)
                    # anticipated thermal-derating risk for this mode at this ambient temp
                    _,_,qgen,over = segment_energy_time(self.robot, m, a["length"], a["slope"],
                                                        a["roughness"], a["surface"], mission.T_amb)
                    therm_risk = 4.0 if over else 0.0
                else:
                    # naive transplant: score modes as if the climate were mild (25 C),
                    # so it never anticipates heat -> the thermal blind spot we test
                    E,dt,sig = self.sur.predict(m, a["length"], a["slope"], a["roughness"],
                                                a["surface"], 25.0)
                    therm_risk = 0.0
                cost = 0.5*E + 0.5*dt + therm_risk + (self.risk_k*sig if not self.mean_only else 0.0)
            else:
                cost = cot(m,a["slope"],a["roughness"],a["surface"])*a["length"]
            pen = self.safety_pen.get((min(i,j),max(i,j),m),0.0) if self.use_safety else 0.0
            scores[m] = 1.0/(cost+1e-3) * np.exp(-pen)
        return scores

    def _construct(self, mission):
        cur=mission.source; route=[cur]; modes=[]; visited={cur}
        for _ in range(mission.max_hops):
            if cur==mission.dest: break
            nbrs=[v for v in self.env.adj[cur] if v not in visited]
            if not nbrs: break
            # edge desirability = pheromone * progress * best-mode heuristic
            probs=[]; cand=[]
            for v in nbrs:
                a=self.env.edge_attr(cur,v)
                ms=self._mode_score(cur,v,a,mission)
                if not ms: continue
                best_m=max(ms,key=ms.get)
                prog=(np.linalg.norm(self.env.pos[cur]-self.env.pos[mission.dest]) -
                      np.linalg.norm(self.env.pos[v]-self.env.pos[mission.dest]))
                heur=ms[best_m]*np.exp(0.15*prog)
                tau=self._pher(cur,v,best_m)
                probs.append((tau**1.0)*(heur**2.0)); cand.append((v,best_m))
            if not cand: break
            probs=np.array(probs); probs/=probs.sum()
            if self.rng.random()<self.lam:
                idx=int(np.argmax(probs))
            else:
                idx=int(self.rng.choice(len(cand),p=probs))
            v,m=cand[idx]; route.append(v); modes.append(m); visited.add(v); cur=v
        recharges=_greedy_recharge(self.robot,self.env,mission,route,modes)
        return route,modes,recharges

    def _learn_safety(self, route, modes, res):
        if not self.use_safety: return
        # RPA: on depletion/thermal/clearance failure, penalize offending edge-modes
        if res["depleted"] or res["thermal_fail"] or res["min_clear"]<0.2 or not res["feasible"]:
            for k in range(len(route)-1):
                i,j=route[k],route[k+1]; key=(min(i,j),max(i,j),modes[k])
                self.safety_pen[key]=self.safety_pen.get(key,0.0)+0.5

    def _update_pheromone(self, best_route, best_modes, best_fit):
        if best_route is None:
            return
        # global evaporation
        for key in list(self.tau.keys()):
            self.tau[key] = max(self.tmin, self.tau[key]*(1-self.rho))
        # reinforce edges of the best solution (MMAS-style, clipped)
        dep = 1.0/(best_fit+1e-3)
        for k in range(len(best_route)-1):
            i, j = best_route[k], best_route[k+1]
            key = (min(i, j), max(i, j), best_modes[k])
            self.tau[key] = float(np.clip(self.tau.get(key, self.tau0) + self.rho*dep,
                                          self.tmin, self.tmax))

    def _intensify(self, route, modes, mission):
        """EAS: local refinement — try swapping each segment's mode for a cheaper feasible one."""
        if not self.use_elite: return route,modes
        best=evaluate_plan(self.robot,self.env,mission,route,modes,
                           _greedy_recharge(self.robot,self.env,mission,route,modes))
        modes=list(modes)
        for k in range(len(modes)):
            i,j=route[k],route[k+1]; a=self.env.edge_attr(i,j)
            for m in GAITS:
                if m==modes[k] or not gait_feasible(m,a["slope"],a["roughness"],a["surface"]):
                    continue
                trial=list(modes); trial[k]=m
                r=evaluate_plan(self.robot,self.env,mission,route,trial,
                                _greedy_recharge(self.robot,self.env,mission,route,trial))
                if r["fitness"]<best["fitness"]:
                    best=r; modes[k]=m
        return route,modes

    def solve(self, mission):
        best=None; best_fit=np.inf; best_route=None; best_modes=None
        stagn=0; curve=[]
        for it in range(self.n_iter):
            it_best=None; it_fit=np.inf; it_r=None; it_m=None
            for _ in range(self.n_ants):
                route,modes,rech=self._construct(mission)
                if len(route)<2: continue
                res=evaluate_plan(self.robot,self.env,mission,route,modes,rech)
                self._learn_safety(route,modes,res)
                if res["fitness"]<it_fit:
                    it_fit=res["fitness"]; it_best=res; it_r=route; it_m=modes
            if it_best is None:
                curve.append(best_fit); continue
            if self.use_elite and it_r is not None:
                it_r,it_m=self._intensify(it_r,it_m,mission)
                it_best=evaluate_plan(self.robot,self.env,mission,it_r,it_m,
                                      _greedy_recharge(self.robot,self.env,mission,it_r,it_m))
                it_fit=it_best["fitness"]
            if it_fit<best_fit:
                best_fit=it_fit; best=it_best; best_route=it_r; best_modes=it_m; stagn=0
            else:
                stagn+=1
            self._update_pheromone(best_route,best_modes,best_fit)
            curve.append(best_fit)
            if stagn>=10: break   # early stopping
        return best_route,best_modes,best,curve


def plan_earp(robot, env, mission, surrogate, seed=0, **flags):
    aco=ACOPlanner(robot,env,surrogate,seed=seed,**flags)
    route,modes,res,curve=aco.solve(mission)
    return route,modes,res,curve

