"""
EARP-Gait large-scale evaluation harness.

Runs, over multiple environments x many missions x seeds:
  E1  main comparison (6 methods x baseline/stress conditions)
  E2  ablation (7 variants)
  E3  ambient-temperature sweep (Oman climate) 25/35/42/48 C
  E4  crowd-density sweep
  E5  robustness (crowd surge, dock outage)
  E6  parameter analysis (ants, rho, lambda)
  E7  surrogate calibration (predicted vs true)
Outputs JSON + CSV to results/.

A learned-policy baseline (Q-learning over a discretized state) stands in for the
RL comparator; it is trained per-environment on the same energy/thermal model.
"""
from __future__ import annotations
import numpy as np, json, time, os, itertools
from collections import Counter, defaultdict
try:
    from .model import Robot, Environment, GAITS, segment_energy_time, gait_feasible, cot
    from .planner import (Surrogate, Mission, plan_earp, plan_fixed_gait, plan_reactive, plan_thermal_reactive,
                          evaluate_plan, dijkstra, _greedy_recharge, ACOPlanner)
except ImportError:  # allow running as a plain script
    from model import Robot, Environment, GAITS, segment_energy_time, gait_feasible, cot
    from planner import (Surrogate, Mission, plan_earp, plan_fixed_gait, plan_reactive, plan_thermal_reactive,
                         evaluate_plan, dijkstra, _greedy_recharge, ACOPlanner)

RNG_MASTER = 20260706

# ----------------------------------------------------------------------
# Simple Q-learning navigation-locomotion baseline
# ----------------------------------------------------------------------
class QPolicy:
    def __init__(self, env, robot, T_amb, seed=0):
        self.env=env; self.robot=robot; self.T=T_amb
        self.rng=np.random.default_rng(seed); self.Q=defaultdict(float)
    def _actions(self, u):
        acts=[]
        for v in self.env.adj[u]:
            a=self.env.edge_attr(u,v)
            for g in GAITS:
                if gait_feasible(g,a["slope"],a["roughness"],a["surface"]):
                    acts.append((v,g))
        return acts
    def train(self, dest, episodes=400, alpha=0.5, gamma=0.9, eps=0.2):
        for ep in range(episodes):
            u=int(self.rng.integers(self.env.n_nodes)); steps=0
            while u!=dest and steps<30:
                acts=self._actions(u)
                if not acts: break
                if self.rng.random()<eps:
                    v,g=acts[self.rng.integers(len(acts))]
                else:
                    v,g=max(acts,key=lambda a:self.Q[(u,a)])
                a=self.env.edge_attr(u,v)
                E,dt,_,over=segment_energy_time(self.robot,g,a["length"],a["slope"],
                                                a["roughness"],a["surface"],self.T)
                rew = -(0.5*E+0.5*dt/30) - (3.0 if over else 0.0)
                if v==dest: rew+=10
                nb=self._actions(v)
                fut=max([self.Q[(v,na)] for na in nb],default=0.0)
                self.Q[(u,(v,g))]+=alpha*(rew+gamma*fut-self.Q[(u,(v,g))])
                u=v; steps+=1
        return self
    def plan(self, mission):
        u=mission.source; route=[u]; modes=[]; visited={u}
        for _ in range(mission.max_hops):
            if u==mission.dest: break
            acts=[a for a in self._actions(u) if a[0] not in visited]
            if not acts: break
            v,g=max(acts,key=lambda a:self.Q[(u,a)])
            route.append(v); modes.append(g); visited.add(v); u=v
        rech=_greedy_recharge(self.robot,self.env,mission,route,modes)
        return route,modes,rech

# ----------------------------------------------------------------------
# Mission generators
# ----------------------------------------------------------------------
def make_missions(env, n, seed, alpha0, T_amb, base_density, surge=0.0, min_hops=4):
    rng=np.random.default_rng(seed)
    D=np.linalg.norm(env.pos[:,None,:]-env.pos[None,:,:],axis=2)
    pairs=[(int(s),int(d)) for s,d in
           np.dstack(np.unravel_index(np.argsort(D.ravel())[::-1],D.shape))[0] if s<d]
    out=[]
    for s,d in pairs:
        path=dijkstra(env,s,d,lambda u,v:1)
        if path and len(path)>=min_hops:
            out.append(Mission(source=s,dest=d,alpha0=alpha0,T_amb=T_amb,
                               base_density=base_density,surge=surge,
                               t0=float(rng.uniform(0,5000))))
        if len(out)>=n: break
    return out

def summarize(results):
    if not results:
        return dict(n=0,succ=0,energy=np.nan,time=np.nan,fit=np.nan,
                    depl=np.nan,therm=np.nan,clear=np.nan)
    return dict(
        n=len(results),
        succ=100*np.mean([r['feasible'] for r in results]),
        energy=np.mean([r['energy'] for r in results]),
        energy_sd=np.std([r['energy'] for r in results]),
        time=np.mean([r['time'] for r in results]),
        fit=np.mean([r['fitness'] for r in results]),
        fit_sd=np.std([r['fitness'] for r in results]),
        depl=100*np.mean([r['depleted'] for r in results]),
        therm=100*np.mean([r['thermal_fail'] for r in results]),
        clear=np.mean([r['min_clear'] for r in results]),
    )

# ----------------------------------------------------------------------
# Method registry — each returns an evaluated result dict (or None)
# ----------------------------------------------------------------------
def eval_result(env, robot, mission, plan):
    if plan is None: return None
    route, modes, rech = plan
    if len(route) < 2: return None
    return evaluate_plan(robot, env, mission, route, modes, rech)

def run_methods(env, robot, sur, qpol, mission, methods, seed=1):
    out = {}
    for name in methods:
        if name == "EARP-Gait":
            _,_,r,_ = plan_earp(env=env, robot=robot, mission=mission, surrogate=sur, seed=seed,
                use_surrogate=True, use_safety=True, use_elite=True, thermal_aware=True)
        elif name == "Fixed-gait":
            r = eval_result(env, robot, mission, plan_fixed_gait(robot, env, mission))
        elif name == "Reactive":
            r = eval_result(env, robot, mission, plan_reactive(robot, env, mission))
        elif name == "Thermal-reactive":
            r = eval_result(env, robot, mission, plan_thermal_reactive(robot, env, mission))
        elif name == "w/o surrogate":
            _,_,r,_ = plan_earp(env=env, robot=robot, mission=mission, surrogate=sur, seed=seed,
                use_surrogate=False, use_safety=True, use_elite=True, thermal_aware=True)
        elif name == "DDTA-ACO (naive)":
            _,_,r,_ = plan_earp(env=env, robot=robot, mission=mission, surrogate=sur, seed=seed,
                use_surrogate=True, use_safety=False, use_elite=False, thermal_aware=False)
        elif name == "RL policy":
            r = eval_result(env, robot, mission, qpol.plan(mission))
        else:
            r = None
        out[name] = r
    return out

ABLATIONS = {
    "EARP-Gait (full)":            dict(use_surrogate=True,  use_safety=True,  use_elite=True,  thermal_aware=True,  mean_only=False),
    "w/o surrogate":               dict(use_surrogate=False, use_safety=True,  use_elite=True,  thermal_aware=True,  mean_only=False),
    "w/o safety learner":          dict(use_surrogate=True,  use_safety=False, use_elite=True,  thermal_aware=True,  mean_only=False),
    "w/o elite intensification":   dict(use_surrogate=True,  use_safety=True,  use_elite=False, thermal_aware=True,  mean_only=False),
    "mean-only surrogate":         dict(use_surrogate=True,  use_safety=True,  use_elite=True,  thermal_aware=True,  mean_only=True),
    "w/o thermal-awareness":       dict(use_surrogate=True,  use_safety=True,  use_elite=True,  thermal_aware=False, mean_only=False),
}

def wilcoxon_vs(ref_vals, other_vals):
    """Wilcoxon rank-sum (Mann-Whitney) p-value; returns nan if degenerate."""
    from scipy.stats import ranksums
    a=np.asarray(ref_vals,float); b=np.asarray(other_vals,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<2 or len(b)<2: return float('nan')
    try: return float(ranksums(a,b).pvalue)
    except Exception: return float('nan')

# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main(n_envs=5, n_missions=24, out_dir="results", quick=False):
    t0=time.time()
    os.makedirs(out_dir, exist_ok=True)
    robot=Robot()
    if quick: n_envs, n_missions = 2, 8
    env_seeds=list(range(100,100+n_envs))
    # pre-build environments + surrogates + RL policies (per env, cached by dest set)
    envs=[]; surs=[]
    for es in env_seeds:
        e=Environment(n_nodes=40, seed=es)
        s=Surrogate().fit(robot,e,np.random.default_rng(es),n=3000)
        envs.append(e); surs.append(s)
    log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
    RESULTS={}

    METHODS=["EARP-Gait","Fixed-gait","Reactive","Thermal-reactive","w/o surrogate","DDTA-ACO (naive)","RL policy"]

    # ---------- E1 main comparison: baseline (mild,low) vs stress (hot,high) ----------
    log("E1 main comparison")
    conds={"baseline":dict(T_amb=25,base_density=0.25,alpha0=0.7),
           "stress":dict(T_amb=44,base_density=0.6,alpha0=0.55)}
    E1={c:{m:[] for m in METHODS} for c in conds}
    E1_raw={c:{m:{'fit':[],'energy':[]} for m in METHODS} for c in conds}
    for ei,(env,sur) in enumerate(zip(envs,surs)):
        # RL policies: train per (env, condition) using the dominant destination set lazily
        for c,cfg in conds.items():
            miss=make_missions(env,n_missions,seed=1000+ei,**cfg)
            # train an RL policy per distinct destination (cache)
            qcache={}
            for m in miss:
                if m.dest not in qcache:
                    qcache[m.dest]=QPolicy(env,robot,cfg["T_amb"],seed=ei).train(m.dest,
                                    episodes=(150 if quick else 300))
                res=run_methods(env,robot,sur,qcache[m.dest],m,METHODS,seed=1)
                for k,v in res.items():
                    if v is not None:
                        E1[c][k].append(v); E1_raw[c][k]['fit'].append(v['fitness']); E1_raw[c][k]['energy'].append(v['energy'])
        log(f"  env {ei+1}/{len(envs)} done")
    RESULTS["E1"]={c:{m:summarize(E1[c][m]) for m in METHODS} for c in conds}
    # significance: EARP vs each other on fitness (stress condition)
    RESULTS["E1_sig"]={m:wilcoxon_vs(E1_raw["stress"]["EARP-Gait"]["fit"],
                                      E1_raw["stress"][m]["fit"]) for m in METHODS if m!="EARP-Gait"}

    # ---------- E2 ablation (stress condition) ----------
    log("E2 ablation")
    E2={v:[] for v in ABLATIONS}; E2_fit={v:[] for v in ABLATIONS}
    for ei,(env,sur) in enumerate(zip(envs,surs)):
        miss=make_missions(env,n_missions,seed=2000+ei,T_amb=44,base_density=0.6,alpha0=0.55)
        for m in miss:
            for vname,flags in ABLATIONS.items():
                _,_,r,_=plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,
                    use_surrogate=flags["use_surrogate"],use_safety=flags["use_safety"],
                    use_elite=flags["use_elite"],thermal_aware=flags["thermal_aware"],
                    mean_only=flags["mean_only"])
                if r is not None: E2[vname].append(r); E2_fit[vname].append(r['fitness'])
        log(f"  env {ei+1}/{len(envs)} done")
    RESULTS["E2"]={v:summarize(E2[v]) for v in ABLATIONS}
    RESULTS["E2_sig"]={v:wilcoxon_vs(E2_fit["EARP-Gait (full)"],E2_fit[v]) for v in ABLATIONS if v!="EARP-Gait (full)"}

    # ---------- E3 temperature sweep: EARP vs thermally-naive ----------
    log("E3 temperature sweep")
    temps=[25,35,42,48]
    E3={T:{"EARP-Gait":[], "Thermally-naive":[]} for T in temps}
    E3_modes={T:{"EARP-Gait":Counter(),"Thermally-naive":Counter()} for T in temps}
    for ei,(env,sur) in enumerate(zip(envs,surs)):
        for T in temps:
            miss=make_missions(env,n_missions,seed=3000+ei,T_amb=T,base_density=0.5,alpha0=0.55)
            for m in miss:
                _,mo1,r1,_=plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,
                    use_surrogate=True,use_safety=True,use_elite=True,thermal_aware=True)
                _,mo0,r0,_=plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,
                    use_surrogate=True,use_safety=True,use_elite=True,thermal_aware=False)
                if r1 is not None: E3[T]["EARP-Gait"].append(r1); E3_modes[T]["EARP-Gait"].update(mo1)
                if r0 is not None: E3[T]["Thermally-naive"].append(r0); E3_modes[T]["Thermally-naive"].update(mo0)
        log(f"  env {ei+1}/{len(envs)} done")
    RESULTS["E3"]={str(T):{k:summarize(E3[T][k]) for k in E3[T]} for T in temps}
    RESULTS["E3_modes"]={str(T):{k:dict(E3_modes[T][k]) for k in E3_modes[T]} for T in temps}

    # ---------- E4 crowd-density sweep ----------
    log("E4 crowd sweep")
    dens=[0.2,0.4,0.6,0.8]
    E4={d:{"EARP-Gait":[],"Reactive":[]} for d in dens}
    for ei,(env,sur) in enumerate(zip(envs,surs)):
        for d in dens:
            miss=make_missions(env,n_missions,seed=4000+ei,T_amb=38,base_density=d,alpha0=0.6)
            for m in miss:
                _,_,r1,_=plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,
                    use_surrogate=True,use_safety=True,use_elite=True,thermal_aware=True)
                r2=eval_result(env,robot,m,plan_reactive(robot,env,m))
                if r1 is not None: E4[d]["EARP-Gait"].append(r1)
                if r2 is not None: E4[d]["Reactive"].append(r2)
    RESULTS["E4"]={str(d):{k:summarize(E4[d][k]) for k in E4[d]} for d in dens}

    # ---------- E5 robustness: crowd surge & dock outage ----------
    log("E5 robustness")
    E5={}
    for scen,kw in {"nominal":dict(surge=0.0),"crowd_surge":dict(surge=0.5)}.items():
        acc={"EARP-Gait":[],"Reactive":[]}
        for ei,(env,sur) in enumerate(zip(envs,surs)):
            miss=make_missions(env,n_missions,seed=5000+ei,T_amb=42,base_density=0.5,alpha0=0.55,**kw)
            for m in miss:
                _,_,r1,_=plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,
                    use_surrogate=True,use_safety=True,use_elite=True,thermal_aware=True)
                r2=eval_result(env,robot,m,plan_reactive(robot,env,m))
                if r1 is not None: acc["EARP-Gait"].append(r1)
                if r2 is not None: acc["Reactive"].append(r2)
        E5[scen]={k:summarize(acc[k]) for k in acc}
    RESULTS["E5"]=E5

    # ---------- E6 parameter analysis ----------
    log("E6 parameter analysis")
    env,sur=envs[0],surs[0]
    miss=make_missions(env,min(12,n_missions),seed=6000,T_amb=42,base_density=0.5,alpha0=0.55)
    E6={"n_ants":{}, "rho":{}, "lam":{}}
    for na in [10,20,30,40]:
        rs=[plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,n_ants=na)[2] for m in miss]
        E6["n_ants"][str(na)]=summarize([r for r in rs if r])
    for rho in [0.05,0.1,0.3,0.5]:
        rs=[plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,rho=rho)[2] for m in miss]
        E6["rho"][str(rho)]=summarize([r for r in rs if r])
    for lam in [0.2,0.5,0.8,0.95]:
        rs=[plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,lam=lam)[2] for m in miss]
        E6["lam"][str(lam)]=summarize([r for r in rs if r])
    RESULTS["E6"]=E6

    # ---------- E7 surrogate calibration ----------
    log("E7 calibration")
    env,sur=envs[0],surs[0]; rng=np.random.default_rng(7)
    keys=list(env.edges.keys()); predE=[]; trueE=[]; predT=[]; trueT=[]
    for _ in range(600):
        i,j=keys[rng.integers(len(keys))]; a=env.edge_attr(i,j)
        m=GAITS[rng.integers(len(GAITS))]
        if not gait_feasible(m,a["slope"],a["roughness"],a["surface"]): continue
        T=rng.uniform(22,50)
        E,dt,_,_=segment_energy_time(robot,m,a["length"],a["slope"],a["roughness"],a["surface"],T)
        pe,pt,_=sur.predict(m,a["length"],a["slope"],a["roughness"],a["surface"],T)
        predE.append(pe); trueE.append(E); predT.append(pt); trueT.append(dt)
    def r2(yp,yt):
        yp=np.array(yp); yt=np.array(yt); ss=((yt-yt.mean())**2).sum()
        return float(1-((yt-yp)**2).sum()/ss)
    RESULTS["E7"]=dict(energy=dict(pred=predE,true=trueE,r2=r2(predE,trueE)),
                       time=dict(pred=predT,true=trueT,r2=r2(predT,trueT)))

    # ---------- convergence curves (for figure) ----------
    log("convergence curves")
    env,sur=envs[0],surs[0]
    miss=make_missions(env,8,seed=9000,T_amb=44,base_density=0.6,alpha0=0.55)
    conv={"EARP-Gait (full)":[], "w/o elite intensification":[]}
    for m in miss:
        _,_,_,c1=plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,use_elite=True)
        _,_,_,c0=plan_earp(env=env,robot=robot,mission=m,surrogate=sur,seed=1,use_elite=False)
        conv["EARP-Gait (full)"].append(c1); conv["w/o elite intensification"].append(c0)
    RESULTS["convergence"]=conv

    RESULTS["_meta"]=dict(n_envs=len(envs),n_missions=n_missions,
                          seconds=round(time.time()-t0,1),quick=quick)
    with open(os.path.join(out_dir,"results.json"),"w") as f:
        json.dump(RESULTS,f,indent=1,default=float)
    log(f"DONE in {time.time()-t0:.1f}s -> {out_dir}/results.json")
    return RESULTS

if __name__=="__main__":
    import sys
    q = "--quick" in sys.argv
    main(quick=q)


