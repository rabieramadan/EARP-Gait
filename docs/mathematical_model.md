# EARP-Gait: Formal Mathematical Model

This document formalizes the Energy-Aware, Anticipatory, Whole-Body Locomotion Planning
problem (EARP-Gait) and the operators used to solve it. The formulation generalizes the
data-driven, traffic-aware ant colony optimization of DDTA-ACO [Wang et al. 2026,
doi:10.34133/icomputing.0361] from electric-vehicle charge-route planning to embodied
legged robotics. Notation is collected in Table N1. All symbols in the implemented
simulator (`model.py`, `planner.py`) correspond one-to-one with the equations below.

---

## N1. Bilayer environment

The environment is a bilayer graph
$$\mathcal{G} = \{\mathcal{G}_l,\ \mathcal{G}_u\}. \tag{1}$$

The **lower (metric) layer** $\mathcal{G}_l=(\mathcal{V}_l,\mathcal{E}_l)$ is a terrain graph.
Each edge $s=(i,j)\in\mathcal{E}_l$ carries a terrain attribute vector
$$\boldsymbol{\theta}_s = \big(d_s,\ \sigma_s,\ r_s,\ u_s\big), \tag{2}$$
where $d_s$ is segment length (m), $\sigma_s$ slope (rad), $r_s\in[0,1]$ roughness, and
$u_s\in\{\text{pave},\text{tile},\text{grass},\text{sand}\}$ surface type. A time-varying
**human-density field** $\rho:\mathcal{E}_l\times\mathbb{R}_{\ge 0}\to[0,\rho_{\max}]$ assigns a
crowd density $\rho_s(t)$ to each edge at time $t$.

The **upper (topological) layer** $\mathcal{G}_u=(\mathcal{V}_u,\mathcal{E}_u)$ is a roadmap over
waypoints, regions, and docking stations $\mathcal{D}\subseteq\mathcal{V}_u$; it manages the logical
sequence of destinations and recharge decisions.

---

## N2. Robot state, locomotion modes, and decisions

The robot state at decision step $k$ is
$$\mathbf{x}_k = \big(p_k,\ v_k,\ \alpha_k,\ m_k\big), \tag{3}$$
with pose $p_k$, base speed $v_k$, state of charge (SoC) $\alpha_k\in[0,1]$, and active
locomotion mode $m_k$.

A **locomotion mode** is drawn from the discrete set
$$\mathcal{M} = \mathcal{G}^{\text{gait}}\times\mathcal{B}^{\text{speed}}\times\mathcal{P}^{\text{posture}}, \tag{4}$$
instantiated here as $\mathcal{M}=\{\text{slow},\text{cruise},\text{fast},\text{roll}\}$,
each with a nominal speed $v(m)$ and base cost of transport $c_0(m)$.

At each traversed segment the planner emits three coupled decisions:
1. the next node $j$ (route selection),
2. a locomotion mode $m\in\mathcal{M}$ (gait/speed/posture),
3. a recharge amount $\Delta\alpha\ge 0$ at docking nodes $j\in\mathcal{D}$.

A **solution** is the triple
$$\Gamma = \big(R,\ \mathbf{m},\ \Pi\big), \tag{5}$$
where $R=(n_0,\dots,n_L)$ is a route with $n_0=$ source and $n_L=$ destination,
$\mathbf{m}=(m_0,\dots,m_{L-1})$ the per-segment mode schedule, and
$\Pi=\{(k,\Delta\alpha_k)\}$ the charging plan.

---

## N3. Cost of transport and segment energy

The mechanical **cost of transport** (dimensionless) of mode $m$ on segment $s$ is
$$\mathrm{COT}(m,s) = c_0(m)\,\Big(1 + \beta_\sigma \max(0,\tan\sigma_s) + \beta_r(m)\,r_s + \beta_u(m,u_s)\Big), \tag{6}$$
with slope, roughness, and surface penalty coefficients $\beta_\sigma,\beta_r,\beta_u$.
The corresponding **mechanical energy** over the segment is
$$E^{\text{loco}}_s = \mathrm{COT}(m,s)\,M\,g\,d_s, \tag{7}$$
for robot mass $M$ and gravity $g$, and the traversal time is
$$\tau_s = \frac{d_s}{v(m_s)}\big(1 + \beta_\rho\,\rho_s(t)\big), \tag{8}$$
where the factor $(1+\beta_\rho\rho_s)$ slows the robot in crowds. The instantaneous
mechanical power is $P^{\text{mech}}_s = E^{\text{loco}}_s/\tau^{0}_s$ with
$\tau^{0}_s=d_s/v(m_s)$.

---

## N4. Thermal energy model (climate coupling)

Waste and environmental heat load. The generated heat power is
$$Q^{\text{gen}}_s = (1-\eta)\,P^{\text{mech}}_s + P^{\text{aux}}, \tag{9}$$
where $\eta$ is drivetrain efficiency and $P^{\text{aux}}$ auxiliary electronics power.
The environmental heat gain against ambient temperature $T$ is
$$Q^{\text{env}}(T) = h A\,(T - T^{\text{target}}), \tag{10}$$
with lumped heat-transfer coefficient $hA$ and pack setpoint $T^{\text{target}}$.

**Cooling power and thermal energy.** The electrical cooling power required is
$$P^{\text{cool}}_s(T) = \frac{\max\!\big(0,\ Q^{\text{gen}}_s + Q^{\text{env}}(T)\big)}{\mathrm{COP}(T)}, \qquad
E^{\text{thermal}}_s = P^{\text{cool}}_s(T)\,\tau_s, \tag{11}$$
where the cooling coefficient of performance falls monotonically with heat,
$$\mathrm{COP}(T) = \max\!\big(\underline{c},\ c_0 - \gamma\,(T-T_0)\big). \tag{12}$$
Because $\mathrm{COP}(T)$ decreases while $Q^{\text{gen}}$ grows with faster/harder gaits,
$E^{\text{thermal}}$ grows **super-linearly** in $T$ and couples directly to mode choice.

**Total segment energy.**
$$\Delta E_s = E^{\text{loco}}_s + E^{\text{thermal}}_s + P^{\text{aux}}\tau_s. \tag{13}$$

**Thermal-derating constraint.** A (segment, mode) pair overheats when the required
cooling power exceeds the sustainable cooling capacity $\overline{P}^{\text{cool}}$:
$$\Phi^{\text{therm}}_s(m,T) \;=\; \mathbb{1}\!\left[\,P^{\text{cool}}_s(T) > \overline{P}^{\text{cool}}\,\right]. \tag{14}$$
This is the mechanism that makes high-heat + fast-gait combinations infeasible while
slow/cruise gaits at the same ambient remain feasible.

---

## N5. State-of-charge dynamics

Usable capacity and charging efficiency are temperature-dependent:
$$Q(T) = Q_{\text{nom}}\,\max\!\big(\underline{q},\ 1 - a\,|T-T_0| - b\max(0,T-T_1)\big), \tag{15}$$
$$\kappa(T) = \max\!\big(\underline{\kappa},\ 1 - \delta\max(0,T-T_2)\big). \tag{16}$$
SoC evolves along the plan as
$$\alpha_{k+1} =
\begin{cases}
\alpha_k - \dfrac{\Delta E_{s_k}}{Q(T)}, & \text{driving segment } s_k,\\[2ex]
\alpha_k + \Delta\alpha_k, & \text{recharge at dock } n_{k}\in\mathcal{D},
\end{cases} \tag{17}$$
and the recharge dwell time at a dock is
$$\tau^{\text{charge}}_k = \frac{\Delta\alpha_k\,Q(T)}{P^{\text{charger}}\,\kappa(T)}. \tag{18}$$

---

## N6. Objective and constraints

**Objective.** Following the weighted, normalized three-term structure of DDTA-ACO's
Eq. (1) — but replacing its monetary charging-expense term with battery energy, since a
robot's binding scarce resource is stored energy — the planner minimizes
$$f(\Gamma) = \omega_E\,\frac{E(\Gamma)}{E_{\min}} + \omega_T\,\frac{T(\Gamma)}{T_{\min}} + \omega_P\,\frac{P(\Gamma)}{P_{\min}}, \tag{19}$$
with
$$E(\Gamma)=\sum_{s\in R}\Delta E_s, \qquad
T(\Gamma)=\sum_{s\in R}\tau_s + \sum_{k}\tau^{\text{charge}}_k, \tag{20}$$
and normalization constants $E_{\min},T_{\min},P_{\min}$. Whereas DDTA-ACO uses equal
weights $\omega_e=\omega_t=\omega_p=1/3$, we deliberately up-weight time
($\omega_E,\omega_T,\omega_P$) so that fast gaits are genuinely tempting; this is the
experimental condition under which thermal-awareness matters.

**Penalty term.** The penalty aggregates soft-constraint violations,
$$P(\Gamma) = \sum_{s\in R}\Big[
\lambda_g\,\Phi^{\text{gait}}_s
+ \lambda_c\max(0,\ \underline{c}\ell - c\ell_s)
+ \lambda_t\,\Phi^{\text{therm}}_s
\Big] + \lambda_f\,\mathbb{1}[\Gamma\ \text{infeasible}], \tag{21}$$
where $\Phi^{\text{gait}}_s=\mathbb{1}[m_s\ \text{infeasible on}\ s]$ and $c\ell_s$ is human clearance.

**Constraints.**
$$0 \le \alpha_k \le 1\ \ \forall k \quad\text{(energy adequacy / SoC band)}, \tag{22}$$
$$\text{gait } m_s \text{ kinematically feasible on } \boldsymbol{\theta}_s, \tag{23}$$
$$\Pr\big[\,c\ell_s < c\ell_{\min}\,\big] \le \epsilon \quad\text{(chance constraint on human clearance)}, \tag{24}$$
$$\textstyle\sum_{s} \tau_s \le \tau^{\text{cont}}_{\max}\ \text{between recharges}\quad\text{(max continuous operation).} \tag{25}$$
Human clearance is modeled as $c\ell_s = \mathrm{clip}(c\ell_{\max}-\rho_s,\,\underline{c\ell},\,c\ell_{\max})$.

---

## N7. Gait-conditioned surrogate (world model)

The surrogate $g_\phi$ predicts, for a segment–mode–context tuple, the distribution of
energy, time, risk, and density:
$$g_\phi\big(\boldsymbol{\theta}_s, m, t, T, \mathbf{x}\big) \mapsto
\big(\hat{\mu}^E, \hat{\sigma}^E,\ \hat{\mu}^\tau,\ \hat{r}^{\text{risk}},\ \hat{\rho}\big). \tag{26}$$
It is trained self-supervised on simulator rollouts by ridge regression on nonlinear
features $\varphi(\cdot)$:
$$\mathbf{w}^E = \arg\min_{\mathbf{w}} \sum_n \big(\varphi_n^\top\mathbf{w} - E_n\big)^2 + \Lambda\lVert\mathbf{w}\rVert^2, \tag{27}$$
with predictive residual $\hat{\sigma}^E = \mathrm{std}(E_n - \varphi_n^\top\mathbf{w}^E)$ used as the
risk term in the (risk-aware) planner. A **mean-only** surrogate sets $\hat{\sigma}^E=0$.

---

## N8. Ant-colony search (MMAS backbone)

**Mode desirability.** For each feasible mode $m$ on edge $s$, the anticipated cost is
$$\hat{\psi}_s(m) = w_E\,\hat{\mu}^E_{s,m} + w_\tau\,\hat{\mu}^\tau_{s,m}
+ \underbrace{\lambda_t\,\Phi^{\text{therm}}_s(m,T)}_{\text{thermal risk (if aware)}}
+ \underbrace{\kappa_r\,\hat{\sigma}^E_{s,m}}_{\text{risk (if not mean-only)}}, \tag{28}$$
and the heuristic desirability is
$$\eta_s(m) = \frac{1}{\hat{\psi}_s(m)+\varepsilon}\,\exp\!\big(-\pi_{s,m}\big), \tag{29}$$
where $\pi_{s,m}\ge 0$ is the learned safety penalty (Sec. N9). A **thermally-naive**
planner scores modes as if $T=T_0$ (mild), i.e. $\Phi^{\text{therm}}\equiv 0$ in (28).

**Transition rule.** Ant $a$ at node $i$ selects candidate $(j,m^\star)$ with
$m^\star=\arg\max_m \eta_{(i,j)}(m)$, using the pseudo-random-proportional rule
$$(j,m^\star) =
\begin{cases}
\arg\max_{(j,m)} \big[\tau_{ij,m}\big]\big[\eta_{ij}(m)\,e^{\zeta\,\Delta_{ij}}\big]^{\!\varsigma}, & q<\lambda,\\[1ex]
\text{sample} \ \propto\ \big[\tau_{ij,m}\big]\big[\eta_{ij}(m)\,e^{\zeta\,\Delta_{ij}}\big]^{\!\varsigma}, & q\ge\lambda,
\end{cases} \tag{30}$$
where $\tau_{ij,m}$ is pheromone, $\Delta_{ij}$ the progress toward the goal, $q\sim U[0,1]$,
and $\lambda$ the exploitation probability.

**Pheromone update (MMAS).** After each iteration, only the global-best solution
$\Gamma^\ast$ deposits, and pheromone is bounded to $[\tau_{\min},\tau_{\max}]$:
$$\tau_{ij,m} \leftarrow \mathrm{clip}\!\Big((1-\rho)\,\tau_{ij,m} + \rho\,\tfrac{1}{f(\Gamma^\ast)+\varepsilon}\,\mathbb{1}[(i,j,m)\in\Gamma^\ast],\ \tau_{\min},\ \tau_{\max}\Big). \tag{31}$$

---

## N9. Experience-learning operators

**Safety / thermal constraint learning (RPA generalization).** Whenever a rollout
violates a margin — depletion, thermal derating, clearance breach, or infeasibility —
the operator increments a penalty field over the offending (edge, mode) tuples:
$$\pi_{s,m} \leftarrow \pi_{s,m} + \Delta\pi\quad
\text{if } \big(\text{depleted} \lor \Phi^{\text{therm}}_s \lor c\ell_s<\underline{c\ell} \lor \neg\text{feasible}\big). \tag{32}$$
Via (29) this proactively steers subsequent ants away from patterns of failure (e.g.
fast trotting on a sun-exposed slope at midday).

**Elite mode intensification (EAS generalization).** For the iteration-best route, the
operator locally refines the mode schedule by coordinate descent over $\mathcal{M}$:
$$\mathbf{m}' = \arg\min_{m'_k\in\mathcal{M}}\ f\big(R,\ (m_0,\dots,m'_k,\dots,m_{L-1}),\ \Pi\big),\quad k=0,\dots,L-1, \tag{33}$$
accepting each swap that lowers $f$. This intensifies search around high-quality joint
plans at low computational cost.

---

## N10. Receding-horizon anticipatory loop

Rather than a single offline plan, EARP-Gait replans on a rolling horizon $H$:

1. **Predict:** query $g_\phi$ over the next $H$ seconds.
2. **Optimize:** solve (19) with the ant-colony engine (N8) + operators (N9), yielding $\Gamma^\ast$.
3. **Execute:** apply the first segment $(n_0\!\to\!n_1, m_0, \Delta\alpha_1)$.
4. **Observe:** update $\mathbf{x}$ from realized energy/time/crowd.
5. Shift the horizon and repeat until $n=$ destination.

This operationalizes real-time interaction and local re-planning — future directions of
DDTA-ACO — as the central control mechanism.

---

## Table N1 — Symbol reference

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| $\mathcal{G}_l,\mathcal{G}_u$ | lower/upper graph layers | $\alpha_k$ | state of charge |
| $\boldsymbol{\theta}_s$ | terrain attributes of segment $s$ | $\rho_s(t)$ | human density on $s$ at $t$ |
| $\mathcal{M}$ | locomotion-mode set | $m_s$ | mode chosen on $s$ |
| $\mathrm{COT}(m,s)$ | cost of transport | $\Delta E_s$ | total segment energy |
| $\mathrm{COP}(T)$ | cooling coefficient of performance | $\overline{P}^{\text{cool}}$ | cooling capacity |
| $Q(T),\kappa(T)$ | capacity, charge efficiency | $\Phi^{\text{therm}}_s$ | thermal-derating indicator |
| $f(\Gamma)$ | objective | $\omega_E,\omega_T,\omega_P$ | objective weights |
| $g_\phi$ | gait-conditioned surrogate | $\hat{\sigma}^E$ | predictive energy std (risk) |
| $\tau_{ij,m}$ | pheromone | $\pi_{s,m}$ | learned safety penalty |
| $\rho$ | pheromone evaporation | $\lambda$ | exploitation probability |

*All functional forms in §N3–N5 are simple, monotonic, and documented, so the model is
transparent; the contribution is the coupling and anticipation of thermal cost within a
whole-body locomotion planner, not an opaque thermal simulator.*
