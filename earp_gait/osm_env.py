"""
Build an EARP-Gait bilayer Environment from real OpenStreetMap geometry.

Converts OSM ways (lat/lon polylines) for a crowded Omani area into the same
graph interface as model.Environment: node positions (projected to metres),
edges with terrain attributes (length, slope, roughness, surface), docking
stations, and a time-varying crowd-density field with hotspots at real
crowded POIs (souq, corniche, mall entrances).

Road/surface geometry is REAL (OpenStreetMap). Slope is synthesized (no DEM
here) and pedestrian *motion* is simulated — no open Omani trajectory data
exists — so crowd dynamics are calibrated to public benchmarks, not measured.
"""
from __future__ import annotations
import json, math
import numpy as np
from dataclasses import dataclass, field

# OSM highway tag -> (surface, base roughness) for a legged robot
HIGHWAY_SURFACE = {
    "pedestrian": ("tile", 0.05), "footway": ("pave", 0.08), "path": ("grass", 0.30),
    "steps": ("tile", 0.55), "living_street": ("pave", 0.10), "residential": ("pave", 0.08),
    "service": ("pave", 0.10), "tertiary": ("pave", 0.06), "secondary": ("pave", 0.05),
    "unclassified": ("pave", 0.12), "track": ("sand", 0.45),
}
# crowded POI keywords -> density boost
CROWD_HINTS = ("souq", "market", "corniche", "mall", "mutrah", "haffa", "beach", "museum")


def _project(lat, lon, lat0, lon0):
    """Equirectangular projection to metres about a local origin."""
    R = 6371000.0
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * R
    y = math.radians(lat - lat0) * R
    return x, y


@dataclass
class OSMEnvironment:
    path: str
    seed: int = 0
    max_nodes: int = 60
    rng: np.random.Generator = field(default=None)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._build()

    def _build(self):
        raw = json.load(open(self.path))
        self.label = raw["label"]; bbox = raw["bbox"]
        lat0 = (bbox[0] + bbox[2]) / 2; lon0 = (bbox[1] + bbox[3]) / 2

        # collect unique nodes (rounded lat/lon) and edges from way geometries
        node_id = {}; pos = []; edge_raw = []
        crowd_nodes = set()
        for w in raw["ways"]:
            tags = w.get("tags", {})
            hw = tags.get("highway", "service")
            surface, rough = HIGHWAY_SURFACE.get(hw, ("pave", 0.12))
            name = (tags.get("name", "") + " " + tags.get("amenity", "")).lower()
            is_crowd = any(h in name for h in CROWD_HINTS) or hw == "pedestrian"
            geom = w["geometry"]
            for a, b in zip(geom[:-1], geom[1:]):
                for p in (a, b):
                    key = (round(p["lat"], 5), round(p["lon"], 5))
                    if key not in node_id:
                        node_id[key] = len(pos)
                        pos.append(_project(p["lat"], p["lon"], lat0, lon0))
                ka = (round(a["lat"], 5), round(a["lon"], 5))
                kb = (round(b["lat"], 5), round(b["lon"], 5))
                i, j = node_id[ka], node_id[kb]
                if i == j:
                    continue
                edge_raw.append((i, j, surface, rough))
                if is_crowd:
                    crowd_nodes.add(i); crowd_nodes.add(j)

        pos = np.array(pos)
        # keep the largest connected component, then subsample to max_nodes
        adj = {k: set() for k in range(len(pos))}
        for i, j, *_ in edge_raw:
            adj[i].add(j); adj[j].add(i)
        # BFS components
        seen = set(); comps = []
        for s in range(len(pos)):
            if s in seen: continue
            stack=[s]; comp=set()
            while stack:
                u=stack.pop()
                if u in seen: continue
                seen.add(u); comp.add(u)
                stack.extend(adj[u]-seen)
            comps.append(comp)
        main = max(comps, key=len)
        # subsample to a CONNECTED subgraph via BFS grown from a crowded seed,
        # so the kept region stays traversable (avoids fragmenting the graph)
        if len(main) > self.max_nodes:
            crowd_in = [n for n in main if n in crowd_nodes]
            seed = crowd_in[0] if crowd_in else next(iter(main))
            keep = set(); frontier = [seed]
            while frontier and len(keep) < self.max_nodes:
                u = frontier.pop(0)
                if u in keep: continue
                keep.add(u)
                nbrs = sorted(adj[u] & main, key=lambda n: (n not in crowd_nodes))
                frontier.extend(n for n in nbrs if n not in keep)
        else:
            keep = set(main)
        # reindex
        remap = {old: new for new, old in enumerate(sorted(keep))}
        self.pos = np.array([pos[o] for o in sorted(keep)])
        n = len(self.pos)
        self.n_nodes = n
        self.crowd_nodes = {remap[c] for c in crowd_nodes if c in remap}

        # build edges among kept nodes
        self.edges = {}
        for i, j, surface, rough in edge_raw:
            if i not in remap or j not in remap: continue
            a, b = remap[i], remap[j]
            key = (min(a, b), max(a, b))
            if key in self.edges: continue
            length = float(np.linalg.norm(self.pos[a] - self.pos[b])) + 1e-6
            if length < 1.0 or length > 400: continue
            # slope synthesized (no DEM): small, deterministic per-edge
            slope = float(self.rng.normal(0, 0.04))
            roughness = float(np.clip(rough + self.rng.normal(0, 0.05), 0, 1))
            self.edges[key] = dict(length=length, slope=slope, roughness=roughness, surface=surface)

        self.adj = {i: [] for i in range(n)}
        for (i, j) in self.edges:
            self.adj[i].append(j); self.adj[j].append(i)
        # drop isolated nodes from adjacency reachability by rebuilding main component
        self._keep_main_component()

        # docks: place at well-connected real intersections, but spread out —
        # greedily pick high-degree nodes while enforcing a minimum spacing so
        # chargers are geographically distributed (not stacked at one junction).
        deg = {i: len(self.adj[i]) for i in range(self.n_nodes)}
        span = self.pos.max(0) - self.pos.min(0)
        min_sep = 0.18 * float(np.linalg.norm(span))  # ~18% of site diagonal
        n_target = max(2, self.n_nodes // 8)
        docks = []
        for cand in sorted(deg, key=deg.get, reverse=True):
            if all(np.linalg.norm(self.pos[cand] - self.pos[d]) >= min_sep for d in docks):
                docks.append(cand)
            if len(docks) >= n_target:
                break
        # if strict spacing yielded too few, relax the separation in steps
        # (rather than stacking chargers) until we have at least 3.
        relax = min_sep
        while len(docks) < max(3, n_target) and relax > 1.0:
            relax *= 0.6
            for cand in sorted(deg, key=deg.get, reverse=True):
                if cand in docks:
                    continue
                if all(np.linalg.norm(self.pos[cand] - self.pos[d]) >= relax for d in docks):
                    docks.append(cand)
                if len(docks) >= n_target:
                    break
        self.docks = set(docks)

    def _keep_main_component(self):
        seen=set(); comps=[]
        for s in range(self.n_nodes):
            if s in seen or not self.adj[s]: continue
            stack=[s]; comp=set()
            while stack:
                u=stack.pop()
                if u in seen: continue
                seen.add(u); comp.add(u); stack.extend(set(self.adj[u])-seen)
            comps.append(comp)
        if not comps: return
        main=max(comps,key=len)
        remap={old:new for new,old in enumerate(sorted(main))}
        self.pos=np.array([self.pos[o] for o in sorted(main)])
        self.n_nodes=len(main)
        newedges={}
        for (i,j),a in self.edges.items():
            if i in remap and j in remap:
                newedges[(min(remap[i],remap[j]),max(remap[i],remap[j]))]=a
        self.edges=newedges
        self.adj={i:[] for i in range(self.n_nodes)}
        for (i,j) in self.edges:
            self.adj[i].append(j); self.adj[j].append(i)
        self.crowd_nodes={remap[c] for c in self.crowd_nodes if c in remap}

    def edge_attr(self, i, j):
        return self.edges[(min(i, j), max(i, j))]

    def crowd_density(self, i, j, t, base_density=0.3, surge=0.0):
        """Time-varying density; real crowded POIs (souq/corniche) get a hotspot boost."""
        key = (min(i, j), max(i, j))
        phase = (hash(key) % 1000) / 1000.0 * 2 * math.pi
        hot = 0.4 if (i in self.crowd_nodes or j in self.crowd_nodes) else 0.0
        d = (base_density + hot) * (0.6 + 0.4 * math.sin(0.001 * t + phase)) + surge
        return float(np.clip(d, 0, 1.4))


if __name__ == "__main__":
    for f in ["osm/muscat_mutrah.json", "osm/salalah_haffa.json"]:
        e = OSMEnvironment(f, seed=1, max_nodes=60)
        span = e.pos.max(0) - e.pos.min(0)
        print(f"{e.label}: {e.n_nodes} nodes, {len(e.edges)} edges, {len(e.docks)} docks, "
              f"{len(e.crowd_nodes)} crowd-nodes, site {span[0]:.0f}x{span[1]:.0f} m")
