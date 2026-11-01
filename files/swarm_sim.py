"""Swarm Drones for Disaster Relief - simulation engine (grid world, 3 strategies)."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

SQ2 = math.sqrt(2.0)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
DCOST = [1.0, 1.0, 1.0, 1.0, SQ2, SQ2, SQ2, SQ2]
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
GRID = 50
STRAT_ID = {"random": 0, "astar": 1, "swarm": 2}

CFG_DEFAULT = dict(battery=160.0, charge_time=10, packets=8, p_near=0.9, p_far=0.4)

ACO_DEFAULT = dict(w0=1.0, w1=1.0, w2=0.6, wa=2.0, wi=0.25, wr=0.6, tau=0.15,
                   rho_v=0.003, rho_a=0.02, bump_sigma=3.0)


@dataclass
class Scenario:
    kind: str
    seed: int
    cost: np.ndarray
    cost_list: list
    survivors: np.ndarray
    base: tuple
    dist_home: np.ndarray
    dist_list: list


def _disc(n, centre, radius):
    rr, cc = np.mgrid[0:n, 0:n]
    return (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2 <= radius ** 2


def dijkstra_home(cost, base):
    n = cost.shape[0]
    d = np.full((n, n), np.inf)
    d[base] = 0.0
    pq = [(0.0, base)]
    while pq:
        dd, (r, c) = heapq.heappop(pq)
        if dd > d[r, c]:
            continue
        for (dr, dc), m in zip(DIRS, DCOST):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and np.isfinite(cost[nr, nc]):
                nd = dd + cost[r, c] * m
                if nd < d[nr, nc]:
                    d[nr, nc] = nd
                    heapq.heappush(pq, (nd, (nr, nc)))
    return d


def _cluster(rng, centre, k, sigma, ok, taken):
    pts = []
    n = ok.shape[0]
    for _ in range(400):
        r = int(round(centre[0] + rng.normal(0, sigma)))
        c = int(round(centre[1] + rng.normal(0, sigma)))
        if 0 <= r < n and 0 <= c < n and ok[r, c] and (r, c) not in taken:
            taken.add((r, c))
            pts.append((r, c))
            if len(pts) == k:
                break
    return pts


def make_scenario(kind: str, seed: int, n_surv: int = 24, n: int = GRID) -> Scenario:
    rng = np.random.default_rng(10_000 + seed)
    cost = np.ones((n, n))
    base = (2, 2)
    rr, cc = np.mgrid[0:n, 0:n]
    river = np.zeros((n, n), bool)

    if kind == "earthquake":
        for _ in range(3):
            centre = rng.integers(12, n - 6, size=2)
            cost[_disc(n, centre, int(rng.integers(5, 9)))] = 2.0
        for _ in range(9):
            h, w = rng.integers(2, 6, size=2)
            r0, c0 = int(rng.integers(0, n - h)), int(rng.integers(0, n - w))
            cost[r0:r0 + h, c0:c0 + w] = np.inf
    elif kind == "flood":
        phase = rng.uniform(0, 2 * np.pi)
        centre_col = n * 0.55 + 11 * np.sin(2 * np.pi * np.arange(n) / n * 1.1 + phase)
        river = np.abs(cc - centre_col[:, None]) <= 5
        cost[river] = 2.0
        for _ in range(6):
            h, w = rng.integers(1, 4, size=2)
            r0, c0 = int(rng.integers(0, n - h)), int(rng.integers(0, n - w))
            cost[r0:r0 + h, c0:c0 + w] = np.inf
    else:
        raise ValueError(kind)

    cost[:6, :6] = 1.0
    dist = dijkstra_home(cost, base)
    ok = np.isfinite(dist) & (np.hypot(rr - base[0], cc - base[1]) >= 12)

    taken: set = set()
    surv: list = []
    far = np.argwhere(ok & (dist >= 18))
    if kind == "earthquake":
        centres: list = []
        tries = 0
        while len(centres) < 4 and tries < 500:
            tries += 1
            c = far[rng.integers(len(far))]
            if all(np.hypot(*(c - o)) >= 12 for o in centres):
                centres.append(c)
        for c in centres:
            surv += _cluster(rng, c, n_surv // 4, 2.2, ok, taken)
    else:
        banks = np.argwhere(river & ok)
        for idx in rng.permutation(len(banks))[:16]:
            r, c = banks[idx]
            taken.add((int(r), int(c)))
            surv.append((int(r), int(c)))
        for _ in range(2):
            c = far[rng.integers(len(far))]
            surv += _cluster(rng, c, 4, 1.8, ok, taken)

    assert len(surv) == n_surv, f"could not place survivors ({len(surv)})"
    return Scenario(kind, seed, cost, cost.tolist(), np.array(surv), base, dist, dist.tolist())


def astar(costl, start, goal):
    n = len(costl)
    if start == goal:
        return []

    def h(r, c):
        dx, dy = abs(r - goal[0]), abs(c - goal[1])
        return (dx + dy) + (SQ2 - 2.0) * min(dx, dy)

    g = {start: 0.0}
    parent = {}
    closed = set()
    pq = [(h(*start), 0.0, start)]
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur == goal:
            path = []
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for (dr, dc), m in zip(DIRS, DCOST):
            qr, qc = cur[0] + dr, cur[1] + dc
            if 0 <= qr < n and 0 <= qc < n:
                cq = costl[qr][qc]
                if cq == math.inf:
                    continue
                ng = gc + cq * m
                if ng < g.get((qr, qc), math.inf):
                    g[(qr, qc)] = ng
                    parent[(qr, qc)] = cur
                    heapq.heappush(pq, (ng + h(qr, qc), ng, (qr, qc)))
    return None


class Drone:
    def __init__(self, i, base, battery, packets):
        self.i = i
        self.pos = base
        self.battery = battery
        self.packets = packets
        self.mode = "search"
        self.timer = 0
        self.last = -1
        self.path: list = []
        self.target = None
        self.energy = 0.0
        self.sorties = 1


@dataclass
class Result:
    strategy: str
    found_curve: np.ndarray
    coverage_curve: np.ndarray
    energy_curve: np.ndarray
    n_found: int
    n_surv: int
    t50: int
    t80: int
    t_all: int
    mean_disc_time: float
    energy: float
    energy_per_found: float
    coverage: float
    wasted: float
    supplied: int
    sorties: int
    positions: np.ndarray | None = None
    cov_frames: list | None = None
    found_frames: list | None = None
    found_times: np.ndarray | None = None


def run(scn: Scenario, strategy: str, n_drones: int = 5, T: int = 300, seed: int = 0,
        aco: dict | None = None, cfg: dict | None = None, record: bool = False) -> Result:
    cfg = {**CFG_DEFAULT, **(cfg or {})}
    P = {**ACO_DEFAULT, **(aco or {})}
    rng = np.random.default_rng([seed, STRAT_ID[strategy], n_drones])

    cost, costl, dist_l = scn.cost, scn.cost_list, scn.dist_list
    n = cost.shape[0]
    free = np.isfinite(cost)
    freel = free.tolist()
    base = scn.base
    surv = scn.survivors
    S = len(surv)
    found = np.zeros(S, bool)
    supplied = np.zeros(S, bool)
    t_found = np.full(S, np.inf)
    covered = np.zeros((n, n), bool)
    swept = np.zeros((n, n), bool)
    n_free = int(free.sum())
    rr, cc = np.mgrid[0:n, 0:n]

    drones = [Drone(i, base, cfg["battery"], cfg["packets"]) for i in range(n_drones)]
    edges = np.linspace(0, n, n_drones + 1).round().astype(int)
    sector = []
    for i in range(n_drones):
        m = np.zeros((n, n), bool)
        m[:, edges[i]:edges[i + 1]] = True
        sector.append(m)

    V = np.where(free, 0.0, 1.0)
    A = np.zeros((n, n))
    decay = np.where(free, 1.0 - P["rho_v"], 1.0)
    PAD = 12
    Up = Gp = None

    stats = dict(search_steps=0, wasted=0)
    new_found: list = []

    def sense(d, t):
        r, c = d.pos
        r0, r1, c0, c1 = max(0, r - 1), min(n, r + 2), max(0, c - 1), min(n, c + 2)
        fw = free[r0:r1, c0:c1]
        blk = covered[r0:r1, c0:c1]
        new = int((~blk & fw).sum())
        blk |= fw
        swept[r0:r1, c0:c1] |= fw
        if strategy == "swarm":
            V[r0:r1, c0:c1] = 1.0
        cheb = np.maximum(np.abs(surv[:, 0] - r), np.abs(surv[:, 1] - c))
        near = (~found) & (cheb <= 2)
        if near.any():
            for s in np.flatnonzero(near):
                p = cfg["p_near"] if cheb[s] <= 1 else cfg["p_far"]
                if rng.random() < p:
                    found[s] = True
                    t_found[s] = t
                    if d.packets > 0:
                        d.packets -= 1
                        supplied[s] = True
                    new_found.append(s)
        return new

    def bm(arr, r, c, rad):
        r += PAD
        c += PAD
        return arr[r - rad:r + rad + 1, c - rad:c + rad + 1].mean()

    def choose_random(d):
        r, c = d.pos
        opts = [k for k, (dr, dc) in enumerate(DIRS)
                if 0 <= r + dr < n and 0 <= c + dc < n and freel[r + dr][c + dc]]
        if d.last in opts and rng.random() < 0.6:
            k = d.last
        else:
            k = opts[int(rng.integers(len(opts)))]
        return (r + DIRS[k][0], c + DIRS[k][1])

    def choose_astar(d):
        r, c = d.pos
        if d.target is not None and (swept[d.target] or d.target == d.pos):
            d.target, d.path = None, []
        if d.path:
            return d.path.pop(0)
        for _ in range(6):
            cand = free & ~swept & sector[d.i]
            if not cand.any():
                cand = free & ~swept
            if not cand.any():
                swept[:] = False
                cand = free & ~swept & sector[d.i]
            idx = np.argwhere(cand)
            dr_, dc_ = np.abs(idx[:, 0] - r), np.abs(idx[:, 1] - c)
            j = int(np.argmin(dr_ + dc_ + (SQ2 - 2.0) * np.minimum(dr_, dc_)))
            tgt = (int(idx[j, 0]), int(idx[j, 1]))
            path = astar(costl, d.pos, tgt)
            if path:
                d.target, d.path = tgt, path
                return d.path.pop(0)
            swept[tgt] = True
        return None

    def choose_swarm(d):
        r, c = d.pos
        others = [o.pos for o in drones if o is not d and o.mode != "charge"]
        scores = np.full(8, -np.inf)
        for k, (dr, dc) in enumerate(DIRS):
            qr, qc = r + dr, c + dc
            if not (0 <= qr < n and 0 <= qc < n and freel[qr][qc]):
                continue
            s = (P["w0"] * bm(Up, qr, qc, 1)
                 + P["w1"] * bm(Up, r + 3 * dr, c + 3 * dc, 2)
                 + P["w2"] * bm(Up, r + 7 * dr, c + 7 * dc, 3)
                 + P["wa"] * bm(Gp, r + 3 * dr, c + 3 * dc, 3))
            if k == d.last:
                s += P["wi"]
            for (orow, ocol) in others:
                dd = max(abs(orow - qr), abs(ocol - qc))
                if dd < 8:
                    s -= P["wr"] * (1.0 - dd / 8.0)
            scores[k] = s
        ok = np.isfinite(scores)
        if not ok.any():
            return None
        z = np.exp((scores[ok] - scores[ok].max()) / P["tau"])
        k = int(np.flatnonzero(ok)[rng.choice(z.size, p=z / z.sum())])
        return (r + DIRS[k][0], c + DIRS[k][1])

    choose = {"random": choose_random, "astar": choose_astar, "swarm": choose_swarm}[strategy]

    def return_step(d):
        r, c = d.pos
        best, bv = None, math.inf
        for k, (dr, dc) in enumerate(DIRS):
            qr, qc = r + dr, c + dc
            if 0 <= qr < n and 0 <= qc < n and freel[qr][qc]:
                step = costl[qr][qc] * DCOST[k]
                v = step + dist_l[qr][qc]
                if v < bv:
                    bv, best = v, ((qr, qc), k, step)
        return best

    found_curve = np.zeros(T + 1, int)
    cov_curve = np.zeros(T + 1)
    en_curve = np.zeros(T + 1)
    pos_hist = np.zeros((T + 1, n_drones, 2), int) if record else None
    cov_frames, found_frames = ([], []) if record else (None, None)

    for d in drones:
        sense(d, 0)
    new_found.clear()

    def snapshot(t):
        found_curve[t] = int(found.sum())
        cov_curve[t] = covered.sum() / n_free
        en_curve[t] = sum(d.energy for d in drones)
        if record:
            pos_hist[t] = [d.pos for d in drones]
            cov_frames.append(covered.copy())
            found_frames.append(found.copy())

    snapshot(0)
    last_t = 0
    for t in range(1, T + 1):
        if found.all():
            break
        new_found.clear()
        if strategy == "swarm":
            V *= decay
            A *= 1.0 - P["rho_a"]
            U = 1.0 - V
            Up = np.pad(U, PAD, constant_values=0.0)
            Gp = np.pad(A * U, PAD, constant_values=0.0)
        for d in drones:
            if d.mode == "charge":
                d.timer -= 1
                if d.timer <= 0:
                    d.mode, d.battery, d.packets = "search", cfg["battery"], cfg["packets"]
                    d.path, d.target, d.last = [], None, -1
                    d.sorties += 1
                continue
            move = None
            if d.mode == "search":
                q = choose(d)
                if q is not None:
                    k = DIR_INDEX[(q[0] - d.pos[0], q[1] - d.pos[1])]
                    step = costl[q[0]][q[1]] * DCOST[k]
                    if d.battery - step >= dist_l[q[0]][q[1]] + 1.0:
                        move = (q, k, step)
                if move is None:
                    d.mode, d.path, d.target = "return", [], None
            if d.mode == "return" and move is None:
                if d.pos == base:
                    d.mode, d.timer = "charge", cfg["charge_time"]
                    continue
                move = return_step(d)
            q, k, step = move
            was_search = d.mode == "search"
            d.pos, d.last = q, k
            d.battery -= step
            d.energy += step
            new = sense(d, t)
            if was_search:
                stats["search_steps"] += 1
                stats["wasted"] += int(new == 0)
            if d.mode == "return" and d.pos == base:
                d.mode, d.timer = "charge", cfg["charge_time"]
        if strategy == "swarm" and new_found:
            for s in new_found:
                A += np.exp(-((rr - surv[s, 0]) ** 2 + (cc - surv[s, 1]) ** 2)
                            / (2 * P["bump_sigma"] ** 2))
        snapshot(t)
        last_t = t

    for t in range(last_t + 1, T + 1):
        found_curve[t] = found_curve[last_t]
        cov_curve[t] = cov_curve[last_t]
        en_curve[t] = en_curve[last_t]
        if record:
            pos_hist[t] = pos_hist[last_t]
            cov_frames.append(cov_frames[-1])
            found_frames.append(found_frames[-1])

    def first(k):
        hit = np.flatnonzero(found_curve >= k)
        return int(hit[0]) if hit.size else T

    nf = int(found.sum())
    energy = float(sum(d.energy for d in drones))
    return Result(
        strategy=strategy, found_curve=found_curve, coverage_curve=cov_curve, energy_curve=en_curve,
        n_found=nf, n_surv=S, t50=first(math.ceil(0.5 * S)), t80=first(math.ceil(0.8 * S)),
        t_all=first(S), mean_disc_time=float(np.minimum(t_found, T).mean()),
        energy=energy, energy_per_found=energy / max(nf, 1), coverage=float(cov_curve[-1]),
        wasted=stats["wasted"] / max(stats["search_steps"], 1), supplied=int(supplied.sum()),
        sorties=sum(d.sorties for d in drones), positions=pos_hist, cov_frames=cov_frames,
        found_frames=found_frames, found_times=t_found,
    )
