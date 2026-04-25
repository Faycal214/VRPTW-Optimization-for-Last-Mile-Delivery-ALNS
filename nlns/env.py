from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

import numpy as np
import torch

from core.baseline import build_baseline_solution
from core.constraints import check_solution_feasibility
from core.evaluation import compute_distance_matrix, compute_objective
from core.model import Route, Solution
from core.parser import VRPTWInstance


@dataclass
class EpisodeStep:
    action: int
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float


class VRPTWRepairEnv:
    def __init__(self, destroy_ratio: float = 0.2, seed: int = 42):
        self.destroy_ratio = destroy_ratio
        self.rng = random.Random(seed)
        self.instance: Optional[VRPTWInstance] = None
        self.distance_matrix: Optional[List[List[float]]] = None
        self.solution: Optional[Solution] = None
        self.unserved: List[int] = []
        self._previous_objective: float = float("inf")

    def reset(self, instance: VRPTWInstance) -> None:
        self.instance = instance
        self.distance_matrix = compute_distance_matrix(instance)

        baseline = build_baseline_solution(instance)
        if not baseline.feasible:
            raise ValueError("Baseline solution is infeasible; cannot start RL episode.")

        self.solution = Solution(routes=[Route(path=list(r)) for r in baseline.routes], unserved_customers=[])
        self._previous_objective = baseline.objective

        # destroy a subset
        all_customers = [n for n in range(1, instance.num_nodes)]
        k = max(1, int(len(all_customers) * self.destroy_ratio))
        self.unserved = self.rng.sample(all_customers, k=k)

        for route in self.solution.routes:
            for cust in list(route.customers):
                if cust in self.unserved:
                    route.path.remove(cust)

        self.solution.unserved_customers = list(self.unserved)

        # clean empty routes
        self.solution.routes = [r for r in self.solution.routes if len(r.path) > 2]
        if not self.solution.routes:
            self.solution.routes = [Route(path=[0, 0])]

    def node_features(self) -> torch.Tensor:
        assert self.instance is not None
        inst = self.instance

        max_x = max(n.x for n in inst.all_nodes) or 1.0
        max_y = max(n.y for n in inst.all_nodes) or 1.0
        max_d = max(n.demand for n in inst.all_nodes) or 1.0
        max_t = max(n.due_date for n in inst.all_nodes) or 1.0
        max_s = max(n.service_time for n in inst.all_nodes) or 1.0

        feats = []
        served_set = set(self._served_customers())

        for n in inst.all_nodes:
            feats.append([
                n.x / max_x,
                n.y / max_y,
                n.demand / max_d,
                n.ready_time / max_t,
                n.due_date / max_t,
                n.service_time / max_s,
                0.0 if n.id in served_set else 1.0,  # unserved flag
            ])
        return torch.tensor(feats, dtype=torch.float32)

    def action_mask(self) -> torch.Tensor:
        assert self.instance is not None
        mask = torch.zeros(self.instance.num_nodes, dtype=torch.bool)
        for cid in self._served_customers():
            mask[cid] = True
        mask[0] = True  # depot not an action
        return mask

    def _served_customers(self) -> List[int]:
        assert self.solution is not None
        served = []
        for route in self.solution.routes:
            served.extend(route.customers)
        return served

    def _best_insert_position(self, customer_id: int) -> Tuple[bool, int, int]:
        assert self.instance is not None and self.solution is not None and self.distance_matrix is not None
        best_route = -1
        best_pos = -1
        best_delta = float("inf")

        for ridx, route in enumerate(self.solution.routes):
            path = route.path
            for pos in range(1, len(path)):
                trial = path[:pos] + [customer_id] + path[pos:]
                feas = check_solution_feasibility([trial], self.instance, self.distance_matrix)
                if not feas.feasible:
                    continue
                before = compute_objective([path], self.instance, self.distance_matrix)[0]
                after = compute_objective([trial], self.instance, self.distance_matrix)[0]
                delta = after - before
                if delta < best_delta:
                    best_delta = delta
                    best_route = ridx
                    best_pos = pos

        if best_route >= 0:
            return True, best_route, best_pos

        # new route if nothing fits
        trial = [0, customer_id, 0]
        feas = check_solution_feasibility([trial], self.instance, self.distance_matrix)
        if feas.feasible:
            return True, len(self.solution.routes), 1

        return False, -1, -1

    def step(self, customer_id: int) -> float:
        assert self.instance is not None and self.solution is not None and self.distance_matrix is not None

        if customer_id not in self.unserved:
            return -5.0

        ok, route_idx, pos = self._best_insert_position(customer_id)
        if not ok:
            self.unserved.remove(customer_id)
            self.solution.unserved_customers = list(self.unserved)
            return -10.0

        if route_idx == len(self.solution.routes):
            self.solution.routes.append(Route(path=[0, customer_id, 0]))
        else:
            self.solution.routes[route_idx].path.insert(pos, customer_id)

        self.unserved.remove(customer_id)
        self.solution.unserved_customers = list(self.unserved)

        current_objective, _ = compute_objective(self.solution.route_paths, self.instance, self.distance_matrix)
        reward = self._previous_objective - current_objective
        self._previous_objective = current_objective
        return float(reward)

    def done(self) -> bool:
        return len(self.unserved) == 0