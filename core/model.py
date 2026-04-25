"""Core VRPTW data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class Route:
    path: List[int] = field(default_factory=lambda: [0, 0])
    load: float = 0.0
    distance: float = 0.0
    total_time: float = 0.0
    waiting_time: float = 0.0
    feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path:
            self.path = [0, 0]
        if self.path[0] != 0:
            self.path.insert(0, 0)
        if self.path[-1] != 0:
            self.path.append(0)

    @property
    def customers(self) -> List[int]:
        return [node for node in self.path if node != 0]

    @property
    def is_empty(self) -> bool:
        return len(self.customers) == 0

    def copy(self) -> "Route":
        return Route(
            path=list(self.path),
            load=self.load,
            distance=self.distance,
            total_time=self.total_time,
            waiting_time=self.waiting_time,
            feasible=self.feasible,
            metadata=dict(self.metadata),
        )


@dataclass
class Solution:
    routes: List[Route] = field(default_factory=list)
    objective: float = float("inf")
    feasible: bool = False
    total_distance: float = 0.0
    total_time: float = 0.0
    vehicles_used: int = 0
    load_variance: float = 0.0
    spatial_variance: float = 0.0
    unserved_customers: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Solution":
        return Solution(
            routes=[r.copy() for r in self.routes],
            objective=self.objective,
            feasible=self.feasible,
            total_distance=self.total_distance,
            total_time=self.total_time,
            vehicles_used=self.vehicles_used,
            load_variance=self.load_variance,
            spatial_variance=self.spatial_variance,
            unserved_customers=list(self.unserved_customers),
            metadata=dict(self.metadata),
        )

    @property
    def route_paths(self) -> List[List[int]]:
        return [route.path for route in self.routes]

    def active_routes(self) -> List[Route]:
        return [r for r in self.routes if len(r.path) > 2]

    def all_customers(self) -> List[int]:
        customers: List[int] = []
        for route in self.routes:
            customers.extend(route.customers)
        return customers

    def to_submission_routes(self) -> List[List[int]]:
        return [route.path for route in self.active_routes()]

    def add_route(self, route: Route) -> None:
        self.routes.append(route)

    def set_objective(self, objective: float) -> None:
        self.objective = objective

    def mark_feasible(self, feasible: bool) -> None:
        self.feasible = feasible


@dataclass
class InsertionMove:
    customer_id: int
    route_index: int
    position: int
    delta_cost: float


@dataclass
class RemovalMove:
    customer_id: int
    route_index: int
    position: int


__all__ = [
    "Customer",
    "Route",
    "Solution",
    "InsertionMove",
    "RemovalMove",
]