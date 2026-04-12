"""VRPTW instance parser (optimized and consistent with core model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import os
import re


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
class VRPTWInstance:
    name: str
    file_path: str
    num_vehicles: int
    capacity: float
    depot: Customer
    customers: List[Customer]
    all_nodes: List[Customer]

    # Cache distance matrix (VERY important for speed)
    _distance_matrix: Optional[List[List[float]]] = field(default=None, init=False, repr=False)

    @property
    def num_customers(self) -> int:
        return len(self.customers)

    @property
    def num_nodes(self) -> int:
        return len(self.all_nodes)

    def distance_matrix(self) -> List[List[float]]:
        if self._distance_matrix is None:
            self._distance_matrix = build_distance_matrix(self)
        return self._distance_matrix


class VRPTWParseError(ValueError):
    pass


# -----------------------------
# Parsing helpers
# -----------------------------

def _clean_lines(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def _find_line_index(lines: List[str], predicate) -> int:
    for i, line in enumerate(lines):
        if predicate(line):
            return i
    raise VRPTWParseError("Required section not found")


def _extract_numbers(line: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", line)]


# -----------------------------
# Main parser
# -----------------------------

def parse_instance(filepath: str) -> VRPTWInstance:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    lines = _clean_lines(filepath)
    name = Path(filepath).stem

    # VEHICLES
    idx = _find_line_index(lines, lambda s: s.upper().startswith("NUMBER"))
    nums = _extract_numbers(lines[idx + 1])
    if len(nums) < 2:
        raise VRPTWParseError("Invalid vehicle line")

    num_vehicles = int(nums[0])
    capacity = float(nums[1])

    # CUSTOMERS
    idx = _find_line_index(lines, lambda s: s.upper().startswith("CUST NO"))
    rows = lines[idx + 1 :]

    customers: List[Customer] = []
    for row in rows:
        nums = _extract_numbers(row)
        if len(nums) < 7:
            continue

        customers.append(
            Customer(
                id=int(nums[0]),
                x=nums[1],
                y=nums[2],
                demand=nums[3],
                ready_time=nums[4],
                due_date=nums[5],
                service_time=nums[6],
            )
        )

    if not customers:
        raise VRPTWParseError("No customers found")

    # CRITICAL: enforce contiguous IDs
    customers.sort(key=lambda c: c.id)

    expected = list(range(len(customers)))
    actual = [c.id for c in customers]
    if actual != expected:
        raise VRPTWParseError("Customer IDs must be 0..N with no gaps")

    depot = customers[0]

    return VRPTWInstance(
        name=name,
        file_path=filepath,
        num_vehicles=num_vehicles,
        capacity=capacity,
        depot=depot,
        customers=customers[1:],
        all_nodes=customers,
    )


def parse_instances_dir(instances_dir: str) -> List[VRPTWInstance]:
    paths = sorted(Path(instances_dir).glob("*.TXT"))
    return [parse_instance(str(p)) for p in paths]


# -----------------------------
# Distance matrix (FAST)
# -----------------------------

def build_distance_matrix(instance: VRPTWInstance) -> List[List[float]]:
    nodes = instance.all_nodes
    n = len(nodes)

    dist = [[0.0] * n for _ in range(n)]

    for i in range(n):
        xi, yi = nodes[i].x, nodes[i].y
        for j in range(n):
            dx = xi - nodes[j].x
            dy = yi - nodes[j].y
            dist[i][j] = (dx * dx + dy * dy) ** 0.5

    return dist


# -----------------------------
# Sanity check
# -----------------------------

def sanity_check_instance(instance: VRPTWInstance) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if instance.depot.id != 0:
        errors.append("Depot must be id 0")

    if instance.all_nodes[0].demand != 0:
        errors.append("Depot demand must be 0")

    if instance.all_nodes[0].service_time != 0:
        errors.append("Depot service time must be 0")

    if instance.num_nodes != instance.num_customers + 1:
        errors.append("Node count mismatch")

    return len(errors) == 0, errors


__all__ = [
    "Customer",
    "VRPTWInstance",
    "VRPTWParseError",
    "parse_instance",
    "parse_instances_dir",
    "build_distance_matrix",
    "sanity_check_instance",
]