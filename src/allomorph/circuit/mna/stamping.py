"""
Allomorph - MNA Circuit Stamping & Disjoint Set Graph
Provides union-find disjoint net partitioning and coil branch representation.
"""

import numpy as np


class DisjointSet:
    """Disjoint-set (Union-Find) with path compression for circuit net grouping."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            gnd_set = {"GND", "gnd", "0", "preamp.gnd"}
            if ry in gnd_set:
                self.parent[rx] = ry
            elif rx in gnd_set:
                self.parent[ry] = rx
            elif ry in {"out", "preamp.in", "preamp.out"}:
                self.parent[rx] = ry
            elif rx in {"out", "preamp.in", "preamp.out"}:
                self.parent[ry] = rx
            else:
                self.parent[rx] = ry


class _CoilBranch:
    """Internal helper representing a coil's branch admittance and terminal nodes."""

    def __init__(
        self,
        pickup_id: str,
        coil_id: str,
        alias_key: str | None,
        t_hot: str,
        t_cold: str,
        L: float,
        Rdc: float,
        Ccoil: float,
        Z_br: np.ndarray,
        Y_br: np.ndarray,
    ) -> None:
        self.pickup_id = pickup_id
        self.coil_id = coil_id
        self.alias_key = alias_key
        self.t_hot = t_hot
        self.t_cold = t_cold
        self.L = L
        self.Rdc = Rdc
        self.Ccoil = Ccoil
        self.Z_br = Z_br
        self.Y_br = Y_br
        self.h_internal: np.ndarray | None = None
        self.y_self: np.ndarray | None = None
        self.y_mutual: np.ndarray | float = 0.0
        self.coupled_key: str | None = None
