from __future__ import annotations

from dataclasses import dataclass, field

from .board import BoardSpec, hole_to_node
from .schemas import SnappedPart


class DisjointSetUnion:
    """Classic weighted DSU with path compression."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def make(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.make(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def all_board_nodes(board: BoardSpec) -> list[str]:
    nodes: list[str] = []
    for c in range(1, board.columns + 1):
        nodes.append(f"col_{c}_left")
        nodes.append(f"col_{c}_right")
    if board.has_rails:
        for prefix in ("top_plus", "top_minus", "bottom_plus", "bottom_minus"):
            if board.split_rails:
                nodes.append(f"{prefix}_a")
                nodes.append(f"{prefix}_b")
            else:
                nodes.append(prefix)
    return nodes


def _rail_nodes(name: str, board: BoardSpec) -> list[str]:
    if board.split_rails:
        return [f"{name}_a", f"{name}_b"]
    return [name]


@dataclass
class NetlistResult:
    """Result of building a breadboard netlist from snapped parts."""

    nets: dict[str, list[str]] = field(default_factory=dict)
    dsu: DisjointSetUnion = field(default_factory=DisjointSetUnion)
    pin_to_net: dict[tuple[str, str], str] = field(default_factory=dict)
    board: BoardSpec | None = None

    def net_of(self, hole_id: str) -> str:
        assert self.board is not None
        return self.dsu.find(hole_to_node(hole_id, self.board))


def build_netlist(
    snapped_parts: list[SnappedPart],
    board: BoardSpec,
    rail_bindings: dict[str, str] | None = None,
) -> NetlistResult:
    """Build a canonical breadboard netlist.

    Returns a NetlistResult containing:
      - nets: canonical net id -> list of labels ("R1.0", "LED1.anode", "VCC")
      - dsu: the DSU over all board nodes
      - pin_to_net: (part_id, pin_name) -> canonical net id
    """
    rail_bindings = rail_bindings or {}
    dsu = DisjointSetUnion()
    for node in all_board_nodes(board):
        dsu.make(node)

    # Phase 1: merge nodes via wires (ignore wires with ambiguous ends).
    for part in snapped_parts:
        if part.type != "wire":
            continue
        if part.ambiguous_pins:
            continue
        try:
            n0 = hole_to_node(part.pins["0"], board)
            n1 = hole_to_node(part.pins["1"], board)
        except (KeyError, ValueError):
            continue
        dsu.union(n0, n1)

    # Phase 2: attach non-wire part pins to canonical nets.
    nets: dict[str, list[str]] = {}
    pin_to_net: dict[tuple[str, str], str] = {}
    for part in snapped_parts:
        if part.type == "wire":
            continue
        for pin_name, hole in part.pins.items():
            if pin_name in part.ambiguous_pins:
                continue
            try:
                node = hole_to_node(hole, board)
            except (KeyError, ValueError, IndexError):
                continue
            dsu.make(node)
            root = dsu.find(node)
            nets.setdefault(root, []).append(f"{part.id}.{pin_name}")
            pin_to_net[(part.id, pin_name)] = root

    # Phase 3: tag rail bindings. A rail label is attached when its root is
    # either (a) touched by at least one part pin, or (b) merged with another
    # bound rail (so VCC/GND short-circuits are visible even with no parts).
    rail_root_to_labels: dict[str, list[str]] = {}
    for rail_name, label in rail_bindings.items():
        for node in _rail_nodes(rail_name, board):
            if node not in dsu.parent:
                continue
            root = dsu.find(node)
            if label not in rail_root_to_labels.setdefault(root, []):
                rail_root_to_labels[root].append(label)

    for root, labels in rail_root_to_labels.items():
        has_parts = root in nets
        is_shorted = len(labels) >= 2
        if not (has_parts or is_shorted):
            continue
        bucket = nets.setdefault(root, [])
        for label in labels:
            if label not in bucket:
                bucket.append(label)

    return NetlistResult(nets=nets, dsu=dsu, pin_to_net=pin_to_net, board=board)
