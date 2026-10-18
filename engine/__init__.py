"""Sherlock Ohms breadboard reconstruction & verification engine."""

from .board import BoardSpec, HoleLUT, hole_to_node
from .netlist import DisjointSetUnion, NetlistResult, build_netlist
from .schemas import (
    Finding,
    InspectionVerdict,
    Keypoint,
    RawDetection,
    ReferenceCircuit,
    SnappedPart,
)
from .snapping import snap_detections
from .verifier import (
    check_connectivity,
    check_placement,
    check_safety,
    verify_circuit,
)

__all__ = [
    "BoardSpec",
    "DisjointSetUnion",
    "Finding",
    "HoleLUT",
    "InspectionVerdict",
    "Keypoint",
    "NetlistResult",
    "RawDetection",
    "ReferenceCircuit",
    "SnappedPart",
    "build_netlist",
    "check_connectivity",
    "check_placement",
    "check_safety",
    "hole_to_node",
    "snap_detections",
    "verify_circuit",
]
