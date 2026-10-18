from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from .board import BoardSpec, hole_to_node
from .netlist import NetlistResult, build_netlist
from .schemas import (
    Finding,
    InspectionVerdict,
    ReferenceCircuit,
    SnappedPart,
)

Mode = Literal["placement", "connectivity"]
POLAR_TYPES = ("led", "cap_electrolytic")


def _ref_part_pins(reference: ReferenceCircuit) -> list[tuple[str, str, dict[str, str]]]:
    out: list[tuple[str, str, dict[str, str]]] = []
    for p in reference.parts:
        if p["type"] == "wire":
            continue
        out.append((p["id"], p["type"], dict(p["pins"])))
    return out


def _ref_netlist(reference: ReferenceCircuit) -> dict[str, list[str]]:
    nets: dict[str, list[str]] = {}
    for p in reference.parts:
        if p["type"] == "wire":
            continue
        for pin_name, net_id in p["pins"].items():
            nets.setdefault(net_id, []).append(f"{p['id']}.{pin_name}")
    for lbl in ("VCC", "GND"):
        if lbl in nets:
            nets[lbl].append(lbl)
    return nets


def _actual_part_pins(
    snapped_parts: list[SnappedPart],
    nl: NetlistResult,
) -> list[tuple[str, str, dict[str, str]]]:
    out: list[tuple[str, str, dict[str, str]]] = []
    for p in snapped_parts:
        if p.type == "wire":
            continue
        pins: dict[str, str] = {}
        for pin_name, _hole in p.pins.items():
            if pin_name in p.ambiguous_pins:
                continue
            net = nl.pin_to_net.get((p.id, pin_name))
            if net is not None:
                pins[pin_name] = net
        out.append((p.id, p.type, pins))
    return out


def _build_iso_graph(
    parts: list[tuple[str, str, dict[str, str]]],
    nets: dict[str, list[str]],
    polar_pin_labels: bool = True,
) -> nx.Graph:
    """Bipartite Part/Net graph with per-pin intermediate nodes.

    Using an intermediate pin node per (part, pin_name) means both pins
    landing on the same net don't collapse into a single edge, and the
    pin's label (anode/cathode/any) is carried as a node attribute.
    """
    g = nx.Graph()
    all_net_ids = set(nets.keys())
    for _pid, _ptype, pins in parts:
        all_net_ids.update(pins.values())

    for net_id in all_net_ids:
        members = nets.get(net_id, [])
        if "VCC" in members:
            label = "VCC"
        elif "GND" in members:
            label = "GND"
        else:
            label = None
        g.add_node(("net", net_id), kind="net", label=label)

    for part_id, part_type, pins in parts:
        g.add_node(("part", part_id), kind="part", type=part_type)
        for pin_name, net_id in pins.items():
            if part_type in POLAR_TYPES and polar_pin_labels:
                pin_label = pin_name
            else:
                pin_label = "any"
            pin_node = ("pin", part_id, pin_name)
            g.add_node(pin_node, kind="pin", pin_label=pin_label)
            g.add_edge(("part", part_id), pin_node)
            if ("net", net_id) not in g.nodes:
                g.add_node(("net", net_id), kind="net", label=None)
            g.add_edge(pin_node, ("net", net_id))
    return g


def _node_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("kind") != b.get("kind"):
        return False
    kind = a["kind"]
    if kind == "part":
        return a.get("type") == b.get("type")
    if kind == "net":
        return a.get("label") == b.get("label")
    if kind == "pin":
        pa = a.get("pin_label", "any")
        pb = b.get("pin_label", "any")
        if pa == "any" or pb == "any":
            return True
        return pa == pb
    return False


def _is_isomorphic(g_ref: nx.Graph, g_act: nx.Graph) -> tuple[bool, dict | None]:
    matcher = GraphMatcher(g_ref, g_act, node_match=_node_match)
    if matcher.is_isomorphic():
        return True, matcher.mapping
    return False, None


# -----------------------------------------------------------------------------
# Safety rules (reference-free)
# -----------------------------------------------------------------------------

def _part_by_id(parts: list[SnappedPart]) -> dict[str, SnappedPart]:
    return {p.id: p for p in parts}


def check_safety(
    snapped_parts: list[SnappedPart],
    nl: NetlistResult,
    board: BoardSpec,
) -> list[Finding]:
    findings: list[Finding] = []
    nets = nl.nets
    pin_to_net = nl.pin_to_net
    parts_by_id = _part_by_id(snapped_parts)

    # POWER_SHORT
    seen_shorts: set[str] = set()
    for net_id, members in nets.items():
        if "VCC" in members and "GND" in members and net_id not in seen_shorts:
            seen_shorts.add(net_id)
            findings.append(
                Finding(
                    code="POWER_SHORT",
                    severity="critical",
                    message=f"Net {net_id} directly connects VCC to GND.",
                    holes=[],
                )
            )

    # SHORTED_PART, REVERSED_POLARITY (via rails)
    for part in snapped_parts:
        if part.type == "resistor":
            n0 = pin_to_net.get((part.id, "0"))
            n1 = pin_to_net.get((part.id, "1"))
            if n0 is not None and n0 == n1:
                findings.append(
                    Finding(
                        code="SHORTED_PART",
                        severity="critical",
                        message=f"Resistor {part.id} is shorted (both pins share a net).",
                        part_id=part.id,
                        holes=[part.pins.get("0", ""), part.pins.get("1", "")],
                    )
                )
        elif part.type in POLAR_TYPES:
            na = pin_to_net.get((part.id, "anode"))
            nc = pin_to_net.get((part.id, "cathode"))
            if na is not None and na == nc:
                findings.append(
                    Finding(
                        code="SHORTED_PART",
                        severity="critical",
                        message=f"{part.type} {part.id} is shorted.",
                        part_id=part.id,
                    )
                )
            if part.type == "led" and na is not None and nc is not None:
                anode_members = nets.get(na, [])
                cathode_members = nets.get(nc, [])
                if "GND" in anode_members and "VCC" in cathode_members:
                    findings.append(
                        Finding(
                            code="REVERSED_POLARITY",
                            severity="critical",
                            message=(
                                f"LED {part.id} is reversed (anode on GND, cathode on VCC)."
                            ),
                            part_id=part.id,
                        )
                    )

        if part.type == "ic_dip":
            p0 = part.pins.get("0")
            p1 = part.pins.get("1")
            if p0 and p1 and p0[0].isalpha() and p1[0].isalpha():
                left_rows = "abcde"
                right_rows = "fghij"
                s0 = "L" if p0[0] in left_rows else "R" if p0[0] in right_rows else None
                s1 = "L" if p1[0] in left_rows else "R" if p1[0] in right_rows else None
                if s0 and s1 and s0 == s1:
                    findings.append(
                        Finding(
                            code="IC_NOT_STRADDLING",
                            severity="critical",
                            message=(
                                f"IC {part.id} does not straddle the center gap "
                                f"(pins {p0}, {p1})."
                            ),
                            part_id=part.id,
                            holes=[p0, p1],
                        )
                    )

    # LED_WITHOUT_RESISTOR
    for part in snapped_parts:
        if part.type != "led":
            continue
        na = pin_to_net.get((part.id, "anode"))
        nc = pin_to_net.get((part.id, "cathode"))
        protected = False
        for net_id in (na, nc):
            if not net_id:
                continue
            for m in nets.get(net_id, []):
                if "." not in m:
                    continue
                pid, _pin = m.rsplit(".", 1)
                other = parts_by_id.get(pid)
                if other is not None and other.type == "resistor":
                    # Make sure the resistor isn't itself shorted on this net.
                    r0 = pin_to_net.get((pid, "0"))
                    r1 = pin_to_net.get((pid, "1"))
                    if r0 != r1:
                        protected = True
                        break
            if protected:
                break
        if not protected:
            findings.append(
                Finding(
                    code="LED_WITHOUT_RESISTOR",
                    severity="critical",
                    message=f"LED {part.id} has no current-limiting resistor in series.",
                    part_id=part.id,
                )
            )

    # DANGLING_WIRE
    for part in snapped_parts:
        if part.type != "wire":
            continue
        if part.ambiguous_pins:
            continue
        try:
            n0 = hole_to_node(part.pins["0"], board)
        except (KeyError, ValueError):
            continue
        root = nl.dsu.find(n0)
        members = nets.get(root, [])
        connected = False
        for m in members:
            if m in ("VCC", "GND"):
                connected = True
                break
            if "." in m:
                pid = m.rsplit(".", 1)[0]
                other = parts_by_id.get(pid)
                if other is not None and other.type != "wire":
                    connected = True
                    break
        if not connected:
            findings.append(
                Finding(
                    code="DANGLING_WIRE",
                    severity="warning",
                    message=f"Wire {part.id} is dangling (not connected to any component).",
                    part_id=part.id,
                    holes=[part.pins.get("0", ""), part.pins.get("1", "")],
                )
            )

    return findings


# -----------------------------------------------------------------------------
# Placement-mode diff (hole-by-hole)
# -----------------------------------------------------------------------------

def _match_parts_by_type(
    ref_parts: list[tuple[str, str, dict[str, str]]],
    act_parts: list[tuple[str, str, dict[str, str]]],
) -> tuple[dict[str, str], list[str], list[str]]:
    """Greedy match reference<->actual parts by exact id if possible, then by type."""
    ref_by_id = {pid: (pid, ptype, pins) for pid, ptype, pins in ref_parts}
    act_by_id = {pid: (pid, ptype, pins) for pid, ptype, pins in act_parts}
    matches: dict[str, str] = {}
    used_actual: set[str] = set()

    for pid in list(ref_by_id):
        if pid in act_by_id:
            matches[pid] = pid
            used_actual.add(pid)

    for pid, ptype, _ in ref_parts:
        if pid in matches:
            continue
        for apid, aptype, _ in act_parts:
            if apid in used_actual:
                continue
            if aptype == ptype:
                matches[pid] = apid
                used_actual.add(apid)
                break

    missing = [pid for pid, _, _ in ref_parts if pid not in matches]
    extra = [apid for apid, _, _ in act_parts if apid not in used_actual]
    return matches, missing, extra


def check_placement(
    snapped_parts: list[SnappedPart],
    reference: ReferenceCircuit,
    board: BoardSpec,
) -> list[Finding]:
    findings: list[Finding] = []
    ref_parts = _ref_part_pins(reference)
    act_parts = [
        (p.id, p.type, dict(p.pins))
        for p in snapped_parts
        if p.type != "wire"
    ]
    matches, missing, extra = _match_parts_by_type(ref_parts, act_parts)
    act_lookup = {pid: pins for pid, _ptype, pins in act_parts}

    for pid in missing:
        findings.append(
            Finding(
                code="MISSING_PART",
                severity="critical",
                message=f"Reference part {pid} is missing from the actual circuit.",
                part_id=pid,
            )
        )
    for pid in extra:
        findings.append(
            Finding(
                code="EXTRA_PART",
                severity="warning",
                message=f"Actual part {pid} is not in the reference circuit.",
                part_id=pid,
            )
        )

    for ref_pid, _ptype, ref_pins in ref_parts:
        if ref_pid not in matches:
            continue
        act_pid = matches[ref_pid]
        act_pins = act_lookup.get(act_pid, {})
        for pin_name, expected_hole in ref_pins.items():
            actual_hole = act_pins.get(pin_name)
            if actual_hole is None:
                findings.append(
                    Finding(
                        code="MISPLACED_PIN",
                        severity="critical",
                        message=(
                            f"Part {act_pid} pin {pin_name!r} is missing "
                            f"(expected hole {expected_hole})."
                        ),
                        part_id=act_pid,
                        expected_holes=[expected_hole],
                    )
                )
                continue
            if actual_hole != expected_hole:
                findings.append(
                    Finding(
                        code="MISPLACED_PIN",
                        severity="critical",
                        message=(
                            f"Part {act_pid} pin {pin_name!r} at {actual_hole}, "
                            f"expected {expected_hole}."
                        ),
                        part_id=act_pid,
                        holes=[actual_hole],
                        expected_holes=[expected_hole],
                    )
                )

    return findings


# -----------------------------------------------------------------------------
# Connectivity-mode diff (bipartite VF2 isomorphism)
# -----------------------------------------------------------------------------

def _find_reversed_leds(
    ref_parts: list[tuple[str, str, dict[str, str]]],
    act_parts: list[tuple[str, str, dict[str, str]]],
    mapping: dict,
) -> list[str]:
    """Given a successful relaxed-isomorphism mapping (polar labels dropped),
    identify LEDs whose anode/cathode roles are swapped relative to reference.
    """
    ref_leds = {pid: pins for pid, ptype, pins in ref_parts if ptype == "led"}
    act_leds = {pid: pins for pid, ptype, pins in act_parts if ptype == "led"}
    reversed_ids: list[str] = []
    # mapping maps ref node -> act node
    for ref_led_id in ref_leds:
        ref_part_node = ("part", ref_led_id)
        act_part_node = mapping.get(ref_part_node)
        if act_part_node is None:
            continue
        act_led_id = act_part_node[1]
        act_pins = act_leds.get(act_led_id, {})
        ref_pins = ref_leds[ref_led_id]
        # For each ref pin ("anode"/"cathode"), look up what the mapping says.
        ref_anode_pin_node = ("pin", ref_led_id, "anode")
        act_anode_pin_mapped = mapping.get(ref_anode_pin_node)
        if act_anode_pin_mapped is None:
            continue
        _kind, _apid, act_pin_name = act_anode_pin_mapped
        # If the actual pin that fills ref's anode role is itself named "cathode"
        # in the actual LED, the actual LED is reversed.
        if act_pin_name == "cathode":
            reversed_ids.append(act_led_id)
    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for lid in reversed_ids:
        if lid not in seen:
            seen.add(lid)
            out.append(lid)
    return out


def _diff_findings(
    ref_parts: list[tuple[str, str, dict[str, str]]],
    act_parts: list[tuple[str, str, dict[str, str]]],
    ref_nets: dict[str, list[str]],
    act_nets: dict[str, list[str]],
) -> list[Finding]:
    findings: list[Finding] = []
    ref_counts = Counter(p[1] for p in ref_parts)
    act_counts = Counter(p[1] for p in act_parts)

    for ptype, count in ref_counts.items():
        delta = count - act_counts.get(ptype, 0)
        if delta > 0:
            ref_ids = [pid for pid, pt, _ in ref_parts if pt == ptype]
            findings.append(
                Finding(
                    code="MISSING_PART",
                    severity="critical",
                    message=f"Missing {delta} {ptype}(s) (expected ids like {ref_ids}).",
                )
            )
    for ptype, count in act_counts.items():
        delta = count - ref_counts.get(ptype, 0)
        if delta > 0:
            act_ids = [pid for pid, pt, _ in act_parts if pt == ptype]
            findings.append(
                Finding(
                    code="EXTRA_PART",
                    severity="warning",
                    message=f"Extra {delta} {ptype}(s) present (ids like {act_ids}).",
                )
            )

    if ref_counts == act_counts:
        # Same part census but non-isomorphic => connectivity mismatch.
        ref_rail_nets = sum(
            1 for _nid, members in ref_nets.items() if "VCC" in members or "GND" in members
        )
        act_rail_nets = sum(
            1 for _nid, members in act_nets.items() if "VCC" in members or "GND" in members
        )
        if ref_rail_nets > act_rail_nets:
            findings.append(
                Finding(
                    code="OPEN_CONNECTION",
                    severity="critical",
                    message=(
                        f"Expected {ref_rail_nets} power-rail connections but found "
                        f"only {act_rail_nets} — some rail wiring is missing."
                    ),
                )
            )
        elif ref_rail_nets < act_rail_nets:
            findings.append(
                Finding(
                    code="UNEXPECTED_CONNECTION",
                    severity="critical",
                    message=(
                        f"Found {act_rail_nets} power-rail connections, reference "
                        f"only has {ref_rail_nets}."
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    code="OPEN_CONNECTION",
                    severity="critical",
                    message=(
                        "Circuit topology does not match the reference "
                        "(nets differ in how parts are tied together)."
                    ),
                )
            )

    return findings


def check_connectivity(
    snapped_parts: list[SnappedPart],
    nl: NetlistResult,
    reference: ReferenceCircuit,
) -> list[Finding]:
    findings: list[Finding] = []
    ref_parts = _ref_part_pins(reference)
    act_parts = _actual_part_pins(snapped_parts, nl)
    ref_nets = _ref_netlist(reference)
    act_nets = nl.nets

    g_ref = _build_iso_graph(ref_parts, ref_nets, polar_pin_labels=True)
    g_act = _build_iso_graph(act_parts, act_nets, polar_pin_labels=True)

    strict_match, _mapping = _is_isomorphic(g_ref, g_act)
    if strict_match:
        return findings

    # Relaxed: ignore anode/cathode distinction so we can detect reversed LEDs.
    g_ref_relaxed = _build_iso_graph(ref_parts, ref_nets, polar_pin_labels=False)
    g_act_relaxed = _build_iso_graph(act_parts, act_nets, polar_pin_labels=False)
    relaxed_match, relaxed_mapping = _is_isomorphic(g_ref_relaxed, g_act_relaxed)
    if relaxed_match and relaxed_mapping is not None:
        reversed_leds = _find_reversed_leds(ref_parts, act_parts, relaxed_mapping)
        if reversed_leds:
            for led_id in reversed_leds:
                findings.append(
                    Finding(
                        code="REVERSED_POLARITY",
                        severity="critical",
                        message=f"LED {led_id} is wired with reversed polarity.",
                        part_id=led_id,
                    )
                )
            return findings
        # Relaxed matched but no polar swap found => structural issue elsewhere.
        findings.append(
            Finding(
                code="REVERSED_POLARITY",
                severity="critical",
                message="Polar part orientation does not match reference.",
            )
        )
        return findings

    findings.extend(_diff_findings(ref_parts, act_parts, ref_nets, act_nets))
    if not findings:
        findings.append(
            Finding(
                code="OPEN_CONNECTION",
                severity="critical",
                message="Actual circuit is not electrically equivalent to the reference.",
            )
        )
    return findings


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def verify_circuit(
    snapped_parts: list[SnappedPart],
    reference: ReferenceCircuit | None = None,
    board: BoardSpec | None = None,
    rail_bindings: dict[str, str] | None = None,
    mode: Mode = "connectivity",
    extra_findings: list[Finding] | None = None,
) -> InspectionVerdict:
    if board is None:
        raise ValueError("verify_circuit requires a BoardSpec")

    nl = build_netlist(snapped_parts, board, rail_bindings)
    findings: list[Finding] = []
    if extra_findings:
        findings.extend(extra_findings)
    findings.extend(check_safety(snapped_parts, nl, board))

    if reference is not None:
        if mode == "placement":
            findings.extend(check_placement(snapped_parts, reference, board))
        else:
            findings.extend(check_connectivity(snapped_parts, nl, reference))

    critical = any(f.severity == "critical" for f in findings)
    passed = not critical
    return InspectionVerdict(
        passed=passed,
        light_color="GREEN" if passed else "RED",
        snapped_parts=snapped_parts,
        actual_netlist=nl.nets,
        findings=findings,
    )
