"""Tests for the Sherlock Ohms engine (schemas, snapping, netlist, verifier)."""

from __future__ import annotations

import pytest

from engine import (
    BoardSpec,
    HoleLUT,
    Keypoint,
    RawDetection,
    ReferenceCircuit,
    build_netlist,
    snap_detections,
    verify_circuit,
)
from engine.snapping import pin_names_for

RAIL_BINDINGS = {"top_plus": "VCC", "top_minus": "GND"}


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------

def _make_det(
    det_id: str,
    part_type: str,
    lut: HoleLUT,
    hole_a: str,
    hole_b: str,
    *,
    conf: float = 0.9,
    kp_conf: float = 0.9,
    jitter_a: tuple[float, float] = (0.0, 0.0),
    jitter_b: tuple[float, float] = (0.0, 0.0),
) -> RawDetection:
    xa, ya = lut.holes[hole_a]
    xb, yb = lut.holes[hole_b]
    xa += jitter_a[0]
    ya += jitter_a[1]
    xb += jitter_b[0]
    yb += jitter_b[1]
    bbox = (
        min(xa, xb) - 5.0,
        min(ya, yb) - 5.0,
        max(xa, xb) + 5.0,
        max(ya, yb) + 5.0,
    )
    return RawDetection(
        id=det_id,
        type=part_type,
        conf=conf,
        bbox=bbox,
        keypoints=(
            Keypoint(x=xa, y=ya, conf=kp_conf),
            Keypoint(x=xb, y=yb, conf=kp_conf),
        ),
    )


def _reference_led_circuit() -> ReferenceCircuit:
    """Classic LED-with-current-limiting-resistor between VCC and GND."""
    return ReferenceCircuit(
        name="led_with_resistor",
        rails={"VCC": "top_plus", "GND": "top_minus"},
        parts=[
            {
                "id": "R1",
                "type": "resistor",
                "pins": {"0": "VCC", "1": "n_bridge"},
            },
            {
                "id": "LED1",
                "type": "led",
                "pins": {"anode": "n_bridge", "cathode": "GND"},
            },
        ],
    )


def _led_actual_detections(lut, col_vcc, col_bridge, col_gnd):
    """Build the detection list for an LED-with-resistor on a given set of columns."""
    return [
        _make_det("W_VCC", "wire", lut, f"top_plus_{col_vcc}", f"a{col_vcc}"),
        _make_det("R1", "resistor", lut, f"a{col_vcc}", f"a{col_bridge}"),
        _make_det("LED1", "led", lut, f"c{col_bridge}", f"c{col_gnd}"),
        _make_det("W_GND", "wire", lut, f"a{col_gnd}", f"top_minus_{col_gnd}"),
    ]


# -----------------------------------------------------------------------------
# Schema / snapping sanity
# -----------------------------------------------------------------------------

def test_board_presets_and_lut_positions():
    board = BoardSpec.full_830()
    assert board.columns == 63
    assert board.has_rails is True
    assert board.pitch_px == pytest.approx(20.32)

    lut = HoleLUT.generate_synthetic(board)
    # Columns on row a should be one pitch apart in x.
    xa, ya = lut.holes["a1"]
    xb, yb = lut.holes["a2"]
    assert yb == pytest.approx(ya)
    assert xb - xa == pytest.approx(board.pitch_px)
    # Half and mini presets.
    assert BoardSpec.half_400().columns == 30
    mini = BoardSpec.mini_170()
    assert mini.columns == 17
    assert mini.has_rails is False
    assert not any(h.startswith("top_plus") for h in HoleLUT.generate_synthetic(mini).holes)


def test_snap_detections_assigns_polar_and_default_pin_names():
    assert pin_names_for("led") == ("anode", "cathode")
    assert pin_names_for("resistor") == ("0", "1")
    board = BoardSpec.half_400()
    lut = HoleLUT.generate_synthetic(board)
    detections = [
        _make_det("R1", "resistor", lut, "a3", "a8"),
        _make_det("LED1", "led", lut, "c8", "c13"),
    ]
    snapped, uncertain = snap_detections(detections, lut)
    assert uncertain == []
    r1 = next(p for p in snapped if p.id == "R1")
    led = next(p for p in snapped if p.id == "LED1")
    assert r1.pins == {"0": "a3", "1": "a8"}
    assert led.pins == {"anode": "c8", "cathode": "c13"}
    assert r1.ambiguous_pins == []
    assert led.ambiguous_pins == []


# -----------------------------------------------------------------------------
# Core verifier scenarios
# -----------------------------------------------------------------------------

def test_exact_match_full_830_passes():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    reference = _reference_led_circuit()

    detections = _led_actual_detections(lut, col_vcc=5, col_bridge=10, col_gnd=15)
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=reference,
        board=board,
        rail_bindings=RAIL_BINDINGS,
        mode="connectivity",
    )
    assert verdict.passed is True
    assert verdict.light_color == "GREEN"
    assert verdict.findings == []
    # VCC and GND labels should be present on the actual netlist.
    all_members = [m for members in verdict.actual_netlist.values() for m in members]
    assert "VCC" in all_members
    assert "GND" in all_members


def test_equivalent_circuit_different_columns_with_jumper_passes_connectivity():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    reference = _reference_led_circuit()

    detections = [
        _make_det("W_VCC", "wire", lut, "top_plus_20", "a20"),
        _make_det("R1", "resistor", lut, "a20", "a25"),
        # Jumper wire bridges n_bridge from col 25 to col 30 — redundant extension.
        _make_det("W_BR", "wire", lut, "a25", "a30"),
        _make_det("LED1", "led", lut, "c30", "c35"),
        _make_det("W_GND", "wire", lut, "a35", "top_minus_35"),
    ]
    snapped, uncertain = snap_detections(detections, lut)
    assert uncertain == []
    verdict = verify_circuit(
        snapped,
        reference=reference,
        board=board,
        rail_bindings=RAIL_BINDINGS,
        mode="connectivity",
    )
    assert verdict.passed is True
    assert verdict.light_color == "GREEN"


def test_cross_board_equivalence_full_830_reference_on_half_400_passes():
    """Reference was authored on a full-size board but the actual wiring is on half_400."""
    reference = _reference_led_circuit()  # Topologically abstract — board-agnostic.
    board = BoardSpec.half_400()
    lut = HoleLUT.generate_synthetic(board)

    detections = _led_actual_detections(lut, col_vcc=3, col_bridge=8, col_gnd=13)
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=reference,
        board=board,
        rail_bindings=RAIL_BINDINGS,
        mode="connectivity",
    )
    assert verdict.passed is True
    assert verdict.light_color == "GREEN"


def test_off_by_one_row_crossing_gap_fails_connectivity():
    """Place LED cathode on the wrong half of the board — misses the GND wire."""
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    reference = _reference_led_circuit()

    detections = [
        _make_det("W_VCC", "wire", lut, "top_plus_5", "f5"),
        _make_det("R1", "resistor", lut, "f5", "f10"),
        # LED cathode should be on h15 (right side) to reach GND wire; placed on e15 instead.
        _make_det("LED1", "led", lut, "h10", "e15"),
        _make_det("W_GND", "wire", lut, "f15", "top_minus_15"),
    ]
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=reference,
        board=board,
        rail_bindings=RAIL_BINDINGS,
        mode="connectivity",
    )
    assert verdict.passed is False
    assert verdict.light_color == "RED"
    assert any(
        f.code in ("OPEN_CONNECTION", "UNEXPECTED_CONNECTION", "MISSING_PART")
        for f in verdict.findings
    )


def test_reversed_led_detected_as_reversed_polarity():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    reference = _reference_led_circuit()

    # Build exact-match wiring, then flip the LED keypoints so anode<->cathode swap.
    detections = _led_actual_detections(lut, col_vcc=5, col_bridge=10, col_gnd=15)
    # Replace LED1 with pins swapped
    detections = [d for d in detections if d.id != "LED1"]
    detections.append(_make_det("LED1", "led", lut, "c15", "c10"))  # anode=c15, cathode=c10
    snapped, _ = snap_detections(detections, lut)

    verdict = verify_circuit(
        snapped,
        reference=reference,
        board=board,
        rail_bindings=RAIL_BINDINGS,
        mode="connectivity",
    )
    assert verdict.passed is False
    assert verdict.light_color == "RED"
    codes = [f.code for f in verdict.findings]
    assert "REVERSED_POLARITY" in codes


def test_power_to_ground_short_detected():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    detections = [
        _make_det("W_SHORT", "wire", lut, "top_plus_5", "top_minus_5"),
        # Add some bystander part so the netlist isn't empty.
        _make_det("R1", "resistor", lut, "a10", "a15"),
    ]
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=None,
        board=board,
        rail_bindings=RAIL_BINDINGS,
    )
    assert verdict.passed is False
    assert verdict.light_color == "RED"
    codes = [f.code for f in verdict.findings]
    assert "POWER_SHORT" in codes


def test_shorted_resistor_detected():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    # Both pins on the same column => same column_left net => shorted.
    detections = [_make_det("R1", "resistor", lut, "a10", "c10")]
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=None,
        board=board,
        rail_bindings=RAIL_BINDINGS,
    )
    assert verdict.passed is False
    assert verdict.light_color == "RED"
    assert any(f.code == "SHORTED_PART" and f.part_id == "R1" for f in verdict.findings)


def test_led_without_resistor_detected():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    detections = [
        _make_det("W_VCC", "wire", lut, "top_plus_5", "a5"),
        _make_det("LED1", "led", lut, "a5", "a10"),
        _make_det("W_GND", "wire", lut, "a10", "top_minus_10"),
    ]
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=None,
        board=board,
        rail_bindings=RAIL_BINDINGS,
    )
    assert verdict.passed is False
    assert verdict.light_color == "RED"
    codes = [f.code for f in verdict.findings]
    assert "LED_WITHOUT_RESISTOR" in codes


def test_ambiguous_pin_when_keypoint_far_from_grid():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    # Push one keypoint 12 px off from the hole — well above the 8 px threshold.
    detections = [
        _make_det(
            "R1",
            "resistor",
            lut,
            "a10",
            "a15",
            jitter_b=(12.0, 0.0),
        )
    ]
    snapped, findings = snap_detections(detections, lut)
    assert len(findings) == 1
    uncertain = findings[0]
    assert uncertain.code == "UNCERTAIN"
    assert uncertain.severity == "warning"
    assert uncertain.part_id == "R1"

    r1 = snapped[0]
    assert "1" in r1.ambiguous_pins
    assert "0" not in r1.ambiguous_pins


# -----------------------------------------------------------------------------
# Netlist / DSU sanity
# -----------------------------------------------------------------------------

def test_netlist_merges_rails_through_wire_and_tags_vcc():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    detections = [
        _make_det("W", "wire", lut, "top_plus_1", "a1"),
        _make_det("R1", "resistor", lut, "a1", "a5"),
    ]
    snapped, _ = snap_detections(detections, lut)
    nl = build_netlist(snapped, board, RAIL_BINDINGS)
    vcc_root = nl.dsu.find("top_plus")
    assert vcc_root == nl.dsu.find("col_1_left")
    assert "VCC" in nl.nets[vcc_root]
    assert "R1.0" in nl.nets[vcc_root]
    other_root = nl.dsu.find("col_5_left")
    assert other_root != vcc_root
    assert "R1.1" in nl.nets[other_root]


def test_placement_mode_detects_misplaced_pin():
    board = BoardSpec.half_400()
    lut = HoleLUT.generate_synthetic(board)
    # Reference uses literal hole ids for placement mode.
    reference = ReferenceCircuit(
        name="placement_ref",
        rails={"VCC": "top_plus", "GND": "top_minus"},
        parts=[
            {"id": "R1", "type": "resistor", "pins": {"0": "a3", "1": "a8"}},
        ],
    )
    # Actual R1 pin 1 is one column off.
    detections = [_make_det("R1", "resistor", lut, "a3", "a9")]
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=reference,
        board=board,
        rail_bindings=RAIL_BINDINGS,
        mode="placement",
    )
    assert verdict.passed is False
    assert verdict.light_color == "RED"
    assert any(
        f.code == "MISPLACED_PIN" and f.expected_holes == ["a8"] and f.holes == ["a9"]
        for f in verdict.findings
    )


def test_ic_not_straddling_center_gap():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    # Both ends on left side (a-e) => not straddling.
    detections = [_make_det("IC1", "ic_dip", lut, "a10", "e10")]
    snapped, _ = snap_detections(detections, lut)
    verdict = verify_circuit(
        snapped,
        reference=None,
        board=board,
        rail_bindings=RAIL_BINDINGS,
    )
    assert verdict.passed is False
    assert any(f.code == "IC_NOT_STRADDLING" for f in verdict.findings)
