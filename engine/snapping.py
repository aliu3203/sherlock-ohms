from __future__ import annotations

from .board import HoleLUT
from .schemas import Finding, RawDetection, SnappedPart

POLAR_PIN_NAMES: dict[str, tuple[str, str]] = {
    "led": ("anode", "cathode"),
    "cap_electrolytic": ("anode", "cathode"),
}
DEFAULT_PIN_NAMES: tuple[str, str] = ("0", "1")


def pin_names_for(part_type: str) -> tuple[str, str]:
    return POLAR_PIN_NAMES.get(part_type, DEFAULT_PIN_NAMES)


def snap_detections(
    detections: list[RawDetection],
    lut: HoleLUT,
    max_snap_px: float = 8.0,
    min_kpt_conf: float = 0.50,
) -> tuple[list[SnappedPart], list[Finding]]:
    """Snap each detection's keypoints to the nearest breadboard hole."""
    snapped: list[SnappedPart] = []
    findings: list[Finding] = []

    for det in detections:
        pin_names = pin_names_for(det.type)
        pins: dict[str, str] = {}
        ambiguous: list[str] = []
        for pin_name, kp in zip(pin_names, det.keypoints):
            hid, dist = lut.nearest(kp.x, kp.y)
            pins[pin_name] = hid
            if dist > max_snap_px or kp.conf < min_kpt_conf:
                ambiguous.append(pin_name)

        if ambiguous:
            findings.append(
                Finding(
                    code="UNCERTAIN",
                    severity="warning",
                    message=(
                        f"Part {det.id} ({det.type}) has uncertain pin snap "
                        f"for {ambiguous}"
                    ),
                    part_id=det.id,
                    holes=[pins[p] for p in ambiguous],
                    bbox=det.bbox,
                )
            )

        snapped.append(
            SnappedPart(
                id=det.id,
                type=det.type,
                pins=pins,
                bbox=det.bbox,
                conf=det.conf,
                ambiguous_pins=ambiguous,
            )
        )

    return snapped, findings
