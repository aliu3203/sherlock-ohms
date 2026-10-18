from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PartType = Literal[
    "wire",
    "resistor",
    "led",
    "cap_ceramic",
    "cap_electrolytic",
    "button",
    "ic_dip",
]

FindingCode = Literal[
    "MISSING_PART",
    "EXTRA_PART",
    "MISPLACED_PIN",
    "OPEN_CONNECTION",
    "UNEXPECTED_CONNECTION",
    "REVERSED_POLARITY",
    "UNCERTAIN",
    "POWER_SHORT",
    "SHORTED_PART",
    "LED_WITHOUT_RESISTOR",
    "IC_NOT_STRADDLING",
    "DANGLING_WIRE",
]

Severity = Literal["critical", "warning"]


class Keypoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    conf: float
    vis: int = 2


class RawDetection(BaseModel):
    id: str
    type: PartType
    conf: float
    bbox: tuple[float, float, float, float]
    keypoints: tuple[Keypoint, Keypoint]


class SnappedPart(BaseModel):
    id: str
    type: str
    pins: dict[str, str]
    bbox: tuple[float, float, float, float]
    conf: float
    ambiguous_pins: list[str] = Field(default_factory=list)


class ReferenceCircuit(BaseModel):
    name: str
    rails: dict[str, str]
    parts: list[dict]


class Finding(BaseModel):
    code: FindingCode
    severity: Severity
    message: str
    part_id: str | None = None
    holes: list[str] = Field(default_factory=list)
    expected_holes: list[str] = Field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None


class InspectionVerdict(BaseModel):
    passed: bool
    light_color: Literal["GREEN", "RED"]
    snapped_parts: list[SnappedPart]
    actual_netlist: dict[str, list[str]]
    findings: list[Finding] = Field(default_factory=list)
