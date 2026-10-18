from __future__ import annotations

import math
from dataclasses import dataclass, field

ROWS_LEFT = "abcde"
ROWS_RIGHT = "fghij"
RAIL_PREFIXES = ("top_plus", "top_minus", "bottom_plus", "bottom_minus")


@dataclass
class BoardSpec:
    """A parametric breadboard layout."""

    preset_id: str
    columns: int
    pitch_mm: float = 2.54
    px_per_mm: float = 8.0
    split_rails: bool = False
    has_rails: bool = True
    width_px: float = 0.0
    height_px: float = 0.0

    @property
    def pitch_px(self) -> float:
        return self.pitch_mm * self.px_per_mm

    @classmethod
    def full_830(cls, split_rails: bool = False) -> "BoardSpec":
        return cls(
            preset_id="full_830",
            columns=63,
            split_rails=split_rails,
            has_rails=True,
            width_px=1400.0,
            height_px=520.0,
        )

    @classmethod
    def half_400(cls, split_rails: bool = False) -> "BoardSpec":
        return cls(
            preset_id="half_400",
            columns=30,
            split_rails=split_rails,
            has_rails=True,
            width_px=720.0,
            height_px=520.0,
        )

    @classmethod
    def mini_170(cls) -> "BoardSpec":
        return cls(
            preset_id="mini_170",
            columns=17,
            has_rails=False,
            width_px=440.0,
            height_px=200.0,
        )


def hole_to_node(hole_id: str, board: BoardSpec) -> str:
    """Map a breadboard hole id to its canonical DSU node name."""
    for prefix in RAIL_PREFIXES:
        full_prefix = prefix + "_"
        if hole_id.startswith(full_prefix):
            col = int(hole_id[len(full_prefix):])
            if board.split_rails:
                half = "a" if col <= board.columns // 2 else "b"
                return f"{prefix}_{half}"
            return prefix
    row = hole_id[0]
    col = int(hole_id[1:])
    if row in ROWS_LEFT:
        return f"col_{col}_left"
    if row in ROWS_RIGHT:
        return f"col_{col}_right"
    raise ValueError(f"Unknown hole id: {hole_id!r}")


@dataclass
class HoleLUT:
    """Rectified hole_id -> (x, y) pixel lookup table."""

    board: BoardSpec
    holes: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def generate_synthetic(cls, board: BoardSpec) -> "HoleLUT":
        lut = cls(board=board)
        pitch = board.pitch_px
        margin_x = pitch
        margin_y = pitch
        y = margin_y

        if board.has_rails:
            for c in range(1, board.columns + 1):
                lut.holes[f"top_plus_{c}"] = (margin_x + (c - 1) * pitch, y)
            y += pitch
            for c in range(1, board.columns + 1):
                lut.holes[f"top_minus_{c}"] = (margin_x + (c - 1) * pitch, y)
            y += 2 * pitch

        for row_idx, row in enumerate(ROWS_LEFT):
            yy = y + row_idx * pitch
            for c in range(1, board.columns + 1):
                lut.holes[f"{row}{c}"] = (margin_x + (c - 1) * pitch, yy)
        y += 5 * pitch
        y += 2 * pitch

        for row_idx, row in enumerate(ROWS_RIGHT):
            yy = y + row_idx * pitch
            for c in range(1, board.columns + 1):
                lut.holes[f"{row}{c}"] = (margin_x + (c - 1) * pitch, yy)
        y += 5 * pitch

        if board.has_rails:
            y += 2 * pitch
            for c in range(1, board.columns + 1):
                lut.holes[f"bottom_plus_{c}"] = (margin_x + (c - 1) * pitch, y)
            y += pitch
            for c in range(1, board.columns + 1):
                lut.holes[f"bottom_minus_{c}"] = (margin_x + (c - 1) * pitch, y)

        return lut

    @classmethod
    def from_reference_clicks(
        cls,
        board: BoardSpec,
        anchors: dict[str, tuple[float, float]],
    ) -> "HoleLUT":
        """Build a LUT from a small set of user-clicked anchor holes.

        Uses the synthetic grid and applies an average translation offset
        inferred from each anchor's (expected, observed) delta. For deployed
        hardware you'd fit a full homography; a translation suffices for
        rectified images.
        """
        synthetic = cls.generate_synthetic(board)
        if not anchors:
            return synthetic
        dxs, dys = [], []
        for hid, (px, py) in anchors.items():
            if hid not in synthetic.holes:
                continue
            sx, sy = synthetic.holes[hid]
            dxs.append(px - sx)
            dys.append(py - sy)
        if not dxs:
            return synthetic
        dx = sum(dxs) / len(dxs)
        dy = sum(dys) / len(dys)
        shifted = {h: (x + dx, y + dy) for h, (x, y) in synthetic.holes.items()}
        return cls(board=board, holes=shifted)

    def nearest(self, x: float, y: float) -> tuple[str, float]:
        best_id = ""
        best_d2 = float("inf")
        for hid, (hx, hy) in self.holes.items():
            d2 = (x - hx) ** 2 + (y - hy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_id = hid
        return best_id, math.sqrt(best_d2)

    def get(self, hole_id: str) -> tuple[float, float] | None:
        return self.holes.get(hole_id)

    def __contains__(self, hole_id: str) -> bool:
        return hole_id in self.holes

    def hole_to_node(self, hole_id: str) -> str:
        return hole_to_node(hole_id, self.board)
