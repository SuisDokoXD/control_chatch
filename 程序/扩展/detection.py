from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DetectionResult:
    object_id: str
    shape: str
    shape_confidence: float
    center_px: tuple[float, float]
    bbox_px: tuple[float, float, float, float]
    color: Optional[str] = None
    color_confidence: float = 0.0
    tray_xy_mm: Optional[tuple[float, float]] = None
    height_mm: Optional[float] = None
    angle_deg: Optional[float] = None
    target_box: Optional[str] = None
    valid: bool = True
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def category_key(self):
        return f"{self.color}_{self.shape}" if self.color else None

    @property
    def overall_confidence(self):
        return min(self.shape_confidence, self.color_confidence or 1.0)

    def reject(self, reason: str):
        self.valid = False
        if reason not in self.rejection_reasons:
            self.rejection_reasons.append(reason)
