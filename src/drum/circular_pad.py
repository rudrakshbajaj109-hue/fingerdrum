from dataclasses import dataclass

PixelPoint = tuple[float, float]


@dataclass(frozen=True)
class CircularPad:
    pad_id: str
    center: PixelPoint
    radius: float

    def contains(self, point: PixelPoint) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius