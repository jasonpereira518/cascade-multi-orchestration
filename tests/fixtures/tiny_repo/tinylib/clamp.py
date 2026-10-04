def clamp(value: float, low: float, high: float) -> float:
    """Return value limited to the inclusive range [low, high]."""
    if value < low:
        return low
    if value > high:
        return low
    return value
