"""Functional coverage tracking for accelerator verification."""
import logging

logger = logging.getLogger(__name__)


class CoveragePoint:
    """A single coverage point with bins.

    Tracks how many times each bin has been hit.
    """

    def __init__(self, name: str, bins: list):
        """
        Args:
            name: Coverage point name.
            bins: List of (label, predicate) tuples. predicate(value) -> bool.
        """
        self.name = name
        self.bins = bins
        self.hit_counts = {label: 0 for label, _ in bins}

    def sample(self, value):
        """Sample a value against all bins."""
        for label, predicate in self.bins:
            if predicate(value):
                self.hit_counts[label] += 1

    @property
    def covered(self) -> bool:
        """True if all bins have been hit at least once."""
        return all(c > 0 for c in self.hit_counts.values())

    @property
    def coverage_pct(self) -> float:
        """Percentage of bins covered."""
        if not self.bins:
            return 100.0
        return sum(1 for c in self.hit_counts.values() if c > 0) / len(self.bins) * 100


class CoverageGroup:
    """A group of coverage points.

    Example:
        >>> cg = CoverageGroup("data_range")
        >>> cg.add_point("value", [
        ...     ("zero", lambda v: v == 0),
        ...     ("positive", lambda v: v > 0),
        ...     ("negative", lambda v: v < 0),
        ...     ("max_int", lambda v: v == 0xFFFFFFFF),
        ... ])
        >>> cg.sample("value", 42)
        >>> cg.sample("value", 0)
        >>> print(cg.report())
    """

    def __init__(self, name: str):
        self.name = name
        self.points: dict = {}  # name -> CoveragePoint

    def add_point(self, name: str, bins: list) -> CoveragePoint:
        """Add a coverage point.

        Args:
            name: Point name.
            bins: List of (label, predicate) tuples.

        Returns:
            The created CoveragePoint.
        """
        cp = CoveragePoint(name, bins)
        self.points[name] = cp
        return cp

    def sample(self, point_name: str, value):
        """Sample a value for a specific coverage point."""
        if point_name in self.points:
            self.points[point_name].sample(value)

    @property
    def covered(self) -> bool:
        """True if all points are fully covered."""
        return all(p.covered for p in self.points.values())

    @property
    def coverage_pct(self) -> float:
        """Overall coverage percentage."""
        if not self.points:
            return 100.0
        return sum(p.coverage_pct for p in self.points.values()) / len(self.points)

    def report(self) -> str:
        """Generate a coverage report."""
        lines = [f"Coverage Group '{self.name}': {self.coverage_pct:.1f}%"]
        for name, point in self.points.items():
            lines.append(f"  {name}: {point.coverage_pct:.1f}%")
            for label, count in point.hit_counts.items():
                marker = "X" if count > 0 else " "
                lines.append(f"    [{marker}] {label}: {count} hits")
        return "\n".join(lines)
