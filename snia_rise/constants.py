"""
Global constants for the `snia_rise` package.

This module is intended to be the single source of truth for package-wide
constants that need to be shared across modules.

Usage:
    from snia_rise.constants import T_PIVOT

Avoid re-defining these values in individual modules; import them from here
instead to ensure consistency across the codebase.
"""

from typing import Final

# Pivot time for the power-law rise (the typical time to reach ~40% of the maximum flux)
EPS: Final[float] = 1e-10
T_PIVOT: Final[float] = 8.0

__all__ = ["T_PIVOT", "EPS"]
