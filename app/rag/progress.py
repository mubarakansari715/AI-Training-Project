"""Progress of long-running work, such as indexing a 200-page PDF.

A Progress callback receives a short stage label ("Embedding passages (420 of
1,194)") and how far along the whole job is, from 0.0 to 1.0. Each step of the
work reports its own 0-1 range; `scaled` maps that into the share of the
whole job the step takes, so nested steps don't need to know about each other.
"""

from __future__ import annotations

from collections.abc import Callable

Progress = Callable[[str, float], None]


def report(progress: Progress | None, label: str, fraction: float) -> None:
    if progress is not None:
        progress(label, min(1.0, max(0.0, fraction)))


def scaled(progress: Progress | None, start: float, end: float) -> Progress | None:
    """A callback whose 0-1 range covers `start`-`end` of `progress`."""
    if progress is None:
        return None
    return lambda label, fraction: progress(label, start + (end - start) * fraction)
