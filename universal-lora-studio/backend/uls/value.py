"""Provenance-tagged values.

Every number that reaches the user carries a record of where it came from.
This is the enforcement mechanism for the project's hardest rule: the UI must
never present a guess as a measurement.

There is deliberately no way to construct a Value without stating an origin,
and no ``.value`` accessor that silently yields ``None`` for an unavailable
reading -- callers must handle absence explicitly via ``.get()`` or ``.or_else()``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Origin(str, enum.Enum):
    """Where a value came from. Rendered verbatim in the UI."""

    DETECTED = "detected"
    """Read from the machine this process is running on."""

    CONFIGURED = "configured"
    """Entered by a person, or loaded from a saved/simulated profile."""

    ESTIMATED = "estimated"
    """Computed by a model. May not match reality."""

    MEASURED = "measured"
    """Sampled from a live run in progress (as opposed to predicted)."""

    UNAVAILABLE = "unavailable"
    """Could not be determined on this platform. Renders as N/A."""


@dataclass(frozen=True)
class Value(Generic[T]):
    """A value plus its provenance, and a note explaining how it was obtained."""

    _value: T | None
    origin: Origin
    note: str = ""
    unit: str = ""
    #: For ESTIMATED values, an optional (low, high) plausible range.
    interval: tuple[float, float] | None = None

    # -- constructors ---------------------------------------------------

    @classmethod
    def detected(cls, value: T, note: str = "", unit: str = "") -> "Value[T]":
        return cls(value, Origin.DETECTED, note, unit)

    @classmethod
    def configured(cls, value: T, note: str = "", unit: str = "") -> "Value[T]":
        return cls(value, Origin.CONFIGURED, note, unit)

    @classmethod
    def measured(cls, value: T, note: str = "", unit: str = "") -> "Value[T]":
        return cls(value, Origin.MEASURED, note, unit)

    @classmethod
    def estimated(
        cls,
        value: T,
        note: str = "",
        unit: str = "",
        interval: tuple[float, float] | None = None,
    ) -> "Value[T]":
        return cls(value, Origin.ESTIMATED, note, unit, interval)

    @classmethod
    def unavailable(cls, note: str = "") -> "Value[Any]":
        """No reading could be obtained. This is a legitimate outcome, not an error."""
        return cls(None, Origin.UNAVAILABLE, note)

    @classmethod
    def probe(
        cls, fn: Callable[[], T], note: str = "", unit: str = "", on_fail: str = ""
    ) -> "Value[T]":
        """Run a detection callable, degrading to UNAVAILABLE instead of raising.

        Detection code runs against drivers and vendor tools that are absent,
        stale or permission-gated on most machines. A failed probe is normal
        and must never take down a page.
        """
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - any probe failure is just absence
            reason = on_fail or f"{type(exc).__name__}: {exc}"
            return cls(None, Origin.UNAVAILABLE, reason)
        if result is None:
            return cls(None, Origin.UNAVAILABLE, on_fail or "probe returned nothing")
        return cls(result, Origin.DETECTED, note, unit)

    # -- access ---------------------------------------------------------

    @property
    def known(self) -> bool:
        return self.origin is not Origin.UNAVAILABLE and self._value is not None

    @property
    def factual(self) -> bool:
        """True only for values that describe something real, not predicted."""
        return self.origin in (Origin.DETECTED, Origin.CONFIGURED, Origin.MEASURED)

    def get(self) -> T:
        if not self.known:
            raise ValueError(f"value is unavailable: {self.note or 'no reason given'}")
        return self._value  # type: ignore[return-value]

    def or_else(self, fallback: T) -> T:
        return self._value if self.known else fallback  # type: ignore[return-value]

    def map(self, fn: Callable[[T], Any]) -> "Value[Any]":
        """Transform the payload, preserving provenance."""
        if not self.known:
            return self
        return Value(fn(self._value), self.origin, self.note, self.unit, self.interval)

    def relabel(self, origin: Origin, note: str = "") -> "Value[T]":
        """Re-stamp provenance.

        Used when a detected reading is overridden by a person: the number may
        survive, but it stops being a measurement.
        """
        return Value(self._value, origin, note or self.note, self.unit, self.interval)

    # -- rendering ------------------------------------------------------

    def display(self) -> str:
        if not self.known:
            return "N/A"
        v = self._value
        if isinstance(v, float):
            text = f"{v:,.2f}".rstrip("0").rstrip(".")
        elif isinstance(v, int):
            text = f"{v:,}"
        else:
            text = str(v)
        return f"{text} {self.unit}".strip()

    def to_dict(self) -> dict[str, Any]:
        """Wire format. ``origin`` is required reading for any renderer."""
        payload: dict[str, Any] = {
            "value": self._value,
            "origin": self.origin.value,
            "display": self.display(),
            "label": ORIGIN_LABELS[self.origin],
        }
        if self.unit:
            payload["unit"] = self.unit
        if self.note:
            payload["note"] = self.note
        if self.interval:
            payload["interval"] = list(self.interval)
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.origin.value}: {self.display()}>"


ORIGIN_LABELS = {
    Origin.DETECTED: "Detected",
    Origin.CONFIGURED: "Configured",
    Origin.ESTIMATED: "Estimated",
    Origin.MEASURED: "Measured",
    Origin.UNAVAILABLE: "Not available",
}


@dataclass
class Report:
    """A named group of values, for a panel or card in the UI."""

    title: str
    items: dict[str, Value[Any]] = field(default_factory=dict)

    def add(self, key: str, value: Value[Any]) -> "Report":
        self.items[key] = value
        return self

    @property
    def completeness(self) -> float:
        """Fraction of fields that produced a reading.

        Surfaced in the UI so a person can tell a sparse report from a
        confident one, rather than reading gaps as zeros.
        """
        if not self.items:
            return 0.0
        return sum(1 for v in self.items.values() if v.known) / len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "completeness": round(self.completeness, 3),
            "items": {k: v.to_dict() for k, v in self.items.items()},
        }
