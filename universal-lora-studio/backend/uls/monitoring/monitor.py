"""Live resource sampling during a run.

Reports what the platform exposes and nothing else. A missing temperature
sensor produces N/A, never a plausible-looking number, because a dashboard
that invents readings is worse than one with gaps in it -- gaps are obvious,
inventions are not.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..hardware.capability import Backend
from ..hardware.detection import engine
from ..hardware.providers.cpu import CPUProvider
from ..value import Value


@dataclass
class Sample:
    timestamp: float
    devices: list[dict[str, Any]]
    cpu: dict[str, Any]
    throughput: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "devices": self.devices,
            "cpu": self.cpu,
            "throughput": self.throughput,
        }


class Monitor:
    """Polls hardware telemetry and keeps a bounded history."""

    def __init__(self, history: int = 600):
        self.history: deque[Sample] = deque(maxlen=history)
        self._provider = None
        self._last_step: tuple[float, int] | None = None

    def _accelerator_provider(self):
        if self._provider is None:
            detection = engine.detect()
            backend = detection.capability.backend
            if backend is Backend.CUDA:
                from ..hardware.providers.nvidia import NVIDIAProvider

                self._provider = NVIDIAProvider()
            elif backend is Backend.ROCM:
                from ..hardware.providers.amd import AMDProvider

                self._provider = AMDProvider()
            else:
                self._provider = CPUProvider()
        return self._provider

    def sample(self) -> Sample:
        provider = self._accelerator_provider()
        try:
            devices = provider.sample()
        except Exception as exc:  # noqa: BLE001
            devices = [
                {
                    "error": f"telemetry unavailable: {type(exc).__name__}",
                }
            ]

        psutil = _psutil()
        if psutil is not None:
            cpu = {
                "utilization_pct": Value.measured(
                    psutil.cpu_percent(interval=None), "psutil", "%"
                ).to_dict(),
                "ram_used_gb": Value.measured(
                    round(psutil.virtual_memory().used / 1024**3, 2), "psutil", "GB"
                ).to_dict(),
                "ram_percent": Value.measured(
                    psutil.virtual_memory().percent, "psutil", "%"
                ).to_dict(),
            }
        else:
            reason = "psutil is not installed"
            cpu = {
                key: Value.unavailable(reason).to_dict()
                for key in ("utilization_pct", "ram_used_gb", "ram_percent")
            }

        sample = Sample(timestamp=time.time(), devices=devices, cpu=cpu)
        self.history.append(sample)
        return sample

    def record_step(self, step: int, tokens_this_step: int) -> dict[str, Any]:
        """Compute throughput from two observed step times.

        Deliberately not reported until a second step has been seen: a
        tokens-per-second figure derived from one data point, or worse from a
        theoretical peak, is exactly the kind of confident-looking fiction this
        dashboard is meant to avoid. The first step is also the slowest, since
        it pays for compilation and allocator warm-up.
        """
        now = time.time()
        if self._last_step is None:
            self._last_step = (now, step)
            return {
                "tokens_per_second": Value.unavailable(
                    "waiting for a second step before measuring"
                ).to_dict(),
                "eta": Value.unavailable("not enough steps to project").to_dict(),
            }

        last_time, last_step = self._last_step
        elapsed = now - last_time
        steps_done = max(1, step - last_step)
        self._last_step = (now, step)

        if elapsed <= 0:
            return {"tokens_per_second": Value.unavailable("clock did not advance").to_dict()}

        per_step = elapsed / steps_done
        return {
            "seconds_per_step": Value.measured(round(per_step, 3), "observed", "s").to_dict(),
            "tokens_per_second": Value.measured(
                round(tokens_this_step / per_step, 1), "observed", "tok/s"
            ).to_dict(),
        }

    def eta(self, step: int, total_steps: int | None) -> Value[str]:
        """Project a finish time from observed pace only."""
        if not total_steps or self._last_step is None or len(self.history) < 2:
            return Value.unavailable("not enough observed steps to project a finish time")
        start = self.history[0].timestamp
        elapsed = time.time() - start
        if step <= 0:
            return Value.unavailable("no steps completed yet")
        remaining = (elapsed / step) * (total_steps - step)
        return Value.estimated(
            _humanise(remaining), "projected from the pace measured so far"
        )

    def peak_device_memory_gb(self) -> Value[float]:
        """The highest device memory use actually observed.

        This is the number to compare against the planner's estimate after a
        run, and the reason the estimator's accuracy can improve over time
        instead of staying a guess forever.
        """
        peak = None
        for sample in self.history:
            for device in sample.devices:
                total = (device.get("total_memory_gb") or {}).get("value")
                free = (device.get("free_memory_gb") or {}).get("value")
                if total is None or free is None:
                    continue
                used = total - free
                peak = used if peak is None else max(peak, used)
        if peak is None:
            return Value.unavailable("no device reported memory during this run")
        return Value.measured(round(peak, 2), "peak observed during the run", "GB")

    def recent(self, count: int = 120) -> list[dict[str, Any]]:
        return [s.to_dict() for s in list(self.history)[-count:]]


def _humanise(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:  # noqa: BLE001
        return None


monitor = Monitor()
