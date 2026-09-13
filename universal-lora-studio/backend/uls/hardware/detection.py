"""Detection engine.

Picks the right provider for this machine, then merges in the host facts
(RAM, CPU, disk) that every environment has regardless of accelerator.

Results are cached, because detection shells out to vendor tools and the
hardware page is polled. The cache is explicitly refreshable -- hardware does
change under you, on cloud instances and when a GPU is passed into a container.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..value import Origin, Report, Value
from .capability import Backend, Capability
from .provider import HardwareProvider, software_environment
from .providers.amd import AMDProvider, XPUProvider
from .providers.apple import AppleProvider
from .providers.cpu import CPUProvider, CustomProvider
from .providers.nvidia import NVIDIAProvider

#: Registration order does not matter; providers are sorted by priority.
PROVIDERS: list[type[HardwareProvider]] = [
    NVIDIAProvider,
    AMDProvider,
    AppleProvider,
    XPUProvider,
    CPUProvider,
]


@dataclass
class Detection:
    capability: Capability
    software: Report
    provider_name: str
    detected_at: float
    considered: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.to_dict(),
            "software": self.software.to_dict(),
            "provider": self.provider_name,
            "detected_at": self.detected_at,
            "considered": self.considered,
        }


class DetectionEngine:
    def __init__(self, providers: list[type[HardwareProvider]] | None = None):
        self._provider_types = providers or PROVIDERS
        self._cache: Detection | None = None

    # -- detection ------------------------------------------------------

    def detect(self, refresh: bool = False) -> Detection:
        if self._cache is not None and not refresh:
            return self._cache

        considered: list[dict[str, Any]] = []
        chosen: HardwareProvider | None = None

        for provider_type in sorted(
            self._provider_types, key=lambda p: p.priority, reverse=True
        ):
            provider = provider_type()
            try:
                is_available = provider.available()
            except Exception as exc:  # noqa: BLE001
                considered.append(
                    {
                        "provider": provider_type.__name__,
                        "available": False,
                        "reason": f"probe raised {type(exc).__name__}",
                    }
                )
                continue
            considered.append(
                {"provider": provider_type.__name__, "available": is_available}
            )
            if is_available and chosen is None:
                chosen = provider

        if chosen is None:  # CPUProvider guarantees this cannot happen
            chosen = CPUProvider()

        capability = chosen.detect()
        capability = _merge_host_facts(capability)
        capability.profile_name = "Current environment"
        capability.origin = Origin.DETECTED

        self._cache = Detection(
            capability=capability,
            software=software_environment(),
            provider_name=type(chosen).__name__,
            detected_at=time.time(),
            considered=considered,
        )
        return self._cache

    def refresh(self) -> Detection:
        return self.detect(refresh=True)

    # -- overrides and simulation ---------------------------------------

    def from_spec(self, spec: dict) -> Capability:
        """Build a capability from values a person entered.

        Used for both saved profiles and the what-if simulator. The result is
        stamped CONFIGURED throughout and is never cached as the current
        environment.
        """
        return CustomProvider(spec).detect()

    def override(self, spec: dict) -> Capability:
        """Apply a partial override on top of what was detected.

        Fields the person did not touch keep their detected provenance; fields
        they changed are re-stamped CONFIGURED. Mixed provenance within one
        profile is normal and the UI renders it per-field.
        """
        base = self.detect().capability
        merged = self.from_spec(
            {
                "name": spec.get("name", f"{base.profile_name} (edited)"),
                "backend": spec.get("backend", base.backend.value),
                "device_count": spec.get("device_count", base.device_count or 1),
                "device_memory_gb": spec.get(
                    "device_memory_gb",
                    base.smallest_device_memory_gb.or_else(None),
                ),
                "device_name": spec.get(
                    "device_name",
                    base.devices[0].name.or_else("Unspecified") if base.devices else None,
                ),
                "system_ram_gb": spec.get("system_ram_gb", base.system_ram_gb.or_else(None)),
                "cpu_cores": spec.get("cpu_cores", base.cpu_cores.or_else(None)),
                "cpu_name": spec.get("cpu_name", base.cpu_name.or_else("Unspecified")),
                "disk_free_gb": spec.get("disk_free_gb", base.disk_free_gb.or_else(None)),
                "precisions": spec.get(
                    "precisions", sorted(p.value for p in base.precisions)
                ),
                "quantization": spec.get(
                    "quantization", sorted(q.value for q in base.quantization)
                ),
                "supports_cpu_offload": spec.get(
                    "supports_cpu_offload", base.supports_cpu_offload
                ),
                "supports_flash_attention": spec.get(
                    "supports_flash_attention", base.supports_flash_attention
                ),
            }
        )
        untouched = [k for k in ("device_memory_gb", "system_ram_gb", "cpu_cores")
                     if k not in spec]
        if untouched:
            merged.notes.append(
                "Values not edited here were carried over from detection: "
                + ", ".join(untouched).replace("_", " ")
                + "."
            )
        return merged


def _merge_host_facts(capability: Capability) -> Capability:
    """Fill in RAM, CPU and disk from the host regardless of accelerator.

    An accelerator provider knows about its own devices but has no business
    reading /proc/cpuinfo, so the host facts come from the CPU provider and are
    merged in here -- without clobbering anything the accelerator provider has
    already established (Apple's unified memory, for instance).
    """
    host = CPUProvider().detect()
    for field_name in (
        "system_ram_gb",
        "free_ram_gb",
        "cpu_name",
        "cpu_cores",
        "cpu_threads",
        "disk_free_gb",
    ):
        current: Value[Any] = getattr(capability, field_name)
        if not current.known:
            setattr(capability, field_name, getattr(host, field_name))

    if capability.backend is not Backend.CPU and not capability.devices:
        capability.notes.append(
            "An accelerator runtime is present but reported no usable devices. "
            "Planning will fall back to system memory."
        )
    return capability


#: Process-wide engine. Detection is read-only and idempotent, so sharing is safe.
engine = DetectionEngine()
