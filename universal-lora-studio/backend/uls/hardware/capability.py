"""Capability model.

The training planner is never allowed to see a device name. It sees only what
a device can *do*: how much memory it has, which numeric formats it supports,
whether it can offload, how many of it there are.

This is what makes the planner portable. A rule written against "24 GB of
device memory with bf16 and 4-bit support" keeps working on hardware that did
not exist when the rule was written; a rule written against a product name
does not.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from ..value import Origin, Value


class Backend(str, enum.Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    XPU = "xpu"
    CPU = "cpu"
    UNKNOWN = "unknown"


class Precision(str, enum.Enum):
    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"

    @property
    def bytes_per_param(self) -> float:
        return {
            Precision.FP32: 4.0,
            Precision.TF32: 4.0,
            Precision.FP16: 2.0,
            Precision.BF16: 2.0,
            Precision.FP8: 1.0,
        }[self]


class Quantization(str, enum.Enum):
    NONE = "none"
    INT8 = "int8"
    NF4 = "nf4"
    FP4 = "fp4"

    @property
    def bytes_per_param(self) -> float:
        """Effective storage cost per weight, including quantization metadata.

        4-bit is not 0.5 bytes in practice: block-wise absmax scales add
        overhead. These figures assume block size 64 with double quantization,
        which is the bitsandbytes default.
        """
        return {
            Quantization.NONE: 0.0,  # caller uses the compute precision instead
            Quantization.INT8: 1.06,
            Quantization.NF4: 0.58,
            Quantization.FP4: 0.58,
        }[self]

    @property
    def is_four_bit(self) -> bool:
        return self in (Quantization.NF4, Quantization.FP4)


@dataclass
class Device:
    """One compute device. Fields are provenance-tagged individually because
    real detection is patchy -- a driver may report total memory but refuse
    temperature, and both outcomes must render truthfully."""

    index: int
    name: Value[str]
    vendor: Value[str]
    backend: Backend
    total_memory_gb: Value[float]
    free_memory_gb: Value[float] = field(default_factory=lambda: Value.unavailable())
    utilization_pct: Value[float] = field(default_factory=lambda: Value.unavailable())
    temperature_c: Value[float] = field(default_factory=lambda: Value.unavailable())
    power_w: Value[float] = field(default_factory=lambda: Value.unavailable())
    compute_capability: Value[str] = field(default_factory=lambda: Value.unavailable())
    memory_bandwidth_gbps: Value[float] = field(
        default_factory=lambda: Value.unavailable()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "backend": self.backend.value,
            "name": self.name.to_dict(),
            "vendor": self.vendor.to_dict(),
            "total_memory_gb": self.total_memory_gb.to_dict(),
            "free_memory_gb": self.free_memory_gb.to_dict(),
            "utilization_pct": self.utilization_pct.to_dict(),
            "temperature_c": self.temperature_c.to_dict(),
            "power_w": self.power_w.to_dict(),
            "compute_capability": self.compute_capability.to_dict(),
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps.to_dict(),
        }


@dataclass
class Capability:
    """What this environment can do. The planner's only view of hardware."""

    backend: Backend
    devices: list[Device] = field(default_factory=list)
    precisions: set[Precision] = field(default_factory=set)
    quantization: set[Quantization] = field(default_factory=set)
    system_ram_gb: Value[float] = field(default_factory=lambda: Value.unavailable())
    free_ram_gb: Value[float] = field(default_factory=lambda: Value.unavailable())
    cpu_cores: Value[int] = field(default_factory=lambda: Value.unavailable())
    cpu_threads: Value[int] = field(default_factory=lambda: Value.unavailable())
    cpu_name: Value[str] = field(default_factory=lambda: Value.unavailable())
    disk_free_gb: Value[float] = field(default_factory=lambda: Value.unavailable())
    supports_cpu_offload: bool = True
    supports_multi_device: bool = False
    supports_flash_attention: bool = False
    #: Where this description came from as a whole: a live machine, a saved
    #: profile, or a hypothetical the user invented in the simulator.
    origin: Origin = Origin.DETECTED
    profile_name: str = "Current environment"
    notes: list[str] = field(default_factory=list)

    # -- derived views --------------------------------------------------

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def has_accelerator(self) -> bool:
        return self.backend is not Backend.CPU and bool(self.devices)

    @property
    def total_device_memory_gb(self) -> Value[float]:
        """Summed device memory, with provenance degraded to match the inputs.

        If any device failed to report, the sum is not a sum -- it is a lower
        bound over a partial set, and says so.
        """
        if not self.devices:
            return Value.unavailable("no accelerator devices")
        known = [d.total_memory_gb for d in self.devices if d.total_memory_gb.known]
        if not known:
            return Value.unavailable("no device reported its memory")
        total = sum(v.get() for v in known)
        origin = min((v.origin for v in known), key=_ORIGIN_RANK.index)
        if len(known) < len(self.devices):
            return Value(
                total,
                origin,
                f"sum over {len(known)} of {len(self.devices)} devices; rest unreported",
                "GB",
            )
        return Value(total, origin, "", "GB")

    @property
    def smallest_device_memory_gb(self) -> Value[float]:
        """The binding constraint for a job that must fit on one device.

        Sharding across uneven devices is limited by the smallest member, so
        this -- not the total -- is what single-device planning uses.
        """
        known = [d.total_memory_gb for d in self.devices if d.total_memory_gb.known]
        if not known:
            return Value.unavailable("no device reported its memory")
        return min(known, key=lambda v: v.get())

    @property
    def training_memory_gb(self) -> Value[float]:
        """The memory pool a training job actually draws on.

        On an accelerator this is device memory. On CPU it is system RAM.
        Planning code asks for this rather than branching on backend.
        """
        if self.has_accelerator:
            return self.smallest_device_memory_gb
        return self.system_ram_gb

    @property
    def preferred_precision(self) -> Precision:
        for candidate in (Precision.BF16, Precision.FP16, Precision.FP32):
            if candidate in self.precisions:
                return candidate
        return Precision.FP32

    def supports(self, precision: Precision) -> bool:
        return precision in self.precisions

    def can_quantize(self, scheme: Quantization) -> bool:
        return scheme is Quantization.NONE or scheme in self.quantization

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "origin": self.origin.value,
            "label": {
                Origin.DETECTED: "Detected",
                Origin.CONFIGURED: "Configured",
                Origin.ESTIMATED: "Estimated",
                Origin.MEASURED: "Measured",
                Origin.UNAVAILABLE: "Not available",
            }[self.origin],
            "backend": self.backend.value,
            "device_count": self.device_count,
            "has_accelerator": self.has_accelerator,
            "devices": [d.to_dict() for d in self.devices],
            "precisions": sorted(p.value for p in self.precisions),
            "quantization": sorted(q.value for q in self.quantization),
            "system_ram_gb": self.system_ram_gb.to_dict(),
            "free_ram_gb": self.free_ram_gb.to_dict(),
            "cpu_name": self.cpu_name.to_dict(),
            "cpu_cores": self.cpu_cores.to_dict(),
            "cpu_threads": self.cpu_threads.to_dict(),
            "disk_free_gb": self.disk_free_gb.to_dict(),
            "total_device_memory_gb": self.total_device_memory_gb.to_dict(),
            "training_memory_gb": self.training_memory_gb.to_dict(),
            "supports_cpu_offload": self.supports_cpu_offload,
            "supports_multi_device": self.supports_multi_device,
            "supports_flash_attention": self.supports_flash_attention,
            "notes": self.notes,
        }


_ORIGIN_RANK = [
    Origin.UNAVAILABLE,
    Origin.ESTIMATED,
    Origin.CONFIGURED,
    Origin.MEASURED,
    Origin.DETECTED,
]
