"""CPU provider, and the custom provider used for overrides and simulation.

The CPU provider always reports available. It is the floor the system lands on
when nothing else is present, and a machine with no accelerator gets a working
studio rather than a disabled one.
"""

from __future__ import annotations

import os
import platform
import shutil

from ...value import Origin, Value
from ..capability import Backend, Capability, Device, Precision, Quantization
from ..provider import HardwareProvider, run_tool, supported_quantization, torch_module


class CPUProvider(HardwareProvider):
    backend = Backend.CPU
    priority = 0  # last resort, and always reachable

    def available(self) -> bool:
        return True

    def detect(self) -> Capability:
        ram_total, ram_free = _memory()
        cores, threads = _cpu_counts()
        notes = [
            "No accelerator was detected. Training on CPU is supported but is "
            "typically one to two orders of magnitude slower, so plans here "
            "favour small models and short sequences.",
        ]

        precisions = {Precision.FP32}
        torch = torch_module()
        if torch is not None:
            # bf16 on CPU needs AVX-512 BF16 or AMX to be worth using; test the
            # operation rather than parsing the CPU model string.
            try:
                a = torch.zeros(8, 8, dtype=torch.bfloat16)
                (a @ a).sum()
                precisions.add(Precision.BF16)
            except Exception:  # noqa: BLE001
                notes.append("bf16 is not usable on this CPU; fp32 will be used.")
        else:
            notes.append(
                "PyTorch is not installed. You can plan and inspect here, but "
                "training cannot start until it is."
            )

        device = Device(
            index=0,
            name=_cpu_name(),
            vendor=_cpu_vendor(),
            backend=Backend.CPU,
            total_memory_gb=ram_total,
            free_memory_gb=ram_free,
            utilization_pct=_cpu_utilization(),
        )

        return Capability(
            backend=Backend.CPU,
            devices=[device],
            precisions=precisions,
            quantization=supported_quantization() & {Quantization.NONE, Quantization.INT8},
            system_ram_gb=ram_total,
            free_ram_gb=ram_free,
            cpu_name=_cpu_name(),
            cpu_cores=cores,
            cpu_threads=threads,
            disk_free_gb=_disk_free(),
            supports_cpu_offload=False,
            supports_multi_device=False,
            notes=notes,
        )


class CustomProvider(HardwareProvider):
    """Hardware described by a person rather than read from a machine.

    Backs two features that must never be confused with detection: overriding
    what was detected, and simulating hardware the user does not own. Every
    value it produces is stamped CONFIGURED, so a simulated 80 GB card can
    never be mistaken in the UI for one that is actually present.
    """

    backend = Backend.UNKNOWN
    priority = -1  # never selected automatically

    def __init__(self, spec: dict):
        self.spec = spec

    def available(self) -> bool:
        return True

    def detect(self) -> Capability:
        spec = self.spec
        backend = Backend(spec.get("backend") or "cuda")
        count = max(1, int(spec.get("device_count") or 1))
        vram = spec.get("device_memory_gb")
        name = spec.get("device_name") or "Custom device"

        devices: list[Device] = []
        if backend is not Backend.CPU and vram:
            for i in range(count):
                devices.append(
                    Device(
                        index=i,
                        name=Value.configured(name),
                        vendor=Value.configured(spec.get("vendor") or "Unspecified"),
                        backend=backend,
                        total_memory_gb=Value.configured(float(vram), "entered", "GB"),
                        free_memory_gb=Value.unavailable(
                            "this profile is not a running machine"
                        ),
                        utilization_pct=Value.unavailable(
                            "this profile is not a running machine"
                        ),
                    )
                )

        # A key present with a null value means "not specified", not "empty set" --
        # serialized profiles round-trip unset fields as null.
        precisions = {
            Precision(p) for p in (spec.get("precisions") or ["fp32", "fp16", "bf16"])
        }
        quant = {
            Quantization(q) for q in (spec.get("quantization") or ["none", "int8", "nf4"])
        }
        ram = spec.get("system_ram_gb")

        return Capability(
            backend=backend,
            devices=devices,
            precisions=precisions,
            quantization=quant,
            system_ram_gb=(
                Value.configured(float(ram), "entered", "GB")
                if ram
                else Value.unavailable("not specified in this profile")
            ),
            free_ram_gb=Value.unavailable("this profile is not a running machine"),
            cpu_name=Value.configured(spec.get("cpu_name") or "Unspecified"),
            cpu_cores=(
                Value.configured(int(spec["cpu_cores"]))
                if spec.get("cpu_cores")
                else Value.unavailable("not specified in this profile")
            ),
            cpu_threads=Value.unavailable("not specified in this profile"),
            disk_free_gb=(
                Value.configured(float(spec["disk_free_gb"]), "entered", "GB")
                if spec.get("disk_free_gb")
                else Value.unavailable("not specified in this profile")
            ),
            supports_cpu_offload=bool(spec.get("supports_cpu_offload", True)),
            supports_multi_device=count > 1,
            supports_flash_attention=bool(spec.get("supports_flash_attention", False)),
            origin=Origin.CONFIGURED,
            profile_name=spec.get("name") or "Custom profile",
            notes=[
                "These values were entered, not measured. Plans built on them are "
                "predictions about a machine this software has not seen.",
            ],
        )


# ---------------------------------------------------------------------------
# platform probes
# ---------------------------------------------------------------------------


def _psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:  # noqa: BLE001
        return None


def _memory() -> tuple[Value[float], Value[float]]:
    psutil = _psutil()
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            return (
                Value.detected(round(mem.total / 1024**3, 2), "psutil", "GB"),
                Value.detected(round(mem.available / 1024**3, 2), "psutil", "GB"),
            )
        except Exception:  # noqa: BLE001
            pass
    # Fall back to POSIX sysconf where psutil is absent.
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return (
            Value.detected(round(total / 1024**3, 2), "sysconf", "GB"),
            Value.unavailable("free memory needs psutil on this platform"),
        )
    except (ValueError, OSError, AttributeError):
        return (
            Value.unavailable("no memory source available on this platform"),
            Value.unavailable("no memory source available on this platform"),
        )


def _cpu_counts() -> tuple[Value[int], Value[int]]:
    psutil = _psutil()
    physical = Value.unavailable("physical core count not available")
    logical = Value.unavailable("logical core count not available")
    if psutil is not None:
        try:
            p = psutil.cpu_count(logical=False)
            if p:
                physical = Value.detected(p, "psutil")
        except Exception:  # noqa: BLE001
            pass
    count = os.cpu_count()
    if count:
        logical = Value.detected(count, "os.cpu_count")
        if not physical.known:
            physical = Value.detected(count, "logical count; SMT layout unknown")
    return physical, logical


def _cpu_name() -> Value[str]:
    system = platform.system()
    if system == "Darwin":
        name = run_tool(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return Value.detected(name, "sysctl")
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return Value.detected(line.split(":", 1)[1].strip(), "/proc/cpuinfo")
        except OSError:
            pass
    processor = platform.processor() or platform.machine()
    if processor:
        return Value.detected(processor, "platform")
    return Value.unavailable("no CPU name source on this platform")


def _cpu_vendor() -> Value[str]:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") and platform.system() == "Darwin":
        return Value.detected("Apple")
    name = _cpu_name()
    if name.known:
        lowered = name.get().lower()
        for needle, vendor in (("intel", "Intel"), ("amd", "AMD"), ("apple", "Apple")):
            if needle in lowered:
                return Value.detected(vendor, "parsed from CPU name")
    return Value.unavailable("CPU vendor could not be determined")


def _cpu_utilization() -> Value[float]:
    psutil = _psutil()
    if psutil is None:
        return Value.unavailable("CPU utilization needs psutil")
    try:
        # Non-blocking: the first call seeds the counter and returns 0.0, so the
        # sampler in monitoring/ calls this repeatedly rather than once.
        return Value.measured(psutil.cpu_percent(interval=None), "psutil", "%")
    except Exception:  # noqa: BLE001
        return Value.unavailable("CPU utilization read failed")


def _disk_free() -> Value[float]:
    try:
        usage = shutil.disk_usage(os.getcwd())
        return Value.detected(round(usage.free / 1024**3, 2), "shutil.disk_usage", "GB")
    except OSError:
        return Value.unavailable("disk usage could not be read")
