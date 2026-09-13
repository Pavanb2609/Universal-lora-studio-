"""The hardware abstraction layer boundary.

Everything vendor-specific lives behind this interface. Nothing above it --
not the planner, not the estimator, not the API -- imports a vendor SDK or
branches on a product name.

Adding support for new silicon means writing one new provider and registering
it. No other file changes.
"""

from __future__ import annotations

import abc
import platform
import shutil
import subprocess
from typing import Any

from ..value import Report, Value
from .capability import Backend, Capability, Device, Precision, Quantization


class HardwareProvider(abc.ABC):
    """Describes one class of compute hardware."""

    backend: Backend
    #: Providers are tried highest-priority first; the first that reports
    #: available wins. Accelerators outrank the CPU fallback.
    priority: int = 0

    @abc.abstractmethod
    def available(self) -> bool:
        """Is this kind of hardware usable right now?

        Must be cheap and must never raise -- it runs on every page load, on
        machines where the relevant driver is absent.
        """

    @abc.abstractmethod
    def detect(self) -> Capability:
        """Build a full capability description. Only called when available()."""

    def get_devices(self) -> list[Device]:
        return self.detect().devices

    def sample(self) -> list[dict[str, Any]]:
        """Live telemetry for the monitoring view.

        Default implementation re-reads the devices. Providers with a cheaper
        telemetry path should override.
        """
        return [d.to_dict() for d in self.get_devices()]


# ---------------------------------------------------------------------------
# shared helpers for providers that shell out to vendor CLI tools
# ---------------------------------------------------------------------------


def run_tool(args: list[str], timeout: float = 5.0) -> str | None:
    """Run a vendor CLI tool, returning None on any failure.

    Vendor tools are missing, renamed, permission-gated or simply hung on a
    large fraction of real machines. Every one of those is ordinary absence,
    not an error worth surfacing.
    """
    if not shutil.which(args[0]):
        return None
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def torch_module():
    """Import torch if present.

    The studio is useful without it -- you can plan a run, inspect a dataset
    and size a model on a laptop with no ML stack installed -- so torch is an
    optional capability, not a hard dependency.
    """
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return torch


def _version_of(module_name: str) -> Value[str]:
    def probe() -> str:
        import importlib

        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "unknown")

    return Value.probe(probe, on_fail=f"{module_name} is not installed")


def software_environment() -> Report:
    """The software half of reproducibility.

    Two identical machines running different PyTorch builds are, for these
    purposes, different machines -- so this is captured alongside hardware in
    every experiment snapshot.
    """
    report = Report("Software environment")
    report.add("os", Value.detected(f"{platform.system()} {platform.release()}"))
    report.add("os_version", Value.detected(platform.version()))
    report.add("architecture", Value.detected(platform.machine()))
    report.add("python", Value.detected(platform.python_version()))
    for name in ("torch", "transformers", "peft", "accelerate", "datasets",
                 "bitsandbytes", "trl"):
        report.add(name, _version_of(name))

    torch = torch_module()
    if torch is None:
        report.add("accelerator_runtime", Value.unavailable("torch is not installed"))
        report.add("flash_attention", Value.unavailable("torch is not installed"))
        return report

    def runtime() -> str:
        if getattr(torch.version, "hip", None):
            return f"ROCm {torch.version.hip}"
        if getattr(torch.version, "cuda", None):
            return f"CUDA {torch.version.cuda}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Metal (MPS)"
        return "CPU only"

    report.add("accelerator_runtime", Value.probe(runtime))
    report.add(
        "flash_attention",
        Value.probe(
            lambda: "available"
            if __import__("importlib").util.find_spec("flash_attn")
            else None,
            on_fail="flash-attn is not installed",
        ),
    )
    return report


def supported_precisions_from_torch(device_index: int = 0) -> set[Precision]:
    """Ask the runtime what it supports rather than inferring from a model name.

    bf16 support in particular cannot be read off a product name reliably --
    it depends on the device, the driver and the build of PyTorch together.
    """
    torch = torch_module()
    if torch is None:
        return {Precision.FP32}
    precisions = {Precision.FP32}
    try:
        if torch.cuda.is_available():
            precisions.add(Precision.FP16)
            if torch.cuda.is_bf16_supported():
                precisions.add(Precision.BF16)
            major, _ = torch.cuda.get_device_capability(device_index)
            if major >= 8:
                precisions.add(Precision.TF32)
            if major >= 9:
                precisions.add(Precision.FP8)
    except Exception:  # noqa: BLE001
        pass
    return precisions


def supported_quantization() -> set[Quantization]:
    """Quantization needs a working bitsandbytes build, which is a separate
    question from having a GPU. Import it and see."""
    schemes = {Quantization.NONE}
    try:
        import bitsandbytes  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        return schemes
    schemes.update({Quantization.INT8, Quantization.NF4, Quantization.FP4})
    return schemes
