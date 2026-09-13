"""AMD ROCm provider, plus Intel XPU.

PyTorch exposes ROCm devices through the same ``torch.cuda`` namespace it uses
for NVIDIA, distinguished by ``torch.version.hip``. That quirk is contained
here so that nothing above the abstraction layer has to know about it.
"""

from __future__ import annotations

import json

from ...value import Value
from ..capability import Backend, Capability, Device, Precision, Quantization
from ..provider import HardwareProvider, run_tool, torch_module


class AMDProvider(HardwareProvider):
    backend = Backend.ROCM
    priority = 90

    def available(self) -> bool:
        torch = torch_module()
        if torch is not None:
            try:
                if getattr(torch.version, "hip", None) and torch.cuda.is_available():
                    return True
            except Exception:  # noqa: BLE001
                pass
        return bool(run_tool(["rocm-smi", "--showid"]))

    def detect(self) -> Capability:
        devices = _devices_from_torch() or _devices_from_rocm_smi() or []
        notes: list[str] = []
        precisions = {Precision.FP32, Precision.FP16}
        torch = torch_module()
        if torch is None:
            notes.append(
                "PyTorch is not installed. Device readings come from rocm-smi; "
                "precision support is unconfirmed."
            )
            precisions = {Precision.FP32}
        else:
            try:
                if torch.cuda.is_bf16_supported():
                    precisions.add(Precision.BF16)
            except Exception:  # noqa: BLE001
                notes.append("Could not confirm bf16 support on this ROCm build.")

        quant = {Quantization.NONE}
        try:
            import bitsandbytes  # type: ignore  # noqa: F401

            quant.update({Quantization.INT8, Quantization.NF4, Quantization.FP4})
        except Exception:  # noqa: BLE001
            notes.append(
                "bitsandbytes is not importable, so 4-bit and 8-bit training are "
                "unavailable. ROCm support in bitsandbytes depends on the build."
            )

        return Capability(
            backend=Backend.ROCM,
            devices=devices,
            precisions=precisions,
            quantization=quant,
            supports_cpu_offload=True,
            supports_multi_device=len(devices) > 1,
            supports_flash_attention=False,
            notes=notes,
        )


def _devices_from_torch() -> list[Device]:
    torch = torch_module()
    if torch is None or not getattr(torch.version, "hip", None):
        return []
    devices: list[Device] = []
    try:
        count = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return []
    for i in range(count):
        try:
            props = torch.cuda.get_device_properties(i)
            free_b, _ = torch.cuda.mem_get_info(i)
            devices.append(
                Device(
                    index=i,
                    name=Value.detected(props.name, "torch (ROCm)"),
                    vendor=Value.detected("AMD"),
                    backend=Backend.ROCM,
                    total_memory_gb=Value.detected(
                        round(props.total_memory / 1024**3, 2), "torch (ROCm)", "GB"
                    ),
                    free_memory_gb=Value.detected(
                        round(free_b / 1024**3, 2), "torch (ROCm)", "GB"
                    ),
                    compute_capability=Value.detected(
                        getattr(props, "gcnArchName", "unknown"), "torch (ROCm)"
                    ),
                )
            )
        except Exception:  # noqa: BLE001
            continue
    return devices


def _devices_from_rocm_smi() -> list[Device]:
    out = run_tool(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"])
    if not out:
        return []
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return []
    devices: list[Device] = []
    for index, (key, entry) in enumerate(sorted(payload.items())):
        if not isinstance(entry, dict):
            continue
        total = _first_number(entry, ("VRAM Total Memory (B)", "vram_total"))
        used = _first_number(entry, ("VRAM Total Used Memory (B)", "vram_used"))
        util = _first_number(entry, ("GPU use (%)", "gpu_use"))
        devices.append(
            Device(
                index=index,
                name=Value.detected(entry.get("Card series", key), "rocm-smi"),
                vendor=Value.detected("AMD"),
                backend=Backend.ROCM,
                total_memory_gb=(
                    Value.detected(round(total / 1024**3, 2), "rocm-smi", "GB")
                    if total
                    else Value.unavailable("rocm-smi did not report VRAM")
                ),
                free_memory_gb=(
                    Value.detected(round((total - used) / 1024**3, 2), "rocm-smi", "GB")
                    if total and used is not None
                    else Value.unavailable("rocm-smi did not report VRAM use")
                ),
                utilization_pct=(
                    Value.measured(util, "rocm-smi", "%")
                    if util is not None
                    else Value.unavailable("utilization not reported")
                ),
            )
        )
    return devices


def _first_number(entry: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in entry:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                continue
    return None


class XPUProvider(HardwareProvider):
    """Intel discrete and integrated GPUs via ``torch.xpu``."""

    backend = Backend.XPU
    priority = 80

    def available(self) -> bool:
        torch = torch_module()
        if torch is None:
            return False
        try:
            return bool(getattr(torch, "xpu", None)) and torch.xpu.is_available()
        except Exception:  # noqa: BLE001
            return False

    def detect(self) -> Capability:
        torch = torch_module()
        devices: list[Device] = []
        try:
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i)
                devices.append(
                    Device(
                        index=i,
                        name=Value.detected(props.name, "torch.xpu"),
                        vendor=Value.detected("Intel"),
                        backend=Backend.XPU,
                        total_memory_gb=Value.detected(
                            round(props.total_memory / 1024**3, 2), "torch.xpu", "GB"
                        ),
                    )
                )
        except Exception:  # noqa: BLE001
            pass
        return Capability(
            backend=Backend.XPU,
            devices=devices,
            precisions={Precision.FP32, Precision.FP16, Precision.BF16},
            quantization={Quantization.NONE},
            supports_cpu_offload=True,
            supports_multi_device=len(devices) > 1,
            notes=[
                "bitsandbytes quantization is not generally available on XPU, so "
                "QLoRA is listed as unsupported here."
            ],
        )
