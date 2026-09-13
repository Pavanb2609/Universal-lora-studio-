"""NVIDIA / CUDA provider.

Two independent detection paths. PyTorch is preferred because it reports what
training will actually be able to use, but it cannot see temperature or power,
and it is often absent. ``nvidia-smi`` fills those gaps and works with no ML
stack installed at all. Where both are present the readings are merged.
"""

from __future__ import annotations

from ...value import Value
from ..capability import Backend, Capability, Device, Precision
from ..provider import (
    HardwareProvider,
    run_tool,
    supported_precisions_from_torch,
    supported_quantization,
    torch_module,
)

_SMI_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "compute_cap",
]


class NVIDIAProvider(HardwareProvider):
    backend = Backend.CUDA
    priority = 100

    def available(self) -> bool:
        torch = torch_module()
        if torch is not None:
            try:
                if torch.cuda.is_available() and torch.version.cuda:
                    return True
            except Exception:  # noqa: BLE001
                pass
        return _query_smi() is not None

    def detect(self) -> Capability:
        smi_rows = _query_smi() or []
        smi_by_index = {row["index"]: row for row in smi_rows}
        devices: list[Device] = []
        notes: list[str] = []

        torch = torch_module()
        torch_count = 0
        if torch is not None:
            try:
                torch_count = torch.cuda.device_count()
            except Exception:  # noqa: BLE001
                torch_count = 0

        count = max(torch_count, len(smi_rows))
        for i in range(count):
            smi = smi_by_index.get(i, {})
            devices.append(_build_device(i, smi, torch))

        if torch is None:
            notes.append(
                "PyTorch is not installed, so readings come from nvidia-smi only. "
                "Precision support cannot be confirmed until it is."
            )
        elif torch_count and len(smi_rows) and torch_count != len(smi_rows):
            notes.append(
                f"PyTorch sees {torch_count} device(s) but nvidia-smi reports "
                f"{len(smi_rows)}. CUDA_VISIBLE_DEVICES is probably set."
            )

        precisions = supported_precisions_from_torch()
        if torch is None:
            precisions = {Precision.FP32}

        cap = Capability(
            backend=Backend.CUDA,
            devices=devices,
            precisions=precisions,
            quantization=supported_quantization(),
            supports_cpu_offload=True,
            supports_multi_device=len(devices) > 1,
            supports_flash_attention=_flash_attention_usable(devices),
            notes=notes,
        )
        return cap

    def sample(self) -> list[dict]:
        rows = _query_smi()
        if rows is None:
            return [d.to_dict() for d in self.get_devices()]
        torch = torch_module()
        return [_build_device(r["index"], r, torch).to_dict() for r in rows]


def _query_smi() -> list[dict] | None:
    out = run_tool(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(_SMI_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    rows: list[dict] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_SMI_FIELDS):
            continue
        row = dict(zip(_SMI_FIELDS, parts))
        try:
            row["index"] = int(row["index"])
        except ValueError:
            continue
        rows.append(row)
    return rows or None


def _num(row: dict, key: str) -> float | None:
    """nvidia-smi writes '[N/A]' for unsupported sensors. Honour that."""
    raw = row.get(key)
    if raw is None or "N/A" in raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _build_device(index: int, smi: dict, torch) -> Device:
    name = Value.unavailable("no source reported a device name")
    total = Value.unavailable("device memory not reported")
    free = Value.unavailable("free memory not reported")
    cc = Value.unavailable()

    if smi.get("name"):
        name = Value.detected(smi["name"], "nvidia-smi")
    total_mib = _num(smi, "memory.total")
    used_mib = _num(smi, "memory.used")
    if total_mib is not None:
        total = Value.detected(round(total_mib / 1024, 2), "nvidia-smi", "GB")
        if used_mib is not None:
            free = Value.detected(
                round((total_mib - used_mib) / 1024, 2), "nvidia-smi", "GB"
            )
    if smi.get("compute_cap"):
        cc = Value.detected(smi["compute_cap"], "nvidia-smi")

    # Prefer torch for memory: it reports the pool the process can really use,
    # which differs from the card's nameplate under MIG or a memory fraction cap.
    if torch is not None:
        try:
            props = torch.cuda.get_device_properties(index)
            name = Value.detected(props.name, "torch.cuda")
            total = Value.detected(
                round(props.total_memory / 1024**3, 2), "torch.cuda", "GB"
            )
            cc = Value.detected(f"{props.major}.{props.minor}", "torch.cuda")
            free_b, _ = torch.cuda.mem_get_info(index)
            free = Value.detected(round(free_b / 1024**3, 2), "torch.cuda", "GB")
        except Exception:  # noqa: BLE001
            pass

    util = _num(smi, "utilization.gpu")
    temp = _num(smi, "temperature.gpu")
    power = _num(smi, "power.draw")

    return Device(
        index=index,
        name=name,
        vendor=Value.detected("NVIDIA"),
        backend=Backend.CUDA,
        total_memory_gb=total,
        free_memory_gb=free,
        utilization_pct=(
            Value.measured(util, "nvidia-smi", "%")
            if util is not None
            else Value.unavailable("utilization not exposed by this driver")
        ),
        temperature_c=(
            Value.measured(temp, "nvidia-smi", "°C")
            if temp is not None
            else Value.unavailable("no temperature sensor reported")
        ),
        power_w=(
            Value.measured(power, "nvidia-smi", "W")
            if power is not None
            else Value.unavailable("power draw not exposed by this driver")
        ),
        compute_capability=cc,
    )


def _flash_attention_usable(devices: list[Device]) -> bool:
    """Flash-attention needs both the package and a device that supports it.

    Checked by capability rather than by model name, so new hardware works
    without a code change.
    """
    try:
        import importlib.util

        if importlib.util.find_spec("flash_attn") is None:
            return False
    except Exception:  # noqa: BLE001
        return False
    for d in devices:
        if d.compute_capability.known:
            try:
                if float(d.compute_capability.get()) >= 8.0:
                    return True
            except ValueError:
                continue
    return False
