"""Apple Silicon / Metal provider.

Unified memory breaks an assumption that runs through most training tooling:
that device memory and system memory are separate pools. Here they are the
same silicon, and the GPU may only address a fraction of it -- the recommended
working set, which the driver enforces.

The capability this provider reports is therefore the working-set limit, not
the machine's total RAM, and the difference is stated in the notes rather than
quietly papered over.
"""

from __future__ import annotations

import platform

from ...value import Value
from ..capability import Backend, Capability, Device, Precision, Quantization
from ..provider import HardwareProvider, run_tool, torch_module


class AppleProvider(HardwareProvider):
    backend = Backend.METAL
    priority = 85

    def available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        torch = torch_module()
        if torch is None:
            # Apple Silicon is still worth describing without torch installed;
            # the user may be planning a run before setting up their stack.
            return platform.machine() == "arm64"
        try:
            return bool(torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            return platform.machine() == "arm64"

    def detect(self) -> Capability:
        notes: list[str] = []
        total_ram = _total_ram_gb()
        working_set = _recommended_working_set_gb()

        if working_set.known and total_ram.known:
            notes.append(
                f"Memory is unified. The GPU may address about "
                f"{working_set.display()} of the machine's {total_ram.display()}, "
                "and that share is what planning uses."
            )
        elif total_ram.known:
            working_set = Value.estimated(
                round(total_ram.get() * 0.75, 2),
                "Metal did not report a working-set limit; using the 75% of unified "
                "memory that Apple's default wired limit approximates",
                "GB",
            )
            notes.append(
                "The GPU working-set limit could not be read, so it is estimated "
                "from total memory. Treat headroom figures as soft."
            )

        chip = Value.probe(
            lambda: (run_tool(["sysctl", "-n", "machdep.cpu.brand_string"]) or None),
            "sysctl",
            on_fail="sysctl did not report a chip name",
        )

        torch = torch_module()
        mps_ready = False
        if torch is not None:
            try:
                mps_ready = bool(torch.backends.mps.is_available())
            except Exception:  # noqa: BLE001
                mps_ready = False
        if not mps_ready:
            notes.append(
                "MPS is not currently usable, so training would fall back to the "
                "CPU. Install a PyTorch build with Metal support to use the GPU."
            )

        device = Device(
            index=0,
            name=chip if chip.known else Value.detected("Apple Silicon GPU"),
            vendor=Value.detected("Apple"),
            backend=Backend.METAL,
            total_memory_gb=working_set,
            free_memory_gb=Value.unavailable(
                "Metal does not expose free GPU memory separately from system memory"
            ),
            utilization_pct=Value.unavailable(
                "GPU utilization is not available without elevated privileges"
            ),
            temperature_c=Value.unavailable("no temperature sensor exposed to userspace"),
        )

        return Capability(
            backend=Backend.METAL,
            devices=[device],
            # fp16 works on MPS; bf16 support is version-dependent and is probed.
            precisions=_mps_precisions(),
            quantization={Quantization.NONE},
            system_ram_gb=total_ram,
            supports_cpu_offload=False,  # offloading is meaningless in a unified pool
            supports_multi_device=False,
            supports_flash_attention=False,
            notes=notes
            + [
                "bitsandbytes has no Metal backend, so QLoRA and 8-bit optimizers "
                "are unavailable. LoRA in fp16 is the practical path here.",
                "CPU offload is not offered: with unified memory there is nowhere "
                "to offload to.",
            ],
        )


def _mps_precisions() -> set[Precision]:
    torch = torch_module()
    precisions = {Precision.FP32, Precision.FP16}
    if torch is None:
        return {Precision.FP32}
    try:
        # bf16 on MPS depends on both macOS and torch version; test it rather
        # than assume either way.
        torch.zeros(1, dtype=torch.bfloat16, device="mps")
        precisions.add(Precision.BF16)
    except Exception:  # noqa: BLE001
        pass
    return precisions


def _total_ram_gb() -> Value[float]:
    def probe() -> float:
        raw = run_tool(["sysctl", "-n", "hw.memsize"])
        if not raw:
            raise RuntimeError("sysctl hw.memsize unavailable")
        return round(int(raw) / 1024**3, 2)

    return Value.probe(probe, "sysctl", "GB")


def _recommended_working_set_gb() -> Value[float]:
    torch = torch_module()
    if torch is None:
        return Value.unavailable("torch is not installed")

    def probe() -> float:
        limit = torch.mps.recommended_max_memory()
        if not limit:
            raise RuntimeError("recommended_max_memory returned 0")
        return round(limit / 1024**3, 2)

    return Value.probe(probe, "torch.mps", "GB", on_fail="Metal working-set limit not exposed")
