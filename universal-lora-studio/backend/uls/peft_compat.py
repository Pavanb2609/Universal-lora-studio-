"""PEFT method compatibility.

Which adapter methods are actually usable depends on three things at once: the
model architecture, what is installed, and what the hardware supports. The UI
asks this module rather than listing every method and letting training fail
twenty minutes in.

A method is never silently hidden. Unavailable methods are returned with the
reason attached, because "why can't I pick QLoRA here" is a question the
interface should answer without being asked twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .estimation.memory import Method
from .estimation.model_spec import ModelSpec
from .hardware.capability import Backend, Capability


@dataclass
class MethodStatus:
    method: Method
    label: str
    available: bool
    summary: str
    reason: str = ""
    recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "label": self.label,
            "available": self.available,
            "summary": self.summary,
            "reason": self.reason,
            "recommended": self.recommended,
        }


LABELS = {
    Method.LORA: "LoRA",
    Method.QLORA: "QLoRA",
    Method.ADALORA: "AdaLoRA",
    Method.DORA: "DoRA",
    Method.PREFIX_TUNING: "Prefix tuning",
    Method.PROMPT_TUNING: "Prompt tuning",
    Method.IA3: "IA³",
    Method.FULL: "Full fine-tuning",
}

SUMMARIES = {
    Method.LORA: "Low-rank adapters on the frozen base model. The default choice.",
    Method.QLORA: "LoRA over a 4-bit base model. Roughly a quarter of the memory.",
    Method.ADALORA: "LoRA that reallocates rank towards the layers that need it.",
    Method.DORA: "Splits the update into direction and magnitude. Often better at low rank.",
    Method.PREFIX_TUNING: "Learns key/value prefixes per layer. Leaves weights untouched.",
    Method.PROMPT_TUNING: "Learns soft input tokens only. Tiny, and limited in what it can change.",
    Method.IA3: "Learns per-channel rescalings. Very few parameters.",
    Method.FULL: "Updates every weight. Not parameter-efficient.",
}

#: Architectures where prefix tuning's per-layer past-key-value interface does
#: not hold. Encoder-decoder and state-space models are the usual exceptions.
_PREFIX_UNSUPPORTED = ("mamba", "rwkv", "ssm", "retnet")


def evaluate(
    model: ModelSpec, capability: Capability, installed: dict[str, bool] | None = None
) -> list[MethodStatus]:
    """Report every method's status for this model in this environment."""
    installed = installed or _installed()
    arch = (model.architecture or "").lower()
    has_peft = installed.get("peft", False)
    has_bnb = installed.get("bitsandbytes", False)
    quant_ok = any(q.is_four_bit for q in capability.quantization) and has_bnb
    memory = capability.training_memory_gb

    statuses: list[MethodStatus] = []

    def add(method: Method, available: bool, reason: str = "", recommended: bool = False):
        statuses.append(
            MethodStatus(
                method, LABELS[method], available, SUMMARIES[method], reason, recommended
            )
        )

    if not has_peft:
        for method in Method:
            add(
                method,
                method is Method.FULL,
                "" if method is Method.FULL else "The peft library is not installed.",
            )
        return statuses

    add(Method.LORA, True, recommended=not quant_ok)

    add(
        Method.QLORA,
        quant_ok,
        ""
        if quant_ok
        else (
            "bitsandbytes is not available here, so the base model cannot be "
            "held in 4-bit."
            if not has_bnb
            else f"The {capability.backend.value} backend has no 4-bit kernels."
        ),
        recommended=quant_ok,
    )

    add(Method.ADALORA, True)
    add(
        Method.DORA,
        True,
        "" if quant_ok else "Works, though it is slower than plain LoRA to train.",
    )

    prefix_ok = not any(token in arch for token in _PREFIX_UNSUPPORTED)
    add(
        Method.PREFIX_TUNING,
        prefix_ok,
        "" if prefix_ok else f"The {model.architecture} architecture does not expose "
        "the per-layer key/value cache this method attaches to.",
    )
    add(Method.PROMPT_TUNING, True)
    add(
        Method.IA3,
        True,
        "Supported, but it has far less capacity than LoRA; best when the task is "
        "a small adjustment to existing behaviour.",
    )

    # Full fine-tuning is offered only when the arithmetic supports it, because
    # it is the one method whose failure mode is discovering after an hour of
    # setup that it was never possible.
    full_reason = ""
    full_ok = True
    if memory.known:
        # weights + gradients + Adam states, all in fp16/fp32 terms
        needed = model.total_params * 16 / 1024**3
        pool = memory.get() * max(1, capability.device_count)
        if needed > pool:
            full_ok = False
            full_reason = (
                f"Full fine-tuning of a {model.size_label} model needs on the order "
                f"of {needed:.0f} GB for weights, gradients and optimizer state. "
                f"This environment has about {pool:.0f} GB."
            )
    else:
        full_ok = False
        full_reason = "Memory capacity is unknown, so this cannot be offered safely."
    if capability.backend is Backend.CPU:
        full_ok = False
        full_reason = "Full fine-tuning on CPU is not practical at any useful scale."
    add(Method.FULL, full_ok, full_reason)

    return statuses


def is_available(method: Method, model: ModelSpec, capability: Capability) -> tuple[bool, str]:
    for status in evaluate(model, capability):
        if status.method is method:
            return status.available, status.reason
    return False, "Unknown method."


def recommended(model: ModelSpec, capability: Capability) -> Method:
    for status in evaluate(model, capability):
        if status.recommended and status.available:
            return status.method
    return Method.LORA


def _installed() -> dict[str, bool]:
    import importlib.util

    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("peft", "bitsandbytes", "transformers", "trl", "accelerate")
    }
