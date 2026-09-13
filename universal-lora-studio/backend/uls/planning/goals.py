"""Goals and budgets: what the person is optimising for, and what they refuse
to spend.

These are separate ideas and are kept separate. A goal expresses a preference
ordering over trade-offs. A budget expresses a hard ceiling that has nothing to
do with what the hardware can physically do -- a person on a shared server may
have 80 GB in front of them and permission to use 20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..estimation.memory import Method, Optimizer
from ..hardware.capability import Quantization
from ..value import Value


class Goal(str, Enum):
    FASTEST = "fastest"
    BALANCED = "balanced"
    MAX_QUALITY = "max_quality"
    MIN_DEVICE_MEMORY = "min_device_memory"
    MIN_SYSTEM_MEMORY = "min_system_memory"
    CUSTOM = "custom"

    @property
    def label(self) -> str:
        return {
            Goal.FASTEST: "Fastest",
            Goal.BALANCED: "Balanced",
            Goal.MAX_QUALITY: "Best quality",
            Goal.MIN_DEVICE_MEMORY: "Smallest memory footprint",
            Goal.MIN_SYSTEM_MEMORY: "Smallest system RAM footprint",
            Goal.CUSTOM: "Custom",
        }[self]

    @property
    def description(self) -> str:
        return {
            Goal.FASTEST: "Finish sooner. Spends memory to avoid recomputation.",
            Goal.BALANCED: "A workable middle: fits comfortably, trains at a reasonable pace.",
            Goal.MAX_QUALITY: "Larger adapters and longer sequences, at the cost of time.",
            Goal.MIN_DEVICE_MEMORY: "Fit in as little accelerator memory as possible.",
            Goal.MIN_SYSTEM_MEMORY: "Keep host RAM use down; avoid offloading.",
            Goal.CUSTOM: "Every parameter is yours to set.",
        }[self]


@dataclass
class Preferences:
    """How a goal biases the starting configuration and the order in which
    concessions get made when memory is short."""

    prefer_checkpointing: bool
    prefer_quantization: bool
    target_rank: int
    target_sequence: int
    target_effective_batch: int
    optimizer: Optimizer
    allow_offload: bool
    #: Levers to pull first when the plan does not fit, cheapest cost first.
    concession_order: list[str] = field(default_factory=list)
    #: Levers to pull when memory is left over, most valuable first. Spare
    #: capacity is not a virtue -- a plan that leaves 60 GB idle is a plan
    #: that trains more slowly or worse than it needed to.
    expansion_order: list[str] = field(default_factory=list)


#: Ordered from the concession that costs least to the one that costs most.
#: Sequence length sits late because truncating past the data's real length
#: silently discards training signal, which is worse than training slowly.
STANDARD_CONCESSIONS = [
    "efficient_attention",
    "batch_size",
    "optimizer_8bit",
    "gradient_checkpointing",
    "quantize_4bit",
    "lora_rank",
    "sequence_length",
    "cpu_offload",
]

PREFERENCES: dict[Goal, Preferences] = {
    Goal.FASTEST: Preferences(
        prefer_checkpointing=False,
        prefer_quantization=False,
        target_rank=16,
        target_sequence=1024,
        target_effective_batch=16,
        optimizer=Optimizer.ADAMW,
        allow_offload=False,
        concession_order=[
            "efficient_attention",
            "quantize_4bit",
            "optimizer_8bit",
            "batch_size",
            "gradient_checkpointing",
            "sequence_length",
            "lora_rank",
            "cpu_offload",
        ],
        expansion_order=[
            "no_checkpointing",
            "batch_size",
            "no_quantization",
            "sequence_length",
        ],
    ),
    Goal.BALANCED: Preferences(
        prefer_checkpointing=True,
        prefer_quantization=True,
        target_rank=16,
        target_sequence=2048,
        target_effective_batch=16,
        optimizer=Optimizer.ADAMW_8BIT,
        allow_offload=False,
        concession_order=STANDARD_CONCESSIONS,
        expansion_order=[
            "batch_size",
            "no_checkpointing",
            "sequence_length",
        ],
    ),
    Goal.MAX_QUALITY: Preferences(
        prefer_checkpointing=True,
        prefer_quantization=False,
        target_rank=64,
        target_sequence=4096,
        target_effective_batch=32,
        optimizer=Optimizer.ADAMW,
        allow_offload=True,
        concession_order=[
            "efficient_attention",
            "batch_size",
            "gradient_checkpointing",
            "optimizer_8bit",
            "quantize_4bit",
            "cpu_offload",
            "sequence_length",
            "lora_rank",
        ],
        expansion_order=[
            "no_offload",
            "no_quantization",
            "lora_rank",
            "sequence_length",
            "batch_size",
        ],
    ),
    Goal.MIN_DEVICE_MEMORY: Preferences(
        prefer_checkpointing=True,
        prefer_quantization=True,
        target_rank=8,
        target_sequence=1024,
        target_effective_batch=16,
        optimizer=Optimizer.ADAMW_8BIT,
        allow_offload=True,
        concession_order=[
            "efficient_attention",
            "gradient_checkpointing",
            "quantize_4bit",
            "batch_size",
            "optimizer_8bit",
            "sequence_length",
            "lora_rank",
            "cpu_offload",
        ],
        expansion_order=[],
    ),
    Goal.MIN_SYSTEM_MEMORY: Preferences(
        prefer_checkpointing=True,
        prefer_quantization=True,
        target_rank=8,
        target_sequence=1024,
        target_effective_batch=8,
        optimizer=Optimizer.ADAMW_8BIT,
        allow_offload=False,  # offloading is exactly what this goal avoids
        concession_order=[
            "efficient_attention",
            "gradient_checkpointing",
            "quantize_4bit",
            "batch_size",
            "optimizer_8bit",
            "sequence_length",
            "lora_rank",
        ],
        expansion_order=[],
    ),
    # Custom starts from the balanced defaults; the point is that the person
    # then edits them, so the starting point should be the unsurprising one.
    Goal.CUSTOM: Preferences(
        prefer_checkpointing=True,
        prefer_quantization=True,
        target_rank=16,
        target_sequence=2048,
        target_effective_batch=16,
        optimizer=Optimizer.ADAMW_8BIT,
        allow_offload=False,
        concession_order=STANDARD_CONCESSIONS,
        expansion_order=["batch_size", "no_checkpointing"],
    ),
}


@dataclass
class Budget:
    """Ceilings a person imposes, independent of what the hardware offers.

    An unset field means "no limit from me" -- it does not mean zero, and it
    does not mean the hardware limit has gone away. Hardware capacity is
    applied separately and always.
    """

    max_device_memory_gb: float | None = None
    max_system_ram_gb: float | None = None
    max_training_hours: float | None = None
    max_disk_gb: float | None = None

    @property
    def active(self) -> bool:
        return any(
            v is not None
            for v in (
                self.max_device_memory_gb,
                self.max_system_ram_gb,
                self.max_training_hours,
                self.max_disk_gb,
            )
        )

    def memory_ceiling(self, hardware_gb: Value[float]) -> tuple[Value[float], str]:
        """The binding memory limit, and which of the two produced it.

        The tighter of budget and hardware wins. Naming the winner matters:
        "you asked for 8 GB" and "the card has 8 GB" call for completely
        different responses from the person reading the plan.
        """
        if self.max_device_memory_gb is None:
            return hardware_gb, "hardware"
        budget = Value.configured(self.max_device_memory_gb, "budget you set", "GB")
        if not hardware_gb.known:
            return budget, "budget"
        if self.max_device_memory_gb <= hardware_gb.get():
            return budget, "budget"
        return hardware_gb, "hardware"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_device_memory_gb": self.max_device_memory_gb,
            "max_system_ram_gb": self.max_system_ram_gb,
            "max_training_hours": self.max_training_hours,
            "max_disk_gb": self.max_disk_gb,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Budget":
        data = data or {}
        return cls(
            max_device_memory_gb=data.get("max_device_memory_gb"),
            max_system_ram_gb=data.get("max_system_ram_gb"),
            max_training_hours=data.get("max_training_hours"),
            max_disk_gb=data.get("max_disk_gb"),
        )


def method_for(prefer_quantization: bool, quantization_available: bool) -> tuple[Method, Quantization]:
    """Pick a PEFT method from capability, never from a device name."""
    if prefer_quantization and quantization_available:
        return Method.QLORA, Quantization.NF4
    return Method.LORA, Quantization.NONE
