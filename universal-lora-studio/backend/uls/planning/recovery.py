"""Recovery from a real out-of-memory failure.

An OOM is different in kind from a failed estimate. The estimate was wrong;
that is information. So this module works from the *measured* allocation at
the moment of failure where the runtime reports it, not from the prediction
that already proved too low.

It proposes and never applies. Silently retrying with different settings would
produce an adapter trained under conditions the person never agreed to and
cannot reconstruct from the experiment record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..estimation.memory import Method, Optimizer, TrainingConfig, estimator
from ..estimation.model_spec import ModelSpec
from ..hardware.capability import Capability, Quantization
from ..value import Value


@dataclass
class Remedy:
    """One proposed change, with its price stated up front."""

    field: str
    current: Any
    proposed: Any
    saving_gb: float
    cost: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "current": _plain(self.current),
            "proposed": _plain(self.proposed),
            "saving_gb": round(self.saving_gb, 2),
            "cost": self.cost,
            "rationale": self.rationale,
        }


@dataclass
class Diagnosis:
    headline: str
    detail: str
    failed_config: TrainingConfig
    remedies: list[Remedy] = field(default_factory=list)
    combined: TrainingConfig | None = None
    combined_estimate_gb: float | None = None
    observed_gb: Value[float] = field(default_factory=lambda: Value.unavailable())
    capacity_gb: Value[float] = field(default_factory=lambda: Value.unavailable())
    estimate_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "detail": self.detail,
            "failed_config": self.failed_config.to_dict(),
            "remedies": [r.to_dict() for r in self.remedies],
            "suggested_config": self.combined.to_dict() if self.combined else None,
            "suggested_estimate_gb": (
                round(self.combined_estimate_gb, 2) if self.combined_estimate_gb else None
            ),
            "observed": self.observed_gb.to_dict(),
            "capacity": self.capacity_gb.to_dict(),
            "estimate_error": self.estimate_error,
            "requires_confirmation": True,
        }


class RecoveryAdvisor:
    def diagnose(
        self,
        config: TrainingConfig,
        model: ModelSpec,
        capability: Capability,
        error_text: str = "",
        step: int | None = None,
    ) -> Diagnosis:
        observed, capacity = _parse_allocation(error_text)
        if not capacity.known:
            capacity = capability.training_memory_gb

        baseline = estimator.estimate(model, config, capacity).breakdown.total_gb

        # Report the estimator's error in whichever direction it went. An
        # overestimate is not a harmless conservatism: it is the reason someone
        # trained at half the batch size they could have used.
        estimate_error = None
        if observed.known:
            gap = observed.get() - baseline
            if gap > 1.0:
                estimate_error = (
                    f"The run reached {observed.display()} where this planner "
                    f"predicted {baseline:.1f} GB — low by {gap:.1f} GB. Treat the "
                    "estimates below as optimistic for this model."
                )
            elif gap < -2.0:
                estimate_error = (
                    f"This planner predicted {baseline:.1f} GB but the run only "
                    f"reached {observed.display()} before failing. The prediction "
                    "was high, so the failure may have come from a memory spike "
                    "rather than steady-state use — a single long sample, or "
                    "evaluation running alongside training."
                )

        detail = _context_detail(config, step)
        remedies = self._build_remedies(config, model, capability, baseline, capacity)

        combined, combined_gb = self._combine(config, model, capability, remedies, capacity)

        return Diagnosis(
            headline="Training stopped: out of memory",
            detail=detail,
            failed_config=config,
            remedies=remedies,
            combined=combined,
            combined_estimate_gb=combined_gb,
            observed_gb=observed,
            capacity_gb=capacity,
            estimate_error=estimate_error,
        )

    # -- candidate changes ------------------------------------------------

    def _build_remedies(
        self,
        config: TrainingConfig,
        model: ModelSpec,
        cap: Capability,
        baseline: float,
        capacity: Value[float],
    ) -> list[Remedy]:
        remedies: list[Remedy] = []

        def saving(**changes) -> float:
            trial = TrainingConfig.from_dict(config.to_dict())
            for key, value in changes.items():
                setattr(trial, key, value)
            return baseline - estimator.estimate(model, trial, capacity).breakdown.total_gb

        if config.batch_size > 1:
            new_batch = max(1, config.batch_size // 2)
            remedies.append(
                Remedy(
                    "batch_size",
                    config.batch_size,
                    new_batch,
                    saving(batch_size=new_batch),
                    "None, if accumulation is raised to compensate. Steps get slower.",
                    "Halving the micro-batch is the first thing to try because "
                    "accumulation can hold the effective batch size constant, so "
                    "the optimisation problem stays the same.",
                )
            )
            remedies.append(
                Remedy(
                    "gradient_accumulation",
                    config.gradient_accumulation,
                    config.gradient_accumulation * 2,
                    0.0,
                    "Slower wall-clock per optimizer step.",
                    "Doubling accumulation alongside the batch change keeps the "
                    f"effective batch at {config.effective_batch}.",
                )
            )

        if not config.gradient_checkpointing:
            remedies.append(
                Remedy(
                    "gradient_checkpointing",
                    False,
                    True,
                    saving(gradient_checkpointing=True),
                    "Roughly 20-30% slower steps.",
                    "Recomputing activations instead of storing them is usually the "
                    "largest saving available that does not change the result.",
                )
            )

        if config.sequence_length > 512:
            new_seq = config.sequence_length // 2
            remedies.append(
                Remedy(
                    "sequence_length",
                    config.sequence_length,
                    new_seq,
                    saving(sequence_length=new_seq),
                    "Samples longer than the new limit get truncated, losing data.",
                    "Activation and logit memory both scale linearly with sequence "
                    "length. Check what fraction of your data this cuts before "
                    "accepting it.",
                )
            )

        quant_ok = any(q.is_four_bit for q in cap.quantization)
        if not config.quantization.is_four_bit and quant_ok:
            remedies.append(
                Remedy(
                    "quantization",
                    config.quantization.value,
                    Quantization.NF4.value,
                    saving(quantization=Quantization.NF4, method=Method.QLORA),
                    "A small, usually acceptable quality loss.",
                    "Moving the frozen base model to 4-bit cuts weight memory by "
                    "about three quarters and is the largest single lever here.",
                )
            )

        if config.optimizer is not Optimizer.ADAMW_8BIT and quant_ok:
            remedies.append(
                Remedy(
                    "optimizer",
                    config.optimizer.value,
                    Optimizer.ADAMW_8BIT.value,
                    saving(optimizer=Optimizer.ADAMW_8BIT),
                    "Negligible in practice.",
                    "8-bit optimizer states are close to free at LoRA scale, though "
                    "the saving is small because the adapter is small.",
                )
            )

        if config.lora_rank > 4:
            new_rank = config.lora_rank // 2
            remedies.append(
                Remedy(
                    "lora_rank",
                    config.lora_rank,
                    new_rank,
                    saving(lora_rank=new_rank),
                    "Less adapter capacity, which can cap final quality.",
                    "Rank barely moves the memory needle at these sizes. It is "
                    "listed for completeness, not because it will rescue this run.",
                )
            )

        if not config.efficient_attention and cap.supports_flash_attention:
            remedies.append(
                Remedy(
                    "efficient_attention",
                    False,
                    True,
                    saving(efficient_attention=True),
                    "None.",
                    "Memory-efficient attention avoids materialising the score "
                    "matrix, which dominates at long sequence lengths.",
                )
            )

        if not config.cpu_offload and cap.supports_cpu_offload:
            remedies.append(
                Remedy(
                    "cpu_offload",
                    False,
                    True,
                    saving(cpu_offload=True),
                    "Several times slower. A last resort.",
                    "Offloading part of the model to host RAM will make almost "
                    "anything fit, at a speed most people will not accept.",
                )
            )

        remedies.sort(key=lambda r: r.saving_gb, reverse=True)
        return remedies

    def _combine(
        self,
        config: TrainingConfig,
        model: ModelSpec,
        cap: Capability,
        remedies: list[Remedy],
        capacity: Value[float],
    ) -> tuple[TrainingConfig | None, float | None]:
        """Assemble the cheapest set of changes that gets back under capacity.

        Presented as one proposal so the person makes a single decision, with
        every component of it listed and reversible.
        """
        if not capacity.known:
            return None, None
        target = capacity.get() * 0.85
        trial = TrainingConfig.from_dict(config.to_dict())
        # Cheapest-first by human cost, not by size of saving.
        order = [
            "efficient_attention",
            "gradient_checkpointing",
            "batch_size",
            "optimizer",
            "quantization",
            "sequence_length",
            "cpu_offload",
        ]
        by_field = {r.field: r for r in remedies}
        for field_name in order:
            current = estimator.estimate(model, trial, capacity).breakdown.total_gb
            if current <= target:
                break
            remedy = by_field.get(field_name)
            if remedy is None:
                continue
            value = remedy.proposed
            if field_name == "quantization":
                trial.quantization = Quantization(value)
                trial.method = Method.QLORA
            elif field_name == "optimizer":
                trial.optimizer = Optimizer(value)
            else:
                setattr(trial, field_name, value)
            if field_name == "batch_size":
                trial.gradient_accumulation = max(
                    1, round(config.effective_batch / max(1, trial.batch_size * trial.device_count))
                )
        final = estimator.estimate(model, trial, capacity).breakdown.total_gb
        return trial, final


def _context_detail(config: TrainingConfig, step: int | None) -> str:
    where = f" at step {step:,}" if step else ""
    return (
        f"The run failed{where} with batch size {config.batch_size}, sequence "
        f"length {config.sequence_length:,} and "
        f"{'checkpointing on' if config.gradient_checkpointing else 'checkpointing off'}. "
        "Nothing has been changed. Review the options below and apply them "
        "yourself if you agree."
    )


_ALLOC_PATTERNS = [
    # CUDA: "Tried to allocate 2.00 GiB ... 23.65 GiB total capacity ... 22.10 GiB already allocated"
    (r"([\d.]+)\s*GiB already allocated", 1.0),
    (r"([\d.]+)\s*MiB already allocated", 1 / 1024),
]
_CAPACITY_PATTERNS = [
    (r"([\d.]+)\s*GiB (?:total capacity|of which)", 1.0),
    (r"GPU\s+\d+\s+has a total capacity of\s+([\d.]+)\s*GiB", 1.0),
]


def _parse_allocation(error_text: str) -> tuple[Value[float], Value[float]]:
    """Pull real numbers out of a runtime OOM message.

    These are measurements the runtime took at the moment of failure, so they
    outrank anything this planner predicted, and are labelled MEASURED to say so.
    """
    if not error_text:
        return (
            Value.unavailable("no error text was captured"),
            Value.unavailable("no error text was captured"),
        )

    observed = Value.unavailable("the error message did not report allocated memory")
    for pattern, scale in _ALLOC_PATTERNS:
        match = re.search(pattern, error_text)
        if match:
            observed = Value.measured(
                round(float(match.group(1)) * scale, 2), "reported by the runtime", "GB"
            )
            break

    capacity = Value.unavailable("the error message did not report total capacity")
    for pattern, scale in _CAPACITY_PATTERNS:
        match = re.search(pattern, error_text)
        if match:
            capacity = Value.measured(
                round(float(match.group(1)) * scale, 2), "reported by the runtime", "GB"
            )
            break
    return observed, capacity


def _plain(value: Any) -> Any:
    return value.value if hasattr(value, "value") and not isinstance(value, (int, float, str, bool)) else value


advisor = RecoveryAdvisor()
