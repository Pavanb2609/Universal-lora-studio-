"""The configuration resolver.

Turns capability + model + data + goal + budget into a runnable configuration,
and records why each value ended up where it did.

The priority chain is absolute and is enforced structurally rather than by
convention: anything a person set explicitly lands in ``config.locked``, and
every write in this module goes through ``_set``, which refuses to touch a
locked field. The resolver cannot silently override a person's choice because
it has no code path that does.

Order of authority:

    1. explicit user settings   (never touched)
    2. resource budget          (a ceiling the person imposed)
    3. hardware capability      (a ceiling physics imposed)
    4. model requirements
    5. dataset characteristics
    6. automatic recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..estimation.memory import (
    Estimate,
    Method,
    Optimizer,
    TrainingConfig,
    estimator,
)
from ..estimation.model_spec import ModelSpec, infer_targets, sanity_check
from ..hardware.capability import Backend, Capability, Precision, Quantization
from ..value import Value
from .goals import PREFERENCES, Budget, Goal, method_for


@dataclass
class Decision:
    """One change the resolver made, and the reason for it.

    Surfaced in full in the UI. A plan a person cannot interrogate is a plan
    they cannot trust, and this is the difference between a recommendation and
    an instruction.
    """

    field: str
    value: Any
    reason: str
    #: Which rung of the priority chain produced this.
    authority: str = "recommendation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "reason": self.reason,
            "authority": self.authority,
        }


@dataclass
class Plan:
    config: TrainingConfig
    estimate: Estimate
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ceiling_source: str = "hardware"
    memory_ceiling: Value[float] | None = None

    @property
    def feasible(self) -> bool:
        return not self.blockers and self.estimate.verdict["status"] in ("fits", "tight")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "config": self.config.to_dict(),
            "estimate": self.estimate.to_dict(),
            "decisions": [d.to_dict() for d in self.decisions],
            "warnings": self.warnings,
            "blockers": self.blockers,
            "ceiling_source": self.ceiling_source,
            "memory_ceiling": (
                self.memory_ceiling.to_dict() if self.memory_ceiling else None
            ),
        }


@dataclass
class DataProfile:
    """What the dataset implies for configuration. All fields optional --
    planning works without a dataset, it just has less to go on."""

    sample_count: int | None = None
    p95_tokens: int | None = None
    max_tokens: int | None = None
    mean_tokens: float | None = None

    @property
    def known(self) -> bool:
        return self.sample_count is not None or self.p95_tokens is not None


class Resolver:
    #: Leave this fraction of memory unclaimed. Allocators fragment, and a plan
    #: that fills the card exactly is a plan that fails in hour three.
    SAFETY_FRACTION = 0.90

    RANK_LADDER = [64, 32, 16, 8, 4]
    SEQUENCE_LADDER = [8192, 4096, 2048, 1024, 512, 256]

    def resolve(
        self,
        capability: Capability,
        model: ModelSpec,
        goal: Goal = Goal.BALANCED,
        budget: Budget | None = None,
        data: DataProfile | None = None,
        user_config: TrainingConfig | None = None,
    ) -> Plan:
        budget = budget or Budget()
        data = data or DataProfile()
        prefs = PREFERENCES[goal]

        cfg = user_config or TrainingConfig()
        decisions: list[Decision] = []
        warnings: list[str] = list(sanity_check(model))
        blockers: list[str] = []

        for name in sorted(cfg.locked):
            decisions.append(
                Decision(
                    name,
                    getattr(cfg, name, None),
                    "You set this explicitly, so it was left alone.",
                    "user",
                )
            )

        # -- rung 3: what the hardware can do --------------------------
        self._apply_capability(cfg, capability, prefs, decisions, warnings)

        # -- rung 4 and 5: model and dataset ---------------------------
        self._apply_model(cfg, model, prefs, decisions)
        self._apply_data(cfg, data, model, decisions, warnings)

        # -- rung 2: the person's budget, then fit ---------------------
        hardware_gb = capability.training_memory_gb
        ceiling, source = budget.memory_ceiling(hardware_gb)
        if source == "budget":
            decisions.append(
                Decision(
                    "memory_ceiling",
                    ceiling.or_else(None),
                    "Your budget is tighter than the hardware, so it sets the limit.",
                    "budget",
                )
            )

        on_cpu = capability.backend is Backend.CPU
        plan_estimate = self._fit(
            cfg, model, capability, ceiling, prefs, decisions, warnings, on_cpu
        )

        if plan_estimate.verdict["status"] in ("unlikely", "infeasible"):
            blockers.append(self._blocker_message(plan_estimate, cfg, model, capability))

        self._check_secondary_budgets(
            cfg, model, capability, budget, warnings, blockers
        )

        return Plan(
            config=cfg,
            estimate=plan_estimate,
            decisions=decisions,
            warnings=warnings,
            blockers=blockers,
            ceiling_source=source,
            memory_ceiling=ceiling,
        )

    # -- the only writer ------------------------------------------------

    #: Rungs of the priority chain that describe a real constraint rather than
    #: a preference. Reasoning from these is recorded even when it changes
    #: nothing, so the trail explains every value rather than only the ones
    #: that happened to need adjusting.
    CONSTRAINT_AUTHORITIES = frozenset({"user", "budget", "hardware", "model", "dataset"})

    @staticmethod
    def _set(
        cfg: TrainingConfig,
        field_name: str,
        value: Any,
        reason: str,
        decisions: list[Decision],
        authority: str = "recommendation",
    ) -> bool:
        """Write a field unless the person locked it.

        Returns whether the write happened, so callers that need a fallback
        (because their preferred lever is unavailable) can move on.
        """
        if cfg.is_locked(field_name):
            return False
        if getattr(cfg, field_name) == value:
            if authority in Resolver.CONSTRAINT_AUTHORITIES and not any(
                d.field == field_name and d.authority == authority for d in decisions
            ):
                decisions.append(Decision(field_name, value, reason, authority))
            return True
        setattr(cfg, field_name, value)
        decisions.append(Decision(field_name, value, reason, authority))
        return True

    # -- rungs -----------------------------------------------------------

    def _apply_capability(
        self,
        cfg: TrainingConfig,
        cap: Capability,
        prefs,
        decisions: list[Decision],
        warnings: list[str],
    ) -> None:
        precision = cap.preferred_precision
        self._set(
            cfg,
            "precision",
            precision,
            f"{precision.value} is the best numeric format this environment reports.",
            decisions,
            "hardware",
        )

        quant_available = any(q.is_four_bit for q in cap.quantization)
        method, quantization = method_for(prefs.prefer_quantization, quant_available)
        if prefs.prefer_quantization and not quant_available:
            warnings.append(
                "4-bit quantization is not available in this environment, so the "
                "plan uses standard LoRA. Weights will be held at full precision, "
                "which needs roughly four times the memory."
            )
        self._set(
            cfg,
            "method",
            method,
            (
                "QLoRA keeps the base model in 4-bit, which is what makes this fit."
                if method is Method.QLORA
                else "Standard LoRA: 4-bit quantization is unavailable or not wanted here."
            ),
            decisions,
            "hardware",
        )
        self._set(cfg, "quantization", quantization, "Follows from the method.", decisions, "hardware")

        self._set(
            cfg,
            "efficient_attention",
            cap.supports_flash_attention,
            (
                "Memory-efficient attention is available and avoids materialising "
                "the score matrix."
                if cap.supports_flash_attention
                else "flash-attn is not installed, so attention scores are materialised. "
                "Installing it would cut memory at long sequence lengths."
            ),
            decisions,
            "hardware",
        )

        if cap.device_count > 1:
            self._set(
                cfg,
                "device_count",
                cap.device_count,
                f"{cap.device_count} devices were detected and can share the work.",
                decisions,
                "hardware",
            )

        if cap.backend is Backend.CPU:
            self._set(
                cfg, "optimizer", Optimizer.ADAMW,
                "8-bit optimizers need a CUDA build of bitsandbytes; on CPU the "
                "standard optimizer is used.",
                decisions, "hardware",
            )
            warnings.append(
                "There is no accelerator here. CPU training works, but expect "
                "hours where a GPU would take minutes. The plan is sized for a "
                "small model and short sequences to keep that tolerable."
            )
        else:
            self._set(
                cfg, "optimizer", prefs.optimizer,
                f"{prefs.optimizer.value} matches the selected goal.",
                decisions,
            )

        if not cap.supports_cpu_offload and cfg.cpu_offload:
            self._set(
                cfg, "cpu_offload", False,
                "This environment cannot offload to host memory.",
                decisions, "hardware",
            )

    def _apply_model(
        self, cfg: TrainingConfig, model: ModelSpec, prefs, decisions: list[Decision]
    ) -> None:
        aggressive = prefs.target_rank >= 32
        targets = infer_targets(model, aggressive)
        self._set(
            cfg,
            "target_modules",
            targets,
            (
                "Targeting attention and MLP projections gives the adapter more "
                "capacity to absorb new knowledge."
                if aggressive
                else "Targeting the attention projections: the usual choice, and "
                "enough for style and instruction following."
            ),
            decisions,
            "model",
        )
        self._set(cfg, "lora_rank", prefs.target_rank, "Starting point for this goal.", decisions)
        self._set(
            cfg, "lora_alpha", prefs.target_rank * 2,
            "Alpha at twice the rank keeps the update scale steady as rank changes.",
            decisions,
        )

        cap_len = model.max_position_embeddings
        target_seq = min(prefs.target_sequence, cap_len)
        if target_seq < prefs.target_sequence:
            self._set(
                cfg, "sequence_length", target_seq,
                f"The model's context limit is {cap_len:,} tokens.",
                decisions, "model",
            )
        else:
            self._set(cfg, "sequence_length", target_seq, "Starting point for this goal.", decisions)

    def _apply_data(
        self,
        cfg: TrainingConfig,
        data: DataProfile,
        model: ModelSpec,
        decisions: list[Decision],
        warnings: list[str],
    ) -> None:
        if not data.known:
            return

        if data.p95_tokens:
            # Fit the bulk of the data rather than its longest outlier: sizing
            # for the maximum wastes memory on every step to serve a handful of
            # samples, and sizing for the mean silently truncates a quarter of
            # the corpus.
            target = _round_up_pow2(data.p95_tokens)
            target = min(target, model.max_position_embeddings)
            reason = (
                f"95% of your samples fit in {data.p95_tokens:,} tokens, so the "
                f"sequence length is set to {target:,} to cover them without "
                "paying for the longest outliers."
            )
            if self._set(cfg, "sequence_length", target, reason, decisions, "dataset"):
                if data.max_tokens and data.max_tokens > target:
                    warnings.append(
                        f"The longest sample is {data.max_tokens:,} tokens and will "
                        f"be truncated at {target:,}. That affects roughly the top "
                        "5% of your data."
                    )

        if data.sample_count is not None:
            if data.sample_count < 200:
                self._set(
                    cfg, "epochs", 5.0,
                    f"With only {data.sample_count:,} samples, more passes are "
                    "needed to learn anything; watch validation loss for overfitting.",
                    decisions, "dataset",
                )
                self._set(
                    cfg, "lora_rank", min(cfg.lora_rank, 8),
                    "A small dataset cannot support a high-capacity adapter without "
                    "memorising it.",
                    decisions, "dataset",
                )
                warnings.append(
                    f"{data.sample_count:,} samples is a small dataset for fine-tuning. "
                    "Expect the adapter to pick up style rather than knowledge."
                )
            elif data.sample_count > 100_000:
                self._set(
                    cfg, "epochs", 1.0,
                    f"{data.sample_count:,} samples is plenty; a single pass is "
                    "usually enough and repeated passes tend to overfit.",
                    decisions, "dataset",
                )

    # -- fitting ---------------------------------------------------------

    def _fit(
        self,
        cfg: TrainingConfig,
        model: ModelSpec,
        cap: Capability,
        ceiling: Value[float],
        prefs,
        decisions: list[Decision],
        warnings: list[str],
        on_cpu: bool,
    ) -> Estimate:
        """Walk the concession ladder until the plan fits, or the levers run out.

        Each concession is applied only if it is not locked and the environment
        supports it, and each one that fires explains itself.
        """
        self._set(
            cfg, "batch_size", max(1, prefs.target_effective_batch // 16),
            "Starting batch size for this goal.", decisions,
        )
        self._set(
            cfg, "gradient_checkpointing", prefs.prefer_checkpointing,
            (
                "Checkpointing trades some speed for a large cut in activation memory."
                if prefs.prefer_checkpointing
                else "Checkpointing is off to keep throughput up."
            ),
            decisions,
        )
        self._rebalance_accumulation(cfg, prefs, decisions)

        estimate = estimator.estimate(model, cfg, ceiling, on_cpu=on_cpu)
        if not ceiling.known:
            warnings.append(
                "This profile does not say how much memory is available, so the "
                "plan could not be checked against a limit."
            )
            return estimate

        limit = ceiling.get() * self.SAFETY_FRACTION

        for lever in prefs.concession_order:
            if estimate.breakdown.total_gb <= limit:
                break
            applied = self._apply_concession(lever, cfg, model, cap, decisions, warnings)
            if applied:
                self._rebalance_accumulation(cfg, prefs, decisions)
                estimate = estimator.estimate(model, cfg, ceiling, on_cpu=on_cpu)

        if estimate.breakdown.total_gb <= limit:
            estimate = self._expand(
                cfg, model, cap, ceiling, limit, prefs, decisions, on_cpu
            )
        return estimate

    def _expand(
        self,
        cfg: TrainingConfig,
        model: ModelSpec,
        cap: Capability,
        ceiling: Value[float],
        limit: float,
        prefs,
        decisions: list[Decision],
        on_cpu: bool,
    ) -> Estimate:
        """Spend leftover memory on whatever the goal values.

        The mirror image of the concession ladder, and the reason the same
        model and dataset produce a different plan on a 24 GB card than on an
        8 GB one. Each step is tried, costed, and reverted if it overshoots, so
        the plan never ends up worse than where it started.
        """
        estimate = estimator.estimate(model, cfg, ceiling, on_cpu=on_cpu)
        if not prefs.expansion_order:
            return estimate

        for lever in prefs.expansion_order:
            # Keep pulling the same lever while it still pays.
            for _ in range(8):
                snapshot = {
                    "batch_size": cfg.batch_size,
                    "gradient_accumulation": cfg.gradient_accumulation,
                    "sequence_length": cfg.sequence_length,
                    "lora_rank": cfg.lora_rank,
                    "lora_alpha": cfg.lora_alpha,
                    "gradient_checkpointing": cfg.gradient_checkpointing,
                    "quantization": cfg.quantization,
                    "method": cfg.method,
                    "cpu_offload": cfg.cpu_offload,
                }
                marker = len(decisions)
                if not self._apply_expansion(lever, cfg, model, cap, prefs, decisions):
                    break
                self._rebalance_accumulation(cfg, prefs, decisions)
                candidate = estimator.estimate(model, cfg, ceiling, on_cpu=on_cpu)
                if candidate.breakdown.total_gb > limit:
                    for key, value in snapshot.items():
                        setattr(cfg, key, value)
                    del decisions[marker:]
                    break
                estimate = candidate
        return estimate

    def _apply_expansion(
        self,
        lever: str,
        cfg: TrainingConfig,
        model: ModelSpec,
        cap: Capability,
        prefs,
        decisions: list[Decision],
    ) -> bool:
        spare = "There is memory to spare"

        if lever == "batch_size":
            if cfg.batch_size >= 64 or cfg.gradient_accumulation <= 1:
                return False
            return self._set(
                cfg, "batch_size", cfg.batch_size * 2,
                f"{spare}, so the batch is doubled to {cfg.batch_size * 2}. Fewer, "
                "larger steps run faster than many small ones.",
                decisions,
            )

        if lever == "no_checkpointing":
            if not cfg.gradient_checkpointing:
                return False
            return self._set(
                cfg, "gradient_checkpointing", False,
                f"{spare}, so activations are kept rather than recomputed. Worth "
                "roughly 20-30% throughput.",
                decisions,
            )

        if lever == "no_quantization":
            if cfg.quantization is Quantization.NONE:
                return False
            changed = self._set(
                cfg, "quantization", Quantization.NONE,
                f"{spare}, so the base model stays at full precision instead of "
                "4-bit. Quantization costs a little accuracy and is unnecessary here.",
                decisions,
            )
            if changed:
                self._set(cfg, "method", Method.LORA, "Follows from unquantized weights.", decisions)
            return changed

        if lever == "no_offload":
            if not cfg.cpu_offload:
                return False
            return self._set(
                cfg, "cpu_offload", False,
                f"{spare}, so nothing needs to be offloaded to host RAM.",
                decisions,
            )

        if lever == "sequence_length":
            higher = [s for s in reversed(self.SEQUENCE_LADDER) if s > cfg.sequence_length]
            if not higher:
                return False
            new_seq = min(higher[0], model.max_position_embeddings, prefs.target_sequence * 2)
            if new_seq <= cfg.sequence_length:
                return False
            return self._set(
                cfg, "sequence_length", new_seq,
                f"{spare}, so sequences extend to {new_seq:,} tokens, which truncates "
                "less of your data.",
                decisions,
            )

        if lever == "lora_rank":
            higher = [r for r in reversed(self.RANK_LADDER) if r > cfg.lora_rank]
            if not higher:
                return False
            new_rank = min(higher[0], 128)
            changed = self._set(
                cfg, "lora_rank", new_rank,
                f"{spare}, so adapter rank rises to {new_rank}, giving it more "
                "capacity to learn from your data.",
                decisions,
            )
            if changed:
                self._set(cfg, "lora_alpha", new_rank * 2, "Alpha tracks rank.", decisions)
            return changed

        return False

    def _apply_concession(
        self,
        lever: str,
        cfg: TrainingConfig,
        model: ModelSpec,
        cap: Capability,
        decisions: list[Decision],
        warnings: list[str],
    ) -> bool:
        over = "The plan does not fit yet"

        if lever == "efficient_attention":
            if cfg.efficient_attention or not cap.supports_flash_attention:
                return False
            return self._set(
                cfg, "efficient_attention", True,
                f"{over}, and memory-efficient attention is free memory here.",
                decisions,
            )

        if lever == "batch_size":
            if cfg.batch_size <= 1:
                return False
            return self._set(
                cfg, "batch_size", max(1, cfg.batch_size // 2),
                f"{over}, so the batch is halved. Gradient accumulation rises to "
                "match, keeping the effective batch size the same.",
                decisions,
            )

        if lever == "optimizer_8bit":
            if cfg.optimizer is Optimizer.ADAMW_8BIT or cap.backend is Backend.CPU:
                return False
            if not any(q.is_four_bit for q in cap.quantization):
                return False  # same bitsandbytes dependency
            return self._set(
                cfg, "optimizer", Optimizer.ADAMW_8BIT,
                f"{over}, so the optimizer moments move to 8-bit. This has little "
                "effect on adapter quality.",
                decisions,
            )

        if lever == "gradient_checkpointing":
            if cfg.gradient_checkpointing:
                return False
            return self._set(
                cfg, "gradient_checkpointing", True,
                f"{over}, so activations are recomputed instead of stored. "
                "Expect roughly 20-30% slower steps.",
                decisions,
            )

        if lever == "quantize_4bit":
            if cfg.quantization.is_four_bit:
                return False
            if not any(q.is_four_bit for q in cap.quantization):
                warnings.append(
                    "4-bit quantization would help here but bitsandbytes is not "
                    "available in this environment."
                )
                return False
            changed = self._set(
                cfg, "quantization", Quantization.NF4,
                f"{over}, so the base model moves to 4-bit. This is the single "
                "biggest saving available and costs a little quality.",
                decisions,
            )
            if changed:
                self._set(cfg, "method", Method.QLORA, "Follows from 4-bit weights.", decisions)
            return changed

        if lever == "lora_rank":
            lower = [r for r in self.RANK_LADDER if r < cfg.lora_rank]
            if not lower:
                return False
            new_rank = lower[0]
            changed = self._set(
                cfg, "lora_rank", new_rank,
                f"{over}, so adapter rank drops to {new_rank}. Lower rank means "
                "less capacity to learn new behaviour.",
                decisions,
            )
            if changed:
                self._set(cfg, "lora_alpha", new_rank * 2, "Alpha tracks rank.", decisions)
            return changed

        if lever == "sequence_length":
            lower = [s for s in self.SEQUENCE_LADDER if s < cfg.sequence_length]
            if not lower:
                return False
            new_seq = lower[0]
            changed = self._set(
                cfg, "sequence_length", new_seq,
                f"{over}, so sequences are cut to {new_seq:,} tokens. Anything "
                "longer in your data will be truncated.",
                decisions,
            )
            if changed:
                warnings.append(
                    f"Sequence length was reduced to {new_seq:,} tokens to fit. "
                    "Check how much of your data this truncates before training."
                )
            return changed

        if lever == "cpu_offload":
            if cfg.cpu_offload or not cap.supports_cpu_offload:
                return False
            changed = self._set(
                cfg, "cpu_offload", True,
                f"{over}, so part of the model moves to host RAM. This is the "
                "last resort: it works, and it is several times slower.",
                decisions,
            )
            if changed:
                warnings.append(
                    "CPU offload is on. It is the slowest option available and is "
                    "only worth it if the alternative is not training at all."
                )
            return changed

        return False

    def _rebalance_accumulation(self, cfg: TrainingConfig, prefs, decisions: list[Decision]) -> None:
        """Hold the effective batch size steady as the micro-batch shrinks.

        Cutting batch size to fit memory would otherwise quietly change the
        optimisation problem, not just its memory footprint. Accumulation keeps
        the gradient statistics the person is actually training with intact.
        """
        if cfg.is_locked("gradient_accumulation"):
            return
        per_device = max(1, cfg.batch_size * max(1, cfg.device_count))
        needed = max(1, round(prefs.target_effective_batch / per_device))
        if needed != cfg.gradient_accumulation:
            cfg.gradient_accumulation = needed
            decisions.append(
                Decision(
                    "gradient_accumulation",
                    needed,
                    f"Set so the effective batch stays near {prefs.target_effective_batch}.",
                    "recommendation",
                )
            )

    # -- reporting --------------------------------------------------------

    def _blocker_message(
        self, estimate: Estimate, cfg: TrainingConfig, model: ModelSpec, cap: Capability
    ) -> str:
        available = estimate.available_gb
        needed = estimate.breakdown.total_gb
        biggest, size = estimate.breakdown.largest_component
        locked = sorted(cfg.locked)
        base = (
            f"Even after exhausting the available adjustments, this plan needs about "
            f"{needed:.1f} GB against {available.display()}. The largest single "
            f"cost is {biggest.lower()} at {size:.1f} GB."
        )
        if locked:
            base += (
                f" You locked {', '.join(locked)}, so those were left as you set them — "
                "unlocking one may be enough."
            )
        else:
            base += (
                f" A smaller base model would be the reliable fix; {model.size_label} "
                "is a lot to ask of this environment."
            )
        return base

    def _check_secondary_budgets(
        self,
        cfg: TrainingConfig,
        model: ModelSpec,
        cap: Capability,
        budget: Budget,
        warnings: list[str],
        blockers: list[str],
    ) -> None:
        disk_needed = estimator.estimate_disk_gb(model, cfg)
        if budget.max_disk_gb is not None and disk_needed > budget.max_disk_gb:
            blockers.append(
                f"This needs roughly {disk_needed:.0f} GB of disk for the base model "
                f"and checkpoints, over your {budget.max_disk_gb:.0f} GB limit. "
                "Keeping fewer checkpoints is the easiest saving."
            )
        elif cap.disk_free_gb.known and disk_needed > cap.disk_free_gb.get():
            blockers.append(
                f"This needs roughly {disk_needed:.0f} GB of disk but only "
                f"{cap.disk_free_gb.display()} is free."
            )

        if budget.max_system_ram_gb is not None and cfg.cpu_offload:
            warnings.append(
                "CPU offload moves weights into host RAM, which works against the "
                "system memory limit you set."
            )


def _round_up_pow2(n: int) -> int:
    """Round a token count up to the next power of two.

    Attention kernels and padding buckets both like powers of two, and the
    memory cost of rounding up is smaller than the cost of a slow path.
    """
    value = 256
    while value < n and value < 131072:
        value *= 2
    return value


resolver = Resolver()
