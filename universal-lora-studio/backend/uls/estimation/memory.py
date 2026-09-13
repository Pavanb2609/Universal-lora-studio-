"""Training memory estimator.

Builds a component breakdown rather than a single number, because the useful
question is never "how much" but "what is eating it" -- the answer decides
whether to cut batch size, cut sequence length, quantize, or checkpoint.

Every figure here is a prediction from a model of memory behaviour, not a
measurement. Real allocators fragment, real kernels allocate scratch, and real
implementations differ. Results carry an uncertainty band and are stamped
ESTIMATED all the way to the screen. During an actual run, the monitor reports
MEASURED values alongside these, so the two can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..hardware.capability import Precision, Quantization
from ..value import Value
from .model_spec import ModelSpec

GB = 1024**3


class Optimizer(str, Enum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    ADAFACTOR = "adafactor"
    SGD = "sgd"
    SGD_MOMENTUM = "sgd_momentum"

    @property
    def bytes_per_trainable_param(self) -> float:
        """Optimizer state cost per trainable parameter.

        AdamW keeps two fp32 moments. Its 8-bit variant quantizes both, which
        is the single cheapest way to cut optimizer memory when the adapter is
        large. Adafactor factors the second moment into row and column
        statistics, making its cost nearly nil at LoRA scale.
        """
        return {
            Optimizer.ADAMW: 8.0,
            Optimizer.ADAMW_8BIT: 2.0,
            Optimizer.ADAFACTOR: 0.5,
            Optimizer.SGD: 0.0,
            Optimizer.SGD_MOMENTUM: 4.0,
        }[self]


class Method(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    DORA = "dora"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"
    IA3 = "ia3"
    FULL = "full"

    @property
    def is_lora_family(self) -> bool:
        return self in (Method.LORA, Method.QLORA, Method.ADALORA, Method.DORA)


@dataclass
class TrainingConfig:
    """A complete, runnable training configuration.

    ``locked`` records which fields a person set explicitly. The planner may
    rewrite anything absent from that set and must never touch anything in it.
    """

    method: Method = Method.QLORA
    quantization: Quantization = Quantization.NF4
    precision: Precision = Precision.BF16
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    batch_size: int = 1
    gradient_accumulation: int = 16
    sequence_length: int = 2048
    learning_rate: float = 2e-4
    epochs: float = 3.0
    optimizer: Optimizer = Optimizer.ADAMW_8BIT
    gradient_checkpointing: bool = True
    efficient_attention: bool = True
    cpu_offload: bool = False
    device_count: int = 1
    seed: int = 42
    locked: set[str] = field(default_factory=set)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation * max(1, self.device_count)

    def lock(self, *fields: str) -> "TrainingConfig":
        self.locked.update(fields)
        return self

    def is_locked(self, name: str) -> bool:
        return name in self.locked

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "quantization": self.quantization.value,
            "precision": self.precision.value,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch": self.effective_batch,
            "sequence_length": self.sequence_length,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "optimizer": self.optimizer.value,
            "gradient_checkpointing": self.gradient_checkpointing,
            "efficient_attention": self.efficient_attention,
            "cpu_offload": self.cpu_offload,
            "device_count": self.device_count,
            "seed": self.seed,
            "locked": sorted(self.locked),
        }

    #: Present in ``to_dict`` output for the UI's benefit but computed, not
    #: stored. Ignored on the way back in so a config round-trips cleanly.
    DERIVED_FIELDS = frozenset({"effective_batch"})

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        cfg = cls()
        for key, value in (data or {}).items():
            if key in cls.DERIVED_FIELDS:
                continue
            if key == "method":
                cfg.method = Method(value)
            elif key == "quantization":
                cfg.quantization = Quantization(value)
            elif key == "precision":
                cfg.precision = Precision(value)
            elif key == "optimizer":
                cfg.optimizer = Optimizer(value)
            elif key == "locked":
                cfg.locked = set(value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class MemoryBreakdown:
    """Per-component estimate, in bytes."""

    base_weights: float = 0.0
    adapter_weights: float = 0.0
    gradients: float = 0.0
    optimizer_state: float = 0.0
    activations: float = 0.0
    logits: float = 0.0
    attention_scores: float = 0.0
    runtime_overhead: float = 0.0
    offloaded_to_host: float = 0.0
    assumptions: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return (
            self.base_weights
            + self.adapter_weights
            + self.gradients
            + self.optimizer_state
            + self.activations
            + self.logits
            + self.attention_scores
            + self.runtime_overhead
        )

    @property
    def total_gb(self) -> float:
        return self.total / GB

    @property
    def components_gb(self) -> dict[str, float]:
        return {
            "Base weights": self.base_weights / GB,
            "Adapter weights": self.adapter_weights / GB,
            "Gradients": self.gradients / GB,
            "Optimizer state": self.optimizer_state / GB,
            "Activations": self.activations / GB,
            "Output logits": self.logits / GB,
            "Attention scores": self.attention_scores / GB,
            "Runtime overhead": self.runtime_overhead / GB,
        }

    @property
    def largest_component(self) -> tuple[str, float]:
        return max(self.components_gb.items(), key=lambda kv: kv[1])

    def to_dict(self) -> dict[str, Any]:
        name, size = self.largest_component
        return {
            "total_gb": round(self.total_gb, 2),
            "components_gb": {k: round(v, 3) for k, v in self.components_gb.items() if v > 0},
            "largest_component": {"name": name, "gb": round(size, 2)},
            "offloaded_to_host_gb": round(self.offloaded_to_host / GB, 2),
            "assumptions": self.assumptions,
        }


#: How far a component estimate can be out. Weight memory is arithmetic and
#: close to exact; activation memory depends on implementation details this
#: code cannot see, so it gets a much wider band.
_UNCERTAINTY = 0.30


@dataclass
class Estimate:
    breakdown: MemoryBreakdown
    config: TrainingConfig
    model: ModelSpec
    available_gb: Value[float]

    @property
    def total(self) -> Value[float]:
        low = self.breakdown.total_gb * (1 - _UNCERTAINTY * 0.5)
        high = self.breakdown.total_gb * (1 + _UNCERTAINTY)
        return Value.estimated(
            round(self.breakdown.total_gb, 2),
            "modelled from architecture and configuration, not measured",
            "GB",
            (round(low, 2), round(high, 2)),
        )

    @property
    def headroom_gb(self) -> Value[float]:
        if not self.available_gb.known:
            return Value.unavailable("memory capacity for this profile is unknown")
        return Value.estimated(
            round(self.available_gb.get() - self.breakdown.total_gb, 2),
            "capacity minus the estimate",
            "GB",
        )

    @property
    def verdict(self) -> dict[str, Any]:
        """A judgement in words, with the reasoning attached.

        Deliberately never a bare yes. The upper end of the band is what
        decides between 'fits' and 'tight', because a plan that only fits at
        the optimistic end will fail on a real allocator.
        """
        if not self.available_gb.known:
            return {
                "status": "unknown",
                "headline": "Cannot judge feasibility",
                "detail": "This profile does not state how much memory is available.",
            }
        available = self.available_gb.get()
        estimate = self.breakdown.total_gb
        high = estimate * (1 + _UNCERTAINTY)
        ratio = estimate / available if available else float("inf")

        if high <= available * 0.9:
            return {
                "status": "fits",
                "headline": "Should fit",
                "detail": (
                    f"Even at the pessimistic end of the estimate "
                    f"({high:.1f} GB) this stays inside {available:.1f} GB."
                ),
            }
        if estimate <= available * 0.92:
            return {
                "status": "tight",
                "headline": "Should fit, with little to spare",
                "detail": (
                    f"The estimate is {estimate:.1f} GB against {available:.1f} GB, "
                    "but the margin is inside the error of this model. A long "
                    "sequence in the data could still overflow it."
                ),
            }
        if ratio <= 1.35:
            return {
                "status": "unlikely",
                "headline": "Likely to run out of memory",
                "detail": (
                    f"The estimate is {estimate:.1f} GB against {available:.1f} GB. "
                    "Reducing sequence length or enabling checkpointing would "
                    "bring it into range."
                ),
            }
        return {
            "status": "infeasible",
            "headline": "Will not fit as configured",
            "detail": (
                f"The estimate is {estimate:.1f} GB against {available:.1f} GB — "
                "over budget by more than a third. This needs a different "
                "method or a smaller model, not a smaller batch."
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total.to_dict(),
            "available": self.available_gb.to_dict(),
            "headroom": self.headroom_gb.to_dict(),
            "verdict": self.verdict,
            "breakdown": self.breakdown.to_dict(),
            "config": self.config.to_dict(),
            "model": self.model.to_dict(),
        }


class MemoryEstimator:
    """Predicts peak training memory for a model and configuration."""

    #: Allocator fragmentation, kernel scratch and the framework's own context.
    #: Measured in the 0.6-1.2 GB range on CUDA; less on CPU.
    BASE_OVERHEAD_GB = 0.9
    CPU_OVERHEAD_GB = 0.4

    def estimate(
        self,
        model: ModelSpec,
        config: TrainingConfig,
        available_gb: Value[float] | None = None,
        on_cpu: bool = False,
    ) -> Estimate:
        b = MemoryBreakdown()
        cfg = config
        assumptions: list[str] = []

        seq = cfg.sequence_length
        batch = cfg.batch_size
        h = model.hidden_size
        layers = model.num_layers
        heads = model.num_attention_heads
        inter = model.intermediate_size

        # -- 1. base weights -------------------------------------------
        if cfg.method is Method.QLORA or cfg.quantization is not Quantization.NONE:
            per_param = cfg.quantization.bytes_per_param or cfg.precision.bytes_per_param
            b.base_weights = model.total_params * per_param
            if cfg.quantization.is_four_bit:
                assumptions.append(
                    "4-bit weights are costed at 0.58 bytes each, which includes "
                    "block-wise scales under double quantization."
                )
        else:
            b.base_weights = model.total_params * cfg.precision.bytes_per_param

        # Sharding splits weights across devices; activations stay per-device.
        devices = max(1, cfg.device_count)
        if devices > 1:
            b.base_weights /= devices
            assumptions.append(
                f"Weights are assumed to be sharded evenly across {devices} devices. "
                "Data-parallel training instead replicates them on every device."
            )

        if cfg.cpu_offload:
            # Offload moves a share of weights to host RAM. The exact share is
            # scheduler-dependent; half is the common default split.
            b.offloaded_to_host = b.base_weights * 0.5
            b.base_weights -= b.offloaded_to_host
            assumptions.append(
                "CPU offload is assumed to move half the weights to host RAM. "
                "This trades roughly 2-5x throughput for the memory saved."
            )

        # -- 2. trainable parameters ------------------------------------
        if cfg.method is Method.FULL:
            trainable = model.total_params
        elif cfg.method.is_lora_family:
            trainable = model.lora_trainable_params(
                cfg.lora_rank, cfg.target_modules or None, use_dora=cfg.method is Method.DORA
            )
        elif cfg.method is Method.IA3:
            trainable = layers * (2 * model.kv_dim + inter)
        elif cfg.method is Method.PREFIX_TUNING:
            trainable = layers * 2 * 32 * h  # 32 virtual tokens per layer, K and V
        elif cfg.method is Method.PROMPT_TUNING:
            trainable = 32 * h
        else:
            trainable = 0

        # Adapter weights are kept in fp32 when the base is quantized, because
        # 4-bit base weights give the update nothing stable to accumulate into.
        adapter_bytes = 4.0 if cfg.quantization is not Quantization.NONE else cfg.precision.bytes_per_param
        b.adapter_weights = trainable * adapter_bytes
        b.gradients = trainable * adapter_bytes
        b.optimizer_state = trainable * cfg.optimizer.bytes_per_trainable_param

        if cfg.method is Method.ADALORA:
            # AdaLoRA carries SVD factors and importance scores alongside.
            b.optimizer_state *= 1.5
            assumptions.append(
                "AdaLoRA's rank allocator keeps extra per-parameter statistics, "
                "costed here at a 50% uplift on optimizer state."
            )

        # -- 3. activations ---------------------------------------------
        act_bytes = 2.0 if cfg.precision in (Precision.FP16, Precision.BF16) else 4.0
        # Per layer, per token, per batch item: the tensors that must be kept
        # for the backward pass. Attention block ~10h, gated MLP ~4h + 6i.
        per_layer_per_token = act_bytes * (10 * h + 4 * h + 3 * inter)
        if not model.gated_mlp:
            per_layer_per_token = act_bytes * (10 * h + 4 * h + 2 * inter)

        if cfg.gradient_checkpointing:
            # Only layer boundaries are kept; one block is rebuilt at a time.
            b.activations = (
                act_bytes * seq * batch * h * layers + per_layer_per_token * seq * batch
            )
            assumptions.append(
                "Gradient checkpointing keeps only layer boundaries, recomputing "
                "the rest. It cuts activation memory sharply and costs roughly "
                "20-30% throughput."
            )
        else:
            b.activations = per_layer_per_token * seq * batch * layers

        # -- 4. attention score matrix ----------------------------------
        if not cfg.efficient_attention:
            # The materialised b×heads×seq×seq matrix. Quadratic in sequence
            # length, and the reason long-context runs fail without flash
            # attention far sooner than the weight maths suggests.
            b.attention_scores = act_bytes * batch * heads * seq * seq * layers
            if cfg.gradient_checkpointing:
                b.attention_scores /= layers  # only one block live at a time
            assumptions.append(
                "Attention scores are materialised because memory-efficient "
                "attention is off. This term grows with the square of sequence "
                "length and usually dominates past 4k tokens."
            )

        # -- 5. output logits --------------------------------------------
        # The logits tensor, plus the fp32 copy the cross-entropy loss makes.
        b.logits = batch * seq * model.vocab_size * (act_bytes + 4.0)
        if b.logits / GB > 1.0:
            assumptions.append(
                f"Output logits alone account for {b.logits / GB:.1f} GB at this "
                f"batch and sequence length, because the vocabulary is "
                f"{model.vocab_size:,} tokens."
            )

        # -- 6. runtime overhead ------------------------------------------
        b.runtime_overhead = (self.CPU_OVERHEAD_GB if on_cpu else self.BASE_OVERHEAD_GB) * GB

        if model.origin.value == "estimated":
            assumptions.append(
                "The model's shape was inferred from a size label rather than "
                "read from a config, so every figure here inherits that guess."
            )

        b.assumptions = assumptions
        return Estimate(
            breakdown=b,
            config=cfg,
            model=model,
            available_gb=available_gb or Value.unavailable("no capacity given"),
        )

    # -- secondary estimates ---------------------------------------------

    def estimate_adapter_size_mb(self, model: ModelSpec, config: TrainingConfig) -> float:
        """On-disk size of the saved adapter. Adapters ship in fp16."""
        trainable = model.lora_trainable_params(
            config.lora_rank, config.target_modules or None,
            use_dora=config.method is Method.DORA,
        )
        return trainable * 2 / 1024**2

    def estimate_disk_gb(
        self, model: ModelSpec, config: TrainingConfig, checkpoints: int = 3
    ) -> float:
        """Working disk: the base model plus retained checkpoints."""
        base_gb = model.total_params * config.precision.bytes_per_param / GB
        if config.quantization.is_four_bit:
            base_gb = model.total_params * 2 / GB  # downloaded in fp16, quantized in memory
        adapter_gb = self.estimate_adapter_size_mb(model, config) / 1024
        # A checkpoint stores the adapter plus its optimizer state.
        return base_gb + adapter_gb * checkpoints * 3

    def estimate_kv_cache_gb(
        self, model: ModelSpec, batch: int, seq: int, precision: Precision = Precision.FP16
    ) -> float:
        """KV cache for inference or generation-based evaluation."""
        return (
            2
            * batch
            * seq
            * model.num_layers
            * model.kv_dim
            * precision.bytes_per_param
            / GB
        )


estimator = MemoryEstimator()
