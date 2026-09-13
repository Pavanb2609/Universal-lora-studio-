"""Model description and parameter accounting.

Two levels of fidelity, and the difference is always visible downstream:

* A **config-derived** spec, built from a real ``config.json``. Parameter counts
  are computed from the architecture and are close to exact.
* A **coarse** spec, built from nothing but "about 7B". Shapes are inferred
  from a scaling relationship, which is good enough to decide whether a run
  will fit and not good enough to trust to two decimal places.

Everything produced from a coarse spec inherits that uncertainty and says so.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..value import Origin, Value


@dataclass
class ModelSpec:
    name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int
    intermediate_size: int
    num_kv_heads: int | None = None
    max_position_embeddings: int = 2048
    architecture: str = "unknown"
    tie_word_embeddings: bool = False
    gated_mlp: bool = True
    #: How the numbers above were obtained.
    origin: Origin = Origin.DETECTED
    source: str = ""
    notes: list[str] = field(default_factory=list)

    # -- shapes ---------------------------------------------------------

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // max(1, self.num_attention_heads)

    @property
    def kv_dim(self) -> int:
        """Width of the K and V projections.

        Under grouped-query attention this is narrower than the hidden size,
        which materially cuts both weight and cache memory. Getting it wrong
        overstates requirements on most modern models.
        """
        return self.kv_heads * self.head_dim

    # -- parameter counts -----------------------------------------------

    @property
    def embedding_params(self) -> int:
        count = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            count *= 2  # separate output head
        return count

    @property
    def attention_params_per_layer(self) -> int:
        h = self.hidden_size
        return (h * h) + (h * self.kv_dim) * 2 + (h * h)  # q, k, v, o

    @property
    def mlp_params_per_layer(self) -> int:
        h, i = self.hidden_size, self.intermediate_size
        return 3 * h * i if self.gated_mlp else 2 * h * i

    @property
    def total_params(self) -> int:
        per_layer = self.attention_params_per_layer + self.mlp_params_per_layer
        norms = 2 * self.hidden_size * self.num_layers + self.hidden_size
        return self.embedding_params + per_layer * self.num_layers + norms

    @property
    def params_value(self) -> Value[int]:
        return Value(
            self.total_params,
            self.origin if self.origin is not Origin.DETECTED else Origin.DETECTED,
            self.source or "computed from architecture",
        )

    @property
    def size_label(self) -> str:
        p = self.total_params
        if p >= 1e12:
            return f"{p / 1e12:.1f}T"
        if p >= 1e9:
            return f"{p / 1e9:.1f}B"
        return f"{p / 1e6:.0f}M"

    # -- LoRA accounting -------------------------------------------------

    def lora_trainable_params(
        self,
        rank: int,
        target_modules: list[str] | None = None,
        use_dora: bool = False,
    ) -> int:
        """Trainable parameter count for a LoRA configuration.

        Computed from the real shapes of the targeted projections rather than
        a fraction-of-total rule of thumb, because the answer differs by a
        factor of three or more depending on whether the MLP is targeted.
        """
        targets = target_modules or DEFAULT_TARGETS
        h, i, kv = self.hidden_size, self.intermediate_size, self.kv_dim
        shapes: dict[str, tuple[int, int]] = {
            "q_proj": (h, h),
            "k_proj": (h, kv),
            "v_proj": (h, kv),
            "o_proj": (h, h),
            "gate_proj": (h, i),
            "up_proj": (h, i),
            "down_proj": (i, h),
            # common aliases in non-Llama architectures
            "query_key_value": (h, h + 2 * kv),
            "dense": (h, h),
            "fc_in": (h, i),
            "fc_out": (i, h),
        }
        total = 0
        for module in targets:
            shape = shapes.get(module)
            if shape is None:
                continue
            fan_in, fan_out = shape
            total += rank * (fan_in + fan_out)
            if use_dora:
                total += fan_out  # one magnitude scalar per output channel
        return total * self.num_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "origin": self.origin.value,
            "source": self.source,
            "size_label": self.size_label,
            "total_params": self.total_params,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.kv_heads,
            "grouped_query_attention": self.kv_heads < self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_word_embeddings,
            "notes": self.notes,
        }

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_config(cls, config: dict, name: str = "", source: str = "") -> "ModelSpec":
        """Build from a Hugging Face ``config.json``.

        Field names vary between architectures, so each value is looked up
        across the known aliases before falling back.
        """
        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if config.get(key) is not None:
                    return config[key]
            return default

        hidden = pick("hidden_size", "n_embd", "d_model", "hidden_dim")
        layers = pick("num_hidden_layers", "n_layer", "num_layers", "n_layers")
        heads = pick("num_attention_heads", "n_head", "num_heads")
        vocab = pick("vocab_size", default=32000)
        if not all((hidden, layers, heads)):
            raise ValueError(
                "config.json is missing hidden size, layer count or head count"
            )

        intermediate = pick("intermediate_size", "ffn_dim", "n_inner", "d_ff")
        arch_list = config.get("architectures") or []
        arch = (arch_list[0] if arch_list else config.get("model_type", "unknown"))
        gated = _is_gated(str(arch), config)
        if not intermediate:
            intermediate = int(hidden * (8 / 3)) if gated else hidden * 4

        notes = []
        if not config.get("intermediate_size"):
            notes.append(
                "The config did not state an MLP width; it was inferred from the "
                "hidden size, so parameter count is approximate."
            )

        return cls(
            name=name or config.get("_name_or_path", "model"),
            hidden_size=int(hidden),
            num_layers=int(layers),
            num_attention_heads=int(heads),
            num_kv_heads=pick("num_key_value_heads", "num_kv_heads"),
            vocab_size=int(vocab),
            intermediate_size=int(intermediate),
            max_position_embeddings=int(
                pick("max_position_embeddings", "n_positions", "max_seq_len", default=2048)
            ),
            architecture=str(arch),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            gated_mlp=gated,
            origin=Origin.DETECTED,
            source=source or "config.json",
            notes=notes,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ModelSpec":
        p = Path(path)
        config_path = p / "config.json" if p.is_dir() else p
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
        return cls.from_config(config, name=p.name, source=str(config_path))

    @classmethod
    def coarse(cls, params: float, name: str = "", vocab_size: int = 32000) -> "ModelSpec":
        """Infer plausible shapes from a parameter count alone.

        Used when a person types "13B" into the simulator with no model in
        hand. Transformer shapes are strongly constrained by total size --
        width scales roughly as the cube root of parameter count at a fixed
        aspect ratio -- so this lands close enough to answer "will it fit",
        and is marked ESTIMATED so nothing downstream treats it as fact.
        """
        params = float(params)
        # Solve the dominant term: P ≈ 12·L·h² with h ≈ 128·L (typical aspect ratio).
        layers = max(2, round((params / (12 * 128**2)) ** (1 / 3)))
        hidden = max(128, round(params / (12 * layers)) ** 0.5)
        hidden = int(round(hidden / 128) * 128) or 128
        heads = max(1, hidden // 128)
        spec = cls(
            name=name or f"{_format_params(params)} model",
            hidden_size=hidden,
            num_layers=int(layers),
            num_attention_heads=heads,
            num_kv_heads=max(1, heads // 4),
            vocab_size=vocab_size,
            intermediate_size=int(round(hidden * 8 / 3 / 128) * 128),
            architecture="generic decoder",
            origin=Origin.ESTIMATED,
            source=f"inferred from a target size of {_format_params(params)}",
            notes=[
                "No model config was supplied, so layer count and width were "
                "inferred from the target size. Real models of the same size "
                "vary, and memory figures will vary with them.",
            ],
        )
        return spec


DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR_TARGETS = DEFAULT_TARGETS + ["gate_proj", "up_proj", "down_proj"]


def _is_gated(architecture: str, config: dict) -> bool:
    """Gated (SwiGLU) MLPs have three matrices; classic MLPs have two.

    Detected from the activation function where stated, since that is the
    property that actually determines the parameter count.
    """
    act = str(config.get("hidden_act") or config.get("activation_function") or "").lower()
    if act:
        return "silu" in act or "swiglu" in act or "geglu" in act
    return bool(re.search(r"llama|mistral|qwen|gemma|phi3|olmo", architecture, re.I))


def _format_params(params: float) -> str:
    if params >= 1e9:
        return f"{params / 1e9:g}B"
    return f"{params / 1e6:g}M"


def parse_size(text: str) -> float | None:
    """Read '7B', '1.5b', '350M' or a raw count into a number of parameters."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([bmk]?)\s*", text, re.I)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    multiplier = {"b": 1e9, "m": 1e6, "k": 1e3, "": 1.0}[match.group(2).lower()]
    return value * multiplier


def infer_targets(spec: ModelSpec, aggressive: bool) -> list[str]:
    """Choose LoRA target modules.

    Targeting the MLP as well as attention roughly triples adapter capacity and
    tends to help on tasks that need new knowledge rather than new formatting;
    attention-only is cheaper and usually enough for style and instruction
    following.
    """
    if "query_key_value" in (spec.architecture or "").lower():
        return ["query_key_value", "dense"]
    return ALL_LINEAR_TARGETS if aggressive else DEFAULT_TARGETS


def sanity_check(spec: ModelSpec) -> list[str]:
    """Warnings a person should see before spending hours on a run."""
    warnings: list[str] = []
    if spec.hidden_size % max(1, spec.num_attention_heads):
        warnings.append(
            "Hidden size is not divisible by the head count; the config may be "
            "misread and estimates could be off."
        )
    if spec.vocab_size > 200_000:
        warnings.append(
            f"A {spec.vocab_size:,}-token vocabulary makes the output logits a "
            "large share of activation memory. Shorter sequences help more than "
            "usual on this model."
        )
    if math.isclose(spec.kv_heads, spec.num_attention_heads) and spec.num_layers > 40:
        warnings.append(
            "This model does not use grouped-query attention, so long sequences "
            "will cost considerably more memory than on a comparable modern model."
        )
    return warnings
