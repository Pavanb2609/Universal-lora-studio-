"""Tests for the invariants this system exists to hold.

These are not coverage tests. Each one pins down a promise that would be easy
to break silently: that nothing invents a reading, that a locked parameter is
never rewritten, and that the same job plans differently on different machines.

Run with ``python -m pytest`` if pytest is installed, or ``python tests.py``
for a plain-stdlib run with no dependencies.
"""

from __future__ import annotations

import unittest

from uls.estimation.memory import (
    Method,
    Optimizer,
    TrainingConfig,
    estimator,
)
from uls.estimation.model_spec import ModelSpec, parse_size
from uls.hardware.capability import Backend, Precision, Quantization
from uls.hardware.detection import engine
from uls.planning.goals import Budget, Goal
from uls.planning.recovery import advisor
from uls.planning.resolver import DataProfile, resolver
from uls.value import Origin, Value

LLAMA3_8B = dict(
    hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
    num_key_value_heads=8, intermediate_size=14336, vocab_size=128256,
    max_position_embeddings=8192, hidden_act="silu",
    architectures=["LlamaForCausalLM"],
)


def gpu(memory_gb, count=1, quant=True, backend="cuda", ram=64):
    return engine.from_spec({
        "name": f"{memory_gb} GB profile", "backend": backend,
        "device_memory_gb": memory_gb, "device_count": count,
        "system_ram_gb": ram, "cpu_cores": 16,
        "quantization": ["none", "int8", "nf4"] if quant else ["none"],
    })


class ValueProvenance(unittest.TestCase):
    def test_unavailable_never_yields_a_number(self):
        v = Value.unavailable("no sensor")
        self.assertFalse(v.known)
        self.assertEqual(v.display(), "N/A")
        with self.assertRaises(ValueError):
            v.get()

    def test_failed_probe_degrades_instead_of_raising(self):
        def boom():
            raise OSError("driver not loaded")

        v = Value.probe(boom)
        self.assertIs(v.origin, Origin.UNAVAILABLE)
        self.assertIn("driver not loaded", v.note)

    def test_estimates_are_never_labelled_as_measurements(self):
        v = Value.estimated(12.5, "modelled", "GB")
        self.assertFalse(v.factual)
        self.assertEqual(v.to_dict()["label"], "Estimated")

    def test_overriding_a_detected_value_strips_its_provenance(self):
        detected = Value.detected(24.0, "torch.cuda", "GB")
        overridden = detected.relabel(Origin.CONFIGURED, "entered by hand")
        self.assertEqual(overridden.or_else(None), 24.0)
        self.assertIs(overridden.origin, Origin.CONFIGURED)


class Detection(unittest.TestCase):
    def test_always_finds_something_to_run_on(self):
        """There is no machine the studio refuses to start on."""
        detection = engine.detect()
        self.assertTrue(detection.capability.devices)
        self.assertIsNotNone(detection.capability.backend)

    def test_simulated_hardware_is_never_marked_detected(self):
        cap = gpu(80)
        self.assertIs(cap.origin, Origin.CONFIGURED)
        for device in cap.devices:
            self.assertIs(device.total_memory_gb.origin, Origin.CONFIGURED)
            # A described machine has no live readings to give.
            self.assertFalse(device.free_memory_gb.known)
            self.assertFalse(device.utilization_pct.known)

    def test_partial_device_reporting_is_flagged_not_silently_summed(self):
        cap = gpu(24, count=2)
        cap.devices[1].total_memory_gb = Value.unavailable("driver refused")
        total = cap.total_device_memory_gb
        self.assertEqual(total.or_else(0), 24)
        self.assertIn("1 of 2", total.note)

    def test_single_device_planning_uses_the_smallest_device(self):
        cap = gpu(48, count=2)
        cap.devices[1].total_memory_gb = Value.configured(16.0, "", "GB")
        self.assertEqual(cap.training_memory_gb.or_else(0), 16.0)


class ModelAccounting(unittest.TestCase):
    def test_parameter_count_matches_the_real_model(self):
        spec = ModelSpec.from_config(LLAMA3_8B)
        self.assertAlmostEqual(spec.total_params / 1e9, 8.03, places=1)
        self.assertEqual(spec.size_label, "8.0B")

    def test_grouped_query_attention_narrows_the_kv_projections(self):
        spec = ModelSpec.from_config(LLAMA3_8B)
        self.assertEqual(spec.kv_dim, 1024)
        self.assertLess(spec.kv_dim, spec.hidden_size)

    def test_lora_parameter_count_matches_peft(self):
        """r=16 on attention projections of Llama-3-8B is 13,631,488 params."""
        spec = ModelSpec.from_config(LLAMA3_8B)
        self.assertEqual(spec.lora_trainable_params(16), 13_631_488)

    def test_targeting_the_mlp_costs_roughly_three_times_as_much(self):
        spec = ModelSpec.from_config(LLAMA3_8B)
        attn = spec.lora_trainable_params(16)
        everything = spec.lora_trainable_params(
            16, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.assertGreater(everything / attn, 2.5)

    def test_coarse_specs_are_marked_as_guesses(self):
        spec = ModelSpec.coarse(7e9)
        self.assertIs(spec.origin, Origin.ESTIMATED)
        self.assertTrue(spec.notes)

    def test_size_parsing(self):
        self.assertEqual(parse_size("7B"), 7e9)
        self.assertEqual(parse_size("350m"), 350e6)
        self.assertIsNone(parse_size("enormous"))


class MemoryEstimation(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec.from_config(LLAMA3_8B)

    def test_qlora_8b_lands_in_the_published_range(self):
        config = TrainingConfig(
            method=Method.QLORA, quantization=Quantization.NF4,
            precision=Precision.BF16, lora_rank=16, sequence_length=2048,
            batch_size=1, gradient_checkpointing=True, efficient_attention=True,
            optimizer=Optimizer.ADAMW_8BIT,
        )
        total = estimator.estimate(self.spec, config).breakdown.total_gb
        self.assertTrue(6.0 < total < 10.0, f"got {total:.1f} GB")

    def test_four_bit_weights_cost_about_a_quarter_of_bf16(self):
        base = TrainingConfig(quantization=Quantization.NONE, method=Method.LORA)
        quantized = TrainingConfig(quantization=Quantization.NF4, method=Method.QLORA)
        ratio = (
            estimator.estimate(self.spec, quantized).breakdown.base_weights
            / estimator.estimate(self.spec, base).breakdown.base_weights
        )
        self.assertTrue(0.25 < ratio < 0.32, f"ratio {ratio:.2f}")

    def test_checkpointing_cuts_activation_memory_sharply(self):
        on = TrainingConfig(gradient_checkpointing=True, sequence_length=2048)
        off = TrainingConfig(gradient_checkpointing=False, sequence_length=2048)
        self.assertLess(
            estimator.estimate(self.spec, on).breakdown.activations,
            estimator.estimate(self.spec, off).breakdown.activations / 5,
        )

    def test_attention_memory_is_quadratic_without_efficient_attention(self):
        short = TrainingConfig(sequence_length=1024, efficient_attention=False,
                               gradient_checkpointing=False)
        long = TrainingConfig(sequence_length=2048, efficient_attention=False,
                              gradient_checkpointing=False)
        ratio = (
            estimator.estimate(self.spec, long).breakdown.attention_scores
            / estimator.estimate(self.spec, short).breakdown.attention_scores
        )
        self.assertAlmostEqual(ratio, 4.0, places=1)

    def test_the_verdict_uses_the_pessimistic_end_of_the_band(self):
        """A plan that only fits if the estimate is generous is not 'fits'."""
        config = TrainingConfig(method=Method.QLORA, quantization=Quantization.NF4,
                                sequence_length=2048, batch_size=1)
        estimate = estimator.estimate(self.spec, config, Value.configured(8.5, "", "GB"))
        self.assertIn(estimate.verdict["status"], ("tight", "unlikely"))

    def test_feasibility_is_unknown_when_capacity_is_unknown(self):
        estimate = estimator.estimate(self.spec, TrainingConfig())
        self.assertEqual(estimate.verdict["status"], "unknown")


class PriorityChain(unittest.TestCase):
    """The promise that the resolver cannot silently overrule a person."""

    def setUp(self):
        self.spec = ModelSpec.from_config(LLAMA3_8B)

    def test_locked_parameters_survive_even_when_the_plan_fails(self):
        config = TrainingConfig(sequence_length=8192, lora_rank=64)
        config.lock("sequence_length", "lora_rank")
        plan = resolver.resolve(gpu(8), self.spec, user_config=config)
        self.assertEqual(plan.config.sequence_length, 8192)
        self.assertEqual(plan.config.lora_rank, 64)
        self.assertFalse(plan.feasible)
        # And it says why it could not help, naming what the person locked.
        self.assertTrue(any("locked" in b for b in plan.blockers))

    def test_a_budget_tighter_than_the_hardware_wins(self):
        plan = resolver.resolve(
            gpu(80), self.spec, budget=Budget(max_device_memory_gb=10)
        )
        self.assertEqual(plan.ceiling_source, "budget")
        self.assertLessEqual(plan.estimate.breakdown.total_gb, 10)

    def test_a_budget_looser_than_the_hardware_does_not_win(self):
        plan = resolver.resolve(
            gpu(12), self.spec, budget=Budget(max_device_memory_gb=200)
        )
        self.assertEqual(plan.ceiling_source, "hardware")

    def test_every_change_is_explained(self):
        plan = resolver.resolve(gpu(8), self.spec)
        changed = {d.field for d in plan.decisions}
        self.assertIn("method", changed)
        for decision in plan.decisions:
            self.assertTrue(decision.reason.strip(), f"{decision.field} has no reason")

    def test_dataset_length_drives_sequence_length(self):
        plan = resolver.resolve(
            gpu(24), self.spec,
            data=DataProfile(sample_count=5000, p95_tokens=3500, max_tokens=9000),
        )
        self.assertEqual(plan.config.sequence_length, 4096)
        self.assertTrue(any(d.authority == "dataset" for d in plan.decisions))

    def test_a_tiny_dataset_lowers_rank_and_raises_epochs(self):
        plan = resolver.resolve(
            gpu(24), self.spec, data=DataProfile(sample_count=80, p95_tokens=500)
        )
        self.assertLessEqual(plan.config.lora_rank, 8)
        self.assertGreater(plan.config.epochs, 3.0)


class HardwareAdaptation(unittest.TestCase):
    """The central claim: the same job plans differently on different machines."""

    def setUp(self):
        self.spec = ModelSpec.from_config(LLAMA3_8B)

    def test_plans_grow_monotonically_with_available_memory(self):
        batches = [
            resolver.resolve(gpu(size), self.spec).config.batch_size
            for size in (8, 12, 24, 48, 80)
        ]
        self.assertEqual(batches, sorted(batches))
        self.assertGreater(batches[-1], batches[0])

    def test_effective_batch_is_preserved_as_the_micro_batch_changes(self):
        """Fitting memory must not quietly change the optimisation problem."""
        small = resolver.resolve(gpu(12), self.spec).config
        large = resolver.resolve(gpu(80), self.spec).config
        self.assertEqual(small.effective_batch, large.effective_batch)

    def test_missing_quantization_support_changes_the_method(self):
        plan = resolver.resolve(gpu(24, quant=False), self.spec)
        self.assertIs(plan.config.method, Method.LORA)
        self.assertIs(plan.config.quantization, Quantization.NONE)
        self.assertTrue(any("4-bit" in w for w in plan.warnings))

    def test_cpu_only_produces_a_plan_rather_than_a_refusal(self):
        cap = engine.from_spec({
            "name": "CPU box", "backend": "cpu", "system_ram_gb": 32,
            "cpu_cores": 8, "quantization": ["none"], "precisions": ["fp32"],
        })
        small = ModelSpec.coarse(350e6, name="small")
        plan = resolver.resolve(cap, small, goal=Goal.MIN_DEVICE_MEMORY)
        self.assertIsNotNone(plan.config)
        self.assertIsNot(plan.config.optimizer, Optimizer.ADAMW_8BIT)
        self.assertTrue(any("CPU" in w or "accelerator" in w for w in plan.warnings))

    def test_goals_produce_materially_different_plans(self):
        cap = gpu(48)
        fast = resolver.resolve(cap, self.spec, goal=Goal.FASTEST).config
        quality = resolver.resolve(cap, self.spec, goal=Goal.MAX_QUALITY).config
        lean = resolver.resolve(cap, self.spec, goal=Goal.MIN_DEVICE_MEMORY).config
        self.assertNotEqual(
            (fast.gradient_checkpointing, fast.lora_rank, fast.sequence_length),
            (quality.gradient_checkpointing, quality.lora_rank, quality.sequence_length),
        )
        self.assertLessEqual(lean.lora_rank, quality.lora_rank)

    def test_minimum_footprint_goals_do_not_spend_spare_memory(self):
        lean_small = resolver.resolve(gpu(12), self.spec, goal=Goal.MIN_DEVICE_MEMORY).config
        lean_huge = resolver.resolve(gpu(80), self.spec, goal=Goal.MIN_DEVICE_MEMORY).config
        self.assertEqual(lean_small.batch_size, lean_huge.batch_size)


class Recovery(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec.from_config(LLAMA3_8B)
        self.failed = TrainingConfig(
            batch_size=4, sequence_length=4096, gradient_checkpointing=False,
            quantization=Quantization.NONE, method=Method.LORA, optimizer=Optimizer.ADAMW,
        )

    def test_real_numbers_are_read_from_the_runtime_error(self):
        diagnosis = advisor.diagnose(
            self.failed, self.spec, gpu(24),
            error_text=("CUDA out of memory. GPU 0 has a total capacity of 23.65 GiB "
                        "of which 1.20 GiB is free. Process has 22.10 GiB already allocated"),
            step=412,
        )
        self.assertEqual(diagnosis.observed_gb.or_else(0), 22.10)
        self.assertIs(diagnosis.observed_gb.origin, Origin.MEASURED)

    def test_nothing_is_applied_automatically(self):
        diagnosis = advisor.diagnose(self.failed, self.spec, gpu(24))
        self.assertTrue(diagnosis.to_dict()["requires_confirmation"])
        # The configuration handed in is untouched.
        self.assertEqual(self.failed.batch_size, 4)
        self.assertEqual(self.failed.sequence_length, 4096)

    def test_every_remedy_states_its_cost(self):
        diagnosis = advisor.diagnose(self.failed, self.spec, gpu(24))
        self.assertTrue(diagnosis.remedies)
        for remedy in diagnosis.remedies:
            self.assertTrue(remedy.cost.strip())
            self.assertTrue(remedy.rationale.strip())

    def test_the_combined_proposal_actually_fits(self):
        diagnosis = advisor.diagnose(self.failed, self.spec, gpu(24))
        self.assertIsNotNone(diagnosis.combined)
        self.assertLess(diagnosis.combined_estimate_gb, 24)

    def test_an_unparseable_error_does_not_invent_numbers(self):
        diagnosis = advisor.diagnose(self.failed, self.spec, gpu(24), error_text="Killed")
        self.assertFalse(diagnosis.observed_gb.known)


class NoFabrication(unittest.TestCase):
    """A sweep over the whole API surface looking for invented readings."""

    def test_no_reading_anywhere_claims_detection_without_a_source(self):
        payloads = [
            engine.detect().capability.to_dict(),
            gpu(24).to_dict(),
            resolver.resolve(gpu(24), ModelSpec.from_config(LLAMA3_8B)).to_dict(),
        ]
        offenders: list[str] = []

        def walk(node, path=""):
            if isinstance(node, dict):
                if "origin" in node and "display" in node:
                    if node["origin"] == "unavailable" and node["display"] != "N/A":
                        offenders.append(f"{path}: unavailable but displays a value")
                    if node["origin"] == "detected" and node.get("value") is None:
                        offenders.append(f"{path}: detected but has no value")
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{path}[{i}]")

        for payload in payloads:
            walk(payload)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
