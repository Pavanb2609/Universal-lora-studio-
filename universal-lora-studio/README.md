# Universal LoRA Studio

Plans, sizes and configures LoRA/QLoRA training runs for any model on whatever
hardware you actually have — or on hardware you are only thinking about buying.

```sh
sh run.sh         # then open http://127.0.0.1:8000
```

Or without the script:

```sh
cd backend
pip install -r requirements.txt
python3 -m uvicorn uls.api.app:app --port 8000
```

No GPU required to start. No PyTorch required to start. The studio detects what
is present, says plainly what is not, and plans accordingly.

---

## The rule everything else follows

Every number the studio shows carries its origin:

| Marker | Meaning |
|---|---|
| **Detected** | Read from this machine, right now |
| **Configured** | Entered by you, or loaded from a saved profile |
| **Estimated** | Computed by a model. May not match reality |
| **Measured** | Sampled from a run in progress |
| **Not available** | Could not be determined here |

This is enforced in the type system, not by convention. `uls/value.py` defines a
`Value` that cannot be constructed without an origin and has no accessor that
quietly returns `None` — callers must handle absence. A failed hardware probe
becomes `Not available` with the reason attached; it never becomes a plausible
default. There is a test that walks every API response looking for a reading
that claims detection without a source.

The reason for the strictness: a training planner that guesses convincingly is
worse than one that admits ignorance, because you find out which it was three
hours into a run.

---

## What it does

**Detects hardware** across NVIDIA, AMD, Intel, Apple Silicon and CPU-only
machines. Each vendor lives behind a provider interface; nothing above the
abstraction layer imports a vendor SDK or branches on a product name. Providers
are ranked and the first that reports available wins, with the CPU provider as
a floor that always reports available — there is no machine this refuses to run
on.

**Estimates memory** component by component: base weights, adapter, gradients,
optimizer state, activations, output logits, attention scores, runtime
overhead. Grouped-query attention, gated MLPs, double-quantized 4-bit storage
and the fp32 loss copy of the logits are all accounted for. The breakdown
matters more than the total — it tells you whether to cut sequence length or
quantize, which are very different decisions.

Calibration, against Llama-3-8B: the parameter count comes out at 8.03B, LoRA
r=16 on attention projections at 13,631,488 trainable parameters (matching what
PEFT reports), and QLoRA at seq 2048 / batch 1 at roughly 7.7 GB, inside the
range people report in practice.

**Plans a run** from hardware + model + dataset + goal + budget, and scales in
both directions. The same 8B model and dataset:

| Memory | Plan produced |
|---|---|
| 8 GB | QLoRA, rank 8, seq 1024, batch 1 × 16 accumulation |
| 12 GB | QLoRA, rank 16, seq 2048, batch 2 × 8 |
| 24 GB | QLoRA, rank 16, seq 2048, batch 4 × 4 |
| 80 GB | QLoRA, rank 16, seq 2048, batch 16 × 1 |

Effective batch size stays constant at 16 throughout. Fitting into memory is
allowed to change how the work is divided; it is not allowed to quietly change
the optimisation problem you are solving.

**Explains itself.** Every value in a plan appears in a decision trail with the
reason and the rung of the priority chain that produced it:

```
1. your explicit settings    (never touched)
2. your resource budget
3. hardware capability
4. model requirements
5. dataset characteristics
6. automatic recommendations
```

This is structural, not a promise. Anything you set lands in `config.locked`,
and every write in the resolver goes through one method that refuses to touch a
locked field. The resolver has no code path that can silently override you.

**Recovers from OOM** by reading the real allocation out of the runtime's error
message — a measurement, which outranks anything the planner predicted — and
proposing named changes with their costs stated. It applies nothing. A silent
retry produces an adapter trained under conditions you never agreed to and
cannot reconstruct afterwards. It will also tell you when its own estimate was
wrong, in either direction.

**Analyses datasets** in JSON, JSONL, CSV, TSV, text, Markdown and Parquet:
schema and role detection, task inference, duplicate detection, and a token
length distribution. Sequence length is set from the 95th percentile rather
than the maximum — sizing for the longest outlier wastes memory on every step,
and sizing for the mean silently truncates a quarter of your corpus.

---

## Layout

```
backend/uls/
  value.py              provenance primitive — start here
  hardware/
    capability.py       what a machine can do, never what it is called
    provider.py         the abstraction layer boundary
    providers/          nvidia · amd + intel · apple · cpu + custom
    detection.py        provider selection, host merge, caching
  estimation/
    model_spec.py       architecture-derived parameter counting
    memory.py           the memory model
  planning/
    goals.py            performance profiles and resource budgets
    resolver.py         the priority chain
    recovery.py         OOM diagnosis
  data/pipeline.py      dataset analysis
  monitoring/monitor.py live telemetry
  db/models.py          entities, including the environment snapshot
  api/app.py            HTTP surface
  peft_compat.py        method availability with reasons
frontend/index.html     zero-build UI
```

`python tests.py` runs 38 tests covering the invariants: provenance integrity,
the priority chain, hardware adaptation, estimator calibration and OOM
handling.

---

## Notable design choices

**Apple Silicon is not a GPU with unified memory bolted on.** The provider
reports the Metal working-set limit rather than total RAM, declines to offer
CPU offload (there is nowhere to offload to), and states that bitsandbytes has
no Metal backend so QLoRA is unavailable. An 8B model in fp16 genuinely does
not fit in an 18 GB working set, and the studio says so rather than producing a
plan that fails later.

**The estimator publishes an uncertainty band** and the feasibility verdict uses
its pessimistic end. A plan that only fits if the estimate is generous is
reported as tight, not as fitting.

**The environment snapshot excludes free memory and utilization** from its
fingerprint. Those move second to second, and including them would make every
run look like it happened on a new machine.

**Throughput is not reported until two steps have been observed.** A
tokens-per-second figure derived from one data point, or from a theoretical
peak, is the kind of confident fiction the whole design is built to avoid. The
first step is also the slowest, since it pays for warm-up.

---

## What is here and what is not

Built and working end to end: hardware detection and abstraction, capability
modelling, hardware profiles, manual override, the simulator, memory
estimation, the configuration resolver, performance profiles, resource budgets,
OOM recovery, PEFT compatibility, dataset analysis, monitoring, the database
schema, the API and the UI.

Scaffolded but not executing: the training worker itself. `db/models.py` has
the job, checkpoint and adapter tables and the API has the shape for it, but
nothing calls into PEFT to run a step yet — that needs a machine with torch and
a GPU to develop against, which this was not built on. The planner it would
consume is finished.

Not started: the adapter lab (composition and fusion), evaluation runners, and
the deployment backends. Each has a table and an API shape waiting.

The frontend is a single HTML file with no build step, rather than the React and
TypeScript stack in the brief. That was a deliberate trade for something that
runs from a clone with no toolchain; the API is a plain JSON surface, so porting
the UI later touches nothing behind it.

Memory estimates are estimates. They are calibrated against published figures
for models I could check, and they carry a band for a reason. Compare them
against the measured peak the monitor records, and trust the measurement.
