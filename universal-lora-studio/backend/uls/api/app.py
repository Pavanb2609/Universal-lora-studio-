"""HTTP API.

The frontend never touches hardware directly. It asks this service, which asks
the abstraction layer, which asks a provider. That separation is what lets the
UI run on a laptop while training runs on a server somewhere else -- the same
endpoints answer either way, and the worker is just another client.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..data import pipeline
from ..estimation.memory import TrainingConfig, estimator
from ..estimation.model_spec import ModelSpec, parse_size
from ..hardware.capability import Capability
from ..hardware.detection import engine
from ..monitoring.monitor import monitor
from ..planning.goals import Budget, Goal
from ..planning.recovery import advisor
from ..planning.resolver import DataProfile, resolver
from .. import peft_compat

app = FastAPI(
    title="Universal LoRA Studio",
    description="Hardware-agnostic planning and training for PEFT adapters.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#: Saved hardware profiles, keyed by id. Swapped for the database table in a
#: deployment; kept in memory here so the studio runs with no setup.
PROFILES: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------


class HardwareSpec(BaseModel):
    name: str = "Custom profile"
    backend: str = "cuda"
    device_name: str | None = None
    device_memory_gb: float | None = None
    device_count: int = 1
    system_ram_gb: float | None = None
    cpu_cores: int | None = None
    cpu_name: str | None = None
    disk_free_gb: float | None = None
    precisions: list[str] | None = None
    quantization: list[str] | None = None
    supports_cpu_offload: bool = True
    supports_flash_attention: bool = False


class ModelRef(BaseModel):
    """A model identified either by a real config or by a size.

    Both are accepted because the two questions people bring here are
    different: "will my model fit" needs the config, "what could I train on a
    24 GB card" does not need a model at all.
    """

    path: str | None = Field(None, description="Directory or config.json on disk")
    config: dict[str, Any] | None = Field(None, description="A config.json as JSON")
    size: str | None = Field(None, description="A size label such as '7B'")
    name: str | None = None


class PlanRequest(BaseModel):
    model: ModelRef
    goal: str = "balanced"
    hardware_profile_id: str | None = None
    hardware: HardwareSpec | None = None
    budget: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    overrides: dict[str, Any] | None = Field(
        None, description="Parameters you set yourself. These are never changed."
    )


class RecoveryRequest(BaseModel):
    model: ModelRef
    config: dict[str, Any]
    error_text: str = ""
    step: int | None = None
    hardware_profile_id: str | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def resolve_model(ref: ModelRef) -> ModelSpec:
    if ref.config:
        return ModelSpec.from_config(ref.config, name=ref.name or "model", source="supplied config")
    if ref.path:
        try:
            return ModelSpec.from_path(ref.path)
        except FileNotFoundError:
            raise HTTPException(404, f"No config.json found at {ref.path}")
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(400, str(exc))
    if ref.size:
        params = parse_size(ref.size)
        if not params:
            raise HTTPException(400, f"'{ref.size}' is not a size this understands. Try '7B'.")
        return ModelSpec.coarse(params, name=ref.name or "")
    raise HTTPException(400, "Give a model path, a config, or a size.")


def resolve_capability(
    profile_id: str | None, spec: HardwareSpec | None
) -> Capability:
    if spec is not None:
        return engine.from_spec(spec.model_dump())
    if profile_id and profile_id != "current":
        stored = PROFILES.get(profile_id)
        if stored is None:
            raise HTTPException(404, f"No hardware profile with id {profile_id}")
        return engine.from_spec(stored)
    return engine.detect().capability


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------


@app.get("/api/hardware/current")
def current_hardware(refresh: bool = False):
    """The machine this backend is running on, as detected."""
    detection = engine.detect(refresh=refresh)
    return detection.to_dict()


@app.post("/api/hardware/refresh")
def refresh_hardware():
    return engine.refresh().to_dict()


@app.get("/api/hardware/profiles")
def list_profiles():
    detection = engine.detect()
    return {
        "current": detection.capability.to_dict(),
        "saved": [
            {"id": pid, **engine.from_spec(spec).to_dict()} for pid, spec in PROFILES.items()
        ],
    }


@app.post("/api/hardware/profiles")
def create_profile(spec: HardwareSpec):
    import uuid

    pid = uuid.uuid4().hex[:12]
    PROFILES[pid] = spec.model_dump()
    return {"id": pid, **engine.from_spec(PROFILES[pid]).to_dict()}


@app.delete("/api/hardware/profiles/{profile_id}")
def delete_profile(profile_id: str):
    if PROFILES.pop(profile_id, None) is None:
        raise HTTPException(404, "No such profile.")
    return {"deleted": profile_id}


@app.post("/api/hardware/override")
def override_hardware(spec: dict[str, Any]):
    """Edit the detected machine.

    The result is a configured profile, not a detected one, and every field the
    person changed says so. Detection is not modified.
    """
    return engine.override(spec).to_dict()


@app.post("/api/hardware/simulate")
def simulate_hardware(spec: HardwareSpec):
    """Describe hardware that may not exist, for planning."""
    capability = engine.from_spec(spec.model_dump())
    return capability.to_dict()


@app.get("/api/hardware/monitor")
def monitor_sample():
    return monitor.sample().to_dict()


@app.get("/api/hardware/monitor/stream")
async def monitor_stream(interval: float = 2.0):
    """Server-sent events carrying live telemetry."""

    async def events():
        while True:
            payload = json.dumps(monitor.sample().to_dict())
            yield f"data: {payload}\n\n"
            await asyncio.sleep(max(0.5, interval))

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


@app.get("/api/goals")
def list_goals():
    return [
        {"id": g.value, "label": g.label, "description": g.description} for g in Goal
    ]


@app.post("/api/plan")
def build_plan(request: PlanRequest):
    """Turn hardware + model + data + goal into a configuration.

    The response carries the plan and the reasoning behind every value in it,
    so nothing here has to be taken on trust.
    """
    model = resolve_model(request.model)
    capability = resolve_capability(request.hardware_profile_id, request.hardware)

    try:
        goal = Goal(request.goal)
    except ValueError:
        raise HTTPException(400, f"'{request.goal}' is not a goal. See /api/goals.")

    user_config = None
    if request.overrides:
        user_config = TrainingConfig.from_dict(request.overrides)
        # Anything explicitly supplied is locked: the resolver may not touch it.
        user_config.lock(*[k for k in request.overrides if k != "locked"])

    data = DataProfile(**(request.dataset or {})) if request.dataset else None
    plan = resolver.resolve(
        capability=capability,
        model=model,
        goal=goal,
        budget=Budget.from_dict(request.budget),
        data=data,
        user_config=user_config,
    )

    payload = plan.to_dict()
    payload["hardware"] = capability.to_dict()
    payload["methods"] = [s.to_dict() for s in peft_compat.evaluate(model, capability)]
    payload["adapter_size_mb"] = round(
        estimator.estimate_adapter_size_mb(model, plan.config), 1
    )
    payload["disk_estimate_gb"] = round(estimator.estimate_disk_gb(model, plan.config), 1)
    return payload


@app.post("/api/plan/compare")
def compare_hardware(request: PlanRequest, profiles: list[HardwareSpec] | None = None):
    """Plan the same job against several machines at once.

    This is the answer to "should I rent a bigger GPU" -- the same model and
    dataset, costed across every profile, side by side.
    """
    model = resolve_model(request.model)
    goal = Goal(request.goal)
    results = []
    candidates: list[tuple[str, Capability]] = [
        ("Current environment", engine.detect().capability)
    ]
    for spec in profiles or []:
        candidates.append((spec.name, engine.from_spec(spec.model_dump())))
    for pid, spec in PROFILES.items():
        candidates.append((spec.get("name", pid), engine.from_spec(spec)))

    for name, capability in candidates:
        plan = resolver.resolve(
            capability=capability,
            model=model,
            goal=goal,
            budget=Budget.from_dict(request.budget),
            data=DataProfile(**(request.dataset or {})) if request.dataset else None,
        )
        results.append(
            {
                "profile": name,
                "origin": capability.origin.value,
                "memory": capability.training_memory_gb.to_dict(),
                "verdict": plan.estimate.verdict,
                "estimate_gb": round(plan.estimate.breakdown.total_gb, 2),
                "config": plan.config.to_dict(),
                "feasible": plan.feasible,
            }
        )
    return {"model": model.to_dict(), "results": results}


@app.post("/api/estimate")
def estimate_memory(request: PlanRequest):
    """Cost one exact configuration, with no automatic adjustment."""
    model = resolve_model(request.model)
    capability = resolve_capability(request.hardware_profile_id, request.hardware)
    config = TrainingConfig.from_dict(request.overrides or {})
    estimate = estimator.estimate(
        model, config, capability.training_memory_gb,
        on_cpu=capability.backend.value == "cpu",
    )
    return estimate.to_dict()


@app.post("/api/recovery")
def suggest_recovery(request: RecoveryRequest):
    """Diagnose a real out-of-memory failure.

    Returns proposals only. Nothing is applied without a further explicit
    request from the person.
    """
    model = resolve_model(request.model)
    capability = resolve_capability(request.hardware_profile_id, None)
    diagnosis = advisor.diagnose(
        config=TrainingConfig.from_dict(request.config),
        model=model,
        capability=capability,
        error_text=request.error_text,
        step=request.step,
    )
    return diagnosis.to_dict()


@app.post("/api/methods")
def list_methods(request: PlanRequest):
    model = resolve_model(request.model)
    capability = resolve_capability(request.hardware_profile_id, request.hardware)
    return [s.to_dict() for s in peft_compat.evaluate(model, capability)]


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


@app.post("/api/datasets/analyse")
def analyse_dataset(payload: dict[str, Any]):
    path = payload.get("path")
    if not path:
        raise HTTPException(400, "Give a path to a dataset file.")
    if not Path(path).exists():
        raise HTTPException(404, f"No file at {path}")
    try:
        analysis = pipeline.analyse(path, tokenizer_name=payload.get("tokenizer"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result = analysis.to_dict()
    result["fingerprint"] = pipeline.fingerprint(path)
    return result


@app.get("/api/environment")
def environment():
    """Everything needed to reproduce a run on another machine."""
    detection = engine.detect()
    from ..db.models import environment_fingerprint

    hardware = detection.capability.to_dict()
    software = detection.software.to_dict()
    return {
        "hardware": hardware,
        "software": software,
        "fingerprint": environment_fingerprint(hardware, software),
    }


@app.get("/api/health")
def health():
    detection = engine.detect()
    return {
        "status": "ok",
        "backend": detection.capability.backend.value,
        "accelerator": detection.capability.has_accelerator,
        "provider": detection.provider_name,
    }


# ---------------------------------------------------------------------------
# static frontend
# ---------------------------------------------------------------------------

FRONTEND = Path(__file__).resolve().parents[3] / "frontend" / "index.html"


@app.get("/")
def index():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    raise HTTPException(404, "The frontend is not built.")
