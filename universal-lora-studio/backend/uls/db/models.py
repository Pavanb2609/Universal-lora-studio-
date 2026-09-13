"""Persistence.

SQLite by default, because a studio should run from a clone with no services
to start. Everything goes through SQLAlchemy's ORM and no SQLite-specific SQL
is used anywhere, so moving to Postgres is a URL change.

The shape worth noting is ``SystemEnvironment``: every experiment points at a
frozen snapshot of the hardware and library versions it ran under. The same
configuration produces different results on different machines, so an
experiment record without its environment is not reproducible, and this schema
makes storing one non-optional.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), default="local")
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    domain: Mapped[str | None] = mapped_column(String(80), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetVersion(Base, TimestampMixin):
    """An immutable snapshot of a dataset.

    Versioned by content hash rather than by a label, so an experiment can
    state exactly which bytes it trained on even after the file on disk moves
    or changes.
    """

    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(Text)
    format: Mapped[str] = mapped_column(String(20))
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    task: Mapped[str] = mapped_column(String(40), default="unknown")
    analysis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dataset: Mapped[Dataset] = relationship(back_populates="versions")


class Model(Base, TimestampMixin):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(40), default="local")  # local | hub
    path_or_repo: Mapped[str] = mapped_column(Text)
    revision: Mapped[str | None] = mapped_column(String(80), nullable=True)
    architecture: Mapped[str | None] = mapped_column(String(120), nullable=True)
    parameter_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class HardwareProfile(Base, TimestampMixin):
    """A named machine description.

    ``origin`` distinguishes a profile captured from a real machine from one a
    person typed in. That distinction is carried on every field inside
    ``capability`` too, and the UI renders it per field -- a profile is allowed
    to be partly measured and partly invented.
    """

    __tablename__ = "hardware_profiles"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    origin: Mapped[str] = mapped_column(String(20), default="configured")
    is_simulation: Mapped[bool] = mapped_column(Boolean, default=False)
    backend: Mapped[str] = mapped_column(String(20), default="cpu")
    capability: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class SystemEnvironment(Base, TimestampMixin):
    """Frozen software and hardware state for one run."""

    __tablename__ = "system_environments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    os: Mapped[str | None] = mapped_column(String(200), nullable=True)
    python_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    accelerator_runtime: Mapped[str | None] = mapped_column(String(80), nullable=True)
    libraries: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    hardware: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    goal: Mapped[str] = mapped_column(String(40), default="balanced")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), nullable=True)
    dataset_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True
    )
    hardware_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("hardware_profiles.id"), nullable=True
    )
    environment_id: Mapped[str | None] = mapped_column(
        ForeignKey("system_environments.id"), nullable=True
    )
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    jobs: Mapped[list["TrainingJob"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class TrainingJob(Base, TimestampMixin):
    __tablename__ = "training_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    worker: Mapped[str] = mapped_column(String(40), default="local")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Populated only when a run fails on memory; holds the proposal the person
    #: has not yet accepted.
    recovery: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    resource_samples: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    experiment: Mapped[Experiment] = relationship(back_populates="jobs")
    checkpoints: Mapped[list["Checkpoint"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Checkpoint(Base, TimestampMixin):
    __tablename__ = "checkpoints"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("training_jobs.id"))
    step: Mapped[int] = mapped_column(Integer)
    path: Mapped[str] = mapped_column(Text)
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    job: Mapped[TrainingJob] = relationship(back_populates="checkpoints")


class Adapter(Base, TimestampMixin):
    """A trained adapter.

    Deliberately not linked to the hardware it needs, only to the hardware it
    happened to be trained on. An adapter is a set of weights; it runs
    anywhere its base model runs.
    """

    __tablename__ = "adapters"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(40), default="1")
    path: Mapped[str] = mapped_column(Text)
    base_model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    method: Mapped[str] = mapped_column(String(30), default="lora")
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_modules: Mapped[list[str]] = mapped_column(JSON, default=list)
    size_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    trained_on: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Evaluation(Base, TimestampMixin):
    __tablename__ = "evaluations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(40), default="metric")  # metric | ab | manual
    adapter_id: Mapped[str | None] = mapped_column(ForeignKey("adapters.id"), nullable=True)
    compared_to: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dataset_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Deployment(Base, TimestampMixin):
    __tablename__ = "deployments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    backend: Mapped[str] = mapped_column(String(40))
    adapter_id: Mapped[str | None] = mapped_column(ForeignKey("adapters.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="stopped")
    endpoint: Mapped[str | None] = mapped_column(String(300), nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# ---------------------------------------------------------------------------
# session management
# ---------------------------------------------------------------------------

DEFAULT_URL = os.environ.get(
    "ULS_DATABASE_URL", f"sqlite:///{os.path.expanduser('~/.uls/studio.db')}"
)

_engine = None
_Session = None


def init(url: str = DEFAULT_URL, echo: bool = False):
    global _engine, _Session
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _engine = create_engine(url, echo=echo, future=True)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def session():
    if _Session is None:
        init()
    return _Session()  # type: ignore[misc]


def environment_fingerprint(hardware: dict, software: dict) -> str:
    """Stable hash over the things that change results.

    Free memory and utilization are excluded on purpose: they move second to
    second and would make every run look like it happened somewhere new.
    """
    import hashlib

    material = {
        "backend": hardware.get("backend"),
        "devices": [
            {"name": d.get("name", {}).get("value"), "memory": d.get("total_memory_gb", {}).get("value")}
            for d in hardware.get("devices", [])
        ],
        "libraries": {
            k: v.get("value") for k, v in (software.get("items") or {}).items()
        },
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
