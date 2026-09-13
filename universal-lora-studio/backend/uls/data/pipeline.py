"""Dataset analysis.

Reads a file, works out what shape it is in, and measures the things that
actually change a training plan: how many samples there are and how long they
run in tokens.

Token counts come from the model's own tokenizer when transformers is
installed. When it is not, they fall back to a characters-per-token ratio and
are stamped ESTIMATED, because a plan built on a guessed sequence length
should not look like one built on a measured distribution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..planning.resolver import DataProfile
from ..value import Report, Value

SUPPORTED_SUFFIXES = {
    ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt", ".md", ".parquet",
}

#: Column names that usually mean the same thing, grouped by role.
ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "instruction": ("instruction", "prompt", "question", "query", "input_text", "human"),
    "input": ("input", "context", "passage", "source", "document"),
    "output": ("output", "response", "answer", "completion", "target", "assistant", "label"),
    "text": ("text", "content", "body", "document_text"),
    "messages": ("messages", "conversation", "conversations", "turns", "dialogue"),
}


@dataclass
class ColumnProfile:
    name: str
    role: str | None
    non_empty: int
    mean_chars: float
    max_chars: int
    sample: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "non_empty": self.non_empty,
            "mean_chars": round(self.mean_chars, 1),
            "max_chars": self.max_chars,
            "sample": self.sample[:180],
        }


@dataclass
class DatasetAnalysis:
    path: str
    fmt: str
    sample_count: int
    columns: list[ColumnProfile] = field(default_factory=list)
    task: str = "unknown"
    task_reason: str = ""
    token_stats: dict[str, Value[Any]] = field(default_factory=dict)
    duplicates: int = 0
    empty_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    tokenizer_source: str = ""

    @property
    def profile(self) -> DataProfile:
        """The subset of this analysis the planner consumes."""
        return DataProfile(
            sample_count=self.sample_count,
            p95_tokens=self.token_stats.get("p95", Value.unavailable()).or_else(None),
            max_tokens=self.token_stats.get("max", Value.unavailable()).or_else(None),
            mean_tokens=self.token_stats.get("mean", Value.unavailable()).or_else(None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.fmt,
            "sample_count": self.sample_count,
            "columns": [c.to_dict() for c in self.columns],
            "task": self.task,
            "task_reason": self.task_reason,
            "tokens": {k: v.to_dict() for k, v in self.token_stats.items()},
            "tokenizer_source": self.tokenizer_source,
            "duplicates": self.duplicates,
            "empty_rows": self.empty_rows,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_rows(path: str | Path, limit: int | None = None) -> tuple[list[dict], str]:
    """Read a dataset into a list of dictionaries plus a format name."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".jsonl", ".ndjson"):
        return list(_read_jsonl(p, limit)), "jsonl"
    if suffix == ".json":
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # A wrapper object around the real list is a common export shape.
            for key in ("data", "rows", "examples", "train", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]
        rows = [r if isinstance(r, dict) else {"text": str(r)} for r in data]
        return rows[:limit] if limit else rows, "json"
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(p, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            rows = []
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                rows.append({k: v for k, v in row.items() if k})
        return rows, suffix.lstrip(".")
    if suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8", errors="replace")
        # Blank-line separated blocks are the near-universal convention for
        # plain-text corpora; a single block means the whole file is one sample.
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        rows = [{"text": c} for c in (chunks[:limit] if limit else chunks)]
        return rows, "text"
    if suffix == ".parquet":
        rows = _read_parquet(p, limit)
        return rows, "parquet"

    raise ValueError(
        f"{suffix or 'this file'} is not a format the studio reads. Supported: "
        + ", ".join(sorted(SUPPORTED_SUFFIXES))
    )


def _read_jsonl(path: Path, limit: int | None) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # counted later as a malformed row
            yield obj if isinstance(obj, dict) else {"text": str(obj)}


def _read_parquet(path: Path, limit: int | None) -> list[dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise ValueError(
            "Reading Parquet needs pyarrow, which is not installed."
        ) from exc
    table = pq.read_table(path)
    rows = table.to_pylist()
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def analyse(
    path: str | Path,
    tokenizer_name: str | None = None,
    sample_size: int = 2000,
) -> DatasetAnalysis:
    rows, fmt = load_rows(path)
    analysis = DatasetAnalysis(path=str(path), fmt=fmt, sample_count=len(rows))

    if not rows:
        analysis.warnings.append("The file parsed, but contained no rows.")
        return analysis

    analysis.columns = _profile_columns(rows, sample_size)
    analysis.task, analysis.task_reason = _infer_task(analysis.columns, rows[0])

    texts = [_row_text(r, analysis.columns) for r in rows]
    analysis.empty_rows = sum(1 for t in texts if not t.strip())
    if analysis.empty_rows:
        analysis.warnings.append(
            f"{analysis.empty_rows:,} rows are empty once the text fields are "
            "joined and will contribute nothing to training."
        )

    seen: set[str] = set()
    duplicates = 0
    for text in texts:
        digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
        seen.add(digest)
    analysis.duplicates = duplicates
    if duplicates:
        share = duplicates / len(texts) * 100
        analysis.warnings.append(
            f"{duplicates:,} rows ({share:.0f}%) are exact duplicates. Removing "
            "them usually helps; repeated samples get weighted more heavily."
        )

    sample = texts if len(texts) <= sample_size else random.Random(0).sample(texts, sample_size)
    stats, source = _token_stats(sample, tokenizer_name)
    analysis.token_stats = stats
    analysis.tokenizer_source = source

    return analysis


def _profile_columns(rows: list[dict], sample_size: int) -> list[ColumnProfile]:
    sample = rows[:sample_size]
    keys: list[str] = []
    for row in sample[:200]:
        for key in row:
            if key not in keys:
                keys.append(key)

    profiles: list[ColumnProfile] = []
    for key in keys:
        values = [_stringify(row.get(key)) for row in sample]
        non_empty = [v for v in values if v.strip()]
        profiles.append(
            ColumnProfile(
                name=key,
                role=_role_for(key),
                non_empty=len(non_empty),
                mean_chars=(sum(len(v) for v in non_empty) / len(non_empty)) if non_empty else 0.0,
                max_chars=max((len(v) for v in non_empty), default=0),
                sample=non_empty[0] if non_empty else "",
            )
        )
    return profiles


def _role_for(column: str) -> str | None:
    lowered = column.strip().lower()
    for role, aliases in ROLE_ALIASES.items():
        if lowered in aliases:
            return role
    return None


def _infer_task(columns: list[ColumnProfile], first_row: dict) -> tuple[str, str]:
    roles = {c.role for c in columns if c.role}

    if "messages" in roles:
        return (
            "chat",
            "A messages column is present, so this is treated as multi-turn chat "
            "data and will be rendered with the model's chat template.",
        )
    if "instruction" in roles and "output" in roles:
        extra = " with a separate context field" if "input" in roles else ""
        return (
            "instruction",
            f"Instruction and output columns are present{extra}, which is the "
            "standard supervised instruction-tuning shape.",
        )
    if "output" in roles and "text" in roles:
        return (
            "completion",
            "A text field and an output field suggest prompt/completion pairs.",
        )
    if roles == {"text"} or (len(columns) == 1 and columns[0].role == "text"):
        return (
            "language_modelling",
            "Only free text is present, so training will be plain next-token "
            "prediction over the whole document.",
        )
    if any(c.name.lower() in ("chosen", "rejected") for c in columns):
        return (
            "preference",
            "Chosen and rejected columns indicate preference data, which needs a "
            "preference trainer rather than plain supervised fine-tuning.",
        )
    return (
        "unknown",
        "No familiar column names were found. Map the columns manually before "
        "training so the right text is used.",
    )


def _row_text(row: dict, columns: list[ColumnProfile]) -> str:
    """Join a row into the text that will actually be tokenized."""
    by_role: dict[str, str] = {}
    for column in columns:
        if column.role:
            by_role.setdefault(column.role, _stringify(row.get(column.name)))

    if "messages" in by_role:
        raw = row.get(_column_named(columns, "messages"))
        if isinstance(raw, list):
            return "\n".join(
                f"{m.get('role', '')}: {m.get('content', '')}"
                for m in raw
                if isinstance(m, dict)
            )
    parts = [by_role.get(role, "") for role in ("instruction", "input", "text", "output")]
    joined = "\n".join(p for p in parts if p)
    if joined:
        return joined
    return "\n".join(_stringify(v) for v in row.values())


def _column_named(columns: list[ColumnProfile], role: str) -> str:
    for column in columns:
        if column.role == role:
            return column.name
    return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


#: Characters per token for English prose under a typical BPE vocabulary.
#: Code and non-Latin scripts sit well below this, which is why the fallback
#: is labelled as an estimate rather than presented as a count.
FALLBACK_CHARS_PER_TOKEN = 3.8


def _token_stats(
    texts: list[str], tokenizer_name: str | None
) -> tuple[dict[str, Value[Any]], str]:
    counts: list[int] | None = None
    source = ""

    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            counts = [len(tok(text, add_special_tokens=False)["input_ids"]) for text in texts]
            source = f"tokenized with {tokenizer_name}"
        except Exception:  # noqa: BLE001
            counts = None

    if counts is None:
        counts = [max(1, round(len(t) / FALLBACK_CHARS_PER_TOKEN)) for t in texts]
        source = (
            "estimated from character count; select a model to tokenize properly"
        )
        maker = lambda v, unit="tokens": Value.estimated(  # noqa: E731
            v, f"~{FALLBACK_CHARS_PER_TOKEN} characters per token assumed", unit
        )
    else:
        maker = lambda v, unit="tokens": Value.detected(v, source, unit)  # noqa: E731

    counts.sort()
    if not counts:
        return {}, source

    def percentile(p: float) -> int:
        index = min(len(counts) - 1, max(0, int(round(p * (len(counts) - 1)))))
        return counts[index]

    return (
        {
            "mean": maker(round(sum(counts) / len(counts), 1)),
            "median": maker(percentile(0.5)),
            "p95": maker(percentile(0.95)),
            "p99": maker(percentile(0.99)),
            "max": maker(counts[-1]),
            "min": maker(counts[0]),
            "total": maker(sum(counts)),
        },
        source,
    )


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------


def split(
    rows: list[dict], train: float = 0.9, validation: float = 0.05, seed: int = 42
) -> dict[str, list[dict]]:
    """Shuffle and split. A fixed seed keeps the split reproducible across runs,
    which matters because comparing two adapters trained on different splits
    tells you nothing."""
    if not 0 < train < 1:
        raise ValueError("The training share must be between 0 and 1.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train)
    val_end = train_end + int(n * validation)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def deduplicate(rows: list[dict], columns: list[ColumnProfile]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        digest = hashlib.sha1(_row_text(row, columns).strip().encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(row)
    return kept, len(rows) - len(kept)


def fingerprint(path: str | Path) -> str:
    """Content hash, used to version a dataset in the experiment record.

    Reproducibility needs the data to be identifiable, and a filename is not an
    identifier -- people edit files in place.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]
