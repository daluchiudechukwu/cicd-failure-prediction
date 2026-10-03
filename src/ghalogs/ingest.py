"""Stage 1 — ingestion.

Converts the two gzipped JSON-lines files GHALogs publishes into columnar
Parquet, projecting away the ~90% of each record that is unusable for
pre-execution prediction (API URLs, avatar links, parsed log insights).

Why streaming rather than `pd.read_json`: `runs.json.gz` is 1.06 GB
compressed and expands to roughly 8 GB of JSON text, and each record is a
deeply nested object with 35 metadata keys, two full repository objects, and a
`log_insights` array that can hold thousands of parsed steps. Loading it whole
is both unnecessary and unreliable on a laptop. Records are therefore parsed
one line at a time, flattened to a fixed schema, and flushed to Parquet in
row-group batches so peak memory stays proportional to the chunk size.

Nothing in this stage filters or engineers. It is a faithful, flattened
projection, so that every later decision is auditable against a stable
intermediate.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ADMITTED_REPO_FIELDS, PipelineConfig

# Fields carried out of each run record. Everything else is dropped here, not
# later, which is what keeps the Parquet file small enough to iterate on.
#
# Post-execution fields are projected deliberately and prefixed `audit_`:
# they are needed to build the exclusion ledger, to measure run duration for
# the cost analysis, and to quantify leakage in the RQ1 comparison. The
# feature contract rejects any column with that prefix, so they cannot reach
# a model by accident.
RUN_SCHEMA = pa.schema(
    [
        # identity
        ("run_id", pa.string()),
        ("repo", pa.string()),
        ("workflow_path", pa.string()),
        ("workflow_id", pa.int64()),
        ("run_number", pa.int64()),
        ("run_attempt", pa.int32()),
        # label
        ("conclusion", pa.string()),
        # timing (run start is pre-execution; run end is not)
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("run_started_at", pa.timestamp("us", tz="UTC")),
        ("audit_updated_at", pa.timestamp("us", tz="UTC")),
        # unstructured trigger-time text
        ("commit_message", pa.string()),
        ("head_branch", pa.string()),
        ("head_sha", pa.string()),
        ("display_title", pa.string()),
        ("workflow_name", pa.string()),
        ("repo_description", pa.string()),
        # commit authorship (pseudonymised downstream)
        ("commit_author_name", pa.string()),
        ("commit_author_email", pa.string()),
        ("commit_timestamp", pa.timestamp("us", tz="UTC")),
        # trigger context
        ("event", pa.string()),
        ("actor_login", pa.string()),
        ("actor_type", pa.string()),
        ("triggering_actor_login", pa.string()),
        ("owner_type", pa.string()),
        ("base_repo_id", pa.int64()),
        ("head_repo_id", pa.int64()),
        ("head_repo_is_fork", pa.bool_()),
        ("n_pull_requests", pa.int32()),
        ("n_referenced_workflows", pa.int32()),
        ("has_previous_attempt", pa.bool_()),
        # post-execution, audit only
        ("audit_status", pa.string()),
        ("audit_total_logs_size", pa.int64()),
        ("audit_n_log_jobs", pa.int32()),
    ]
)

REPO_SCHEMA = pa.schema(
    [
        ("repo", pa.string()),
        ("selected", pa.bool_()),
        ("language", pa.string()),
        ("repo_created_at", pa.timestamp("us", tz="UTC")),
        ("default_branch", pa.string()),
        ("has_license", pa.bool_()),
        ("has_wiki", pa.bool_()),
        ("is_fork", pa.bool_()),
        ("n_topics", pa.int32()),
    ]
)


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    """Yield one decoded record per line without materialising the file."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp to a UTC-aware datetime.

    GHALogs mixes two formats: run timestamps end in `Z` while
    `repo.createdAt` is naive. Both are UTC, so naive values are localised
    rather than discarded.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def flatten_run(doc: dict) -> dict:
    """Project one nested run record onto the flat RUN_SCHEMA.

    Defensive throughout: `head_commit` is null for 133 runs, `actor` and
    `head_repository` can be absent, and `log_insights` may be missing for
    10.8% of runs. Missing values become None rather than raising, because
    exclusion decisions belong to the filtering stage where they can be
    counted.
    """
    meta = doc.get("metadata") or {}
    head_commit = meta.get("head_commit") or {}
    author = head_commit.get("author") or {}
    actor = meta.get("actor") or {}
    triggering_actor = meta.get("triggering_actor") or {}
    base_repo = meta.get("repository") or {}
    head_repo = meta.get("head_repository") or {}
    owner = base_repo.get("owner") or {}
    log_insights = doc.get("log_insights") or []

    return {
        "run_id": doc.get("_id"),
        "repo": doc.get("repository_name"),
        "workflow_path": doc.get("workflow_path"),
        "workflow_id": meta.get("workflow_id"),
        "run_number": doc.get("run_number"),
        "run_attempt": doc.get("run_attempt", 1),
        "conclusion": meta.get("conclusion"),
        "created_at": parse_timestamp(meta.get("created_at")),
        "run_started_at": parse_timestamp(meta.get("run_started_at")),
        "audit_updated_at": parse_timestamp(meta.get("updated_at")),
        "commit_message": head_commit.get("message"),
        "head_branch": meta.get("head_branch"),
        "head_sha": meta.get("head_sha"),
        "display_title": meta.get("display_title"),
        "workflow_name": meta.get("name"),
        "repo_description": base_repo.get("description"),
        "commit_author_name": author.get("name"),
        "commit_author_email": author.get("email"),
        "commit_timestamp": parse_timestamp(head_commit.get("timestamp")),
        "event": meta.get("event"),
        "actor_login": actor.get("login"),
        "actor_type": actor.get("type"),
        "triggering_actor_login": triggering_actor.get("login"),
        "owner_type": owner.get("type"),
        "base_repo_id": base_repo.get("id"),
        "head_repo_id": head_repo.get("id"),
        "head_repo_is_fork": head_repo.get("fork"),
        "n_pull_requests": len(meta.get("pull_requests") or []),
        "n_referenced_workflows": len(meta.get("referenced_workflows") or []),
        "has_previous_attempt": meta.get("previous_attempt_url") is not None,
        "audit_status": meta.get("status"),
        "audit_total_logs_size": doc.get("total_logs_size"),
        "audit_n_log_jobs": len(log_insights),
    }


def flatten_repository(doc: dict) -> dict:
    """Project one repository record, keeping only contract-admitted fields.

    Crawl-time counters (stargazers, forks, commits, openIssues,
    total_runs_90d, codeLines) are omitted at source. They are a single
    snapshot taken after most runs completed, so including them would attribute
    later information to an earlier run.
    """
    repo = doc.get("repo") or {}
    assert set(ADMITTED_REPO_FIELDS) >= {
        "mainLanguage",
        "createdAt",
        "defaultBranch",
        "license",
        "hasWiki",
        "isFork",
        "topics",
    }
    return {
        "repo": doc.get("_id"),
        "selected": bool(doc.get("selected")),
        "language": repo.get("mainLanguage"),
        "repo_created_at": parse_timestamp(repo.get("createdAt")),
        "default_branch": repo.get("defaultBranch"),
        "has_license": bool(repo.get("license")),
        "has_wiki": bool(repo.get("hasWiki")),
        "is_fork": bool(repo.get("isFork")),
        "n_topics": len(repo.get("topics") or []),
    }


def _write_streaming(
    records: Iterator[dict],
    schema: pa.Schema,
    destination: Path,
    chunk_rows: int,
    label: str,
) -> int:
    """Buffer `chunk_rows` records at a time and append them as a row group."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    total = 0

    def flush(rows: list[dict]) -> None:
        nonlocal writer
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(destination, schema, compression="zstd")
        writer.write_table(table)

    try:
        for record in records:
            buffer.append(record)
            if len(buffer) >= chunk_rows:
                flush(buffer)
                total += len(buffer)
                buffer = []
                print(f"  {label}: {total:,} rows", end="\r", flush=True)
        flush(buffer)
        total += len(buffer)
    finally:
        if writer is not None:
            writer.close()

    print(f"  {label}: {total:,} rows written to {destination}")
    return total


def ingest_repositories(source: Path, destination: Path, config: PipelineConfig) -> int:
    """Flatten repositories.json.gz to Parquet."""
    print("ingesting repositories ...")
    return _write_streaming(
        (flatten_repository(doc) for doc in read_jsonl_gz(source)),
        REPO_SCHEMA,
        destination,
        config.ingest_chunk_rows,
        "repositories",
    )


def ingest_runs(source: Path, destination: Path, config: PipelineConfig) -> int:
    """Flatten runs.json.gz to Parquet."""
    print("ingesting runs ...")
    return _write_streaming(
        (flatten_run(doc) for doc in read_jsonl_gz(source)),
        RUN_SCHEMA,
        destination,
        config.ingest_chunk_rows,
        "runs",
    )


def load_runs(path: Path) -> pd.DataFrame:
    """Read the ingested runs table.

    573,993 usable rows at this width fit comfortably in memory, so later
    stages work on a single frame. Should the population grow, the filtering
    and feature stages are written to be chunk-friendly.
    """
    return pq.read_table(path).to_pandas()
