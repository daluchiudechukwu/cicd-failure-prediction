#!/usr/bin/env python3
"""Extract pre-execution features from GHALogs runs into a Parquet table.

Produces one row per usable run (conclusion in {success, failure}) with three
feature groups kept separate so that ablations are a column selection rather
than a re-run:

    static_*  pre-execution signals that need no execution history at all
              (branch name, trigger event, actor, time of day, repo context)
    text_*    raw unstructured strings, retained for later text modelling
    hist_*    causally ordered history features (previous outcome, streaks)

Every feature is justified in docs/02-feature-contract.md. Nothing derived
from logs, run duration, or crawl-time repository snapshots is included.

Usage:
    python scripts/extract_features.py --data-dir /path/to/ghalogs \
        --out data/features.parquet
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

USABLE_CONCLUSIONS = ("success", "failure")

# Branch-name intent markers. Deliberately a short, readable list: it doubles
# as the hand-written keyword baseline that learned text models must beat.
BRANCH_MARKERS = (
    "dependabot",
    "renovate",
    "feature",
    "fix",
    "hotfix",
    "bugfix",
    "release",
    "chore",
    "refactor",
    "test",
    "docs",
    "wip",
    "revert",
    "merge",
    "develop",
)

# Commit-message intent markers, following conventional-commit vocabulary.
MESSAGE_MARKERS = (
    "fix",
    "feat",
    "chore",
    "refactor",
    "test",
    "docs",
    "bump",
    "update",
    "revert",
    "merge",
    "wip",
    "hotfix",
    "breaking",
    "deprecate",
    "security",
)

DEFAULT_BRANCHES = ("main", "master", "trunk", "default")

VERSION_RE = re.compile(r"\bv?\d+\.\d+(\.\d+)?\b")
ISSUE_REF_RE = re.compile(r"#\d+")


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO timestamp to a UTC-aware datetime.

    GHALogs mixes formats: run timestamps carry a trailing Z while
    repo.createdAt is naive. Both are UTC, so naive values are localised
    rather than rejected.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_repository_context(path: Path) -> dict[str, dict]:
    """Collect only the slow-moving repository fields the contract admits.

    Crawl-time counters (stars, forks, commits, open issues, total_runs_90d)
    are deliberately excluded: they post-date most runs.
    """
    context: dict[str, dict] = {}
    for doc in read_jsonl_gz(path):
        if not doc.get("selected"):
            continue
        repo = doc.get("repo") or {}
        context[doc["_id"]] = {
            "language": repo.get("mainLanguage"),
            "created_at": repo.get("createdAt"),
            "default_branch": repo.get("defaultBranch"),
            "has_license": bool(repo.get("license")),
            "has_wiki": bool(repo.get("hasWiki")),
            "n_topics": len(repo.get("topics") or []),
        }
    return context


def build_row(doc: dict, repo_context: dict) -> dict | None:
    meta = doc.get("metadata") or {}
    conclusion = meta.get("conclusion")
    if conclusion not in USABLE_CONCLUSIONS:
        return None

    created = parse_ts(meta.get("created_at"))
    if created is None:
        return None

    branch = (meta.get("head_branch") or "").strip()
    branch_lower = branch.lower()
    message = ((meta.get("head_commit") or {}).get("message") or "").strip()
    message_lower = message.lower()
    subject = message.split("\n", 1)[0]
    actor = (meta.get("actor") or {}).get("login") or ""
    workflow_name = meta.get("name") or ""
    repo_name = doc["repository_name"]
    context = repo_context.get(repo_name, {})

    repo_created = parse_ts(context.get("created_at"))
    repo_age_days = (created - repo_created).days if repo_created else None

    row = {
        # identifiers and label
        "repo": repo_name,
        "workflow_path": doc["workflow_path"],
        "run_number": doc["run_number"],
        "run_attempt": doc.get("run_attempt", 1),
        "created_at": created,
        "failed": int(conclusion == "failure"),
        # raw text, retained for transformer work
        "text_commit_message": message,
        "text_branch": branch,
        "text_display_title": meta.get("display_title") or "",
        "text_workflow_name": workflow_name,
        "text_repo_description": (meta.get("repository") or {}).get("description") or "",
        # trigger context
        "static_event": meta.get("event") or "unknown",
        "static_hour": created.hour,
        "static_weekday": created.weekday(),
        "static_is_weekend": int(created.weekday() >= 5),
        "static_is_office_hours": int(9 <= created.hour < 18),
        # actor
        "static_actor_is_bot": int(actor.endswith("[bot]") or "bot" in actor.lower()),
        "static_actor_type_bot": int((meta.get("actor") or {}).get("type") == "Bot"),
        "static_owner_is_org": int(
            ((meta.get("repository") or {}).get("owner") or {}).get("type") == "Organization"
        ),
        # provenance of the change
        "static_from_fork": int(bool((meta.get("head_repository") or {}).get("fork"))),
        "static_cross_repo": int(
            (meta.get("head_repository") or {}).get("id")
            != (meta.get("repository") or {}).get("id")
        ),
        "static_has_pr": int(bool(meta.get("pull_requests"))),
        "static_n_referenced_workflows": len(meta.get("referenced_workflows") or []),
        "static_is_rerun": int(doc.get("run_attempt", 1) > 1),
        # branch-name shape
        "static_branch_len": len(branch),
        "static_branch_depth": branch.count("/"),
        "static_branch_is_default": int(
            branch_lower in DEFAULT_BRANCHES
            or branch == (context.get("default_branch") or "\0")
        ),
        "static_branch_has_digits": int(any(c.isdigit() for c in branch)),
        "static_branch_has_version": int(bool(VERSION_RE.search(branch_lower))),
        # commit-message shape
        "static_msg_len": len(message),
        "static_msg_subject_len": len(subject),
        "static_msg_n_lines": message.count("\n") + 1 if message else 0,
        "static_msg_has_body": int("\n\n" in message),
        "static_msg_is_merge": int(message_lower.startswith("merge")),
        "static_msg_has_issue_ref": int(bool(ISSUE_REF_RE.search(message))),
        "static_msg_has_version": int(bool(VERSION_RE.search(message_lower))),
        "static_msg_is_conventional": int(bool(re.match(r"^[a-z]+(\([^)]*\))?!?:", message_lower))),
        "static_msg_n_words": len(message.split()),
        # workflow identity
        "static_workflow_name_len": len(workflow_name),
        "static_workflow_path_depth": doc["workflow_path"].count("/"),
        # repository context (slow-moving fields only)
        "static_language": context.get("language") or "unknown",
        "static_repo_age_days": repo_age_days,
        "static_repo_has_license": int(bool(context.get("has_license"))),
        "static_repo_has_wiki": int(bool(context.get("has_wiki"))),
        "static_repo_n_topics": context.get("n_topics") or 0,
    }

    for marker in BRANCH_MARKERS:
        row[f"static_branch_kw_{marker}"] = int(marker in branch_lower)
    for marker in MESSAGE_MARKERS:
        row[f"static_msg_kw_{marker}"] = int(marker in message_lower)

    return row


def prior_rate(failures: pd.Series, counts: pd.Series) -> pd.Series:
    """Failure rate over prior runs, with -1 marking "no prior runs"."""
    counts = counts.to_numpy(dtype="float64")
    rates = np.divide(
        failures.to_numpy(dtype="float64"),
        counts,
        out=np.full(len(counts), -1.0),
        where=counts > 0,
    )
    return pd.Series(rates, index=failures.index)


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach causally ordered history features.

    All aggregates use only runs strictly earlier than the target run. An
    adjacent predecessor is one whose run_number is exactly one lower, so
    gaps left by the five-runs-per-workflow sampling are not mistaken for
    adjacency.
    """
    frame = frame.sort_values(["repo", "workflow_path", "run_number", "run_attempt"]).reset_index(
        drop=True
    )
    group = frame.groupby(["repo", "workflow_path"], sort=False)

    prev_failed = group["failed"].shift(1)
    prev_run_number = group["run_number"].shift(1)
    adjacent = (frame["run_number"] - prev_run_number) == 1

    frame["hist_has_prev"] = adjacent.fillna(False).astype(int)
    frame["hist_prev_failed"] = prev_failed.where(adjacent).fillna(-1).astype(int)

    # Expanding failure rate over earlier runs of the same workflow.
    # -1 encodes "no prior runs" rather than being imputed to a central value.
    frame["hist_wf_prior_runs"] = group.cumcount()
    cum_failures = group["failed"].cumsum() - frame["failed"]
    frame["hist_wf_prior_failure_rate"] = prior_rate(
        cum_failures, frame["hist_wf_prior_runs"]
    )

    # Same, pooled across all workflows in the repository.
    frame = frame.sort_values(["repo", "created_at"]).reset_index(drop=True)
    repo_group = frame.groupby("repo", sort=False)
    frame["hist_repo_prior_runs"] = repo_group.cumcount()
    repo_cum_failures = repo_group["failed"].cumsum() - frame["failed"]
    frame["hist_repo_prior_failure_rate"] = prior_rate(
        repo_cum_failures, frame["hist_repo_prior_runs"]
    )

    # Hours since the workflow's previous run.
    frame = frame.sort_values(["repo", "workflow_path", "run_number", "run_attempt"]).reset_index(
        drop=True
    )
    prev_created = frame.groupby(["repo", "workflow_path"], sort=False)["created_at"].shift(1)
    gap = (frame["created_at"] - prev_created).dt.total_seconds() / 3600.0
    frame["hist_hours_since_prev"] = gap.fillna(-1.0)

    # Regime label, used for stratified reporting rather than as a feature.
    regime = pd.Series("A_cold_start", index=frame.index)
    regime[frame["hist_prev_failed"] == 0] = "B_prev_success"
    regime[frame["hist_prev_failed"] == 1] = "C_prev_failure"
    frame["regime"] = regime

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/features.parquet"))
    args = parser.parse_args()

    print("loading repository context ...")
    repo_context = load_repository_context(args.data_dir / "repositories.json.gz")
    print(f"  {len(repo_context)} selected repositories")

    print("extracting run features ...")
    rows = []
    skipped: collections.Counter = collections.Counter()
    for doc in read_jsonl_gz(args.data_dir / "runs.json.gz"):
        row = build_row(doc, repo_context)
        if row is None:
            skipped[(doc.get("metadata") or {}).get("conclusion")] += 1
            continue
        rows.append(row)
    print(f"  kept {len(rows)} runs; excluded {dict(skipped)}")

    frame = pd.DataFrame(rows)
    frame = add_history_features(frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} with {len(frame)} rows and {len(frame.columns)} columns")
    print(f"failure rate: {100 * frame['failed'].mean():.2f}%")
    print(f"regimes:\n{frame['regime'].value_counts()}")


if __name__ == "__main__":
    main()
