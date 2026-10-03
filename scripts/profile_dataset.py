#!/usr/bin/env python3
"""Profile the GHALogs metadata files to establish the feasibility baseline.

Reads the two gzipped JSON-lines metadata files published with GHALogs
(Zenodo record 14796970) and reports the statistics that constrain the
experimental design: label distribution, field coverage, history
availability, and the strength of the previous-outcome baseline.

The logs archive (~142 GB) is NOT required.

Usage:
    python scripts/profile_dataset.py --data-dir /path/to/ghalogs
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from typing import Iterator

# Conclusions that GHALogs retains. Only these two carry an interpretable
# pass/fail signal; the remainder are excluded from the modelling population.
USABLE_CONCLUSIONS = ("success", "failure")

UNSTRUCTURED_FIELDS = {
    "head_commit.message": lambda m: (m.get("head_commit") or {}).get("message"),
    "head_branch": lambda m: m.get("head_branch"),
    "display_title": lambda m: m.get("display_title"),
    "workflow_name": lambda m: m.get("name"),
    "actor.login": lambda m: (m.get("actor") or {}).get("login"),
    "repository.description": lambda m: (m.get("repository") or {}).get("description"),
}

BRANCH_BUCKETS = (
    "dependabot",
    "renovate",
    "main",
    "master",
    "develop",
    "release",
    "hotfix",
    "feature",
    "fix",
    "chore",
)


def read_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def bucket_branch(branch: str) -> str:
    lowered = (branch or "").lower()
    for token in BRANCH_BUCKETS:
        if token in lowered:
            return token
    return "other"


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "n/a"


def profile_repositories(path: Path) -> None:
    total = selected = 0
    languages: collections.Counter = collections.Counter()
    for doc in read_jsonl_gz(path):
        total += 1
        if doc.get("selected"):
            selected += 1
            languages[(doc.get("repo") or {}).get("mainLanguage")] += 1

    print("\n== repositories.json.gz ==")
    print(f"documents: {total}")
    print(f"selected (met the >=30 runs/90d criterion): {selected}")
    print(f"top languages among selected: {languages.most_common(8)}")


def profile_runs(path: Path) -> None:
    conclusions: collections.Counter = collections.Counter()
    events: collections.Counter = collections.Counter()
    attempts: collections.Counter = collections.Counter()
    coverage: collections.Counter = collections.Counter()
    months: collections.Counter = collections.Counter()
    branch_outcomes = collections.defaultdict(list)
    event_outcomes = collections.defaultdict(list)
    repo_outcomes = collections.defaultdict(list)
    workflow_runs = collections.defaultdict(list)
    message_lengths: list[int] = []
    has_log_insights = 0
    total_runs = 0

    for doc in read_jsonl_gz(path):
        total_runs += 1
        meta = doc.get("metadata") or {}
        conclusion = meta.get("conclusion")
        conclusions[conclusion] += 1
        attempts[doc.get("run_attempt", 1)] += 1
        if doc.get("log_insights"):
            has_log_insights += 1

        for name, getter in UNSTRUCTURED_FIELDS.items():
            if getter(meta):
                coverage[name] += 1

        created = meta.get("created_at")
        if created:
            months[created[:7]] += 1

        if conclusion not in USABLE_CONCLUSIONS:
            continue

        failed = int(conclusion == "failure")
        repo = doc["repository_name"]
        events[meta.get("event")] += 1
        event_outcomes[meta.get("event")].append(failed)
        branch_outcomes[bucket_branch(meta.get("head_branch") or "")].append(failed)
        repo_outcomes[repo].append(failed)
        workflow_runs[(repo, doc["workflow_path"])].append(
            (doc["run_number"], doc.get("run_attempt", 1), failed)
        )
        message = (meta.get("head_commit") or {}).get("message")
        if message:
            message_lengths.append(len(message))

    usable = sum(conclusions[c] for c in USABLE_CONCLUSIONS)

    print("\n== runs.json.gz ==")
    print(f"run documents: {total_runs}")
    print(f"conclusion distribution: {conclusions.most_common()}")
    print(f"usable (success|failure): {usable} -> failure rate {pct(conclusions['failure'], usable)}")
    print(f"run_attempt distribution: {attempts.most_common(6)}")
    reruns = total_runs - attempts[1]
    print(f"re-run attempts (run_attempt > 1): {reruns} ({pct(reruns, total_runs)})")
    print(f"log_insights present: {has_log_insights} ({pct(has_log_insights, total_runs)})")
    print(f"runs by month: {sorted(months.items())}")

    print("\n-- unstructured field coverage (share of all run documents) --")
    for name in UNSTRUCTURED_FIELDS:
        print(f"  {name:28s} {coverage[name]:7d}  {pct(coverage[name], total_runs)}")

    if message_lengths:
        ordered = sorted(message_lengths)
        print(
            "\ncommit message length (chars): "
            f"mean {statistics.mean(ordered):.0f}, "
            f"median {statistics.median(ordered):.0f}, "
            f"p95 {ordered[int(0.95 * len(ordered))]}"
        )

    print("\n-- failure rate by trigger event (n >= 2000) --")
    for event, outcomes in sorted(event_outcomes.items(), key=lambda kv: -len(kv[1])):
        if len(outcomes) >= 2000:
            print(f"  {event:22s} n={len(outcomes):7d} failure={pct(sum(outcomes), len(outcomes))}")

    print("\n-- failure rate by branch-name bucket --")
    for name, outcomes in sorted(branch_outcomes.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name:12s} n={len(outcomes):7d} failure={pct(sum(outcomes), len(outcomes))}")

    profile_history(workflow_runs)
    profile_repository_heterogeneity(repo_outcomes)


def profile_history(workflow_runs: dict) -> None:
    """Partition runs by the availability and value of the preceding outcome.

    A predecessor only counts when its run_number is exactly one lower, so
    that gaps created by the five-runs-per-workflow sampling are not silently
    treated as adjacency.
    """
    cold = cold_failed = 0
    after_success = after_success_failed = 0
    after_failure = after_failure_failed = 0
    runs_per_workflow: collections.Counter = collections.Counter()

    for runs in workflow_runs.values():
        runs_per_workflow[len(runs)] += 1
        ordered = sorted(runs)
        for index, (run_number, _attempt, failed) in enumerate(ordered):
            if index == 0 or run_number - ordered[index - 1][0] > 1:
                cold += 1
                cold_failed += failed
            elif ordered[index - 1][2] == 0:
                after_success += 1
                after_success_failed += failed
            else:
                after_failure += 1
                after_failure_failed += failed

    total = cold + after_success + after_failure
    print("\n-- runs per workflow --")
    print(f"  distribution: {sorted(runs_per_workflow.items())}")
    counts = [n for n, c in runs_per_workflow.items() for _ in range(c)]
    print(f"  mean {statistics.mean(counts):.2f}, median {statistics.median(counts):.1f}")

    print("\n-- prediction regimes --")
    print(f"  A cold start (no adjacent predecessor) n={cold:7d} ({pct(cold, total)}) failure={pct(cold_failed, cold)}")
    print(
        f"  B previous run succeeded               n={after_success:7d} "
        f"({pct(after_success, total)}) failure={pct(after_success_failed, after_success)}"
    )
    print(
        f"  C previous run failed                  n={after_failure:7d} "
        f"({pct(after_failure, total)}) failure={pct(after_failure_failed, after_failure)}"
    )

    # Straw-man: copy the previous outcome. Only defined on regimes B and C.
    true_pos = after_failure_failed
    false_pos = after_failure - after_failure_failed
    false_neg = after_success_failed
    true_neg = after_success - after_success_failed
    decided = true_pos + false_pos + false_neg + true_neg
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    all_failures = cold_failed + after_success_failed + after_failure_failed

    print("\n-- straw-man baseline: predict the previous run's outcome (regimes B and C) --")
    print(f"  decided runs: {decided}")
    print(f"  accuracy {100 * (true_pos + true_neg) / decided:.2f}%")
    print(f"  failure-class precision {100 * precision:.2f}%, recall {100 * recall:.2f}%, F1 {100 * f1:.2f}%")
    print(
        f"  misses every one of the {after_success_failed} regime-B failures "
        f"({pct(after_success_failed, all_failures)} of all failures)"
    )


def profile_repository_heterogeneity(repo_outcomes: dict) -> None:
    rates = [sum(v) / len(v) for v in repo_outcomes.values() if len(v) >= 10]
    if not rates:
        return
    print("\n-- repository heterogeneity (repos with >= 10 usable runs) --")
    print(f"  repositories: {len(rates)}")
    print(
        f"  failure rate mean {100 * statistics.mean(rates):.1f}%, "
        f"median {100 * statistics.median(rates):.1f}%, "
        f"sd {100 * statistics.pstdev(rates):.1f}%"
    )
    never = sum(1 for r in rates if r == 0)
    mostly = sum(1 for r in rates if r > 0.5)
    print(f"  never failed: {pct(never, len(rates))}; failed more than half the time: {pct(mostly, len(rates))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory holding repositories.json.gz and runs.json.gz",
    )
    parser.add_argument("--skip-repositories", action="store_true")
    args = parser.parse_args()

    runs_path = args.data_dir / "runs.json.gz"
    if not runs_path.exists():
        raise SystemExit(f"missing {runs_path}")

    if not args.skip_repositories:
        repositories_path = args.data_dir / "repositories.json.gz"
        if repositories_path.exists():
            profile_repositories(repositories_path)

    profile_runs(runs_path)


if __name__ == "__main__":
    main()
