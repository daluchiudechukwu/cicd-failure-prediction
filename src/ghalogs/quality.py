"""Stage 2 — noise filtering and data-quality auditing.

Every exclusion is counted into a ledger rather than applied silently. The
ledger is the exclusion table for the methodology chapter: a reader can see
exactly how 580,641 ingested runs become the modelling population, and a
reviewer can check that nothing convenient was quietly dropped.

The filters implemented here address problems measured in the dataset, not
hypothetical ones:

- 6,648 runs carry a conclusion that is not an execution outcome
- 133 runs have a null head_commit, 19 an empty head_branch
- 14 runs have updated_at earlier than created_at
- 15,385 runs (2.65%) are re-run attempts whose labels are partly
  non-deterministic because of flakiness
- 69.9% of runs share a head_sha with another run, because one commit
  triggers several workflows; 47.7% of all failures share a commit with a
  success, which sets a hard ceiling on commit-text-only models

The last point is a property of the task rather than a defect, so it is
measured and annotated rather than filtered away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DROPPED_CONCLUSIONS, FAILURE_LABEL, FilterConfig, USABLE_CONCLUSIONS


@dataclass
class FilterLedger:
    """Running record of how many rows each rule removed, and why."""

    entries: list[dict] = field(default_factory=list)
    _running_total: int | None = None

    def start(self, n: int) -> None:
        self._running_total = n
        self.entries.append({"step": "ingested", "removed": 0, "remaining": n, "reason": ""})

    def record(self, step: str, removed: int, remaining: int, reason: str) -> None:
        self.entries.append(
            {"step": step, "removed": removed, "remaining": remaining, "reason": reason}
        )
        self._running_total = remaining

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.entries)
        start = frame.loc[0, "remaining"]
        frame["pct_of_ingested"] = (100 * frame["removed"] / start).round(4)
        return frame

    def report(self) -> str:
        frame = self.to_frame()
        lines = ["exclusion ledger:"]
        for row in frame.itertuples():
            if row.step == "ingested":
                lines.append(f"  {row.step:28s} {row.remaining:>8,}")
            else:
                lines.append(
                    f"  {row.step:28s} -{row.removed:>7,} -> {row.remaining:>8,}"
                    f"  ({row.pct_of_ingested:.3f}%)  {row.reason}"
                )
        return "\n".join(lines)


def _apply(
    frame: pd.DataFrame, keep: pd.Series, ledger: FilterLedger, step: str, reason: str
) -> pd.DataFrame:
    """Apply a boolean keep-mask and log the effect."""
    keep = keep.fillna(False).astype(bool)
    removed = int((~keep).sum())
    filtered = frame[keep].reset_index(drop=True)
    ledger.record(step, removed, len(filtered), reason)
    return filtered


def filter_runs(
    runs: pd.DataFrame,
    repositories: pd.DataFrame,
    config: FilterConfig,
) -> tuple[pd.DataFrame, FilterLedger]:
    """Reduce the ingested runs to the modelling population.

    Order matters: structural integrity first, then label validity, then
    population scoping, then optional robustness filters. Each stage therefore
    reports against a population that the previous stage has already made
    coherent.
    """
    ledger = FilterLedger()
    ledger.start(len(runs))

    # ---- structural integrity -------------------------------------------
    # A duplicate run_id would double-count a run and could place the same
    # observation on both sides of a split. Measured count is zero, but the
    # check is cheap and protects against a future dataset revision.
    before = len(runs)
    runs = runs.drop_duplicates(subset="run_id", keep="first").reset_index(drop=True)
    ledger.record(
        "duplicate run_id", before - len(runs), len(runs), "same run ingested more than once"
    )

    runs = _apply(
        runs,
        runs["repo"].notna() & runs["workflow_path"].notna() & runs["created_at"].notna(),
        ledger,
        "missing identity/timestamp",
        "cannot be ordered or grouped without repo, workflow and created_at",
    )

    if config.drop_inconsistent_timestamps:
        consistent = runs["audit_updated_at"].isna() | (
            runs["audit_updated_at"] >= runs["created_at"]
        )
        runs = _apply(
            runs,
            consistent,
            ledger,
            "impossible timestamps",
            "run ended before it started; corrupt metadata",
        )

    # ---- label validity --------------------------------------------------
    for conclusion, reason in DROPPED_CONCLUSIONS.items():
        runs = _apply(
            runs,
            runs["conclusion"] != conclusion,
            ledger,
            f"conclusion={conclusion}",
            reason,
        )

    runs = _apply(
        runs,
        runs["conclusion"].isin(USABLE_CONCLUSIONS),
        ledger,
        "unrecognised conclusion",
        "conclusion outside the documented value set",
    )

    # ---- required pre-execution inputs -----------------------------------
    # Dropping these is a judgement: a run with no commit message could be
    # modelled with an explicit "empty message" marker. They are removed
    # because at 133 of 573,993 rows the choice cannot affect any conclusion,
    # and keeping them complicates every text code path.
    if config.require_head_commit:
        runs = _apply(
            runs,
            runs["commit_message"].notna(),
            ledger,
            "null head_commit",
            "no commit object, so the primary unstructured feature is absent",
        )
    if config.require_head_branch:
        runs = _apply(
            runs,
            runs["head_branch"].notna() & (runs["head_branch"].astype(str).str.len() > 0),
            ledger,
            "empty head_branch",
            "branch name is a core pre-execution signal",
        )
    if config.require_head_sha:
        runs = _apply(
            runs,
            runs["head_sha"].notna(),
            ledger,
            "null head_sha",
            "needed to group runs triggered by the same commit",
        )

    # ---- population scoping ----------------------------------------------
    if config.require_selected_repository:
        selected = set(repositories.loc[repositories["selected"], "repo"])
        runs = _apply(
            runs,
            runs["repo"].isin(selected),
            ledger,
            "repository not selected",
            "outside GHALogs' documented sampling frame (>=30 runs per 90 days, non-fork)",
        )

    # ---- optional robustness filters -------------------------------------
    if config.drop_rerun_attempts:
        runs = _apply(
            runs,
            runs["run_attempt"] == 1,
            ledger,
            "re-run attempt",
            "flaky-contaminated labels; sensitivity analysis only",
        )
    if config.drop_dynamic_event:
        runs = _apply(
            runs,
            runs["event"] != "dynamic",
            ledger,
            "event=dynamic",
            "mostly GitHub Pages deployment with atypical 2.46% failure rate",
        )

    runs[FAILURE_LABEL] = (runs["conclusion"] == "failure").astype("int8")
    return runs, ledger


def annotate_commit_groups(runs: pd.DataFrame) -> pd.DataFrame:
    """Group runs triggered by the same commit and flag ambiguous labels.

    One push commonly starts several workflows: 69.9% of runs share a
    `(repo, head_sha)` key with at least one other run. Two consequences, both
    handled here.

    First, leakage. Runs in the same commit group have identical commit text
    and near-identical metadata. A split that separates them puts effectively
    the same input on both sides, so `commit_group` must be respected when
    splitting (repository grouping already implies this, since a commit cannot
    span repositories).

    Second, an irreducible error floor. In 25.1% of multi-run commit groups
    the outcomes disagree: the same commit passes one workflow and fails
    another. 47.7% of all failures share a commit with a success. No model
    that sees only commit-level information can separate those cases, which
    bounds the achievable error at 6.42% of runs against a 15.88% base rate.
    Workflow identity is therefore a required input, not an optional extra.
    """
    runs = runs.copy()
    runs["commit_group"] = runs["repo"].astype(str) + "@" + runs["head_sha"].astype(str)

    grouped = runs.groupby("commit_group")[FAILURE_LABEL]
    group_size = grouped.transform("size")
    group_failures = grouped.transform("sum")

    runs["commit_group_size"] = group_size.astype("int32")
    runs["commit_group_is_mixed"] = (
        (group_failures > 0) & (group_failures < group_size)
    ).astype("int8")
    # Minority share within the group: the fraction of this group's runs that
    # a commit-text-only model must necessarily misclassify.
    minority = np.minimum(group_failures, group_size - group_failures)
    runs["commit_group_minority"] = minority.astype("int32")
    return runs


def quality_report(runs: pd.DataFrame) -> str:
    """Summarise the properties that constrain modelling and reporting."""
    n = len(runs)
    failures = int(runs[FAILURE_LABEL].sum())
    multi = runs["commit_group_size"] > 1
    mixed = runs["commit_group_is_mixed"] == 1
    # Each group's minority count is repeated across its rows; divide it out.
    irreducible = int(
        (runs["commit_group_minority"] / runs["commit_group_size"]).sum()
    )
    reruns = int((runs["run_attempt"] > 1).sum())

    lines = [
        "data quality report:",
        f"  runs: {n:,}   failures: {failures:,} ({100 * failures / n:.2f}%)",
        f"  repositories: {runs['repo'].nunique():,}   "
        f"workflows: {runs.groupby(['repo', 'workflow_path']).ngroups:,}   "
        f"commit groups: {runs['commit_group'].nunique():,}",
        "",
        "  commit-level label ambiguity:",
        f"    runs sharing a commit with another run: {int(multi.sum()):,} "
        f"({100 * multi.mean():.1f}%)",
        f"    runs in commit groups with mixed outcomes: {int(mixed.sum()):,} "
        f"({100 * mixed.mean():.1f}%)",
        f"    failures sharing a commit with a success: "
        f"{int(runs.loc[mixed, FAILURE_LABEL].sum()):,} "
        f"({100 * runs.loc[mixed, FAILURE_LABEL].sum() / failures:.1f}% of failures)",
        f"    irreducible error for a commit-text-only model: ~{irreducible:,} runs "
        f"({100 * irreducible / n:.2f}%)",
        "",
        "  label noise from flakiness:",
        f"    re-run attempts (run_attempt > 1): {reruns:,} ({100 * reruns / n:.2f}%)",
        f"    failure rate among re-runs: "
        f"{100 * runs.loc[runs['run_attempt'] > 1, FAILURE_LABEL].mean():.2f}%",
    ]
    return "\n".join(lines)
