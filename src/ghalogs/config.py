"""Central configuration for the GHALogs pre-execution pipeline.

Every threshold that affects the modelling population lives here rather than
being scattered through the code, so that the methodology chapter can cite a
single source and so that sensitivity analyses are a config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Label definition
# --------------------------------------------------------------------------

# GHALogs retains six conclusion values. Only these two carry an interpretable
# pass/fail signal for the prediction task.
USABLE_CONCLUSIONS: tuple[str, ...] = ("success", "failure")

# Conclusions we drop, with the reason recorded for the exclusion table.
DROPPED_CONCLUSIONS: dict[str, str] = {
    "skipped": "workflow matched no triggering condition; never executed",
    "cancelled": "human or system aborted the run; outcome unknown and "
    "cancellation may itself be a failure signal (selection bias)",
    "action_required": "blocked awaiting manual approval; not an execution outcome",
    "startup_failure": "workflow never started, usually invalid YAML; a distinct "
    "failure mode fully determined by pre-execution state",
}

FAILURE_LABEL = "failed"


# --------------------------------------------------------------------------
# Text handling
# --------------------------------------------------------------------------

# Commit messages: measured mean 180 chars, median 57, p95 591. 9,172 runs
# exceed 1000 chars, mostly bot-generated changelogs. Truncate generously
# enough to keep the p95 intact but bounded enough to cap tokeniser cost.
MAX_COMMIT_MESSAGE_CHARS = 2000
MAX_BRANCH_CHARS = 200
MAX_DESCRIPTION_CHARS = 500

# Token budget for the transformer input assembled from the text fields.
TRANSFORMER_MAX_TOKENS = 256

# Placeholders substituted for redacted spans. Chosen to be single tokens in
# most BPE vocabularies and to remain human-readable in explanations.
REDACTION_TOKENS = {
    "email": "<EMAIL>",
    "url": "<URL>",
    "sha": "<SHA>",
    "secret": "<SECRET>",
    "mention": "<USER>",
}


# --------------------------------------------------------------------------
# Noise filtering
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    """Switches for the noise filters, each with a measured justification."""

    # 133 runs carry a null head_commit and 19 an empty head_branch. Without
    # these the core unstructured features do not exist.
    require_head_commit: bool = True
    require_head_branch: bool = True
    require_head_sha: bool = True

    # 14 runs have updated_at earlier than created_at, which is impossible and
    # signals corrupt metadata.
    drop_inconsistent_timestamps: bool = True

    # Keep only repositories that met GHALogs' own selection criterion, so the
    # population matches the dataset's documented sampling frame.
    require_selected_repository: bool = True

    # 15,385 runs (2.65%) are re-run attempts. Roughly two thirds of reruns are
    # flaky in the GitHub Actions literature, so their labels are partly
    # non-deterministic. Default is to keep and flag them, because dropping
    # them removes real traffic; set to True for the sensitivity analysis.
    drop_rerun_attempts: bool = False

    # The `dynamic` event is almost entirely GitHub Pages deployment (13,515 of
    # 29,585 runs are on gh-pages) with an atypical 2.46% failure rate. Kept by
    # default, flagged so it can be excluded in a robustness check.
    drop_dynamic_event: bool = False


DEFAULT_FILTERS = FilterConfig()


# --------------------------------------------------------------------------
# Feature engineering vocabularies
# --------------------------------------------------------------------------

# Branch-name intent markers. Deliberately short and readable: this list
# doubles as the hand-written keyword baseline that learned text models must
# beat. Measured failure rates span 12.9% (main) to 27.4% (renovate).
BRANCH_MARKERS: tuple[str, ...] = (
    "dependabot",
    "renovate",
    "feature",
    "feat",
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
    "staging",
    "gh-pages",
    "backport",
    "ci",
)

# Conventional-commit and common intent vocabulary for commit messages.
MESSAGE_MARKERS: tuple[str, ...] = (
    "fix",
    "feat",
    "chore",
    "refactor",
    "test",
    "docs",
    "style",
    "perf",
    "build",
    "ci",
    "bump",
    "update",
    "upgrade",
    "revert",
    "merge",
    "wip",
    "hotfix",
    "breaking",
    "deprecate",
    "security",
    "typo",
    "cleanup",
    "release",
)

# Workflow-name vocabulary. Important because the same commit frequently
# passes one workflow and fails another, so workflow identity is not optional.
WORKFLOW_MARKERS: tuple[str, ...] = (
    "test",
    "build",
    "lint",
    "ci",
    "cd",
    "deploy",
    "release",
    "publish",
    "docs",
    "coverage",
    "security",
    "codeql",
    "benchmark",
    "nightly",
    "matrix",
    "e2e",
    "integration",
    "format",
    "docker",
    "label",
)

DEFAULT_BRANCH_NAMES: tuple[str, ...] = ("main", "master", "trunk", "default", "develop")

# Trigger events that cannot involve a new code change. For these the head
# commit is unchanged from the previous run, so commit text carries no
# incremental information and failures must originate outside the repository.
NON_CODE_CHANGE_EVENTS: tuple[str, ...] = (
    "schedule",
    "workflow_dispatch",
    "repository_dispatch",
    "issues",
    "issue_comment",
    "workflow_run",
    "status",
    "watch",
    "deployment",
    "deployment_status",
    "public",
    "label",
    "milestone",
)


# --------------------------------------------------------------------------
# Repository context
# --------------------------------------------------------------------------

# The only repo.* fields admitted by the feature contract: fixed at creation
# or slow-moving enough that the crawl-time snapshot approximates the value at
# run time. Every counter (stars, forks, commits, issues, total_runs_90d) is
# excluded because it post-dates most runs.
ADMITTED_REPO_FIELDS: tuple[str, ...] = (
    "mainLanguage",
    "createdAt",
    "defaultBranch",
    "license",
    "hasWiki",
    "isFork",
    "topics",
)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Grouping keys that must not straddle a train/test boundary.

    Repository grouping is primary: 32.4% of repositories never fail, so a
    row-level split lets a model score well by memorising repository identity.

    Commit grouping matters because 69.9% of runs share a head_sha with at
    least one other run (one commit triggers several workflows). A split that
    separates those runs puts near-identical text on both sides.
    """

    group_key: str = "repo"
    secondary_group_key: str = "commit_group"
    n_splits: int = 5
    validation_fraction: float = 0.2
    seed: int = 42


DEFAULT_SPLITS = SplitConfig()


@dataclass(frozen=True)
class PipelineConfig:
    filters: FilterConfig = field(default_factory=lambda: DEFAULT_FILTERS)
    splits: SplitConfig = field(default_factory=lambda: DEFAULT_SPLITS)
    ingest_chunk_rows: int = 50_000
