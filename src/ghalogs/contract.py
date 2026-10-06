"""Stage 5 — executable enforcement of the pre-execution feature contract.

`docs/02-feature-contract.md` states in prose which fields a model may see.
Prose is not enforcement. This module turns the contract into assertions that
run before every experiment, so a leak is caught at the gate rather than
discovered after the models are trained.

Three mechanisms, in increasing strength:

1. **Prefix allowlist.** Only `static_*` and `hist_*` columns may be used as
   features. Anything else — identifiers, labels, raw text, audit columns — is
   refused by name. An allowlist is used rather than a denylist because the
   failure mode of a denylist is silence: the field you forgot to ban becomes
   a feature.

2. **Named-field denial.** Explicit rejection of the fields known to leak,
   including the two GitHub Actions-specific mechanisms described in the
   contract: log-derived workflow structure (truncated by fail-fast, so it
   encodes the outcome) and crawl-time repository counters (which post-date
   most runs). Denial is by substring, which over-matches: a *lagged*
   post-execution field describes an earlier run and is legitimately
   observable at prediction time. Those are admitted through a named
   exemption list, never by renaming the column to evade the filter.

3. **Statistical screening.** Any single feature whose univariate ROC-AUC
   against the label exceeds a threshold is flagged. This catches leaks that
   neither list anticipated. It is a screen, not a proof: a legitimately
   strong feature can trip it, which is why tripping it demands a written
   justification rather than automatic removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import FAILURE_LABEL

# Columns permitted as model inputs, by prefix.
FEATURE_PREFIXES: tuple[str, ...] = ("static_", "hist_")

# Non-feature columns that legitimately travel with the frame: identifiers for
# grouping, the label, raw text for the transformer tokeniser, and the regime
# label used for stratified reporting.
PERMITTED_NON_FEATURES: frozenset[str] = frozenset(
    {
        "run_id",
        "repo",
        "workflow_path",
        "workflow_id",
        "run_number",
        "run_attempt",
        "head_sha",
        "commit_group",
        "commit_group_size",
        "commit_group_is_mixed",
        "commit_group_minority",
        "created_at",
        "run_started_at",
        "regime",
        FAILURE_LABEL,
        "conclusion",
        # Raw text: consumed by the tokeniser, never fed to a tabular model.
        "commit_message",
        "head_branch",
        "display_title",
        "workflow_name",
        "repo_description",
        "transformer_input",
        # Pseudonymised identifiers, used for grouping and history only.
        "actor_login",
        "triggering_actor_login",
        "commit_author_id",
    }
)

# Fields that must never appear as a feature, with the leakage class from
# docs/02-feature-contract.md. Matched as substrings so that derived columns
# (`audit_updated_at`, `duration_sec`, ...) are caught too.
FORBIDDEN_SUBSTRINGS: dict[str, str] = {
    "conclusion": "type 1 — the label itself",
    "audit_": "type 2 — projected for auditing only, never a feature",
    "updated_at": "type 2 — run end time; differencing it yields the duration",
    "duration": "type 2 — execution-dependent",
    "log_insights": "type 2 — parsed from logs produced during execution",
    "total_logs_size": "type 2 — log volume tracks stack traces, so it proxies the label",
    "n_log_jobs": "type 2 — job count from logs is truncated by fail-fast",
    "n_log_steps": "type 2 — step count from logs is truncated by fail-fast",
    "image_version": "type 2 — runner image resolved at runtime",
    "token_permissions": "type 2 — printed by the runner at runtime",
    "stargazers": "type 3 — crawl-time counter post-dating the run",
    "forks": "type 3 — crawl-time counter post-dating the run",
    "watchers": "type 3 — crawl-time counter post-dating the run",
    "open_issues": "type 3 — crawl-time counter post-dating the run",
    "total_issues": "type 3 — crawl-time counter post-dating the run",
    "pushed_at": "type 3 — describes activity after most runs",
    "last_commit": "type 3 — can post-date the run by months",
    "total_runs_90d": "type 3 — the dataset's own selection criterion, an "
    "aggregate over the runs being predicted",
    "nb_runs": "type 3 — window aggregate spanning the test set",
    "code_lines": "type 3 — measured on the crawl-time tree, not the tree at head_sha",
    # Derived from the previous outcome, so circular as an input.
    "regime": "derived from the label of the preceding run",
}

# Prefixes that mark a feature as describing a strictly earlier run rather than
# the run being predicted. An exemption below is only honoured for a name
# carrying one of these, so the lag is always visible in the column name.
LAG_PREFIXES: tuple[str, ...] = ("hist_prev", "hist_prior")

# Features that contain a forbidden substring yet are admissible, because they
# describe a run that had *already completed* when the target run was
# triggered.
#
# This closes a gap in the substring denial above. "Duration" is forbidden
# because a run's own duration is execution-dependent; the duration of the
# preceding run is a different quantity, observable at prediction time, and
# exactly the signal an early-warning framing rests on. A substring match
# cannot tell the two apart, so the distinction is made explicitly here rather
# than by picking a column name that slips past the filter.
#
# Admissibility depends on a gate, not just on a name: `precursor.py` blanks
# every one of these when the predecessor had not finished before the target
# run's `created_at`, and `assert_precursor_is_observable` checks the
# invariant. Without that gate these would be future information.
LAGGED_FIELD_EXEMPTIONS: dict[str, str] = {
    "hist_prev_duration_sec": "duration of the preceding run, gated on it having "
    "completed before this run was triggered",
    "hist_prev_duration_log": "log1p of the same gated quantity",
    "hist_prev2_duration_sec": "duration of the run two positions earlier, same gate",
    "hist_prev_duration_ratio": "preceding duration over the mean of the ones before "
    "it; the degradation-trend signal",
    "hist_prior_duration_mean": "expanding mean of durations strictly earlier than "
    "the preceding run",
    "hist_prior_duration_slope": "least-squares slope over up to four earlier "
    "durations, all completed before the trigger",
}

# Univariate ROC-AUC above which a feature is flagged for manual review.
SINGLE_FEATURE_AUC_LIMIT = 0.95

# Features allowed to exceed the screen, each with a recorded reason. Keeping
# this explicit and empty-by-default means every exemption is a deliberate,
# reviewable decision.
SCREEN_EXEMPTIONS: dict[str, str] = {}


@dataclass
class ContractReport:
    feature_columns: list[str]
    flagged: dict[str, float]

    @property
    def ok(self) -> bool:
        return not self.flagged

    def describe(self) -> str:
        lines = [f"contract: {len(self.feature_columns)} admissible features"]
        static = sum(1 for c in self.feature_columns if c.startswith("static_"))
        history = sum(1 for c in self.feature_columns if c.startswith("hist_"))
        lines.append(f"  static_*: {static}   hist_*: {history}")
        lagged = [c for c in self.feature_columns if _is_exempt_lagged_field(c)]
        if lagged:
            lines.append(
                f"  lagged-field exemptions in use: {len(lagged)} "
                f"({', '.join(sorted(lagged))})"
            )
        if self.flagged:
            lines.append(f"  FLAGGED by the AUC screen (limit {SINGLE_FEATURE_AUC_LIMIT}):")
            for name, auc in sorted(self.flagged.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {name}: univariate AUC {auc:.4f}")
        else:
            lines.append("  no feature exceeds the univariate AUC screen")
        return "\n".join(lines)


def _is_exempt_lagged_field(column: str) -> bool:
    """Is `column` an explicitly justified lagged field?

    Both conditions are required: a recorded justification *and* a name that
    advertises the lag. Either alone is too easy to satisfy by accident — a
    bare exemption list would let `hist_duration_sec` through on a typo, and a
    bare prefix rule would exempt anything merely named `hist_prev_*`.
    """
    return column in LAGGED_FIELD_EXEMPTIONS and column.startswith(LAG_PREFIXES)


def select_features(frame: pd.DataFrame) -> list[str]:
    """Return the admissible feature columns, refusing anything unexpected.

    Raises rather than silently dropping, because a column that is neither an
    allowed feature nor an explicitly permitted passenger means the pipeline
    produced something the contract has not considered.
    """
    features: list[str] = []
    unexpected: list[str] = []

    for column in frame.columns:
        if column.startswith(FEATURE_PREFIXES):
            features.append(column)
        elif column in PERMITTED_NON_FEATURES:
            continue
        else:
            unexpected.append(column)

    if unexpected:
        raise ValueError(
            "columns present that the feature contract does not recognise: "
            f"{sorted(unexpected)}. Add them to PERMITTED_NON_FEATURES with a "
            "justification, give them a static_/hist_ prefix, or drop them."
        )

    violations = {
        column: reason
        for column in features
        for substring, reason in FORBIDDEN_SUBSTRINGS.items()
        if substring in column and not _is_exempt_lagged_field(column)
    }
    if violations:
        raise ValueError(
            "forbidden fields reached the feature set:\n"
            + "\n".join(f"  {name}: {reason}" for name, reason in violations.items())
        )

    if not features:
        raise ValueError("no admissible features found")
    return sorted(features)


def screen_single_feature_auc(
    frame: pd.DataFrame, features: list[str], sample: int = 100_000, seed: int = 0
) -> dict[str, float]:
    """Flag features that predict the label almost perfectly on their own.

    A univariate AUC near 1.0 almost always means the feature is a transformed
    copy of the outcome. Computed on a subsample because the screen runs on
    every experiment and exact AUC is unnecessary for a tripwire.

    AUC is used rather than Pearson correlation because it is invariant to
    monotone transformations and is defined for binary and skewed features,
    both of which dominate this feature set.
    """
    labels = frame[FAILURE_LABEL].to_numpy()
    if len(frame) > sample:
        rng = np.random.default_rng(seed)
        index = rng.choice(len(frame), size=sample, replace=False)
        labels = labels[index]
    else:
        index = np.arange(len(frame))

    if labels.min() == labels.max():
        return {}

    flagged: dict[str, float] = {}
    for column in features:
        if column in SCREEN_EXEMPTIONS:
            continue
        values = frame[column].to_numpy()[index]
        if not np.issubdtype(values.dtype, np.number):
            continue
        values = np.nan_to_num(values.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0)
        if values.min() == values.max():
            continue
        auc = roc_auc_score(labels, values)
        # Direction-agnostic: a feature that perfectly predicts success is as
        # much of a leak as one that perfectly predicts failure.
        auc = max(auc, 1.0 - auc)
        if auc > SINGLE_FEATURE_AUC_LIMIT:
            flagged[column] = float(auc)
    return flagged


def enforce(frame: pd.DataFrame, strict: bool = True) -> ContractReport:
    """Run the full contract check over a feature frame."""
    features = select_features(frame)
    flagged = screen_single_feature_auc(frame, features)
    report = ContractReport(feature_columns=features, flagged=flagged)
    if strict and flagged:
        raise AssertionError(
            "the univariate AUC screen flagged features as possible leaks:\n"
            + report.describe()
            + "\n\nEither remove them, or record a justification in "
            "contract.SCREEN_EXEMPTIONS explaining why a legitimate "
            "pre-execution feature is this predictive."
        )
    return report
