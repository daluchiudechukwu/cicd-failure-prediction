"""Stage 4b — execution-history features, computed causally.

History features are legitimately pre-execution: whether the previous run of a
workflow passed is a fact available the instant the next run is queued. They
are also the strongest signal in the dataset — history alone reaches PR-AUC
0.669 against 0.295 for all the static features combined.

That strength is exactly why they are dangerous. Two mistakes are easy and
both silently invalidate results:

1. **Whole-dataset aggregation.** Computing a per-repository failure rate over
   the entire table and joining it back leaks future outcomes into every row.
   Between-repository failure-rate standard deviation is 18.4% and 32.4% of
   repositories never fail, so such a feature would dominate the model and
   produce a result that collapses in deployment. Every aggregate here is an
   expanding window over strictly earlier rows.

2. **Imputing missing history.** 25.5% of runs have no adjacent predecessor.
   Filling those with a global median tells the model that an unknown history
   looks average, which it does not: the cold-start failure rate is 18.0%
   against 5.0% for runs following a success. Missingness is therefore carried
   as an explicit sentinel (-1) plus a boolean flag, so the model can learn
   the cold-start regime as its own thing.

A third subtlety is specific to GHALogs. The dataset keeps at most the five
most recent runs per workflow, so predecessors are capped at four and
`run_number` gaps are common. A predecessor is only accepted when its
`run_number` is exactly one lower; otherwise the run is treated as cold start.
Without that check, a run numbered 200 would be treated as the direct
successor of run 150 and the "previous outcome" feature would be months stale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAILURE_LABEL

WORKFLOW_KEY = ["repo", "workflow_path"]
# Sentinel for "this quantity is undefined because there is no history".
NO_HISTORY = -1.0


def _expanding_prior_rate(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.Series, pd.Series]:
    """Count and failure rate over all strictly earlier rows within `keys`.

    Implemented as a cumulative sum minus the current row, which is an
    expanding window that provably excludes the current observation. The
    caller must have sorted the frame chronologically within each group.
    """
    group = frame.groupby(keys, sort=False)
    prior_count = group.cumcount()
    prior_failures = group[FAILURE_LABEL].cumsum() - frame[FAILURE_LABEL]

    counts = prior_count.to_numpy(dtype="float64")
    rate = np.divide(
        prior_failures.to_numpy(dtype="float64"),
        counts,
        out=np.full(len(counts), NO_HISTORY),
        where=counts > 0,
    )
    return prior_count.astype("int32"), pd.Series(rate, index=frame.index, dtype="float32")


def build_history_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Attach causally ordered history features and the regime label.

    Returns the input frame with `hist_*` columns and a `regime` column added.
    The regime label is for stratified reporting, never a model input: it is
    derived from the previous outcome, so feeding it to a model would be
    circular.
    """
    # Chronological order within each workflow. run_number then run_attempt is
    # more reliable than created_at here, because re-runs of an earlier run can
    # be queued after a later run has already finished.
    runs = runs.sort_values(
        WORKFLOW_KEY + ["run_number", "run_attempt"], kind="mergesort"
    ).reset_index(drop=True)
    workflow_group = runs.groupby(WORKFLOW_KEY, sort=False)

    # ---- immediate predecessor ------------------------------------------
    prev_outcome = workflow_group[FAILURE_LABEL].shift(1)
    prev_run_number = workflow_group["run_number"].shift(1)
    prev_created = workflow_group["created_at"].shift(1)
    # Only an exactly-adjacent run counts, for the five-run sampling reason
    # described in the module docstring.
    is_adjacent = (runs["run_number"] - prev_run_number).eq(1).fillna(False)

    out = pd.DataFrame(index=runs.index)
    out["hist_has_prev_run"] = is_adjacent.astype("int8")
    out["hist_prev_failed"] = (
        prev_outcome.where(is_adjacent).fillna(NO_HISTORY).astype("int8")
    )
    gap_hours = (runs["created_at"] - prev_created).dt.total_seconds().div(3600.0)
    out["hist_hours_since_prev"] = (
        gap_hours.where(is_adjacent).fillna(NO_HISTORY).clip(upper=24 * 90).astype("float32")
    )

    # ---- failure streak immediately before this run ----------------------
    # Capped at four by construction, since no workflow has more than five
    # runs in the dataset. Reported as such wherever the feature is described.
    streak = np.zeros(len(runs), dtype="int8")
    outcomes = runs[FAILURE_LABEL].to_numpy()
    adjacent = is_adjacent.to_numpy()
    for i in range(1, len(runs)):
        if adjacent[i]:
            streak[i] = streak[i - 1] + 1 if outcomes[i - 1] == 1 else 0
    out["hist_prev_failure_streak"] = streak

    # ---- expanding aggregates at three granularities ---------------------
    # Workflow level: the most specific signal, but the five-run cap means at
    # most four observations contribute.
    wf_count, wf_rate = _expanding_prior_rate(runs, WORKFLOW_KEY)
    out["hist_wf_prior_runs"] = wf_count
    out["hist_wf_prior_failure_rate"] = wf_rate

    # Repository level: pools every workflow in the repository, giving a
    # median of 15 prior observations and so a far more stable estimate. This
    # is what makes some prediction possible even at workflow cold start.
    runs_by_time = runs.sort_values(["repo", "created_at"], kind="mergesort")
    repo_count, repo_rate = _expanding_prior_rate(runs_by_time, ["repo"])
    out["hist_repo_prior_runs"] = repo_count.reindex(runs.index)
    out["hist_repo_prior_failure_rate"] = repo_rate.reindex(runs.index)

    # Actor level, scoped to the repository. The scoping is not cosmetic.
    #
    # Pooling an actor's history across repositories breaks the primary
    # splitting protocol. Features are computed once over the whole table,
    # while evaluation groups by repository; a bot such as Dependabot triggers
    # runs in thousands of repositories, so an unscoped actor failure rate for
    # a training row would be computed partly from runs belonging to test-fold
    # repositories. That is future information leakage by the back door, and
    # it is invisible to a per-row temporal check because every contributing
    # run genuinely is earlier in time.
    #
    # Scoping to (repo, actor) keeps the feature inside the group boundary, so
    # it remains valid under repository-grouped cross-validation. The unscoped
    # variant is only safe if features are recomputed inside each fold; see
    # docs/06-data-pipeline.md.
    #
    # It is also a fairness hazard regardless of scope: a model that flags a
    # change because of who wrote it is a performance-management tool, not an
    # engineering one. Pseudonymised upstream, and reported in its own
    # ablation so the dissertation can argue against using it.
    actor_by_time = runs.sort_values(["repo", "actor_login", "created_at"], kind="mergesort")
    actor_count, actor_rate = _expanding_prior_rate(actor_by_time, ["repo", "actor_login"])
    out["hist_actor_prior_runs"] = actor_count.reindex(runs.index)
    out["hist_actor_prior_failure_rate"] = actor_rate.reindex(runs.index)

    # ---- regime label, for reporting only --------------------------------
    regime = pd.Series("A_cold_start", index=runs.index, dtype="object")
    regime[out["hist_prev_failed"] == 0] = "B_prev_success"
    regime[out["hist_prev_failed"] == 1] = "C_prev_failure"

    result = pd.concat([runs, out], axis=1)
    result["regime"] = regime.astype("category")
    return result


def assert_history_is_causal(frame: pd.DataFrame) -> None:
    """Fail loudly if any history feature could have seen the future.

    Two independent checks, because the expanding-window implementation and
    the adjacency rule can each fail in isolation.

    Check 1: a run with no prior runs in its group must carry the sentinel,
    never a real rate. If an expanding window were replaced by a whole-group
    aggregate, the first row of each group would gain a value and this fires.

    Check 2: the previous-outcome feature must equal the actual outcome of the
    exactly-preceding run whenever one exists, and the sentinel otherwise.
    """
    first_in_workflow = frame["hist_wf_prior_runs"] == 0
    bad = frame.loc[first_in_workflow, "hist_wf_prior_failure_rate"].ne(NO_HISTORY)
    if bad.any():
        raise AssertionError(
            f"{int(bad.sum())} runs with no prior workflow runs carry a real failure rate; "
            "an aggregate is reading the whole group instead of earlier rows only"
        )

    first_in_repo = frame["hist_repo_prior_runs"] == 0
    bad = frame.loc[first_in_repo, "hist_repo_prior_failure_rate"].ne(NO_HISTORY)
    if bad.any():
        raise AssertionError(
            f"{int(bad.sum())} runs with no prior repository runs carry a real failure rate"
        )

    ordered = frame.sort_values(WORKFLOW_KEY + ["run_number", "run_attempt"], kind="mergesort")
    group = ordered.groupby(WORKFLOW_KEY, sort=False)
    expected_prev = group[FAILURE_LABEL].shift(1)
    adjacent = (ordered["run_number"] - group["run_number"].shift(1)).eq(1).fillna(False)
    expected = expected_prev.where(adjacent).fillna(NO_HISTORY).astype("int8")
    mismatch = ordered["hist_prev_failed"].ne(expected)
    if mismatch.any():
        raise AssertionError(
            f"{int(mismatch.sum())} rows where hist_prev_failed disagrees with the "
            "actual preceding run outcome"
        )
