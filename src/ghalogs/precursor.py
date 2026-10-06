"""Stage 4c — precursor features, and the observability constraint they expose.

`history.py` answers "did the previous run of this workflow fail?". This
module asks the question an early-warning framing actually depends on: is
there anything in the *shape* of the preceding runs — durations drifting up,
cadence changing, a sibling workflow breaking first, the previous run needing
a re-run — that signals an upcoming failure before it happens?

Two things make this more than a feature-engineering exercise.

**Lagged post-execution fields are pre-execution.** A run's own duration and
log volume are execution-dependent, and the feature contract rightly forbids
them. The duration of a run that *finished before the target run was
triggered* is a different quantity: it was observable at prediction time. The
contract's substring denial cannot distinguish the two, so these enter through
an explicit, named exemption (`contract.LAGGED_FIELD_EXEMPTIONS`) rather than
by choosing a column name that slips past the filter.

**The predecessor is not always observable.** This is the finding that
motivated the module. CI is concurrent: a developer pushes again while the
previous run is still executing, and GitHub Actions runs both. When that
happens the previous run's outcome does not exist at the moment the target run
is triggered, so any feature derived from it — including the `hist_prev_failed`
that dominates every published model, and the previous-outcome straw man
itself — is reading a value from the future.

Every lagged feature here is therefore gated on
`predecessor finished < target created_at`, and the gate is exposed as
`hist_prev_completed_before_trigger` so the size of the effect can be measured
rather than assumed. `build_observable_history` applies the same gate to the
baseline history features, which is what makes a deployment-honest comparison
possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAILURE_LABEL

WORKFLOW_KEY = ["repo", "workflow_path"]
NO_HISTORY = -1.0

# How far back the burst and sibling windows look. One day is long enough to
# catch an upstream dependency breaking a repository's workflows in sequence,
# and short enough that it is not simply the repository's long-run failure
# rate restated.
BURST_WINDOW_HOURS = 24.0

# Depth cap. GHALogs keeps at most five runs per workflow, so no target run can
# have more than four predecessors; the trend features are computed over
# whatever subset of those is observable.
MAX_LAG_DEPTH = 4


def _as_datetime64(values: pd.Series) -> np.ndarray:
    """UTC timestamps as `datetime64`, which numpy can compare and subtract.

    pandas hands back a tz-aware column as an object array of `Timestamp`s.
    numpy cannot subtract a `timedelta64` from that, and comparing it falls
    back to a Python-level loop. Every timestamp in GHALogs is UTC, so
    dropping the uniform zone is lossless. `NaT` survives, and comparisons
    against it are false, which is the conservative direction for a gate.
    """
    return values.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


def _duration_seconds(frame: pd.DataFrame) -> pd.Series:
    """Wall-clock duration of each run, for use as a *lagged* quantity only.

    Never a feature for the run it describes. Every caller below shifts it by
    at least one position within the workflow and gates it on completion.
    """
    return (frame["audit_updated_at"] - frame["created_at"]).dt.total_seconds().clip(lower=0.0)


def _slope(values: np.ndarray) -> float:
    """Least-squares slope of `values` against position.

    Returns the sentinel when fewer than two observations exist, which is the
    common case rather than the exception: the five-run sampling cap means most
    target runs have one or two usable predecessors, not four.
    """
    if len(values) < 2:
        return NO_HISTORY
    positions = np.arange(len(values), dtype="float64")
    centred = positions - positions.mean()
    denominator = float((centred**2).sum())
    if denominator == 0.0:
        return NO_HISTORY
    return float((centred * (values - values.mean())).sum() / denominator)


def _lagged_duration_window(runs: pd.DataFrame, group) -> pd.DataFrame:
    """Duration statistics over predecessors that had finished at trigger time.

    Computed in one explicit pass rather than with `expanding()`, because every
    contributing run must be gated *individually* on having completed before
    the target run was triggered. CI runs finish out of order — a long run
    started earlier can still be going when a shorter later one has finished —
    so the immediate predecessor having completed does not imply that the runs
    before it had. An expanding window cannot express that condition, and
    quietly including an unfinished run is the exact leak this module exists to
    measure.
    """
    duration = runs["_duration"].to_numpy(dtype="float64")
    finished = _as_datetime64(runs["audit_updated_at"])
    triggered = _as_datetime64(runs["created_at"])
    run_number = runs["run_number"].to_numpy()

    previous = np.full(len(runs), np.nan)
    second = np.full(len(runs), np.nan)
    prior_mean = np.full(len(runs), np.nan)
    slope = np.full(len(runs), NO_HISTORY)

    for positions in group.indices.values():
        positions = np.sort(positions)
        for offset in range(1, len(positions)):
            row = positions[offset]
            window = positions[max(0, offset - MAX_LAG_DEPTH) : offset]
            observed = window[finished[window] < triggered[row]]
            if len(observed) == 0:
                continue

            values = duration[observed]
            slope[row] = _slope(values)
            prior_mean[row] = float(values.mean())

            # The immediate-predecessor features are pinned to the positionally
            # preceding row, not merely to the newest *observed* predecessor.
            # The distinction matters for re-runs: when attempt 2 of run 5 is
            # still executing, attempt 1 of run 5 is an adjacent, completed run,
            # so a "newest observed with adjacent run_number" rule would quietly
            # substitute it. That disagreed with the `observable` gate on 54
            # rows and the assertion caught it. Pinning both to `shift(1)` keeps
            # one definition of "the previous run" throughout.
            #
            # Adjacency is still required, for the sampling-gap reason in
            # history.py: run 200 is not the successor of run 150.
            immediate = positions[offset - 1]
            if (
                finished[immediate] < triggered[row]
                and run_number[row] - run_number[immediate] == 1
            ):
                previous[row] = duration[immediate]
                if offset >= 2:
                    earlier = positions[offset - 2]
                    if finished[earlier] < triggered[row]:
                        second[row] = duration[earlier]

    return pd.DataFrame(
        {"previous": previous, "second": second, "prior_mean": prior_mean, "slope": slope},
        index=runs.index,
    )


def _repo_window_counts(runs: pd.DataFrame, window_hours: float) -> tuple[pd.Series, pd.Series]:
    """Runs, and failures, that *completed* in the window before each trigger.

    Scoped to the repository and keyed on completion time rather than start
    time, because a run still in flight has told the developer nothing yet.
    Implemented per repository with `searchsorted` over the completion-ordered
    array, so the cost is a sort per repository rather than a cross join.

    Matching is on the half-open interval `[created_at - window, created_at)`,
    so a run completing at exactly the trigger instant is excluded along with
    everything later.
    """
    total = pd.Series(0.0, index=runs.index, dtype="float64")
    failed = pd.Series(0.0, index=runs.index, dtype="float64")
    window = np.timedelta64(int(window_hours * 3600), "s")

    for _, block in runs.groupby("repo", sort=False):
        usable = block[block["audit_updated_at"].notna()]
        if usable.empty:
            continue

        ordered = usable.sort_values("audit_updated_at", kind="mergesort")
        finish = _as_datetime64(ordered["audit_updated_at"])
        # Leading zero so that cumulative[i] is the failure count strictly
        # before position i.
        cumulative = np.concatenate(
            [[0.0], np.cumsum(ordered[FAILURE_LABEL].to_numpy(dtype="float64"))]
        )

        trigger = _as_datetime64(block["created_at"])
        upper = np.searchsorted(finish, trigger, side="left")
        lower = np.searchsorted(finish, trigger - window, side="left")

        total.loc[block.index] = (upper - lower).astype("float64")
        failed.loc[block.index] = cumulative[upper] - cumulative[lower]

    return total, failed


def _last_completed_other_workflow_failed(runs: pd.DataFrame) -> pd.Series:
    """Did the most recently completed run of a *different* workflow fail?

    The cleanest candidate for a genuine leading indicator. A dependency bump
    or an expired credential tends to break a repository's workflows one after
    another, so a sibling failing is a warning that is available before this
    workflow runs and is not simply this workflow's own history restated.
    """
    out = pd.Series(NO_HISTORY, index=runs.index, dtype="float64")

    for _, block in runs.groupby("repo", sort=False):
        usable = block[block["audit_updated_at"].notna()]
        if usable.empty:
            continue
        ordered = usable.sort_values("audit_updated_at", kind="mergesort")
        finish = _as_datetime64(ordered["audit_updated_at"])
        outcome = ordered[FAILURE_LABEL].to_numpy()
        path = ordered["workflow_path"].to_numpy()

        trigger = _as_datetime64(block["created_at"])
        cut = np.searchsorted(finish, trigger, side="left")
        own = block["workflow_path"].to_numpy()

        values = np.full(len(block), NO_HISTORY)
        for position in range(len(block)):
            # Walk back from the trigger to the newest completed run belonging
            # to some other workflow. Bounded so that a repository with many
            # concurrent workflows cannot make this quadratic.
            limit = max(0, cut[position] - MAX_LAG_DEPTH * 4)
            for j in range(cut[position] - 1, limit - 1, -1):
                if path[j] != own[position]:
                    values[position] = float(outcome[j])
                    break
        out.loc[block.index] = values

    return out


def build_precursor_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Return `run_id` plus the lagged precursor features.

    Expects the output of the `prepare` stage (`runs_clean.parquet`), because
    the lagged quantities are built from the `audit_*` columns that the feature
    stage drops.
    """
    required = {
        "run_id",
        "repo",
        "workflow_path",
        "run_number",
        "run_attempt",
        "created_at",
        "audit_updated_at",
        "actor_login",
        "head_branch",
        FAILURE_LABEL,
    }
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"precursor features need absent columns: {sorted(missing)}")

    runs = runs.sort_values(
        WORKFLOW_KEY + ["run_number", "run_attempt"], kind="mergesort"
    ).reset_index(drop=True)
    runs = runs.assign(_duration=_duration_seconds(runs))
    group = runs.groupby(WORKFLOW_KEY, sort=False)

    prev_run_number = group["run_number"].shift(1)
    adjacent = (runs["run_number"] - prev_run_number).eq(1).fillna(False)

    prev_finish = group["audit_updated_at"].shift(1)
    # The observability gate. A missing predecessor timestamp counts as *not*
    # observed, which is the conservative direction.
    completed = (prev_finish < runs["created_at"]).fillna(False)
    observable = adjacent & completed

    out = pd.DataFrame({"run_id": runs["run_id"]}, index=runs.index)
    out["hist_prev_completed_before_trigger"] = observable.astype("int8")

    # ---- deployment-honest previous outcome ------------------------------
    # Identical to hist_prev_failed except that an in-flight predecessor is
    # treated as no history, which is what a real predictor would see.
    prev_failed = group[FAILURE_LABEL].shift(1)
    out["hist_prev_failed_observable"] = (
        prev_failed.where(observable).fillna(NO_HISTORY).astype("int8")
    )

    # ---- lagged duration, and its trend ----------------------------------
    window = _lagged_duration_window(runs, group)
    out["hist_prev_duration_sec"] = window["previous"].fillna(NO_HISTORY).astype("float32")
    # log1p as well as the raw value: durations span seconds to hours and the
    # trees split more usefully on the compressed scale.
    out["hist_prev_duration_log"] = (
        np.log1p(window["previous"]).fillna(NO_HISTORY).astype("float32")
    )
    out["hist_prev2_duration_sec"] = window["second"].fillna(NO_HISTORY).astype("float32")
    out["hist_prior_duration_mean"] = window["prior_mean"].fillna(NO_HISTORY).astype("float32")
    out["hist_prior_duration_slope"] = window["slope"].astype("float32")

    # Ratio of the last duration to the mean of the observable ones before it:
    # the classic degradation signal.
    ratio = window["previous"] / window["prior_mean"].replace(0.0, np.nan)
    out["hist_prev_duration_ratio"] = (
        ratio.replace([np.inf, -np.inf], np.nan)
        .fillna(NO_HISTORY)
        .clip(upper=100)
        .astype("float32")
    )

    # ---- lagged instability markers --------------------------------------
    prev_attempt = group["run_attempt"].shift(1)
    out["hist_prev_was_rerun"] = (
        prev_attempt.gt(1).where(observable).fillna(NO_HISTORY).astype("int8")
    )

    if "audit_total_logs_size" in runs.columns:
        prev_logs = group["audit_total_logs_size"].shift(1).where(observable)
        out["hist_prev_log_bytes_log"] = (
            np.log1p(prev_logs.clip(lower=0)).fillna(NO_HISTORY).astype("float32")
        )
    else:
        out["hist_prev_log_bytes_log"] = np.float32(NO_HISTORY)

    # ---- cadence ---------------------------------------------------------
    runs["_gap_hours"] = group["created_at"].diff().dt.total_seconds().div(3600.0)
    gap_group = runs.groupby(WORKFLOW_KEY, sort=False)["_gap_hours"]
    prior_gap_mean = gap_group.transform(lambda s: s.shift(1).expanding().mean())

    out["hist_prior_gap_mean_hours"] = (
        prior_gap_mean.where(adjacent).fillna(NO_HISTORY).clip(upper=24 * 90).astype("float32")
    )
    gap_ratio = runs["_gap_hours"] / prior_gap_mean.replace(0.0, np.nan)
    out["hist_gap_vs_prior_mean"] = (
        gap_ratio.replace([np.inf, -np.inf], np.nan)
        .where(adjacent)
        .fillna(NO_HISTORY)
        .clip(upper=100)
        .astype("float32")
    )

    # ---- how much history exists at all ----------------------------------
    # The unbroken chain of adjacent predecessors: the honest answer to "how
    # many historical builds does this prediction get to see?"
    depth = np.zeros(len(runs), dtype="int8")
    adjacent_values = adjacent.to_numpy()
    for i in range(1, len(runs)):
        if adjacent_values[i]:
            depth[i] = min(depth[i - 1] + 1, MAX_LAG_DEPTH)
    out["hist_adjacent_depth"] = depth

    # ---- contemporaneous repository context ------------------------------
    window_total, window_failed = _repo_window_counts(runs, BURST_WINDOW_HOURS)
    out["hist_repo_runs_prev_24h"] = window_total.astype("float32")
    out["hist_repo_failures_prev_24h"] = window_failed.astype("float32")
    rate = np.divide(
        window_failed.to_numpy(),
        window_total.to_numpy(),
        out=np.full(len(runs), NO_HISTORY),
        where=window_total.to_numpy() > 0,
    )
    out["hist_repo_failure_rate_prev_24h"] = pd.Series(rate, index=runs.index).astype("float32")
    out["hist_sibling_workflow_failed"] = _last_completed_other_workflow_failed(runs).astype(
        "float32"
    )

    # ---- identity churn --------------------------------------------------
    prev_actor = group["actor_login"].shift(1)
    out["hist_prev_actor_changed"] = (
        prev_actor.ne(runs["actor_login"]).where(adjacent).fillna(NO_HISTORY).astype("int8")
    )
    prev_branch = group["head_branch"].shift(1)
    out["hist_prev_branch_changed"] = (
        prev_branch.ne(runs["head_branch"]).where(adjacent).fillna(NO_HISTORY).astype("int8")
    )

    return out


def build_observable_history(features: pd.DataFrame, precursor: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the history features with their deployment-observable form.

    Returns a copy of `features` in which every quantity derived from the
    immediate predecessor is blanked to the sentinel when that predecessor had
    not completed at trigger time. The difference between this and the original
    frame is the amount of measured performance that depends on reading an
    in-flight run's outcome.
    """
    merged = features[["run_id"]].merge(
        precursor[["run_id", "hist_prev_completed_before_trigger"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    )
    observable = (
        merged["hist_prev_completed_before_trigger"].fillna(0).astype(bool).to_numpy()
    )

    out = features.copy()
    blanked = {
        "hist_prev_failed": NO_HISTORY,
        "hist_hours_since_prev": NO_HISTORY,
        "hist_prev_failure_streak": 0,
        "hist_has_prev_run": 0,
    }
    for column, sentinel in blanked.items():
        if column in out.columns:
            values = out[column].to_numpy().copy()
            values[~observable] = sentinel
            out[column] = values.astype(out[column].dtype)
    return out


def assert_precursor_is_observable(frame: pd.DataFrame) -> None:
    """Fail loudly if a lagged feature exists where no predecessor had finished.

    The gate is applied column by column in `build_precursor_features`, so a
    single missed `.where(observable)` would silently admit a value from a run
    that was still executing. This checks the invariant on the assembled frame
    rather than trusting each call site.

    Only the immediate-predecessor columns are checked. The window aggregates
    (`hist_prior_*`) are gated per contributing element and can legitimately
    exist when the *adjacent* predecessor was unobservable but an earlier run
    had finished.
    """
    unobservable = frame["hist_prev_completed_before_trigger"] == 0
    gated = [
        "hist_prev_failed_observable",
        "hist_prev_duration_sec",
        "hist_prev_duration_log",
        "hist_prev2_duration_sec",
        "hist_prev_was_rerun",
        "hist_prev_log_bytes_log",
    ]
    for column in gated:
        if column not in frame.columns:
            continue
        offending = frame.loc[unobservable, column].ne(NO_HISTORY)
        if offending.any():
            raise AssertionError(
                f"{int(offending.sum())} rows carry a real value for {column} even though "
                "no predecessor had completed when the run was triggered"
            )
