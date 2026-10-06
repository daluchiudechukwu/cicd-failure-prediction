"""Tests for the pipeline's correctness-critical behaviour.

These are not coverage tests. Each one pins down a property whose silent
failure would invalidate the dissertation's results:

- the contract refuses leaky and unrecognised columns
- the AUC screen actually detects a planted leak
- history features never read the future
- the five-run sampling gap is not mistaken for adjacency
- redaction removes personal data before it can reach a feature

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghalogs import contract, history, precursor, quality, textproc  # noqa: E402
from ghalogs.config import FAILURE_LABEL, FilterConfig  # noqa: E402

BASE = datetime(2023, 9, 1, tzinfo=timezone.utc)


def make_runs(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal runs frame with sensible defaults for unset fields."""
    defaults = {
        "repo": "o/r",
        "workflow_path": ".github/workflows/ci.yml",
        "run_attempt": 1,
        "actor_login": "alice",
        "head_sha": "abc123",
    }
    records = []
    for index, row in enumerate(rows):
        record = dict(defaults)
        record.update(row)
        record.setdefault("run_number", index + 1)
        record.setdefault("created_at", BASE + timedelta(hours=index))
        record.setdefault("run_id", f"run-{index}")
        records.append(record)
    frame = pd.DataFrame(records)
    frame[FAILURE_LABEL] = frame[FAILURE_LABEL].astype("int8")
    return frame


# ---------------------------------------------------------------------------
# Contract enforcement
# ---------------------------------------------------------------------------


def base_feature_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(n)],
            "repo": ["o/r"] * n,
            FAILURE_LABEL: rng.integers(0, 2, n).astype("int8"),
            "static_branch_len": rng.integers(3, 40, n),
            "hist_prev_failed": rng.integers(-1, 2, n),
        }
    )


def test_contract_accepts_a_clean_frame():
    report = contract.enforce(base_feature_frame())
    assert report.ok
    assert report.feature_columns == ["hist_prev_failed", "static_branch_len"]


def test_contract_rejects_unrecognised_columns():
    """A column that is neither a feature nor a permitted passenger must raise.

    This is the allowlist doing its job: silence here is how a forgotten
    field becomes a feature.
    """
    frame = base_feature_frame()
    frame["some_new_column"] = 1
    with pytest.raises(ValueError, match="does not recognise"):
        contract.enforce(frame)


@pytest.mark.parametrize(
    "column",
    [
        "static_total_logs_size",
        "static_n_log_jobs",
        "hist_duration_mean",
        "static_stargazers",
        "static_total_runs_90d",
        "static_updated_at_hour",
    ],
)
def test_contract_rejects_known_leaky_fields(column):
    """Each named leakage mechanism is refused even behind a valid prefix."""
    frame = base_feature_frame()
    frame[column] = 1
    with pytest.raises(ValueError, match="forbidden fields"):
        contract.enforce(frame)


def test_auc_screen_detects_a_planted_leak():
    """The screen must catch a leak that neither list anticipated.

    A near-copy of the label is given an innocuous name and a valid prefix, so
    it passes both the allowlist and the denylist. Only the statistical screen
    can find it.
    """
    frame = base_feature_frame(2000)
    rng = np.random.default_rng(1)
    noise = rng.random(len(frame)) < 0.02
    frame["static_innocuous_name"] = np.where(noise, 1 - frame[FAILURE_LABEL], frame[FAILURE_LABEL])
    with pytest.raises(AssertionError, match="possible leaks"):
        contract.enforce(frame, strict=True)

    report = contract.enforce(frame, strict=False)
    assert "static_innocuous_name" in report.flagged
    assert report.flagged["static_innocuous_name"] > contract.SINGLE_FEATURE_AUC_LIMIT


def test_auc_screen_is_direction_agnostic():
    """A feature that perfectly predicts *success* is just as much of a leak."""
    frame = base_feature_frame(2000)
    frame["static_inverted"] = 1 - frame[FAILURE_LABEL]
    report = contract.enforce(frame, strict=False)
    assert "static_inverted" in report.flagged


# ---------------------------------------------------------------------------
# History causality
# ---------------------------------------------------------------------------


def test_previous_outcome_matches_the_preceding_run():
    frame = make_runs(
        [
            {FAILURE_LABEL: 0},
            {FAILURE_LABEL: 1},
            {FAILURE_LABEL: 1},
            {FAILURE_LABEL: 0},
        ]
    )
    result = history.build_history_features(frame)
    # First run has no predecessor; the rest copy the preceding outcome.
    assert result["hist_prev_failed"].tolist() == [-1, 0, 1, 1]
    assert result["hist_has_prev_run"].tolist() == [0, 1, 1, 1]
    assert result["regime"].tolist() == [
        "A_cold_start",
        "B_prev_success",
        "C_prev_failure",
        "C_prev_failure",
    ]


def test_non_adjacent_run_numbers_are_treated_as_cold_start():
    """GHALogs keeps only five runs per workflow, so run_number gaps are common.

    Run 200 is not the successor of run 150. Without the adjacency check the
    "previous outcome" feature would be months stale and the cold-start regime
    would be understated.
    """
    frame = make_runs([{"run_number": 150, FAILURE_LABEL: 1}, {"run_number": 200, FAILURE_LABEL: 0}])
    result = history.build_history_features(frame)
    assert result["hist_has_prev_run"].tolist() == [0, 0]
    assert result["hist_prev_failed"].tolist() == [-1, -1]
    assert result["regime"].tolist() == ["A_cold_start", "A_cold_start"]


def test_first_run_in_a_group_carries_the_sentinel_not_an_average():
    """The classic leak: aggregating over the whole group instead of earlier rows."""
    frame = make_runs([{FAILURE_LABEL: 1}, {FAILURE_LABEL: 1}, {FAILURE_LABEL: 1}])
    result = history.build_history_features(frame)
    assert result.loc[0, "hist_wf_prior_failure_rate"] == -1.0
    assert result.loc[0, "hist_repo_prior_failure_rate"] == -1.0
    # Row 1 sees exactly one prior run, which failed.
    assert result.loc[1, "hist_wf_prior_failure_rate"] == pytest.approx(1.0)


def test_prior_failure_rate_excludes_the_current_row():
    """A run's own outcome must not appear in its own history aggregate."""
    frame = make_runs([{FAILURE_LABEL: 0}, {FAILURE_LABEL: 0}, {FAILURE_LABEL: 1}])
    result = history.build_history_features(frame)
    # Row 2 fails, but its prior rate is over rows 0 and 1, which both passed.
    assert result.loc[2, "hist_wf_prior_failure_rate"] == pytest.approx(0.0)


def test_failure_streak_counts_only_adjacent_failures():
    frame = make_runs(
        [{FAILURE_LABEL: 1}, {FAILURE_LABEL: 1}, {FAILURE_LABEL: 1}, {FAILURE_LABEL: 0}, {FAILURE_LABEL: 1}]
    )
    result = history.build_history_features(frame)
    assert result["hist_prev_failure_streak"].tolist() == [0, 1, 2, 3, 0]


def test_causality_assertion_catches_a_whole_group_aggregate():
    """Simulate the mistake and confirm the assertion fires."""
    frame = make_runs([{FAILURE_LABEL: 0}, {FAILURE_LABEL: 1}])
    result = history.build_history_features(frame)
    # Overwrite with a whole-group mean, the leak we are guarding against.
    result["hist_wf_prior_failure_rate"] = result[FAILURE_LABEL].mean()
    with pytest.raises(AssertionError, match="reading the whole group"):
        history.assert_history_is_causal(result)


def test_actor_history_does_not_pool_across_repositories():
    """Actor history must stay inside the repository group boundary.

    Features are built once over the whole table while evaluation groups by
    repository. An actor failure rate pooled across repositories would let a
    training row read outcomes from test-fold repositories — leakage that no
    per-row temporal check can detect, because every contributing run really
    is earlier in time.
    """
    frame = make_runs(
        [
            # Same actor fails twice in repo A ...
            {"repo": "o/a", "run_number": 1, "actor_login": "bot", FAILURE_LABEL: 1},
            {"repo": "o/a", "run_number": 2, "actor_login": "bot", FAILURE_LABEL: 1},
            # ... then appears for the first time in repo B.
            {"repo": "o/b", "run_number": 1, "actor_login": "bot", FAILURE_LABEL: 0},
        ]
    )
    result = history.build_history_features(frame).sort_values(["repo", "run_number"])
    repo_b = result[result["repo"] == "o/b"].iloc[0]
    assert repo_b["hist_actor_prior_runs"] == 0
    assert repo_b["hist_actor_prior_failure_rate"] == -1.0


def test_history_is_causal_on_a_realistic_multi_workflow_frame():
    frame = make_runs(
        [
            {"repo": "o/a", "workflow_path": "ci.yml", "run_number": 1, FAILURE_LABEL: 0},
            {"repo": "o/a", "workflow_path": "ci.yml", "run_number": 2, FAILURE_LABEL: 1},
            {"repo": "o/a", "workflow_path": "release.yml", "run_number": 7, FAILURE_LABEL: 1},
            {"repo": "o/b", "workflow_path": "ci.yml", "run_number": 3, FAILURE_LABEL: 0},
            {"repo": "o/b", "workflow_path": "ci.yml", "run_number": 4, FAILURE_LABEL: 0},
        ]
    )
    result = history.build_history_features(frame)
    history.assert_history_is_causal(result)


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------


def test_redaction_removes_personal_data_and_secrets():
    text = (
        "Fix login for alice@example.com, see https://ci.example.com/build/9 "
        "cc @bob token ghp_abcdefghijklmnopqrstuvwxyz0123 ref deadbeefcafe123"
    )
    redacted = textproc.redact(text)
    assert "alice@example.com" not in redacted
    assert "ci.example.com" not in redacted
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123" not in redacted
    assert "@bob" not in redacted
    # The type markers survive, because "mentions an email" is usable signal.
    assert "<EMAIL>" in redacted
    assert "<URL>" in redacted
    assert "<SECRET>" in redacted
    assert "<USER>" in redacted
    assert "<SHA>" in redacted


def test_redaction_runs_before_truncation():
    """An address near the end of a long message must still be removed.

    If truncation came first, the same address would be redacted in short
    messages and silently retained in long ones.
    """
    text = "x" * 1990 + " contact alice@example.com for details"
    cleaned = textproc.normalise(textproc.redact(text), max_chars=2000)
    assert "alice@example.com" not in cleaned


def test_normalise_strips_control_characters_and_marks_truncation():
    assert "\x00" not in textproc.normalise("a\x00b", max_chars=100)
    assert textproc.normalise("y" * 50, max_chars=10).endswith("<TRUNC>")
    # Non-ASCII content is legitimate and must survive.
    assert "日本語" in textproc.normalise("fix 日本語 encoding", max_chars=100)


def test_unicode_is_canonicalised_before_redaction():
    """Fullwidth characters must not be a way around the redaction patterns.

    An address written with fullwidth punctuation is not matched by the email
    pattern until NFKC folds it to ASCII. Normalising after redacting left 7
    such addresses in the processed corpus, which is why canonicalisation runs
    first.
    """
    obfuscated = "contact alice\uff20example\uff0ecom please"
    cleaned = textproc.normalise(obfuscated, max_chars=200)
    assert "<EMAIL>" in cleaned
    assert "alice" not in cleaned


def test_branch_names_are_not_redacted():
    """Redacting identifiers would destroy the strongest static signal."""
    frame = pd.DataFrame(
        {
            "commit_message": ["see https://example.com"],
            "display_title": ["t"],
            "head_branch": ["renovate/lodash-4.x"],
            "workflow_name": ["CI"],
            "repo_description": ["d"],
            "actor_login": ["renovate[bot]"],
            "triggering_actor_login": ["renovate[bot]"],
            "commit_author_email": ["a@b.com"],
            "commit_author_name": ["A"],
        }
    )
    cleaned = textproc.clean_text_columns(frame, salt="s")
    assert cleaned.loc[0, "head_branch"] == "renovate/lodash-4.x"
    # Dependency references survive the email pattern because it requires an
    # alphabetic TLD; otherwise `checkout@v3...v4` would become `<EMAIL>`.
    assert textproc.redact("bump actions/checkout@v3...v4") == "bump actions/checkout@v3...v4"
    # The message, which is prose, is still redacted.
    assert "<URL>" in cleaned.loc[0, "commit_message"]
    # Bot detection happens before hashing, so the flag survives.
    assert cleaned.loc[0, "actor_login_is_bot"] == 1
    assert "renovate" not in cleaned.loc[0, "actor_login"]


def test_pseudonymise_is_stable_and_salt_dependent():
    assert textproc.pseudonymise("alice", "s1") == textproc.pseudonymise("alice", "s1")
    assert textproc.pseudonymise("alice", "s1") != textproc.pseudonymise("alice", "s2")
    assert textproc.pseudonymise("alice", "s1") != textproc.pseudonymise("bob", "s1")
    assert "alice" not in textproc.pseudonymise("alice", "s1")


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------


def test_filter_drops_non_outcome_conclusions_and_records_them():
    runs = pd.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(6)],
            "repo": ["o/r"] * 6,
            "workflow_path": ["ci.yml"] * 6,
            "run_number": range(6),
            "run_attempt": [1] * 6,
            "conclusion": [
                "success",
                "failure",
                "skipped",
                "cancelled",
                "action_required",
                "startup_failure",
            ],
            "created_at": [BASE] * 6,
            "audit_updated_at": [BASE] * 6,
            "commit_message": ["m"] * 6,
            "head_branch": ["main"] * 6,
            "head_sha": ["abc"] * 6,
        }
    )
    repositories = pd.DataFrame({"repo": ["o/r"], "selected": [True]})
    filtered, ledger = quality.filter_runs(runs, repositories, FilterConfig())

    assert len(filtered) == 2
    assert set(filtered["conclusion"]) == {"success", "failure"}
    ledger_frame = ledger.to_frame()
    # Every exclusion is accounted for in the ledger, which becomes the
    # methodology chapter's exclusion table.
    assert ledger_frame["removed"].sum() == 4
    assert "conclusion=cancelled" in set(ledger_frame["step"])


def test_filter_drops_impossible_timestamps_and_missing_text():
    runs = pd.DataFrame(
        {
            "run_id": ["a", "b", "c"],
            "repo": ["o/r"] * 3,
            "workflow_path": ["ci.yml"] * 3,
            "run_number": [1, 2, 3],
            "run_attempt": [1, 1, 1],
            "conclusion": ["success"] * 3,
            "created_at": [BASE, BASE, BASE],
            # Run "b" finished before it started.
            "audit_updated_at": [BASE, BASE - timedelta(hours=1), BASE],
            # Run "c" has no commit object.
            "commit_message": ["m", "m", None],
            "head_branch": ["main"] * 3,
            "head_sha": ["abc"] * 3,
        }
    )
    repositories = pd.DataFrame({"repo": ["o/r"], "selected": [True]})
    filtered, _ = quality.filter_runs(runs, repositories, FilterConfig())
    assert filtered["run_id"].tolist() == ["a"]


def test_commit_group_annotation_detects_mixed_outcomes():
    """One commit triggering several workflows with disagreeing outcomes.

    This is the 47.7%-of-failures case: identical commit text, different
    labels. The annotation is what lets the ceiling be reported rather than
    mistaken for model error.
    """
    runs = make_runs(
        [
            {"workflow_path": "ci.yml", "head_sha": "s1", FAILURE_LABEL: 0},
            {"workflow_path": "lint.yml", "head_sha": "s1", FAILURE_LABEL: 1},
            {"workflow_path": "ci.yml", "head_sha": "s2", FAILURE_LABEL: 0},
        ]
    )
    annotated = quality.annotate_commit_groups(runs)
    assert annotated["commit_group_is_mixed"].tolist() == [1, 1, 0]
    assert annotated["commit_group_size"].tolist() == [2, 2, 1]
    # One of the two runs on s1 is necessarily misclassified by any model that
    # sees only commit-level information.
    assert annotated["commit_group_minority"].tolist() == [1, 1, 0]


def make_timed_runs(rows: list[dict]) -> pd.DataFrame:
    """Build a runs frame carrying the completion timestamps precursors need.

    Separate from `make_runs` because the precursor features turn on the
    relationship between one run's completion and the next run's trigger, so
    every test here has to state both explicitly.
    """
    defaults = {
        "repo": "o/r",
        "workflow_path": "ci.yml",
        "run_attempt": 1,
        "actor_login": "alice",
        "head_branch": "main",
        "head_sha": "abc123",
        "audit_total_logs_size": 1000,
    }
    records = []
    for index, row in enumerate(rows):
        record = dict(defaults)
        record.update(row)
        record.setdefault("run_number", index + 1)
        record.setdefault("created_at", BASE + timedelta(hours=index))
        record.setdefault("audit_updated_at", record["created_at"] + timedelta(minutes=10))
        record.setdefault("run_id", f"run-{index}")
        records.append(record)
    frame = pd.DataFrame(records)
    frame[FAILURE_LABEL] = frame[FAILURE_LABEL].astype("int8")
    for column in ("created_at", "audit_updated_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def precursor_by_run(frame: pd.DataFrame) -> pd.DataFrame:
    return precursor.build_precursor_features(frame).set_index("run_id")


# ---------------------------------------------------------------------------
# Lagged-field exemptions in the contract
# ---------------------------------------------------------------------------


def test_contract_admits_a_justified_lagged_field():
    """A predecessor's duration was observable at prediction time.

    The substring denial cannot tell it apart from the run's own duration, so
    the exemption list has to carry the distinction.
    """
    frame = base_feature_frame()
    frame["hist_prev_duration_sec"] = 42.0
    report = contract.enforce(frame)
    assert "hist_prev_duration_sec" in report.feature_columns
    assert "lagged-field exemptions in use: 1" in report.describe()


def test_contract_refuses_a_lagged_name_without_a_recorded_justification():
    """Naming a column `hist_prev_*` must not be enough to admit it.

    Otherwise the exemption degenerates into a naming convention that anyone
    can satisfy by accident.
    """
    frame = base_feature_frame()
    frame["hist_prev_duration_unlisted"] = 1.0
    with pytest.raises(ValueError, match="forbidden fields"):
        contract.enforce(frame)


def test_contract_refuses_an_exempt_name_that_does_not_advertise_the_lag(monkeypatch):
    """The justification alone is not enough either; the name must show the lag.

    Guards against a typo in the exemption list silently admitting a field that
    describes the run being predicted.
    """
    monkeypatch.setitem(
        contract.LAGGED_FIELD_EXEMPTIONS, "static_duration_sec", "typo in the list"
    )
    frame = base_feature_frame()
    frame["static_duration_sec"] = 1.0
    with pytest.raises(ValueError, match="forbidden fields"):
        contract.enforce(frame)


# ---------------------------------------------------------------------------
# Precursor observability
# ---------------------------------------------------------------------------


def test_in_flight_predecessor_is_not_observable():
    """The previous run is often still running when the next one is triggered.

    This is the mechanism the baseline history features miss: `hist_prev_failed`
    reports an outcome that did not exist at prediction time. Here run 1 is
    still executing an hour after run 2 was queued.
    """
    frame = make_timed_runs(
        [
            {
                "run_number": 1,
                "created_at": BASE,
                "audit_updated_at": BASE + timedelta(hours=2),
                FAILURE_LABEL: 1,
            },
            {
                "run_number": 2,
                "created_at": BASE + timedelta(hours=1),
                "audit_updated_at": BASE + timedelta(hours=3),
                FAILURE_LABEL: 0,
            },
        ]
    )
    result = precursor_by_run(frame)
    second = result.loc["run-1"]
    assert second["hist_prev_completed_before_trigger"] == 0
    assert second["hist_prev_failed_observable"] == -1
    assert second["hist_prev_duration_sec"] == -1

    # The baseline history feature, by contrast, happily reports the outcome.
    leaky = history.build_history_features(frame).set_index("run_id")
    assert leaky.loc["run-1", "hist_prev_failed"] == 1


def test_completed_predecessor_yields_its_duration():
    frame = make_timed_runs(
        [
            {
                "run_number": 1,
                "created_at": BASE,
                "audit_updated_at": BASE + timedelta(minutes=10),
                FAILURE_LABEL: 1,
            },
            {
                "run_number": 2,
                "created_at": BASE + timedelta(hours=1),
                FAILURE_LABEL: 0,
            },
        ]
    )
    result = precursor_by_run(frame)
    second = result.loc["run-1"]
    assert second["hist_prev_completed_before_trigger"] == 1
    assert second["hist_prev_failed_observable"] == 1
    assert second["hist_prev_duration_sec"] == pytest.approx(600.0)


def test_duration_window_excludes_an_unfinished_earlier_run():
    """Per-element gating, not a gate on the immediate predecessor alone.

    Runs finish out of order: run 2 is a long one still executing, while run 1
    finished quickly. Run 3 therefore has no observable *immediate*
    predecessor, but one observable earlier run, so the window aggregate exists
    while the immediate-predecessor features do not. An `expanding()` window
    could not express this and would include the unfinished run.
    """
    frame = make_timed_runs(
        [
            {
                "run_number": 1,
                "created_at": BASE,
                "audit_updated_at": BASE + timedelta(minutes=30),
                FAILURE_LABEL: 0,
            },
            {
                "run_number": 2,
                "created_at": BASE + timedelta(minutes=10),
                "audit_updated_at": BASE + timedelta(hours=5),
                FAILURE_LABEL: 0,
            },
            {
                "run_number": 3,
                "created_at": BASE + timedelta(hours=1),
                FAILURE_LABEL: 1,
            },
        ]
    )
    third = precursor_by_run(frame).loc["run-2"]
    assert third["hist_prev_completed_before_trigger"] == 0
    assert third["hist_prev_duration_sec"] == -1
    assert third["hist_prior_duration_mean"] == pytest.approx(1800.0)


def test_observability_assertion_catches_an_ungated_lagged_value():
    """Simulate a missed `.where(observable)` and confirm the guard fires."""
    frame = make_timed_runs(
        [
            {
                "run_number": 1,
                "created_at": BASE,
                "audit_updated_at": BASE + timedelta(hours=2),
                FAILURE_LABEL: 1,
            },
            {
                "run_number": 2,
                "created_at": BASE + timedelta(hours=1),
                FAILURE_LABEL: 0,
            },
        ]
    )
    result = precursor.build_precursor_features(frame)
    precursor.assert_precursor_is_observable(result)

    result.loc[result["hist_prev_completed_before_trigger"] == 0, "hist_prev_duration_sec"] = 99.0
    with pytest.raises(AssertionError, match="no predecessor had completed"):
        precursor.assert_precursor_is_observable(result)


def test_observable_history_blanks_an_in_flight_predecessor():
    frame = make_timed_runs(
        [
            {
                "run_number": 1,
                "created_at": BASE,
                "audit_updated_at": BASE + timedelta(hours=2),
                FAILURE_LABEL: 1,
            },
            {
                "run_number": 2,
                "created_at": BASE + timedelta(hours=1),
                FAILURE_LABEL: 0,
            },
        ]
    )
    features = history.build_history_features(frame)
    pre = precursor.build_precursor_features(frame)
    honest = precursor.build_observable_history(features, pre).set_index("run_id")

    assert honest.loc["run-1", "hist_prev_failed"] == -1
    assert honest.loc["run-1", "hist_has_prev_run"] == 0
    assert honest.loc["run-1", "hist_prev_failure_streak"] == 0


def test_repository_window_counts_only_runs_that_had_finished():
    """A run still in flight has told the developer nothing yet.

    Workflow `c` starts before the target but finishes long after it, so it
    must not appear in the target's 24-hour window even though it started
    earlier.
    """
    frame = make_timed_runs(
        [
            {
                "workflow_path": "a.yml",
                "run_number": 1,
                "created_at": BASE,
                "audit_updated_at": BASE + timedelta(minutes=10),
                FAILURE_LABEL: 1,
            },
            {
                "workflow_path": "c.yml",
                "run_number": 1,
                "created_at": BASE + timedelta(minutes=30),
                "audit_updated_at": BASE + timedelta(hours=5),
                FAILURE_LABEL: 1,
            },
            {
                "workflow_path": "b.yml",
                "run_number": 1,
                "created_at": BASE + timedelta(hours=1),
                FAILURE_LABEL: 0,
            },
        ]
    )
    result = precursor_by_run(frame)
    target = result.loc["run-2"]
    assert target["hist_repo_runs_prev_24h"] == 1
    assert target["hist_repo_failures_prev_24h"] == 1
    assert target["hist_repo_failure_rate_prev_24h"] == pytest.approx(1.0)
    # The newest completed run of another workflow failed, so the sibling
    # signal fires.
    assert target["hist_sibling_workflow_failed"] == 1

    # The first run in the repository has no completed predecessor at all.
    first = result.loc["run-0"]
    assert first["hist_repo_runs_prev_24h"] == 0
    assert first["hist_sibling_workflow_failed"] == -1


def test_adjacent_depth_resets_on_a_run_number_gap():
    """Depth measures the unbroken chain, which the five-run cap keeps short."""
    frame = make_timed_runs(
        [
            {"run_number": 1, FAILURE_LABEL: 0},
            {"run_number": 2, FAILURE_LABEL: 0},
            {"run_number": 3, FAILURE_LABEL: 0},
            {"run_number": 10, FAILURE_LABEL: 0},
        ]
    )
    result = precursor_by_run(frame)
    assert result["hist_adjacent_depth"].tolist() == [0, 1, 2, 0]


def test_precursor_requires_the_audit_columns():
    """Fail with a clear message rather than a KeyError deep in the pass."""
    frame = make_timed_runs([{FAILURE_LABEL: 0}]).drop(columns=["audit_updated_at"])
    with pytest.raises(ValueError, match="absent columns"):
        precursor.build_precursor_features(frame)


def test_identifier_columns_are_redacted_for_addresses_only():
    """An audit of real output found a contributor's email used as a branch name.

    Identifier columns must still have addresses and credentials removed, but
    a full redaction pass would rewrite `renovate/lodash-4.x` and destroy the
    strongest static signal in the dataset.
    """
    assert textproc.redact_identifiers("fix/alice@example.com/retry") == "fix/<EMAIL>/retry"
    assert textproc.redact_identifiers("renovate/lodash-4.x") == "renovate/lodash-4.x"
    # A URL-like ref is left intact, unlike in prose mode.
    assert "<URL>" not in textproc.normalise("feature/http-client", 100, mode="identifier")
    with pytest.raises(ValueError, match="unknown redaction mode"):
        textproc.normalise("x", 100, mode="bogus")
