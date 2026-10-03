"""Stage 4a — static feature engineering.

"Static" means: computable from the trigger event and the repository's
slow-moving attributes alone, with no reference to any previous run. These are
the features that remain available when a workflow has no history, which is
25.5% of the dataset and the regime where outcome-copying heuristics have zero
skill.

Four families, each with a stated rationale:

1. **Branch-name shape and intent.** Measured failure rates range from 12.9%
   on `main`/`master` to 27.4% on `renovate/` branches, so branch naming is
   directly predictive. The keyword flags also constitute the hand-written
   baseline that learned text models must beat.
2. **Commit-message shape.** Length, structure, and conventional-commit
   conformance are proxies for change discipline. Deliberately shape-based
   rather than semantic, so that the transformer models have somewhere to add
   value by reading the actual words.
3. **Trigger context.** Event type separates failure rates from 2.5%
   (`dynamic`) to 20.9% (`workflow_dispatch`). Fork, bot, and PR provenance
   capture who is proposing the change and under what trust rules.
4. **Workflow identity.** Essential, not optional: the same commit frequently
   passes one workflow and fails another, so a model blind to which workflow
   is running is bounded well away from perfect accuracy.

No feature here reads a log, a run duration, or a crawl-time repository
counter. Compliance is asserted by `contract.py`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import (
    BRANCH_MARKERS,
    DEFAULT_BRANCH_NAMES,
    MESSAGE_MARKERS,
    NON_CODE_CHANGE_EVENTS,
    WORKFLOW_MARKERS,
)

VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
ISSUE_REF_RE = re.compile(r"#\d+")
CONVENTIONAL_RE = re.compile(r"^[a-z]+(?:\([^)]*\))?!?:\s")
TICKET_RE = re.compile(r"\b[A-Z]{2,10}-\d+\b")


def _keyword_flags(values: pd.Series, markers: tuple[str, ...], prefix: str) -> pd.DataFrame:
    """One binary column per marker, matched case-insensitively as a substring.

    Substring rather than token matching is intentional: `fix` should fire on
    `hotfixes` and `bugfix/`, and branch names are rarely tokenised cleanly.
    """
    lowered = values.fillna("").astype(str).str.lower()
    return pd.DataFrame(
        {f"{prefix}{marker}": lowered.str.contains(marker, regex=False).astype("int8")
         for marker in markers},
        index=values.index,
    )


def branch_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Shape and intent signals from the ref that triggered the run."""
    branch = runs["head_branch"].fillna("").astype(str)
    lowered = branch.str.lower()
    default_branch = runs.get("default_branch", pd.Series("", index=runs.index)).fillna("")

    out = pd.DataFrame(index=runs.index)
    out["static_branch_len"] = branch.str.len().astype("int16")
    # Depth distinguishes `main` from `feature/team/JIRA-123/retry-logic`:
    # deeper refs tend to be topic branches rather than integration branches.
    out["static_branch_depth"] = branch.str.count("/").astype("int8")
    out["static_branch_n_tokens"] = (
        branch.str.count(r"[-_/]").add(1).astype("int8")
    )
    out["static_branch_is_default"] = (
        lowered.isin(DEFAULT_BRANCH_NAMES) | branch.eq(default_branch)
    ).astype("int8")
    out["static_branch_has_digits"] = branch.str.contains(r"\d", regex=True).astype("int8")
    out["static_branch_has_version"] = lowered.str.contains(VERSION_RE).astype("int8")
    out["static_branch_has_ticket"] = branch.str.contains(TICKET_RE).astype("int8")
    out["static_branch_is_uppercase_heavy"] = (
        branch.str.count(r"[A-Z]").div(branch.str.len().clip(lower=1)) > 0.3
    ).astype("int8")
    return pd.concat([out, _keyword_flags(branch, BRANCH_MARKERS, "static_branch_kw_")], axis=1)


def message_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Structural properties of the commit message.

    Shape only. Reading the semantics is left to the text models, so that the
    ablation in RQ2 measures something real rather than re-encoding the same
    information twice.
    """
    message = runs["commit_message"].fillna("").astype(str)
    lowered = message.str.lower()
    subject = message.str.split("\n").str[0].fillna("")

    out = pd.DataFrame(index=runs.index)
    out["static_msg_len"] = message.str.len().clip(upper=5000).astype("int32")
    out["static_msg_raw_len"] = (
        runs.get("raw_commit_message_chars", pd.Series(0, index=runs.index))
        .fillna(0)
        .clip(upper=100_000)
        .astype("int32")
    )
    out["static_msg_was_truncated"] = message.str.endswith("<TRUNC>").astype("int8")
    out["static_msg_subject_len"] = subject.str.len().clip(upper=500).astype("int16")
    out["static_msg_n_lines"] = message.str.count("\n").add(1).astype("int16")
    out["static_msg_n_words"] = message.str.split().str.len().fillna(0).astype("int16")
    # A blank line after the subject is the git convention for a structured
    # message, and a weak proxy for care taken over the change.
    out["static_msg_has_body"] = message.str.contains("\n\n", regex=False).astype("int8")
    out["static_msg_is_conventional"] = lowered.str.contains(CONVENTIONAL_RE).astype("int8")
    out["static_msg_is_merge"] = lowered.str.startswith("merge").astype("int8")
    out["static_msg_is_revert"] = lowered.str.startswith("revert").astype("int8")
    out["static_msg_has_issue_ref"] = message.str.contains(ISSUE_REF_RE).astype("int8")
    out["static_msg_has_ticket"] = message.str.contains(TICKET_RE).astype("int8")
    out["static_msg_has_version"] = lowered.str.contains(VERSION_RE).astype("int8")
    out["static_msg_has_breaking_marker"] = message.str.contains(
        r"BREAKING[ -]CHANGE|!:", regex=True
    ).astype("int8")
    # Redaction markers survive as features: "mentions a URL" is plausible
    # signal (dependency bumps cite changelogs) while the URL itself is not.
    out["static_msg_has_url_marker"] = message.str.contains("<URL>", regex=False).astype("int8")
    out["static_msg_has_email_marker"] = message.str.contains(
        "<EMAIL>", regex=False
    ).astype("int8")
    out["static_msg_has_sha_marker"] = message.str.contains("<SHA>", regex=False).astype("int8")
    out["static_msg_has_coauthor"] = lowered.str.contains(
        "co-authored-by", regex=False
    ).astype("int8")
    out["static_msg_has_signoff"] = lowered.str.contains(
        "signed-off-by", regex=False
    ).astype("int8")
    out["static_msg_is_terse"] = (out["static_msg_n_words"] <= 2).astype("int8")
    return pd.concat([out, _keyword_flags(message, MESSAGE_MARKERS, "static_msg_kw_")], axis=1)


def workflow_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Which workflow is running.

    Load-bearing rather than incidental. 69.9% of runs share a commit with
    another run and 47.7% of failures share a commit with a success, so the
    difference between those runs is the workflow, not the change. A model
    without these features has an error floor of 6.42% on a task with a 15.88%
    base rate.
    """
    name = runs["workflow_name"].fillna("").astype(str)
    path = runs["workflow_path"].fillna("").astype(str)

    out = pd.DataFrame(index=runs.index)
    out["static_workflow_name_len"] = name.str.len().clip(upper=200).astype("int16")
    out["static_workflow_path_depth"] = path.str.count("/").astype("int8")
    out["static_workflow_is_reusable_caller"] = (
        runs["n_referenced_workflows"].fillna(0) > 0
    ).astype("int8")
    out["static_workflow_n_referenced"] = (
        runs["n_referenced_workflows"].fillna(0).clip(upper=20).astype("int8")
    )
    # Run number proxies how established this workflow is, and is known before
    # the run starts because GitHub assigns it at trigger time.
    out["static_run_number"] = runs["run_number"].fillna(0).clip(upper=100_000).astype("int32")
    out["static_run_number_log"] = np.log1p(out["static_run_number"]).astype("float32")
    return pd.concat(
        [out, _keyword_flags(name + " " + path, WORKFLOW_MARKERS, "static_wf_kw_")], axis=1
    )


def trigger_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Event type, actor provenance, and timing."""
    created = pd.to_datetime(runs["created_at"], utc=True)

    out = pd.DataFrame(index=runs.index)
    # Timing. Interpreted with care: UTC does not align with any single
    # author's working day, so these are weak signals and are included mainly
    # to be tested rather than believed.
    out["static_hour_utc"] = created.dt.hour.astype("int8")
    out["static_weekday"] = created.dt.weekday.astype("int8")
    out["static_is_weekend"] = (created.dt.weekday >= 5).astype("int8")
    out["static_is_office_hours_utc"] = created.dt.hour.between(9, 17).astype("int8")
    # Cyclical encoding so that 23:00 and 00:00 are adjacent for the linear
    # and neural models; trees are indifferent but unharmed.
    out["static_hour_sin"] = np.sin(2 * np.pi * created.dt.hour / 24).astype("float32")
    out["static_hour_cos"] = np.cos(2 * np.pi * created.dt.hour / 24).astype("float32")

    # Provenance of the change.
    out["static_is_bot"] = (
        runs.get("actor_login_is_bot", pd.Series(0, index=runs.index)).fillna(0)
        | runs["actor_type"].eq("Bot").astype("int8")
    ).astype("int8")
    out["static_owner_is_org"] = runs["owner_type"].eq("Organization").astype("int8")
    out["static_head_repo_is_fork"] = runs["head_repo_is_fork"].fillna(False).astype("int8")
    out["static_is_cross_repo"] = (
        runs["head_repo_id"].fillna(-1) != runs["base_repo_id"].fillna(-1)
    ).astype("int8")
    out["static_has_pull_request"] = (runs["n_pull_requests"].fillna(0) > 0).astype("int8")
    out["static_n_pull_requests"] = (
        runs["n_pull_requests"].fillna(0).clip(upper=10).astype("int8")
    )
    # A re-run is known to be a re-run before it starts, so this is admissible.
    out["static_is_rerun"] = (runs["run_attempt"].fillna(1) > 1).astype("int8")
    out["static_run_attempt"] = runs["run_attempt"].fillna(1).clip(upper=6).astype("int8")

    # Whether the trigger can involve a code change at all. Roughly a fifth of
    # runs cannot, and for those the commit text is identical to the previous
    # run's, so commit-derived features carry no new information. Modelling
    # this explicitly prevents the confound.
    out["static_is_code_change_event"] = (
        ~runs["event"].isin(NON_CODE_CHANGE_EVENTS)
    ).astype("int8")

    # Lag between authoring the commit and the run being queued: a long lag
    # suggests a rebase, a reverted branch, or a delayed trigger.
    commit_ts = pd.to_datetime(runs["commit_timestamp"], utc=True, errors="coerce")
    lag = (created - commit_ts).dt.total_seconds().div(60.0)
    out["static_commit_to_trigger_mins"] = lag.clip(lower=-60, upper=10_080).fillna(-1).astype(
        "float32"
    )

    # Categoricals left as strings; the modelling code declares them as
    # pandas `category` dtype so LightGBM handles them natively and one-hot
    # encoding is only applied for the linear models.
    out["static_event"] = runs["event"].fillna("unknown").astype(str)
    out["static_actor_type"] = runs["actor_type"].fillna("unknown").astype(str)
    return out


def repository_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Slow-moving repository attributes only.

    Repository *age at the moment of the run* is derived by differencing the
    creation timestamp against the run's own timestamp, which is legitimate
    because creation time is fixed. Every crawl-time counter is excluded, so
    there is no star count, fork count, commit count, or `total_runs_90d`
    here. See `docs/02-feature-contract.md` section 4.2.
    """
    created = pd.to_datetime(runs["created_at"], utc=True)
    repo_created = pd.to_datetime(runs.get("repo_created_at"), utc=True, errors="coerce")

    out = pd.DataFrame(index=runs.index)
    age_days = (created - repo_created).dt.days
    out["static_repo_age_days"] = age_days.fillna(-1).clip(lower=-1, upper=10_000).astype("int32")
    out["static_repo_age_log"] = np.log1p(age_days.fillna(0).clip(lower=0)).astype("float32")
    out["static_repo_has_license"] = (
        runs.get("has_license", pd.Series(False, index=runs.index)).fillna(False).astype("int8")
    )
    out["static_repo_has_wiki"] = (
        runs.get("has_wiki", pd.Series(False, index=runs.index)).fillna(False).astype("int8")
    )
    out["static_repo_n_topics"] = (
        runs.get("n_topics", pd.Series(0, index=runs.index)).fillna(0).clip(upper=25).astype("int8")
    )
    out["static_repo_desc_len"] = (
        runs["repo_description"].fillna("").astype(str).str.len().clip(upper=500).astype("int16")
    )
    out["static_language"] = (
        runs.get("language", pd.Series("unknown", index=runs.index)).fillna("unknown").astype(str)
    )
    return out


def build_static_features(runs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate every static family into one frame."""
    blocks = [
        branch_features(runs),
        message_features(runs),
        workflow_features(runs),
        trigger_features(runs),
        repository_features(runs),
    ]
    features = pd.concat(blocks, axis=1)
    duplicated = features.columns[features.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"duplicate static feature names: {duplicated}")
    return features
