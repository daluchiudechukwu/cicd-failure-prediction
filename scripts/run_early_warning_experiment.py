#!/usr/bin/env python3
"""Does CI failure have early warning signals, and could you act on them?

`run_baseline_experiment.py` established that pre-execution prediction is
possible at all. This script tests the three further claims an AIOps framing
adds on top of feasibility, each of which can fail independently:

1. **"Patterns from thousands of historical builds."** How many historical
   builds does a prediction actually get to see, and does having more of them
   improve anything? Reported as the distribution of history available at
   prediction time, and as PR-AUC stratified by the depth of the predecessor
   chain.

2. **"Early warning signals that precede failures."** Is there a leading
   indicator — durations drifting upward, cadence changing, a sibling workflow
   breaking first, the previous run needing a re-run — beyond the previous
   outcome? Measured as the uplift from the `precursor.py` features, with
   repository-level bootstrap intervals, in the regime where a warning would
   matter.

   This also quantifies a leakage mechanism the feature contract did not
   previously cover: the preceding run is often still executing when the next
   is triggered, so its outcome is not observable at prediction time. The
   `*_obs` conditions re-run the comparison with that gate applied.

3. **"Enabling preventive actions."** Discrimination is not actionability.
   What matters for a policy is the operating point, so this reports precision
   at fixed recall levels, the alert rate each implies, and the break-even
   cost ratio at which acting becomes net positive.

Usage:
    export PYTHONPATH=src
    python scripts/run_early_warning_experiment.py \
        --features data/features.parquet --runs data/runs_clean.parquet
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

from ghalogs import contract, precursor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LABEL = "failed"
GROUP = "repo"

# Recall levels at which to report precision. Chosen to span the plausible
# policy range, from a conservative gate that fires only on near-certainties
# to a sweep that tries to catch most failures.
RECALL_TARGETS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)

BOOTSTRAP_RESAMPLES = 200


# --------------------------------------------------------------------------
# Feature conditions
# --------------------------------------------------------------------------

def columns_for(frame: pd.DataFrame, condition: str, precursor_columns: list[str]) -> list[str]:
    """Resolve a condition name to its feature columns.

    `hist_*` is split into the baseline history block and the precursor block
    so the ablation is a column selection rather than a second pipeline run.
    """
    static = sorted(c for c in frame.columns if c.startswith("static_"))
    history = sorted(
        c for c in frame.columns if c.startswith("hist_") and c not in precursor_columns
    )
    pre = sorted(c for c in precursor_columns if c in frame.columns)

    if condition in ("both", "both_obs"):
        return static + history
    if condition == "both_pre":
        return static + history + pre
    if condition == "pre_only":
        # Static features plus the leading indicators but *without* any
        # previous-outcome feature, to isolate whether precursors stand alone.
        return static + [c for c in pre if "prev_failed" not in c]
    raise ValueError(condition)


def frame_for(frame: pd.DataFrame, observable: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Return the frame a condition trains on.

    `both_obs` uses the same feature set as `both` over a frame whose history
    columns are blanked wherever the predecessor had not finished. The
    difference between the two is the quantity of interest.
    """
    return observable if condition == "both_obs" else frame


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------

def prepare(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = frame[columns].copy()
    for column in data.columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].astype("category")
    return data


def out_of_fold_scores(
    frame: pd.DataFrame, columns: list[str], n_splits: int, seed: int, label: str
) -> np.ndarray:
    """Repository-grouped out-of-fold probabilities from one LightGBM family.

    Hyperparameters match `run_baseline_experiment.py`, so every comparison
    here isolates the inputs rather than the architecture.
    """
    import lightgbm as lgb

    features = prepare(frame, columns)
    labels = frame[LABEL].to_numpy()
    groups = frame[GROUP].to_numpy()
    scores = np.full(len(frame), np.nan)

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(features, labels, groups), 1
    ):
        train_groups = groups[train_idx]
        unique_groups = np.unique(train_groups)
        rng = np.random.default_rng(seed + fold)
        validation = set(
            rng.choice(unique_groups, size=max(1, len(unique_groups) // 5), replace=False)
        )
        is_validation = np.isin(train_groups, list(validation))
        fit_idx = train_idx[~is_validation]
        val_idx = train_idx[is_validation]

        model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            features.iloc[fit_idx],
            labels[fit_idx],
            eval_set=[(features.iloc[val_idx], labels[val_idx])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        scores[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
        print(f"  {label}: fold {fold}/{n_splits} done", flush=True)

    return scores


def straw_scores(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Copy the previous outcome; abstain at the base rate when unavailable."""
    scores = np.full(len(frame), float(frame[LABEL].mean()))
    previous = frame[column].to_numpy()
    scores[previous == 1] = 1.0
    scores[previous == 0] = 0.0
    return scores


# --------------------------------------------------------------------------
# Analysis 1 — how much history is there, and does it help?
# --------------------------------------------------------------------------

def analyse_history_depth(frame: pd.DataFrame, score_column: str) -> dict:
    """Quantify available history, then test whether more of it helps.

    Stratification is restricted to regime B. In regime C the previous run
    already failed, so depth is confounded with an almost-determined label; in
    regime A there is no history to have a depth.
    """
    per_workflow = frame.groupby(["repo", "workflow_path"]).size()
    per_repo = frame.groupby("repo").size()

    summary = {
        "runs_per_workflow": {
            "mean": float(per_workflow.mean()),
            "median": float(per_workflow.median()),
            "max": int(per_workflow.max()),
        },
        "runs_per_repository": {
            "mean": float(per_repo.mean()),
            "median": float(per_repo.median()),
            "p95": float(per_repo.quantile(0.95)),
            "max": int(per_repo.max()),
        },
        # The honest answer to "how many historical builds does the model see?"
        # is the number of prior runs available at prediction time, not the
        # size of the dataset.
        "prior_runs_at_prediction_time": {
            "workflow_scope_median": float(frame["hist_wf_prior_runs"].median()),
            "workflow_scope_mean": float(frame["hist_wf_prior_runs"].mean()),
            "repo_scope_median": float(frame["hist_repo_prior_runs"].median()),
            "repo_scope_mean": float(frame["hist_repo_prior_runs"].mean()),
            "repo_scope_p95": float(frame["hist_repo_prior_runs"].quantile(0.95)),
            "repo_scope_share_over_100": float((frame["hist_repo_prior_runs"] > 100).mean()),
            "repo_scope_share_over_1000": float((frame["hist_repo_prior_runs"] > 1000).mean()),
        },
    }

    rows = []
    regime_b = frame[frame["regime"] == "B_prev_success"]
    for depth, block in regime_b.groupby("hist_adjacent_depth", observed=True):
        if len(block) < 500:
            continue
        labels = block[LABEL].to_numpy()
        scores = block[score_column].to_numpy()
        if labels.min() == labels.max():
            continue
        base = float(labels.mean())
        average_precision = float(average_precision_score(labels, scores))
        rows.append(
            {
                "adjacent_depth": int(depth),
                "n": int(len(block)),
                "base_rate": base,
                "pr_auc": average_precision,
                "pr_auc_lift": average_precision / base if base else float("nan"),
                "roc_auc": float(roc_auc_score(labels, scores)),
            }
        )
    summary["skill_by_depth_regime_b"] = rows
    return summary


# --------------------------------------------------------------------------
# Analysis 2 — is the predecessor even observable?
# --------------------------------------------------------------------------

def analyse_observability(frame: pd.DataFrame) -> dict:
    """How often is the previous run still in flight when the next is queued?

    Restricted to runs that have an adjacent predecessor at all, since that is
    the population for which a previous-outcome feature claims to exist.
    """
    has_prev = frame["hist_has_prev_run"] == 1
    population = frame[has_prev]
    observable = population["hist_prev_completed_before_trigger"] == 1
    unobservable = population[~observable]

    return {
        "runs_with_adjacent_predecessor": int(len(population)),
        "share_of_all_runs": float(has_prev.mean()),
        "predecessor_still_running_at_trigger": int((~observable).sum()),
        "share_unobservable": float((~observable).mean()),
        "share_unobservable_of_all_runs": float(
            (~observable).sum() / len(frame) if len(frame) else float("nan")
        ),
        # If the in-flight cases were benign these two would match.
        "failure_rate_when_observable": float(population.loc[observable, LABEL].mean()),
        "failure_rate_when_unobservable": float(unobservable[LABEL].mean()),
        "regime_mix_when_unobservable": {
            str(k): int(v) for k, v in unobservable["regime"].value_counts().items()
        },
    }


# --------------------------------------------------------------------------
# Analysis 3 — actionability
# --------------------------------------------------------------------------

def operating_points(labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Precision and alert rate at each target recall.

    Reported this way round — fix recall, read off precision — because a team
    choosing a policy starts from "what fraction of failures must we catch?",
    not from a probability threshold.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one more point than thresholds; drop the
    # trailing (recall=0, precision=1) sentinel so the indices line up.
    precision, recall = precision[:-1], recall[:-1]

    rows = []
    for target in RECALL_TARGETS:
        feasible = np.flatnonzero(recall >= target)
        if len(feasible) == 0:
            continue
        # Among thresholds achieving the target recall, take the most precise.
        best = feasible[np.argmax(precision[feasible])]
        threshold = float(thresholds[best])
        flagged = scores >= threshold
        achieved = float(precision[best])
        rows.append(
            {
                "target_recall": target,
                "achieved_recall": float(recall[best]),
                "precision": achieved,
                "threshold": threshold,
                "alert_rate": float(flagged.mean()),
                "alerts_per_1000_runs": float(1000 * flagged.mean()),
                "true_catches_per_1000_runs": float(1000 * (flagged & (labels == 1)).mean()),
                # Acting is net positive only if a correct warning is worth at
                # least this multiple of what a false alarm costs. From
                # p*V >= (1-p)*C, so V/C >= (1-p)/p.
                "false_alarms_per_true_catch": (
                    float((1 - achieved) / achieved) if achieved > 0 else float("inf")
                ),
            }
        )
    return rows


def cost_context(runs: pd.DataFrame) -> dict:
    """Observed run durations, so operating points can be read in minutes.

    Durations appear only here, for the cost model. They are never features;
    `precursor.py` admits them solely in lagged, completion-gated form.

    Percentiles matter more than the mean: the distribution is extremely
    right-skewed, so a mean duration implies a typical run far longer than
    most, and a saving computed from it would be fiction.
    """
    duration = (runs["audit_updated_at"] - runs["created_at"]).dt.total_seconds().div(60.0)
    usable = duration[duration.notna() & (duration >= 0)]
    failed = usable[runs.loc[usable.index, LABEL] == 1]
    passed = usable[runs.loc[usable.index, LABEL] == 0]

    def spread(values: pd.Series) -> dict:
        return {
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "p99": float(values.quantile(0.99)),
            "mean": float(values.mean()),
        }

    total = float(usable.sum())
    return {
        "minutes_all": spread(usable),
        "minutes_failed": spread(failed),
        "minutes_passed": spread(passed),
        "share_under_5_minutes": float((usable < 5).mean()),
        "share_under_10_minutes": float((usable < 10).mean()),
        "total_hours_all_runs": total / 60.0,
        "total_hours_failed_runs": float(failed.sum()) / 60.0,
        "failed_share_of_total_minutes": float(failed.sum() / total) if total else float("nan"),
    }


def precursor_univariate(frame: pd.DataFrame, precursor_columns: list[str]) -> list[dict]:
    """Univariate discrimination of each precursor feature, within regime B.

    The direct test of "are there early warning signals?". A leading indicator
    worth the name should separate the classes on its own, among runs whose
    previous run passed. Coverage is reported alongside, because a feature that
    discriminates well on 3% of rows is not a usable warning.

    ROC-AUC is folded to >= 0.5 so the number reads as discrimination
    regardless of sign; the direction is in `mean_when_failed` versus
    `mean_when_passed`.
    """
    regime_b = frame[frame["regime"] == "B_prev_success"]
    labels = regime_b[LABEL].to_numpy()
    rows = []
    for column in sorted(precursor_columns):
        values = regime_b[column]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        numeric = values.to_numpy(dtype="float64")
        present = numeric != -1.0
        if np.unique(numeric).size < 2:
            continue
        auc = roc_auc_score(labels, numeric)
        rows.append(
            {
                "feature": column,
                "roc_auc": float(max(auc, 1.0 - auc)),
                "coverage": float(present.mean()),
                "mean_when_failed": float(numeric[labels == 1].mean()),
                "mean_when_passed": float(numeric[labels == 0].mean()),
            }
        )
    return sorted(rows, key=lambda row: -row["roc_auc"])


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def bootstrap_delta(
    frame: pd.DataFrame,
    column_a: str,
    column_b: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Repository-level bootstrap interval for PR-AUC(b) - PR-AUC(a).

    Resampling repositories rather than runs, because runs are nested inside
    repositories and a run-level interval would be far too narrow: the
    between-repository standard deviation in failure rate is 18.4%.
    """
    repos = frame[GROUP].to_numpy()
    unique_repos = np.unique(repos)
    order = np.argsort(repos, kind="mergesort")
    sorted_repos = repos[order]
    starts = np.searchsorted(sorted_repos, unique_repos, side="left")
    ends = np.searchsorted(sorted_repos, unique_repos, side="right")
    index_by_repo = {
        repo: order[start:end] for repo, start, end in zip(unique_repos, starts, ends)
    }

    labels = frame[LABEL].to_numpy()
    scores_a = frame[column_a].to_numpy()
    scores_b = frame[column_b].to_numpy()

    point_a = float(average_precision_score(labels, scores_a))
    point_b = float(average_precision_score(labels, scores_b))

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        drawn = rng.choice(unique_repos, size=len(unique_repos), replace=True)
        index = np.concatenate([index_by_repo[repo] for repo in drawn])
        y = labels[index]
        if y.min() == y.max():
            continue
        deltas.append(
            average_precision_score(y, scores_b[index])
            - average_precision_score(y, scores_a[index])
        )

    deltas = np.asarray(deltas)
    return {
        "pr_auc_a": point_a,
        "pr_auc_b": point_b,
        "delta_point": point_b - point_a,
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "share_of_resamples_positive": float((deltas > 0).mean()),
        "resamples": int(len(deltas)),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def regime_table(frame: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    rows = []
    for condition in conditions:
        column = f"score_{condition}"
        for regime in ["ALL", "A_cold_start", "B_prev_success", "C_prev_failure"]:
            mask = (
                np.ones(len(frame), dtype=bool)
                if regime == "ALL"
                else (frame["regime"] == regime).to_numpy()
            )
            labels = frame.loc[mask, LABEL].to_numpy()
            scores = frame.loc[mask, column].to_numpy()
            if len(labels) < 100 or labels.min() == labels.max():
                continue
            base = float(labels.mean())
            average_precision = float(average_precision_score(labels, scores))
            rows.append(
                {
                    "condition": condition,
                    "regime": regime,
                    "n": int(len(labels)),
                    "base_rate": base,
                    "pr_auc": average_precision,
                    "pr_auc_lift": average_precision / base if base else float("nan"),
                    "roc_auc": float(roc_auc_score(labels, scores)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--runs", type=Path, default=Path("data/runs_clean.parquet"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-repos", type=int, default=0, help="repository cap for a smoke run"
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    runs = pd.read_parquet(args.runs)

    # Sampling happens before feature building rather than after. Every
    # precursor feature is computed within a repository, so restricting the
    # repository set first changes nothing about the values and makes a smoke
    # run take seconds instead of minutes.
    if args.sample_repos:
        chosen = (
            frame[GROUP]
            .drop_duplicates()
            .sample(n=min(args.sample_repos, frame[GROUP].nunique()), random_state=args.seed)
        )
        frame = frame[frame[GROUP].isin(chosen)].reset_index(drop=True)
        runs = runs[runs[GROUP].isin(chosen)].reset_index(drop=True)

    print("building precursor features ...", flush=True)
    pre = precursor.build_precursor_features(runs)
    precursor.assert_precursor_is_observable(pre)
    print(f"  {len(pre.columns) - 1} precursor features, observability assertions passed")

    precursor_columns = [c for c in pre.columns if c != "run_id"]
    frame = frame.merge(pre, on="run_id", how="left", validate="one_to_one")

    # The contract runs over the merged frame, so the lagged exemptions are
    # exercised on real data rather than only in the unit tests.
    report = contract.enforce(frame, strict=True)
    print(report.describe())

    observable = precursor.build_observable_history(frame, pre)

    print(
        f"\nrows: {len(frame):,}  repositories: {frame[GROUP].nunique():,}  "
        f"failure rate: {100 * frame[LABEL].mean():.2f}%"
    )

    conditions = ["both", "both_obs", "both_pre", "pre_only"]
    for condition in conditions:
        columns = columns_for(frame, condition, precursor_columns)
        source = frame_for(frame, observable, condition)
        print(f"\n{condition}: {len(columns)} features", flush=True)
        frame[f"score_{condition}"] = out_of_fold_scores(
            source, columns, args.n_splits, args.seed, condition
        )

    frame["score_straw"] = straw_scores(frame, "hist_prev_failed")
    frame["score_straw_obs"] = straw_scores(frame, "hist_prev_failed_observable")

    args.out.mkdir(parents=True, exist_ok=True)

    table = regime_table(frame, conditions + ["straw", "straw_obs"])
    table.to_csv(args.out / "early_warning_regimes.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== PR-AUC by regime and condition ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    regime_b = frame[frame["regime"] == "B_prev_success"].reset_index(drop=True)

    results = {
        "population": {
            "runs": int(len(frame)),
            "repositories": int(frame[GROUP].nunique()),
            "failure_rate": float(frame[LABEL].mean()),
            "n_splits": args.n_splits,
        },
        "history_depth": analyse_history_depth(frame, "score_both_pre"),
        "observability": analyse_observability(frame),
        "precursor_uplift_regime_b": bootstrap_delta(
            regime_b, "score_both", "score_both_pre", seed=args.seed
        ),
        "precursor_uplift_all": bootstrap_delta(
            frame, "score_both", "score_both_pre", seed=args.seed
        ),
        "observable_history_cost_all": bootstrap_delta(
            frame, "score_both_obs", "score_both", seed=args.seed
        ),
        "precursor_univariate_regime_b": precursor_univariate(frame, precursor_columns),
        "cost_context": cost_context(runs),
        "operating_points_regime_b": operating_points(
            regime_b[LABEL].to_numpy(), regime_b["score_both_pre"].to_numpy()
        ),
        "operating_points_all": operating_points(
            frame[LABEL].to_numpy(), frame["score_both_pre"].to_numpy()
        ),
        "regime_table": table.to_dict(orient="records"),
    }

    with open(args.out / "early_warning_results.json", "w") as handle:
        json.dump(results, handle, indent=2)

    print("\n=== history available at prediction time ===")
    print(json.dumps(results["history_depth"], indent=2))
    print("\n=== predecessor observability ===")
    print(json.dumps(results["observability"], indent=2))
    print("\n=== precursor features, univariate discrimination in regime B ===")
    print(
        pd.DataFrame(results["precursor_univariate_regime_b"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print("\n=== precursor uplift, regime B (repository bootstrap) ===")
    print(json.dumps(results["precursor_uplift_regime_b"], indent=2))
    print("\n=== cost of honest (observable-only) history, pooled ===")
    print(json.dumps(results["observable_history_cost_all"], indent=2))
    print("\n=== operating points, regime B ===")
    print(
        pd.DataFrame(results["operating_points_regime_b"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print("\n=== run durations, for the cost model ===")
    print(json.dumps(results["cost_context"], indent=2))
    print(f"\nwrote {args.out / 'early_warning_results.json'}")


if __name__ == "__main__":
    main()
