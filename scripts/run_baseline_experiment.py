#!/usr/bin/env python3
"""Feasibility experiment: can pre-execution metadata alone predict failure?

Compares four input conditions under one repository-grouped protocol, so that
no repository appears in both train and test:

    static   pre-execution signals needing no execution history
    hist     execution-history features only
    both     static + history
    straw    copy the previous run's outcome (no training)

Reported per prediction regime as well as pooled, because the regimes have
failure rates of roughly 18%, 5%, and 69% and a pooled number hides which
problem is actually being solved.

Primary metric is average precision (PR-AUC) on the failure class. Accuracy is
reported only for comparability with prior work; at a 15.9% base rate it is
dominated by the majority class.

Usage:
    python scripts/run_baseline_experiment.py --features data/features.parquet
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

CATEGORICAL = ["static_event", "static_language"]
LABEL = "failed"
GROUP = "repo"


def feature_columns(frame: pd.DataFrame, condition: str) -> list[str]:
    static = [c for c in frame.columns if c.startswith("static_")]
    history = [c for c in frame.columns if c.startswith("hist_")]
    if condition == "static":
        return static
    if condition == "hist":
        return history
    if condition == "both":
        return static + history
    raise ValueError(condition)


def prepare(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = frame[columns].copy()
    for column in CATEGORICAL:
        if column in data.columns:
            data[column] = data[column].astype("category")
    return data


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Score a set of predictions. `scores` are probabilities in [0, 1]."""
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    base_rate = float(y_true.mean())
    average_precision = float(average_precision_score(y_true, scores))
    return {
        "n": int(len(y_true)),
        "base_rate": base_rate,
        "pr_auc": average_precision,
        # How many times better than guessing at the base rate.
        "pr_auc_lift": average_precision / base_rate if base_rate else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores)) if 0 < base_rate < 1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((predictions == y_true).mean()),
        "brier": float(brier_score_loss(y_true, scores)),
        "threshold": float(threshold),
    }


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising failure-class F1, chosen on validation data only."""
    candidates = np.quantile(scores, np.linspace(0.50, 0.999, 120))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.unique(candidates):
        predictions = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, predictions, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def run_condition(frame: pd.DataFrame, condition: str, n_splits: int, seed: int) -> pd.DataFrame:
    import lightgbm as lgb

    columns = feature_columns(frame, condition)
    features = prepare(frame, columns)
    labels = frame[LABEL].to_numpy()
    groups = frame[GROUP].to_numpy()

    out_of_fold = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels, groups), 1):
        # Carve a validation slice out of training, also grouped by repository,
        # so the threshold is never chosen on test data.
        train_groups = frame[GROUP].to_numpy()[train_idx]
        unique_groups = np.unique(train_groups)
        rng = np.random.default_rng(seed + fold)
        validation_groups = set(
            rng.choice(unique_groups, size=max(1, len(unique_groups) // 5), replace=False)
        )
        is_validation = np.isin(train_groups, list(validation_groups))
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
            eval_X=features.iloc[val_idx],
            eval_y=labels[val_idx],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        out_of_fold[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
        print(f"  {condition}: fold {fold}/{n_splits} done", flush=True)

    frame = frame.assign(**{f"score_{condition}": out_of_fold})
    return frame


def straw_man_scores(frame: pd.DataFrame) -> np.ndarray:
    """Copy the previous run's outcome; abstain at the base rate when absent."""
    scores = np.full(len(frame), float(frame[LABEL].mean()))
    previous = frame["hist_prev_failed"].to_numpy()
    scores[previous == 1] = 1.0
    scores[previous == 0] = 0.0
    return scores


def report(frame: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    rows = []
    # One threshold per condition, fitted on a 20% repository slice held out
    # from reporting, mimicking deployment-time threshold selection.
    repos = frame[GROUP].unique()
    rng = np.random.default_rng(0)
    calibration_repos = set(rng.choice(repos, size=max(1, len(repos) // 5), replace=False))
    is_calibration = frame[GROUP].isin(calibration_repos).to_numpy()

    for condition in conditions:
        column = f"score_{condition}"
        scores = frame[column].to_numpy()
        labels = frame[LABEL].to_numpy()
        threshold = pick_threshold(labels[is_calibration], scores[is_calibration])

        held = ~is_calibration
        rows.append({"condition": condition, "regime": "ALL", **evaluate(labels[held], scores[held], threshold)})
        for regime in sorted(frame["regime"].unique()):
            mask = held & (frame["regime"] == regime).to_numpy()
            if mask.sum() < 100:
                continue
            rows.append(
                {"condition": condition, "regime": regime, **evaluate(labels[mask], scores[mask], threshold)}
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0, help="optional row cap for a smoke run")
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    if args.sample:
        repos = frame[GROUP].drop_duplicates().sample(
            n=min(args.sample, frame[GROUP].nunique()), random_state=args.seed
        )
        frame = frame[frame[GROUP].isin(repos)].reset_index(drop=True)
    print(f"rows: {len(frame)}, repositories: {frame[GROUP].nunique()}, failure rate: {100 * frame[LABEL].mean():.2f}%")

    conditions = ["static", "hist", "both"]
    for condition in conditions:
        frame = run_condition(frame, condition, args.n_splits, args.seed)
    frame["score_straw"] = straw_man_scores(frame)

    results = report(frame, conditions + ["straw"])
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "feasibility_results.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    shown = ["condition", "regime", "n", "base_rate", "pr_auc", "pr_auc_lift", "roc_auc", "mcc", "precision", "recall", "f1", "accuracy"]
    print("\n=== results ===")
    print(results[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    with open(args.out / "feasibility_results.json", "w") as handle:
        json.dump(results.to_dict(orient="records"), handle, indent=2)
    print(f"\nwrote {args.out / 'feasibility_results.csv'}")


if __name__ == "__main__":
    main()
