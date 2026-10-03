"""Pipeline orchestration and command-line entry point.

Three stages, each writing an inspectable artefact so that a failure in one
does not require re-running the others:

    ingest    runs.json.gz / repositories.json.gz  ->  runs_raw.parquet, repositories.parquet
    prepare   runs_raw.parquet                     ->  runs_clean.parquet + filter_ledger.csv
    features  runs_clean.parquet                   ->  features.parquet + contract report

Run everything with:

    python -m ghalogs.pipeline all --data-dir <ghalogs> --out-dir data

The salt used to pseudonymise actor and author identifiers is read from the
GHALOGS_SALT environment variable. It must not be committed: without it the
pseudonyms cannot be reversed, which is the point of having it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from . import contract, features, history, ingest, quality, textproc
from .config import FAILURE_LABEL, PipelineConfig

DEFAULT_SALT_ENV = "GHALOGS_SALT"


def stage_ingest(data_dir: Path, out_dir: Path, config: PipelineConfig) -> None:
    ingest.ingest_repositories(
        data_dir / "repositories.json.gz", out_dir / "repositories.parquet", config
    )
    ingest.ingest_runs(data_dir / "runs.json.gz", out_dir / "runs_raw.parquet", config)


def stage_prepare(out_dir: Path, config: PipelineConfig, salt: str) -> pd.DataFrame:
    """Filter noise, annotate commit groups, and clean the text columns."""
    runs = ingest.load_runs(out_dir / "runs_raw.parquet")
    repositories = pd.read_parquet(out_dir / "repositories.parquet")

    runs, ledger = quality.filter_runs(runs, repositories, config.filters)
    print(ledger.report())
    ledger.to_frame().to_csv(out_dir / "filter_ledger.csv", index=False)

    runs = quality.annotate_commit_groups(runs)
    print()
    print(quality.quality_report(runs))

    # Join the contract-admitted repository attributes. A left join keeps the
    # run population fixed; repositories missing from the context table simply
    # get nulls, which the feature builders translate into sentinels.
    runs = runs.merge(
        repositories.drop(columns=["selected"]), on="repo", how="left", validate="many_to_one"
    )

    runs = textproc.clean_text_columns(runs, salt=salt)
    runs.to_parquet(out_dir / "runs_clean.parquet", index=False)
    print(f"\nwrote {out_dir / 'runs_clean.parquet'} ({len(runs):,} rows)")
    return runs


def stage_features(out_dir: Path) -> pd.DataFrame:
    """Build static and history features, then enforce the contract."""
    runs = pd.read_parquet(out_dir / "runs_clean.parquet")

    print("building history features ...")
    runs = history.build_history_features(runs)
    history.assert_history_is_causal(runs)
    print("  causality assertions passed")

    print("building static features ...")
    static = features.build_static_features(runs)

    # Assemble: identifiers and label, raw text for the tokeniser, then the
    # two feature blocks. Column order is stable so artefacts diff cleanly.
    keep = [
        "run_id",
        "repo",
        "workflow_path",
        "run_number",
        "run_attempt",
        "head_sha",
        "commit_group",
        "commit_group_size",
        "commit_group_is_mixed",
        "commit_group_minority",
        "created_at",
        "regime",
        FAILURE_LABEL,
        "commit_message",
        "head_branch",
        "display_title",
        "workflow_name",
        "repo_description",
        "actor_login",
        "commit_author_id",
    ]
    hist_columns = [c for c in runs.columns if c.startswith("hist_")]
    frame = pd.concat([runs[keep + hist_columns], static], axis=1)
    frame["transformer_input"] = textproc.build_transformer_input(runs)

    report = contract.enforce(frame, strict=True)
    print()
    print(report.describe())

    destination = Path(out_dir) / "features.parquet"
    frame.to_parquet(destination, index=False)
    pd.Series(report.feature_columns).to_csv(
        Path(out_dir) / "feature_manifest.csv", index=False, header=["feature"]
    )
    print(f"\nwrote {destination} ({len(frame):,} rows, {len(frame.columns)} columns)")
    print(f"failure rate: {100 * frame[FAILURE_LABEL].mean():.2f}%")
    print("regimes:")
    print(frame["regime"].value_counts().to_string())
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["ingest", "prepare", "features", "all"], help="pipeline stage to run"
    )
    parser.add_argument("--data-dir", type=Path, help="directory holding the GHALogs .json.gz files")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--salt",
        default=None,
        help=f"pseudonymisation salt; defaults to ${DEFAULT_SALT_ENV}",
    )
    parser.add_argument("--drop-reruns", action="store_true", help="sensitivity analysis")
    parser.add_argument("--drop-dynamic", action="store_true", help="sensitivity analysis")
    args = parser.parse_args()

    config = PipelineConfig()
    if args.drop_reruns or args.drop_dynamic:
        from dataclasses import replace

        config = PipelineConfig(
            filters=replace(
                config.filters,
                drop_rerun_attempts=args.drop_reruns,
                drop_dynamic_event=args.drop_dynamic,
            )
        )

    salt = args.salt or os.environ.get(DEFAULT_SALT_ENV)
    if salt is None and args.stage in ("prepare", "all"):
        raise SystemExit(
            f"set ${DEFAULT_SALT_ENV} (or pass --salt) so author and actor "
            "identifiers can be pseudonymised. Keep the value out of the repository."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("ingest", "all"):
        if args.data_dir is None:
            raise SystemExit("--data-dir is required for the ingest stage")
        stage_ingest(args.data_dir, args.out_dir, config)
    if args.stage in ("prepare", "all"):
        stage_prepare(args.out_dir, config, salt or "")
    if args.stage in ("features", "all"):
        stage_features(args.out_dir)


if __name__ == "__main__":
    main()
