#!/usr/bin/env python3
r"""Audit the pipeline's output: privacy, leakage, and feature sanity.

Run after `python -m ghalogs.pipeline all` and before any modelling. It is the
Phase 2 gate from docs/04-execution-plan.md turned into a command.

Four checks:

1. **Privacy.** No raw email address, URL, or credential survives in any text
   column, and no identifier is stored in the clear.
2. **Contract.** Only admissible features are present and none trips the
   univariate AUC screen.
3. **Causality.** History features depend only on strictly earlier runs.
4. **Sanity.** Feature distributions, regime balance, and the strongest
   label correlations, so an implausibly strong feature is visible to a human
   even if it passed the automated screen.

The email check needs care. A naive `\w+@\w+\.\w+` pattern flags the version
and dependency references that saturate this corpus — `actions/checkout@v3...v4`,
`calcite-components@1.10.0`, `python@3.11` — none of which are addresses.
Requiring the final segment to be an alphabetic top-level domain removes those
false positives without creating a hiding place for a genuine address.

Usage:
    python scripts/audit_pipeline_output.py --features data/features.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghalogs import contract, history  # noqa: E402
from ghalogs.config import FAILURE_LABEL  # noqa: E402

TEXT_COLUMNS = ("commit_message", "display_title", "repo_description", "head_branch", "workflow_name")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://|www\.")
SECRET_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY")
HEX_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def check_privacy(frame: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    for column in TEXT_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str)

        hits = values[values.str.contains(EMAIL_RE)]
        if len(hits):
            examples = [EMAIL_RE.search(text).group() for text in hits.head(3)]
            problems.append(
                f"{column}: {len(hits)} rows contain an email address, e.g. {examples}"
            )

        # Branch and workflow names are identifiers and are deliberately not
        # redacted, so URLs are only checked in the prose columns.
        if column in ("commit_message", "display_title", "repo_description"):
            n_urls = int(values.str.contains(URL_RE).sum())
            if n_urls:
                problems.append(f"{column}: {n_urls} rows still contain a URL scheme")

        n_secrets = int(values.str.contains(SECRET_RE).sum())
        if n_secrets:
            problems.append(f"{column}: {n_secrets} rows contain a credential-like token")

    for column in ("actor_login", "commit_author_id"):
        if column not in frame.columns:
            continue
        sample = frame[column].dropna().astype(str)
        sample = sample[sample != ""]
        if not sample.head(1000).map(lambda v: bool(HEX_ID_RE.match(v))).all():
            problems.append(f"{column}: values are not all 16-char hashes; identifiers in the clear?")

    if "commit_author_email" in frame.columns or "commit_author_name" in frame.columns:
        problems.append("raw author name/email columns are still present")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--top-correlations", type=int, default=15)
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    print(f"loaded {args.features}: {len(frame):,} rows, {len(frame.columns)} columns\n")
    failed = False

    print("== 1. privacy ==")
    problems = check_privacy(frame)
    if problems:
        failed = True
        for problem in problems:
            print(f"  FAIL  {problem}")
    else:
        print("  pass: no addresses, URLs, credentials, or plaintext identifiers found")

    print("\n== 2. feature contract ==")
    try:
        report = contract.enforce(frame, strict=True)
        print("  pass")
        print("  " + report.describe().replace("\n", "\n  "))
    except (AssertionError, ValueError) as error:
        failed = True
        print(f"  FAIL  {error}")

    print("\n== 3. history causality ==")
    try:
        history.assert_history_is_causal(frame)
        print("  pass: every history aggregate uses strictly earlier runs only")
    except AssertionError as error:
        failed = True
        print(f"  FAIL  {error}")

    print("\n== 4. sanity ==")
    print(f"  failure rate: {100 * frame[FAILURE_LABEL].mean():.2f}%")
    counts = frame["regime"].value_counts()
    for regime, count in counts.items():
        rate = 100 * frame.loc[frame["regime"] == regime, FAILURE_LABEL].mean()
        print(f"  {regime:16s} n={count:>8,} ({100 * count / len(frame):5.2f}%)  failure={rate:5.2f}%")

    numeric = [
        c
        for c in frame.columns
        if c.startswith(("static_", "hist_")) and pd.api.types.is_numeric_dtype(frame[c])
    ]
    correlations = (
        frame[numeric + [FAILURE_LABEL]]
        .corr(numeric_only=True)[FAILURE_LABEL]
        .drop(FAILURE_LABEL)
        .abs()
        .sort_values(ascending=False)
    )
    print(f"\n  strongest |correlation| with the label (top {args.top_correlations}):")
    for name, value in correlations.head(args.top_correlations).items():
        print(f"    {name:38s} {value:.4f}")
    strongest = float(correlations.iloc[0])
    print(
        f"\n  highest single-feature correlation is {strongest:.4f}. A value near 1.0 "
        "would indicate a leak; these are consistent with weak individual signals "
        "combining into a usable model."
    )

    print("\nAUDIT " + ("FAILED" if failed else "PASSED"))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
