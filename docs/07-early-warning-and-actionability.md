# Early warning signals and actionability

`docs/05-feasibility-results.md` established that pre-execution prediction
works. It did not test the three further claims that an AIOps framing adds on
top of feasibility, each of which can fail independently:

> By analysing patterns from **thousands of historical builds**, identify
> **early warning signals that precede failures**, enabling **preventive
> actions** such as targeted testing, resource scaling, or automated
> rollbacks.

This document tests all three on the full dataset. Two fail, one partly holds,
and the testing turned up a leakage mechanism that none of the previous
documents covered.

Reproduce with:

```sh
export PYTHONPATH=src; export GHALOGS_SALT="$(openssl rand -hex 16)"
python -m ghalogs.pipeline all --data-dir <ghalogs> --out-dir data
python scripts/run_early_warning_experiment.py \
    --features data/features.parquet --runs data/runs_clean.parquet
```

## Setup

- **Data:** the full 567,814-run modelling population across 28,297
  repositories, failure rate 15.87%.
- **Features:** 151 contract-admitted features — 122 static, 10 baseline
  history, 19 new precursor features from `src/ghalogs/precursor.py`. Six
  lagged fields enter through the named exemption described below. No feature
  trips the univariate AUC screen.
- **Model:** LightGBM, identical hyperparameters to the feasibility probe, so
  every comparison isolates the inputs.
- **Splitting:** 5-fold `GroupKFold` on repository; all scores out-of-fold.
- **Metric:** PR-AUC on the failure class. Unlike `docs/05`, no 20% slice is
  withheld for threshold fitting, because PR-AUC needs no threshold; pooled
  figures therefore differ from that document in the third decimal (0.7142
  here against 0.716 there) and are not a disagreement.

Four input conditions, plus two training-free baselines:

| Condition | Inputs |
| --- | --- |
| `both` | 122 static + 10 history. The `docs/05` reference condition |
| `both_obs` | Same features, with history blanked wherever the predecessor had not finished before this run was triggered |
| `both_pre` | `both` + the 19 precursor features |
| `pre_only` | Static + precursors, with every previous-outcome feature removed |
| `straw` | Copy the previous run's outcome |
| `straw_obs` | Copy it only when it was observable; otherwise abstain at the base rate |

### Headline table

| Condition | Pooled | A — cold start | B — previous run passed | C — previous run failed |
| --- | --- | --- | --- | --- |
| Base rate | 0.1587 | 0.1804 | 0.0497 | 0.6923 |
| `straw` | 0.4861 | 0.1804 | 0.0497 | 0.6923 |
| `straw_obs` | 0.4324 | 0.1804 | 0.0509 | 0.6835 |
| `pre_only` | 0.5136 | 0.3997 | 0.1299 | 0.7847 |
| `both_obs` | 0.7118 | 0.4780 | 0.2151 | 0.8964 |
| `both` | 0.7142 | 0.4783 | 0.2239 | 0.8982 |
| `both_pre` | **0.7212** | **0.4786** | **0.2473** | **0.9048** |

---

## 1. "Thousands of historical builds" — not available, but depth does help

### The quantity is wrong by three orders of magnitude

| Scope | Median | Mean | p95 | Max |
| --- | --- | --- | --- | --- |
| Runs per workflow in the dataset | 5 | 4.67 | — | 5 |
| Runs per repository in the dataset | 15 | 20.07 | 55 | 2,100 |
| **Prior runs of the same workflow, at prediction time** | **2** | 1.93 | — | 4 |
| **Prior runs of the same repository, at prediction time** | **11** | 28.44 | 86 | — |

The median prediction is made with **two** previous runs of the same workflow
and **eleven** of the same repository in view. Only 3.96% of runs have more
than 100 prior repository runs available, and **0.22% have more than 1,000**.

"Thousands of historical builds" describes the dataset in aggregate, not the
evidence behind any individual prediction. GHALogs retains at most the five
most recent runs per workflow, so this is a property of the sampling frame and
no amount of modelling recovers it. Any proposal phrased around per-project
build history at that scale needs a different dataset.

### But more history is better, as far as the data can show

Stratifying regime B by the length of the unbroken chain of adjacent
predecessors:

| Adjacent predecessors | n | Base rate | PR-AUC | PR-AUC lift | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| 1 | 100,883 | 5.59% | 0.2143 | 3.84× | 0.7924 |
| 2 | 91,580 | 5.29% | 0.2409 | 4.55× | 0.8116 |
| 3 | 84,787 | 4.82% | 0.2665 | 5.53× | 0.8371 |
| 4 | 79,183 | 3.97% | 0.2847 | 7.17× | 0.8524 |

PR-AUC rises 33% relative from depth 1 to depth 4 and ROC-AUC gains 6.0
points, while the base rate *falls* — so this is a real improvement in
discrimination, not an artefact of class balance.

Two cautions. The comparison is observational, not controlled: deeper chains
belong to workflows that ran repeatedly without a sampling gap, which skews
towards busier and more established workflows, so part of the gain is
selection rather than information. And the trend is measured over the range
1–4. It is consistent with the premise that more history helps, and it says
nothing about whether 100 or 1,000 would help more — the dataset truncates
exactly where the question becomes interesting.

---

## 2. "Early warning signals that precede failures" — the temporal ones are not there

`precursor.py` adds 19 features covering the standard leading-indicator
hypotheses: durations drifting upward, cadence changing, the repository
entering a failure burst, a sibling workflow breaking first, the previous run
needing a re-run, the branch or author changing.

### The uplift is real, robust, and small

In regime B — new breakage, the only regime where a warning would change
anything — adding the precursor block moves PR-AUC from 0.2239 to 0.2473:

| | Value |
| --- | --- |
| Delta | **+0.0234** (+10.4% relative) |
| Repository bootstrap 95% CI | [+0.0195, +0.0266] |
| Resamples with a positive delta | 200 / 200 |

Pooled, the same change is +0.0070 (0.7142 to 0.7212). The interval excludes
zero comfortably, so the effect is not noise. It is also not a breakthrough:
regime B precision and recall remain in the range Section 3 describes.

### Which signals carry it, and which are empty

Univariate ROC-AUC within regime B, folded to ≥ 0.5, with coverage (the share
of rows where the feature is not a missing-history sentinel):

| Feature | ROC-AUC | Coverage | Mean when failed | Mean when passed |
| --- | --- | --- | --- | --- |
| `hist_prior_duration_mean` | **0.631** | 0.952 | 2,913 s | 1,587 s |
| `hist_prev_branch_changed` | **0.625** | 1.000 | 0.654 | 0.405 |
| `hist_prev_duration_sec` | 0.607 | 0.873 | 2,024 s | 1,231 s |
| `hist_prev_actor_changed` | 0.575 | 1.000 | 0.407 | 0.257 |
| `hist_prev_log_bytes_log` | 0.564 | 0.788 | 9.30 | 9.00 |
| `hist_prev2_duration_sec` | 0.539 | 0.647 | 2,180 s | 1,114 s |
| `hist_prior_gap_mean_hours` | 0.538 | 0.742 | 45.8 h | 52.1 h |
| `hist_adjacent_depth` | 0.534 | 1.000 | 2.27 | 2.41 |
| `hist_repo_failures_prev_24h` | 0.529 | 1.000 | 1.16 | 1.14 |
| `hist_repo_runs_prev_24h` | 0.529 | 1.000 | 7.99 | 11.11 |
| `hist_gap_vs_prior_mean` | 0.526 | 0.740 | 11.84 | 7.16 |
| `hist_prior_duration_slope` | 0.505 | 0.678 | −639 | −212 |
| `hist_sibling_workflow_failed` | 0.504 | 0.870 | — | — |
| `hist_prev_duration_ratio` | 0.504 | 0.873 | 0.723 | 0.747 |
| `hist_repo_failure_rate_prev_24h` | 0.502 | 0.729 | — | — |
| `hist_prev_was_rerun` | 0.501 | 0.873 | — | — |

The pattern is consistent and is the main finding of this document. **Every
feature that encodes a temporal precursor is flat at chance:**

- duration *trend* (`hist_prior_duration_slope`, 0.505) and duration
  *anomaly* (`hist_prev_duration_ratio`, 0.504) — builds do not measurably
  creep upward before they break
- failure *cascade* (`hist_sibling_workflow_failed`, 0.504) — a sibling
  workflow breaking first does not warn you
- failure *burst* (`hist_repo_failure_rate_prev_24h`, 0.502) — recent
  repository-level trouble does not warn you either
- prior *instability* (`hist_prev_was_rerun`, 0.501)

What carries the uplift instead is **cross-sectional**: how long this
workflow's builds usually take, and whether the branch or the author changed
since the last run. `hist_prior_duration_mean` is the strongest precursor at
0.631, but a long-running workflow is a complex workflow — that is a
statement about what kind of pipeline this is, not a signal that something is
degrading. The same applies to branch and actor churn, which mark a change of
context rather than a trajectory.

A second check points the same way. `pre_only` — static features plus every
precursor, but no previous-outcome feature — reaches PR-AUC 0.1299 in regime
B, against 0.121 for the static features alone (`docs/05`). Stripped of the
previous outcome, the whole precursor block is worth roughly nine points in
the third decimal.

**Conclusion.** There is no evidence in GHALogs for leading indicators that
precede CI failure. The measurable signal is contextual, not temporal: what
this pipeline is and what changed about the trigger, not a degradation
trajectory that a monitoring system could watch. A dissertation may state
that as a finding — it is a clean negative result against a widely assumed
mechanism — but it cannot promise an early-warning system built on it.

---

## 3. A fourth leakage class: the predecessor is often still running

Testing the above required knowing *when* each predecessor finished, which
surfaced a mechanism the feature contract did not cover.

CI is concurrent. A developer pushes again while the previous run is still
executing, and GitHub Actions runs both. When that happens the previous run's
outcome **does not exist** at the moment the next run is triggered, so every
feature derived from it is reading the future — including the
`hist_prev_failed` that dominates every published model, and the
previous-outcome straw man itself.

| | Value |
| --- | --- |
| Runs with an adjacent predecessor | 423,323 (74.6% of all runs) |
| Of those, predecessor still running at trigger time | **57,083 (13.5%)** |
| As a share of all runs | 10.1% |
| Failure rate when the predecessor was observable | 14.40% |
| Failure rate when it was **not** | **19.78%** |

The affected runs are not a benign slice: they fail half again as often as the
rest, which is what you would expect, since pushing again before the previous
build finishes is itself a marker of churn or of a developer who already knows
something is wrong.

### What it costs

For the full model, little — it has other features to fall back on:

| Condition | Pooled PR-AUC |
| --- | --- |
| `both` (reads in-flight predecessors) | 0.7142 |
| `both_obs` (observable history only) | 0.7118 |
| Delta | 0.0024, 95% CI [0.0018, 0.0031] |

For the baseline everyone benchmarks against, considerably more:

| Condition | Pooled PR-AUC | Regime C PR-AUC | Regime C ROC-AUC |
| --- | --- | --- | --- |
| `straw` | 0.4861 | 0.6923 | 0.5000 |
| `straw_obs` | **0.4324** | 0.6835 | **0.4788** |

Restricted to outcomes it could actually have seen, the previous-outcome
heuristic loses 5.4 PR-AUC points pooled — an 11.0% relative drop — and in
regime C it falls *below* the base rate, with ROC-AUC under 0.5.

This is a small, specific, and checkable addition to the published
three-type leakage taxonomy, which is feature-level and cannot express it:
`hist_prev_failed` is a legitimate feature that becomes future information for
10.1% of rows depending on a timestamp comparison. It also slightly
*strengthens* the case for modelling, since the heuristic's advantage shrinks
once it is held to the same standard as the model.

### How it is enforced

The contract denies post-execution fields by substring, which cannot
distinguish a run's own duration from a predecessor's. Lagged fields now enter
through `contract.LAGGED_FIELD_EXEMPTIONS`, which requires **both** a recorded
justification and a name that advertises the lag — a bare exemption list would
admit `hist_duration_sec` on a typo, and a bare prefix rule would exempt
anything named `hist_prev_*`. Every lagged value is gated per contributing
element on that run having completed before the target's `created_at`, and
`assert_precursor_is_observable` checks the invariant on the assembled frame.

The gate is not cosmetic. An `expanding()` window cannot express it, because
runs finish out of order: a long run started earlier can still be going when a
shorter later one has finished, so the immediate predecessor having completed
does not imply that the runs before it had. The first implementation used
"newest observed predecessor with an adjacent run number", which disagreed
with the gate on 54 rows — re-runs, where attempt 1 of run 5 is adjacent and
complete while attempt 2 is still going. The assertion caught it.

---

## 4. "Preventive actions" — a narrow gate is viable, broad coverage is not

Discrimination is not actionability. Operating points in regime B, using
`both_pre`:

| Target recall | Precision | Alerts per 1,000 runs | True catches per 1,000 | False alarms per catch |
| --- | --- | --- | --- | --- |
| 5% | **56.9%** | 4.4 | 2.5 | 0.76 |
| 10% | **46.0%** | 10.8 | 5.0 | 1.18 |
| 25% | 32.7% | 38.1 | 12.4 | 2.06 |
| 50% | 20.6% | 120.6 | 24.9 | 3.85 |
| 75% | 13.2% | 283.4 | 37.3 | 6.60 |
| 90% | 9.0% | 499.1 | 44.7 | 10.15 |

This is more encouraging than PR-AUC 0.247 suggests, and the shape matters
more than any single number. A **narrow, high-confidence gate is feasible**:
catching 5% of new breakages at 56.9% precision means roughly four alerts per
thousand runs, three of which are right. Catching 10% at 46.0% precision costs
about one false alarm per catch. What is not available is coverage — at 50%
recall you are firing on 12% of all runs and wrong four times out of five.

Acting is net positive only when a correct warning is worth at least the
"false alarms per catch" multiple of what a false alarm costs. That is 1.18 at
10% recall and 3.85 at 50%.

### The duration distribution decides it

Run durations, in minutes:

| | Median | p75 | p90 | p99 | Mean |
| --- | --- | --- | --- | --- | --- |
| All runs | 2.42 | 8.80 | 28.27 | 798.94 | 62.47 |
| Failed runs | 3.45 | 15.55 | 70.75 | 2,414.20 | 201.58 |
| Passed runs | 2.27 | 7.98 | 23.98 | 307.07 | 36.23 |

64.8% of runs finish in under five minutes and 77.0% in under ten. **For the
median run there is nothing to save**: it completes in under three minutes, so
a pre-execution warning buys no meaningful compute and no meaningful feedback
latency. Quote the mean (62 minutes) and the economics look transformative;
quote the median (2.4 minutes) and they vanish. The mean is the wrong
statistic for a distribution this skewed.

But the skew is itself the opportunity. **Failed runs consume 51.2% of all CI
minutes** — 302,711 of 591,172 hours — because failures are both longer and
heavy-tailed (p90 of 70.75 minutes against 23.98 for passes). The value is
concentrated in the tail, so the defensible policy is not "warn on predicted
failures" but "warn on predicted failures whose expected duration is large".
That makes run-duration prediction a prerequisite for the cost case, not a
side quest.

### The three proposed actions, assessed

**Targeted testing — out of reach with this dataset.** The prediction is at
workflow-run granularity; targeting tests needs test-level outcomes and a
mapping from changed code to affected tests. GHALogs has no diffs, and 47.7%
of failures share a commit with a success, so the model cannot even identify
*which workflow* will break from commit content alone — workflow identity has
to be supplied as an input. Test-level targeting is several data sources away.

**Resource scaling — right idea, wrong predictor.** Scaling runners is a
response to *load and duration*, not to failure. Failure prediction is the
wrong signal, and GHALogs supports duration prediction far better: durations
are present for every run and the log archive carries per-step timings. If an
operational angle is wanted, predicting cost and duration is both more
tractable and more directly useful than predicting failure, and Section 4
shows the cost case depends on it anyway.

**Automated rollbacks — incoherent with "prior to execution".** A rollback
presupposes something was deployed; a pre-execution prediction fires before
the pipeline has run, so there is nothing to roll back. Even granting a
post-deployment variant, the best operating points here are wrong 43% of the
time at 5% recall, and automatically reverting a correct change on that basis
is worse than the failure it avoids.

---

## 5. Design consequences

1. **Drop "thousands of historical builds" from any framing.** The median
   prediction sees two prior runs of the workflow. Report the measured
   distribution instead, and cite the five-run sampling cap as the reason.
2. **Keep the depth result.** It supports the premise that history helps, with
   the stated caveat that the range is 1–4 and the comparison is
   observational.
3. **Report the precursor result as a negative one.** Temporal leading
   indicators — trend, anomaly, burst, cascade, prior instability — are all at
   chance. The +0.0234 uplift in regime B is real but comes from
   cross-sectional context, and saying so is more valuable than implying an
   early-warning mechanism that the measurements do not support.
4. **Add predecessor observability to the leakage chapter.** It is a fourth
   mechanism alongside the two GitHub Actions-specific ones in `docs/02`, it
   costs the straw man 11% of its pooled PR-AUC, and it is enforced in code.
5. **Evaluate at operating points, never at PR-AUC alone.** The viable
   artefact is a narrow high-precision gate, which the aggregate metric hides.
6. **Make duration prediction part of the cost analysis.** Half of all CI
   minutes go into failed runs, but the median run is 2.4 minutes, so the
   policy has to be conditioned on expected duration to be net positive.
