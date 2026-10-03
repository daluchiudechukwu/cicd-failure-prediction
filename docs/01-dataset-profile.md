# Dataset profile: what GHALogs actually contains

All figures below were measured directly from the GHALogs metadata files
(Zenodo record 14796970, version 1.0.1) and are reproducible with
`python scripts/profile_dataset.py --data-dir <dir>`. They differ in several
material respects from the dataset paper's description, so the numbers here —
not the paper's abstract — should be cited in the dissertation's dataset
chapter.

Only `repositories.json.gz` (69 MB) and `runs.json.gz` (1.06 GB) are needed.
The 142 GB `github_run_logs.zip` archive is required only for the
failure-cause annotation described in `docs/04-execution-plan.md`.

## 1. Scale and population

| Property | Measured value |
| --- | --- |
| Run documents | 580,641 |
| Usable runs (`success` or `failure`) | 573,993 |
| Distinct workflows | 123,784 |
| Distinct repositories | 28,658 |
| Repository documents (search frontier) | 240,781, of which 28,322 `selected: true` |
| Observation window (`metadata.created_at`) | 2023-06-30 to 2023-11-14 |

The paper reports 513k runs; the published metadata file contains 580,641. The
discrepancy is accounted for by runs the paper excluded from its headline
count. Report the measured figure and the filter you applied.

## 2. Label distribution

| `metadata.conclusion` | Count | Share |
| --- | --- | --- |
| `success` | 482,830 | 83.15% |
| `failure` | 91,163 | 15.70% |
| `skipped` | 4,113 | 0.71% |
| `cancelled` | 1,713 | 0.30% |
| `action_required` | 447 | 0.08% |
| `startup_failure` | 375 | 0.06% |

After restricting to `success` and `failure`, the failure rate is **15.88%**.

Two corrections to the literature follow from this table. First, the dataset
paper states that only `success`, `failure`, and `timed_out` were collected;
in fact `timed_out` never occurs, while `skipped`, `cancelled`,
`action_required`, and `startup_failure` all do. Second, `startup_failure`
denotes a workflow that never executed — usually invalid YAML — which is the
one failure mode that is *fully* determined by pre-execution state. It is too
rare (0.06%) to model separately but is worth a sentence in the discussion.

Excluding `cancelled` runs introduces a known selection bias: a developer who
cancels a run they expect to fail removes a positive case from the sample.
State this as a threat to construct validity.

## 3. Trigger events and the no-code-change problem

| Event | n | Failure rate |
| --- | --- | --- |
| `push` | 195,593 | 14.16% |
| `pull_request` | 188,156 | 20.40% |
| `schedule` | 85,953 | 16.22% |
| `dynamic` | 29,585 | 2.46% |
| `workflow_dispatch` | 19,851 | 20.86% |
| `pull_request_target` | 18,580 | 7.53% |
| `release` | 11,586 | 15.92% |
| `issues` | 8,778 | 7.79% |
| `workflow_run` | 6,644 | 16.80% |
| `issue_comment` | 4,386 | 9.48% |

Roughly a fifth of runs (`schedule`, `workflow_dispatch`, `issues`,
`issue_comment`, `workflow_run`, `repository_dispatch`) are **not triggered by
a code change**. For these, the head commit is identical to that of the
previous run, so commit-derived text carries no incremental information and any
observed failure must be caused by something outside the repository — an
upstream dependency that moved, an expired credential, a flaky network call.
Any claim that unstructured commit metadata predicts failure must therefore be
reported separately for change-triggered and non-change-triggered runs, or it
will be confounded.

## 4. Unstructured field coverage

Every text field that can be read at trigger time is close to universally
populated, which is the central feasibility result for this project.

| Field | Coverage |
| --- | --- |
| `metadata.head_commit.message` | 99.96% |
| `metadata.head_branch` | 100.00% |
| `metadata.display_title` | 100.00% |
| `metadata.name` (workflow name) | 99.98% |
| `metadata.actor.login` | 100.00% |
| `metadata.repository.description` | 98.52% |

Commit messages have a mean length of 180 characters, a median of 57, and a
95th percentile of 591. A 256-token encoder window therefore covers almost the
entire distribution, which keeps transformer fine-tuning cheap. Note that
`metadata.repository.description` exists only inside the run documents;
`repositories.json.gz` does not carry a description field.

`metadata.pull_requests` is non-empty for only 33.1% of runs and
`metadata.referenced_workflows` for 6.1%, so neither can be a required input.

## 5. Branch names carry signal

Bucketing `metadata.head_branch` by substring gives a better-than-twofold
spread in failure rate with no model at all:

| Bucket | n | Failure rate |
| --- | --- | --- |
| `renovate` | 14,235 | 27.36% |
| `feature` | 8,056 | 23.06% |
| `chore` | 1,982 | 22.65% |
| `dependabot` | 37,661 | 20.98% |
| `release` | 9,121 | 16.92% |
| `hotfix` | 595 | 17.48% |
| `fix` | 23,891 | 15.79% |
| `develop` | 15,068 | 14.97% |
| `main` | 131,306 | 12.90% |
| `master` | 120,431 | 12.89% |
| other | 211,647 | 17.45% |

This is direct evidence that trigger-time text is predictive, and it also
shows why hand-crafted keyword buckets are a necessary baseline: if a
fine-tuned transformer cannot beat a ten-rule regex on branch names, the
representation learning has added nothing.

## 6. History availability and the three prediction regimes

GHALogs retains at most the five most recent runs of each workflow (measured
mean 4.67, median 5). Build history is therefore shallow, but it is not
absent: counting only predecessors whose `run_number` is exactly one lower,
74.5% of usable runs have an immediate in-dataset predecessor.

Partitioning on that predecessor splits the task into three regimes with
radically different difficulty:

| Regime | Definition | n | Share | Failure rate |
| --- | --- | --- | --- | --- |
| **A** | No adjacent predecessor (cold start) | 146,185 | 25.47% | 18.03% |
| **B** | Previous run succeeded (new breakage) | 360,085 | 62.73% | 4.96% |
| **C** | Previous run failed (still broken) | 67,723 | 11.80% | 69.31% |

## 7. The baseline that must be beaten

Copying the previous run's outcome — no training, no features — achieves the
following on regimes B and C combined (427,808 runs):

- accuracy **90.97%**
- failure-class precision **69.31%**, recall **72.44%**, F1 **70.84%**

This is the most consequential number in the profile. The strongest published
GHALogs result for pre-execution prediction is 83.30% accuracy from a Random
Forest on 29 clean tabular features; that is **7.7 percentage points below a
baseline that requires no model at all**. Any result in this dissertation that
is reported as accuracy, without this baseline alongside it, will be read as a
null result by an examiner who knows the dataset.

The baseline's weakness is precisely where the research value lies: because it
predicts "success" whenever the previous run succeeded, it misses **all 17,859
regime-B failures, which are 19.6% of every failure in the dataset**. These are
the newly introduced breakages — the cases where early warning would actually
change a developer's behaviour. Together with the 26,357 cold-start failures in
regime A, roughly 48% of all failures sit in the two regimes where outcome
autocorrelation provides no answer and trigger-time metadata is the only
remaining signal.

Within regime B, failure rates by event show where the headroom concentrates:

| Event (regime B only) | n | Failure rate |
| --- | --- | --- |
| `pull_request` | 109,499 | 8.29% |
| `workflow_dispatch` | 10,447 | 8.17% |
| `release` | 6,906 | 5.86% |
| `workflow_run` | 3,836 | 4.74% |
| `push` | 126,897 | 4.23% |
| `pull_request_target` | 12,552 | 2.49% |
| `schedule` | 57,198 | 2.20% |
| `dynamic` | 21,334 | 0.62% |

## 8. Repository heterogeneity forces grouped splitting

Among the 21,864 repositories with at least ten usable runs, the mean failure
rate is 15.4%, the median 10.0%, and the standard deviation 18.4%. **32.4% of
these repositories never fail once**, and 4.7% fail more than half the time.

A model can therefore score well by memorising repository identity rather than
learning anything about change risk. Because runs are nested inside workflows
inside repositories, a random row-level train/test split leaks repository
identity across the boundary and inflates every metric. Splitting must be
grouped by repository, and confidence intervals must be computed by
bootstrapping over repositories rather than over runs.

## 9. Flakiness sets a ceiling on achievable accuracy

15,385 runs (2.65%) have `run_attempt > 1`, closely matching the ~3.2% rerun
rate reported in the GitHub Actions flaky-build literature, which also finds
that about two thirds of rerun builds are genuinely flaky. A non-trivial
fraction of `failure` labels are therefore not deterministic functions of
pre-execution state and cannot be predicted by any model from any
pre-execution feature set. Quantify this as an irreducible error floor rather
than treating it as model error, and analyse `run_attempt > 1` runs separately.

## 10. Temporal structure is skewed and constrains the split

Monthly run volume rises steeply across the window:

| Month | Runs |
| --- | --- |
| 2023-06 | 92 |
| 2023-07 | 28,450 |
| 2023-08 | 45,477 |
| 2023-09 | 123,145 |
| 2023-10 | 146,929 |
| 2023-11 | 236,548 |

Because only the five most recent runs per workflow were captured, and capture
proceeded over time, early months are not a random sample of workflows — they
are the workflows that happened to be scraped early and then went quiet. A
single chronological cut therefore does not produce two exchangeable halves,
and the 4.5-month window is far too short to study concept drift. Treat the
repository-grouped split as primary and the temporal split as a secondary
robustness check, and say explicitly that longitudinal drift is out of scope.

## 11. Summary of design consequences

1. The modelling population is the 573,993 `success`/`failure` runs; document
   every exclusion.
2. Report performance per regime (A, B, C), never pooled only.
3. Report the previous-outcome straw man beside every model.
4. Group splits by repository; bootstrap over repositories.
5. Separate change-triggered from non-change-triggered events before claiming
   that commit text is predictive.
6. Treat the 142 GB log archive as a source of labels and evaluation oracles,
   never as a source of features.
