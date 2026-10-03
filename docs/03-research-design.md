# Research design: gaps, questions, and contributions

## 1. Stated aim

> To develop an explainable machine learning framework that predicts
> CI/CD pipeline failures prior to execution by extracting predictive data
> from unstructured metadata.

The aim is sound but under-specified in one respect that determines whether
the dissertation has a defensible contribution. "Predicts CI/CD pipeline
failures" is already solved in the average case: copying the previous run's
outcome achieves 90.97% accuracy and 0.708 failure-class F1 on this dataset
(see `docs/01-dataset-profile.md`, Section 7). Framed as "predict the
outcome", the project competes against a one-line heuristic it is unlikely to
beat, and against published GHALogs results (83.30% accuracy) that already sit
below that heuristic.

The aim becomes defensible when narrowed to the cases the heuristic cannot
touch. Proposed refinement:

> To develop and explain a pre-execution CI/CD failure prediction framework
> that recovers failures invisible to outcome-history heuristics, by learning
> from unstructured trigger-time metadata, and to establish whether the
> resulting explanations are faithful to the true causes of failure.

This keeps every element the original aim promises — pre-execution,
unstructured metadata, explainability — while targeting the 48% of failures
(44,216 runs) that occur either at cold start or immediately after a
successful run, where history-based prediction is structurally blind.

## 2. Research gaps

Each gap is stated with the specific prior work it is defined against, so the
literature review can be written directly from this section.

### G1. Pre-execution prediction on GitHub Actions from unstructured metadata is unexplored

Work on GitHub Actions failure prediction uses structured tabular features, or
post-execution signals, or both. The transformer approach closest to this
project (GitSense, ISSTA 2024) models commit artefacts — code diffs and commit
messages — but is evaluated on Travis-era data, requires the diff, and offers
no explanation component. Pre-trained encoders have been applied to
just-in-time defect prediction with commit code and messages, but defect
prediction and build-outcome prediction are different tasks with different
label semantics. No study applies pre-trained language models to
trigger-time GitHub Actions metadata.

### G2. The temporal-leakage taxonomy is incomplete for GitHub Actions

The three-type taxonomy (direct outcome encoding, execution-dependent metrics,
future information leakage) is feature-level and derived largely from
TravisTorrent's tabular schema. It does not cover two mechanisms specific to
this dataset and platform, both documented in `docs/02-feature-contract.md`,
Section 4:

- **Fail-fast structural truncation.** Workflow structure read from logs is
  truncated by the outcome, because failing jobs cancel their siblings and
  abort their remaining steps. "Number of steps" becomes a label proxy. The
  same quantity read from the YAML at `head_sha` is legitimate. No prior work
  distinguishes declared from observed workflow structure.
- **Snapshot aggregate leakage.** Every `repo.*` field is a single crawl-time
  snapshot post-dating most runs. The published GHALogs analysis excludes four
  such fields (stars, forks, watchers, open issues) but retains others —
  commit counts, line counts, activity timestamps, and the
  `total_runs_90d` field that was itself the dataset's selection criterion.

### G3. No leakage-controlled comparison of model families exists

Reported numbers across the literature are not comparable: they use different
datasets, different splits, different leakage exposure, and different metrics.
No study evaluates classical, deep, and pre-trained transformer models on one
dataset under one pre-execution contract with one splitting protocol. Without
that, "deep learning helps" is untestable.

### G4. Explanations in this domain are never validated against ground truth

Explainable AI for build prediction is named as future work in the
leakage-taxonomy study and, where SHAP or LIME appear in CI research, the
explanations are assessed for plausibility rather than correctness. Yet
GHALogs ships execution logs for the same runs, which means the actual cause
of each failure is recoverable. The question "do the model's attributions
point at the thing that actually broke?" is answerable here and has not been
asked.

### G5. Reported metrics do not translate into deployment decisions

The literature reports accuracy and F1. Neither tells a team whether to act on
a prediction. Missing throughout: calibrated probabilities, reliability
analysis, explicit cost models over the asymmetric error costs of CI (a missed
failure wastes compute; a false alarm wastes developer attention and erodes
trust), and threshold selection tied to those costs. At a 15.88% base rate,
accuracy is close to uninformative.

### G6. Cold start is acknowledged but never measured

History features are the strongest known predictors and are unavailable for
new workflows and new repositories — 25.5% of this dataset. The hypothesis
that unstructured metadata degrades more gracefully than history features in
this regime is plausible, consequential, and untested.

### G7. Label noise from flakiness is not quantified as a performance ceiling

2.65% of runs in GHALogs are re-run attempts, and the GitHub Actions
flaky-build literature finds roughly two thirds of rerun builds to be flaky.
Some share of `failure` labels is therefore not a deterministic function of
pre-execution state. Studies report accuracy as though the attainable maximum
were 100%.

## 3. Research questions

Six questions, each tied to the gaps it closes and to a concrete, falsifiable
output. RQ1–RQ3 and RQ5 are the core; RQ4 and RQ6 are the depth the
dissertation needs to be more than a benchmark.

**RQ1 — What is actually knowable before execution, and what does pretending
otherwise cost?** *(G2)*
Which GHALogs fields satisfy the pre-execution contract, and how much
measured performance is attributable to violating it? Sub-question: how large
is the inflation specifically from log-derived workflow structure versus
YAML-derived structure, and from snapshot aggregates?
*Output:* an auditable field-by-field classification, an executable leakage
test suite, and a quantified inflation figure per leakage class.

**RQ2 — How much predictive signal does unstructured trigger-time metadata
carry, and is it complementary to structured features?** *(G1)*
Compare text-only, structured-only, and fused inputs. The text-only condition
must also be compared against a hand-written keyword baseline over branch
names and commit messages, since branch-name bucketing alone separates 12.9%
from 27.4% failure rates with no learning at all.
*Output:* an ablation establishing the incremental value of unstructured
metadata, and whether learned representations beat keyword rules.

**RQ3 — Under a single leakage-controlled protocol, how do classical,
deep, and pre-trained transformer models compare?** *(G3)*
Two classical, two deep, two transformer architectures, compared on
discrimination, calibration, and cost, with statistical testing and effect
sizes, all against the previous-outcome straw man.
*Output:* the first like-for-like comparison on this dataset, including the
negative results.

**RQ4 — How much does systematic optimisation of the best deep or transformer
model add?** *(G3)*
Having selected the best deep or transformer model on validation data,
quantify the gain from hyperparameter search, input-length and tokenisation
choices, class-imbalance handling, fusion strategy, and parameter-efficient
fine-tuning — separating genuine gains from search noise by repeating with
multiple seeds.
*Output:* an optimisation study with an honest accounting of how much of the
improvement survives seed variance.

**RQ5 — Are the explanations faithful, and do they agree with the real cause
of failure?** *(G4)*
Three levels. (a) Faithfulness: do comprehensiveness and sufficiency metrics
show that the highlighted tokens drive the prediction? (b) Correctness: do
attributions align with the failure cause recovered from the execution log?
(c) Cost of transparency: how much accuracy is given up by using an
intrinsically interpretable model instead of the optimised one?
*Output:* the first ground-truth-validated explanation evaluation in CI
failure prediction, and an explicit accuracy–interpretability trade-off curve.

**RQ6 — Where does the framework generalise, and where does it fail?** *(G5,
G6, G7)*
Performance under repository-grouped holdout, on unseen programming
languages, in the cold-start regime, on non-change-triggered events, and on
re-run attempts. Plus the deployment question: under a stated cost model, at
what threshold does acting on predictions become net-positive, and what does
the reliability diagram look like before and after calibration?
*Output:* a scoped validity statement and a cost-threshold analysis that says
plainly whether the framework is deployable.

## 4. Hypotheses

Stating these in advance, and reporting them whichever way they fall, is what
separates a dissertation from a leaderboard exercise. Several are expected to
be refuted; that is the point.

| ID | Hypothesis | Expectation |
| --- | --- | --- |
| H1 | Removing log-derived structural features reduces measured performance by a margin larger than the 0.48pp previously reported for GHALogs | Supported — the previously reported figure did not include structural features |
| H2 | Unstructured metadata alone outperforms the previous-outcome straw man on failure-class F1 in regime B | Uncertain; the central empirical risk of the project |
| H3 | Fusing unstructured and structured features beats either alone | Likely supported, modest margin |
| H4 | A fine-tuned transformer beats gradient boosting over the same information | Uncertain; on short text with strong tabular signal, boosting is a hard baseline |
| H5 | Unstructured metadata degrades less than history features in the cold-start regime | Likely supported; this is the framework's main argument for existence |
| H6 | Token attributions agree with log-derived failure causes above chance but well below human agreement | Likely supported |
| H7 | Uncalibrated model probabilities are miscalibrated enough to make fixed-threshold decisions materially worse than calibrated ones | Likely supported; consistent with CI triage findings |

## 5. Objectives

1. Construct and publish a reproducible, leakage-audited, pre-execution
   feature set over the 573,993 usable GHALogs runs, with the audit enforced
   in code.
2. Quantify the performance inflation attributable to each leakage class,
   extending the existing taxonomy with the two GitHub Actions-specific
   mechanisms.
3. Establish trivial, heuristic, and straw-man baselines before any model is
   trained, and report them beside every subsequent result.
4. Train and evaluate two classical, two deep, and two transformer models
   under one protocol, with grouped splits and repository-level bootstrap
   intervals.
5. Select the strongest deep or transformer model on validation data and
   optimise it systematically, separating real gains from seed variance.
6. Produce global, local, and token-level explanations; evaluate them for
   faithfulness and for agreement with log-derived failure causes.
7. Deliver a calibrated, cost-aware deployment analysis and a scoped statement
   of where the framework does and does not generalise.

## 6. Contributions

1. **The regime decomposition.** Showing that pre-execution CI prediction is
   three different problems — cold start, new breakage, still broken — with
   failure rates of 18.0%, 5.0%, and 69.3%, and that a trivial heuristic
   solves the third while being blind to the first two. This reframes the
   task and explains why high reported accuracies have not translated into
   deployed tools.
2. **Two new leakage mechanisms** (fail-fast structural truncation, snapshot
   aggregate leakage) with quantified inflation, extending the published
   taxonomy, plus an executable audit suite.
3. **The first evaluation of pre-trained language models on trigger-time
   GitHub Actions metadata**, including the negative result if the keyword
   baseline or gradient boosting wins.
4. **Ground-truth explanation validation**, using the dataset's logs as an
   oracle for whether attributions identify real causes — turning
   explainability from a presentational layer into a tested claim.
5. **A deployment-oriented evaluation** with calibration and an explicit cost
   model, replacing accuracy-first reporting.
6. **A reproducible artefact bundle**: feature contract as code, dataset
   profiler, trained model configurations, and the pre-registered protocol.

## 7. Reconciling the explainability stance

The repository's original framing prioritised "operational transparency over
black-box deep learning". The revised plan trains deep and transformer models
and optimises the best one. These are reconcilable, but only if the position
is stated deliberately rather than left as a contradiction an examiner will
find:

Interpretable models are not the framework's output; they are its control
condition. The dissertation measures the price of transparency rather than
assuming it. Concretely: logistic regression and a shallow tree establish the
fully transparent end of the spectrum; gradient boosting with SHAP occupies
the middle; the optimised transformer with token attribution occupies the
opaque end. RQ5(c) reports the accuracy difference across that spectrum, and
RQ5(a–b) tests whether the opaque end's explanations are actually trustworthy.
If the transformer's gain over logistic regression is small, or its
explanations prove unfaithful, the conclusion is that the interpretable model
should be deployed — and that conclusion is now evidence-backed rather than
asserted. `README.md` has been updated to this framing.

## 8. Scope exclusions

State these explicitly; each one is a question an examiner will otherwise ask.

- **Concept drift over time.** The window spans 2023-06-30 to 2023-11-14 with
  heavily skewed monthly volume. Too short and too uneven for drift analysis.
- **Platform generalisation.** GitHub Actions only. No Jenkins, GitLab CI, or
  CircleCI. Travis-era results are cited but not re-run, because TravisTorrent
  predates GitHub Actions and the two platforms have different leakage
  profiles.
- **Private and enterprise repositories.** The dataset is public repositories
  with 100+ stars and 30+ runs per 90 days. Conclusions do not extend to
  proprietary codebases or to low-activity projects.
- **Source code content.** No diffs in the dataset; any diff-level work is the
  optional Tier B strand on a subsample.
- **Root-cause prediction.** Failure causes are used for explanation
  validation. Predicting the cause is an extension, not a core question.
- **Live deployment.** The cost analysis is simulated against recorded
  durations, not measured in a production pipeline.
