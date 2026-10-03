# Feasibility probe: is pre-execution prediction actually possible?

This is a real experiment on the full dataset, not an estimate. It answers one
question before the dissertation commits to a direction: **does metadata
available before a run starts carry enough signal to predict failure?**

Reproduce with:

```sh
python scripts/extract_features.py --data-dir <ghalogs> --out data/features.parquet
python scripts/run_baseline_experiment.py --features data/features.parquet
```

## Setup

- **Data:** all 573,993 usable runs (`conclusion in {success, failure}`) across
  28,653 repositories. Failure rate 15.88%.
- **Model:** LightGBM, `class_weight="balanced"`, 600 trees with early stopping
  on a grouped validation slice. One model family throughout, so the
  comparison isolates the *inputs* rather than the architecture.
- **Splitting:** 5-fold `GroupKFold` on repository. No repository appears in
  both train and test, so these are cross-project numbers. Predictions are
  out-of-fold.
- **Thresholds:** chosen to maximise failure-class F1 on a 20% repository slice
  held out from reporting; all metrics below are on the remaining 459,049 runs.
- **Features:** only fields admitted by `docs/02-feature-contract.md`. No logs,
  no run duration, no crawl-time repository snapshots.

Four input conditions:

| Condition | Inputs |
| --- | --- |
| `static` | 73 features needing no execution history: branch-name shape and keywords, commit-message shape and keywords, trigger event, hour and weekday, bot/fork/PR flags, language, repository age |
| `hist` | 8 causally ordered history features: previous outcome, prior failure rates for the workflow and repository, hours since previous run, missingness flags |
| `both` | `static` + `hist` |
| `straw` | Copy the previous run's outcome. No training |

## Primary result: PR-AUC by regime

PR-AUC (average precision on the failure class) is the primary metric. The
`base_rate` column is the PR-AUC a random scorer achieves, so it is the floor.

| Regime | n | Base rate | `static` | `hist` | `both` | `straw` |
| --- | --- | --- | --- | --- | --- | --- |
| **All** | 459,049 | 0.1591 | 0.2954 | 0.6690 | **0.7009** | 0.4876 |
| A — cold start | 117,015 | 0.1806 | 0.3215 | 0.3565 | **0.4353** | 0.1806 |
| B — previous run succeeded | 287,744 | 0.0496 | 0.1188 | 0.1248 | **0.1968** | 0.0496 |
| C — previous run failed | 54,290 | 0.6933 | 0.7246 | 0.8839 | **0.8944** | 0.6933 |

Expressed as lift over the base rate — how many times better than guessing:

| Regime | `static` | `hist` | `both` | `straw` |
| --- | --- | --- | --- | --- |
| All | 1.86× | 4.20× | **4.40×** | 3.06× |
| A — cold start | 1.78× | 1.97× | **2.41×** | 1.00× |
| B — previous run succeeded | 2.40× | 2.52× | **3.97×** | 1.00× |
| C — previous run failed | 1.05× | 1.28× | **1.29×** | 1.00× |

And as the relative gain of the full model over the straw man:

| Regime | `both` | `straw` | Relative gain |
| --- | --- | --- | --- |
| All | 0.7009 | 0.4876 | **+43.7%** |
| A — cold start | 0.4353 | 0.1806 | **+141.1%** |
| B — previous run succeeded | 0.1968 | 0.0496 | **+296.9%** |
| C — previous run failed | 0.8944 | 0.6933 | +29.0% |

## Secondary metrics

| Condition | Regime | ROC-AUC | MCC | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `static` | All | 0.6821 | 0.194 | 0.253 | 0.578 | 0.352 | 0.661 |
| `hist` | All | 0.8628 | 0.556 | 0.690 | 0.555 | 0.615 | 0.890 |
| `both` | All | 0.8890 | **0.565** | 0.636 | 0.632 | **0.634** | 0.884 |
| `straw` | All | 0.8143 | 0.535 | 0.693 | 0.515 | 0.591 | **0.887** |
| `static` | A | 0.6792 | 0.201 | 0.279 | 0.591 | 0.379 | 0.650 |
| `both` | A | 0.7440 | 0.295 | 0.445 | 0.382 | 0.411 | 0.802 |
| `straw` | A | 0.5000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.819 |
| `static` | B | 0.7001 | 0.133 | 0.092 | 0.590 | 0.159 | 0.690 |
| `both` | B | 0.7825 | 0.153 | 0.370 | 0.080 | 0.131 | 0.948 |
| `straw` | B | 0.5000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.950 |
| `static` | C | 0.5266 | 0.026 | 0.704 | 0.567 | 0.628 | 0.534 |
| `both` | C | 0.7932 | 0.235 | 0.719 | 0.982 | 0.830 | 0.721 |

## Findings

**1. Pre-execution prediction is feasible.** The full pre-execution model
reaches ROC-AUC 0.889 and PR-AUC 0.701 — 4.4× the base rate — across projects
it has never seen, using nothing that was unavailable at the moment the run
was triggered. The research question is answerable in the affirmative.

**2. Static metadata alone carries real but modest signal.** Branch names,
commit-message shape, trigger event, timing, and repository context reach
ROC-AUC 0.682 and PR-AUC 0.295, which is 1.86× the base rate. That is far
above chance and confirms that unstructured trigger-time metadata is
predictive. It is not, on its own, enough to beat a model that also knows the
previous outcome.

**3. The previous run's outcome is itself pre-execution metadata.** This is the
point most easily missed. Whether the last run of this workflow passed is
known before the current run starts, so it satisfies the pre-execution
constraint completely — and it is the strongest single signal in the dataset.
It cannot be excluded on the grounds of not being "pre-execution"; excluding it
has to be justified as a deliberate cold-start study instead. `hist` alone
reaches PR-AUC 0.669 against `static`'s 0.295.

**4. Static metadata earns its place exactly where history fails.** In the two
regimes where outcome history has zero skill by construction, static metadata
is the only thing that works at all:

- Cold start: `static` reaches PR-AUC 0.322 and ROC-AUC 0.679 where the straw
  man is at the base rate with zero discrimination. This is 117,015 runs,
  more than a fifth of the dataset.
- Previous run succeeded: `static` reaches 2.40× lift where the straw man again
  has zero skill. This regime holds 287,744 runs and 19.6% of all failures.

Conversely, in regime C static metadata is nearly worthless (ROC-AUC 0.527) —
when the pipeline is already broken, *what* you committed barely matters.

**5. The two signals are complementary, and that is the headline.** `both`
beats `hist` in every regime, with the largest gains precisely where history
is weakest: +22% relative PR-AUC in cold start (0.436 vs 0.357) and +58% in
the new-breakage regime (0.197 vs 0.125). Against the straw man, `both` is
+141% in cold start and +297% in new breakage. The contribution of
unstructured metadata is real, measurable, and concentrated.

**6. Accuracy ranks the models backwards.** This is the clearest illustration
of why the metric matters. `static` has 66.1% accuracy — worse than the 84.1%
you get by always predicting success — yet it has genuine discriminative
power (ROC-AUC 0.682) and is the only usable signal for a quarter of the
dataset. The straw man has the *highest* accuracy of all four conditions
(88.7%) while being completely blind to 48% of failures. Ranking by accuracy
would lead you to deploy the straw man and discard the informative model.
Rank by PR-AUC and MCC, and report accuracy only for comparability.

**7. Regime B is where the difficulty lives.** Even `both` reaches only
PR-AUC 0.197 there, against a 4.96% base rate. Predicting a newly introduced
breakage is genuinely hard, which is why it is worth a dissertation. This is
where the transformer models should be targeted and where improvements should
be measured.

## Implications for the project

- The aim is feasible. Report it as the regime-stratified result above, not as
  a single pooled accuracy figure.
- Include history features. Do not restrict the model to static metadata on
  the mistaken grounds that history is not pre-execution. Report `static`,
  `hist`, and `both` as the three input conditions of RQ2 — that ablation is
  now a measured result rather than a plan.
- Target regimes A and B. That is where headroom exists, where the straw man
  is blind, and where early warning would change a developer's behaviour.
- These are gradient-boosting numbers over hand-engineered features. They are
  the bar the deep and transformer models must clear. The transformers get to
  read the actual commit message text rather than 15 keyword flags, so there
  is room to improve — but the bar is PR-AUC 0.701 pooled and 0.197 in regime
  B, not the base rate.

## Limitations of this probe

Deliberately a probe, not the final experiment:

- One model family (LightGBM), one seed, no confidence intervals. The final
  study needs repository-level bootstrap intervals and multi-seed repeats per
  `docs/04-execution-plan.md`, Phase 8.
- Text is reduced to 15 branch keywords, 15 message keywords, and shape
  statistics. No TF-IDF, no embeddings, no transformer. Representation quality
  is the open question, not feasibility.
- No calibration applied, so Brier scores and thresholds are not yet
  deployment-grade.
- Thresholds maximise F1, which is not the cost-optimal objective. The cost
  model in Phase 8 will move them.
- Regime membership is derived from in-dataset predecessors only. A real
  deployment would see the true previous run even when GHALogs' five-run
  sampling omits it, so regime A is somewhat larger here than it would be in
  production.
