# Pre-Execution CI/CD Failure Prediction from Unstructured Metadata

A master's dissertation project developing an explainable machine learning
framework that predicts GitHub Actions workflow failures **before the pipeline
runs**, using only metadata available at the moment the run is triggered.

## Aim

> To develop and explain a pre-execution CI/CD failure prediction framework
> that recovers failures invisible to outcome-history heuristics, by learning
> from unstructured trigger-time metadata, and to establish whether the
> resulting explanations are faithful to the true causes of failure.

## Why the framing is narrow

Predicting a workflow's outcome, in the average case, is already solved by a
heuristic that needs no model: copying the previous run's outcome achieves
**90.97% accuracy and 0.708 failure-class F1** on the GHALogs dataset. The
strongest published pre-execution result on the same dataset is 83.30%
accuracy — below that heuristic.

Measured on all 573,993 usable runs, the task decomposes into three regimes:

| Regime | Share of runs | Failure rate | Does outcome history help? |
| --- | --- | --- | --- |
| Cold start (no prior run) | 25.5% | 18.0% | No — nothing to copy |
| Previous run succeeded | 62.7% | 5.0% | No — the heuristic misses every failure here |
| Previous run failed | 11.8% | 69.3% | Yes — trivially |

Around **48% of all failures** fall in the first two regimes, where
autocorrelation is blind and trigger-time metadata is the only remaining
signal. That is the target of this work.

## Feasibility is established, not assumed

A gradient-boosting model over the 132 pre-execution features, evaluated with
5-fold repository-grouped cross-validation on 567,814 runs, reaches **PR-AUC
0.716 and ROC-AUC 0.898** — 4.5× the base rate on projects it has never seen.
Against the previous-outcome heuristic it is **+165% in the cold-start regime
and +356% in the new-breakage regime**, the two regimes the heuristic cannot
address at all. Full results in
[`docs/05-feasibility-results.md`](docs/05-feasibility-results.md).

One measured constraint shapes the whole design: 69.9% of runs share a commit
with another run, and **47.7% of all failures share a commit with a success**
because one push triggers several workflows. Commit text alone therefore
cannot separate nearly half the positive class — workflow identity is a
required input, and any commit-level model has an irreducible error floor of
6.42%.

## What the failures do not have: precursors

Failures were also tested for the leading indicators an AIOps framing assumes.
Nineteen precursor features covering duration drift, cadence change, failure
bursts, cross-workflow cascades, and prior re-runs raise regime-B PR-AUC from
0.224 to 0.247 (bootstrap CI [+0.019, +0.027]) — real, but **every explicitly
temporal signal sits at chance**: duration trend 0.505, sibling-workflow
failure 0.504, 24-hour failure burst 0.502, previous-run re-run 0.501. The
uplift comes from cross-sectional context — how long this workflow usually
takes, whether the branch or author changed — not from a degradation
trajectory.

Two further measured constraints follow. The median prediction sees **two**
prior runs of its workflow and eleven of its repository, because GHALogs keeps
only five runs per workflow; 0.22% of runs have more than a thousand. And the
median run finishes in **2.4 minutes**, so warning about it saves nothing —
although failed runs consume 51.2% of all CI minutes, so the value is
concentrated in a long tail and any cost case has to be conditioned on
expected duration.

Testing this also surfaced a fourth leakage mechanism. CI is concurrent, so
for **10.1% of runs the previous run was still executing** when the next was
triggered — its outcome did not exist at prediction time. Held to outcomes it
could actually have seen, the previous-outcome heuristic loses 11.0% of its
pooled PR-AUC and drops below the base rate in the still-broken regime. See
[`docs/07-early-warning-and-actionability.md`](docs/07-early-warning-and-actionability.md).

## Approach

Unstructured trigger-time artefacts — commit messages, branch names, run
titles, workflow names, repository descriptions — are modelled with classical,
deep, and pre-trained transformer architectures under a single
leakage-controlled protocol, then the strongest deep or transformer model is
optimised and explained.

Two commitments shape the design:

**Logs are an oracle, not an input.** GHALogs ships 142 GB of execution logs.
Logs are produced *during* execution, so they are excluded from every feature
set by construction. They are used instead to label the true cause of each
failure, which makes it possible to test whether the model's explanations
point at what actually broke — rather than merely whether they look plausible.

**Interpretable models are the control condition, not the conclusion.** The
project measures the price of transparency rather than assuming it. Logistic
regression anchors the transparent end of the spectrum, gradient boosting with
SHAP the middle, and the optimised transformer with token attribution the
opaque end. If the opaque model's advantage is small, or its explanations
prove unfaithful, the recommendation is the interpretable model — backed by
evidence instead of preference.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/01-dataset-profile.md`](docs/01-dataset-profile.md) | Measured properties of GHALogs and the design constraints they impose, including several corrections to the published description |
| [`docs/02-feature-contract.md`](docs/02-feature-contract.md) | What may and may not enter a model, mapped field by field to the dataset JSON, including two GitHub Actions-specific leakage mechanisms |
| [`docs/03-research-design.md`](docs/03-research-design.md) | Research gaps, questions, hypotheses, objectives, contributions, and scope exclusions |
| [`docs/04-execution-plan.md`](docs/04-execution-plan.md) | Eight phases with exit gates, model selection, evaluation protocol, explainability study, risk register, and chapter mapping |
| [`docs/05-feasibility-results.md`](docs/05-feasibility-results.md) | A real experiment on all 573,993 runs establishing that pre-execution prediction works, and the bar the deep models must clear |
| [`docs/06-data-pipeline.md`](docs/06-data-pipeline.md) | The ingestion, filtering, and feature-engineering pipeline, with the rationale behind each design decision |
| [`docs/07-early-warning-and-actionability.md`](docs/07-early-warning-and-actionability.md) | Whether failures have temporal precursors, how much history a prediction actually sees, what operating points are achievable, and a fourth leakage mechanism |

## Running the pipeline

Download the two metadata files from Zenodo record
[14796970](https://doi.org/10.5281/zenodo.10154920) — `repositories.json.gz`
(69 MB) and `runs.json.gz` (1.06 GB). The 142 GB log archive is not required.

```sh
pip install -r requirements.txt
export PYTHONPATH=src
export GHALOGS_SALT="$(openssl rand -hex 16)"   # keep out of version control

# Profile the raw dataset
python scripts/profile_dataset.py --data-dir /path/to/ghalogs

# Ingest -> filter -> engineer features (about 2m45s total on 4 cores)
python -m ghalogs.pipeline all --data-dir /path/to/ghalogs --out-dir data

# Gate: privacy, feature contract, history causality, sanity
python scripts/audit_pipeline_output.py --features data/features.parquet

# Feasibility experiment
python scripts/run_baseline_experiment.py --features data/features.parquet

# Early warning signals, predecessor observability, and operating points
python scripts/run_early_warning_experiment.py \
    --features data/features.parquet --runs data/runs_clean.parquet

python -m pytest tests/ -q
```

Every figure quoted in this README and in `docs/` is output from these
commands. See [`docs/06-data-pipeline.md`](docs/06-data-pipeline.md) for what
each stage does and why.

## Dataset

Moriconi, F., Durieux, T., Falleri, J.-R., Francillon, A., and Troncy, R.
*GHALogs: Large-Scale Dataset of GitHub Actions Runs.* MSR '25.

```bibtex
@inproceedings{msr25_ghalogs,
  author    = {Moriconi, Florent and Durieux, Thomas and Falleri, Jean-R{\'e}mi
               and Francillon, Aur{\'e}lien and Troncy, Raphael},
  title     = {GHALogs: Large-Scale Dataset of GitHub Actions Runs},
  booktitle = {Proceedings of the 22nd International Conference on Mining
               Software Repositories},
  series    = {MSR '25},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Ottawa, Canada}
}
```
