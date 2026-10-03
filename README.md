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

A gradient-boosting model over pre-execution features only, evaluated with
5-fold repository-grouped cross-validation on all 573,993 runs, reaches
**PR-AUC 0.701 and ROC-AUC 0.889** — 4.4× the base rate on projects it has
never seen. Against the previous-outcome heuristic it is **+141% in the
cold-start regime and +297% in the new-breakage regime**, the two regimes the
heuristic cannot address at all. Full results and the reproduction commands
are in [`docs/05-feasibility-results.md`](docs/05-feasibility-results.md).

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

## Reproducing the dataset profile

Download the two metadata files from Zenodo record
[14796970](https://doi.org/10.5281/zenodo.10154920) — `repositories.json.gz`
(69 MB) and `runs.json.gz` (1.06 GB). The 142 GB log archive is not required
for the profile.

```sh
python scripts/profile_dataset.py --data-dir /path/to/ghalogs
```

Every figure quoted in this README and in `docs/01-dataset-profile.md` is
produced by that script.

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
