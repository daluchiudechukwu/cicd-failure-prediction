# Pre-execution feature contract

This document is the project's single source of truth on what may enter a
model. Every field is mapped to its exact path in the GHALogs JSON so that the
audit is checkable rather than rhetorical. The rule is absolute:

> A feature is admissible only if its value could have been computed by an
> observer standing at the moment `metadata.run_started_at` occurs, knowing
> only the repository state at `metadata.head_sha` and the outcomes of runs
> that had already completed.

Published work in this area reports 95–99% accuracy using features that
violate this rule; the corrective literature classifies the violations as
direct outcome encoding, execution-dependent metrics, and future information
leakage. Two further violation classes specific to GitHub Actions are
identified in Section 4 below and, as far as the surveyed literature goes,
have not been described before. Documenting them is one of this
dissertation's methodological contributions.

## 1. Admissible: unstructured trigger-time text (Tier A)

These are the project's primary inputs. All are present in `runs.json.gz` and
all exceed 98% coverage.

| Field | Path | Why it is pre-execution |
| --- | --- | --- |
| Commit message | `metadata.head_commit.message` | Written by the author before the push that triggered the run |
| Branch / ref name | `metadata.head_branch` | Part of the triggering event payload |
| Run display title | `metadata.display_title` | Derived from the commit subject or PR title at trigger time |
| Workflow name | `metadata.name` | Declared in the YAML at `head_sha` |
| Workflow file path | `metadata.workflow_path` / `metadata.path` | Declared in the repository at `head_sha` |
| Repository description | `metadata.repository.description` | Repository-level prose, slow-moving |
| Repository topics | `repo.topics` in `repositories.json.gz` | Repository-level labels, slow-moving |
| Commit author name/email | `metadata.head_commit.author.*` | Recorded in the commit object itself |

Treat the author identity fields with care. They are legitimately available
before execution, but a model that predicts failure from *who* wrote the
commit raises a fairness problem and produces explanations a developer cannot
act on. Pseudonymise them (salted hash) and include them only in an explicit
ablation whose purpose is to measure, and then argue against, their use.

## 2. Admissible: structured trigger-time context (Tier A)

| Feature | Path | Note |
| --- | --- | --- |
| Trigger event | `metadata.event` | Categorical, 15+ levels |
| Run attempt | `run_attempt` | `> 1` identifies a re-run; analyse separately |
| Run number | `run_number` | Position in the workflow's own history |
| Actor type | `metadata.actor.type` | `User` vs `Bot` vs `Organization` |
| Actor is bot | derived from `metadata.actor.login` suffix `[bot]` | Dependabot/Renovate detection |
| Fork indicator | `metadata.head_repository.fork` | PRs from forks run under different trust rules |
| Head repo differs from base | `metadata.head_repository.id != metadata.repository.id` | Cross-fork contribution |
| Associated PR present | `metadata.pull_requests` non-empty | Only 33% coverage; encode as a flag, not a required value |
| Referenced reusable workflows | `metadata.referenced_workflows` | 6% coverage; count only |
| Owner is organisation | `metadata.repository.owner.type` | Organisational vs personal project |
| Repository is private | `metadata.repository.private` | Constant (all public) — drop |
| Trigger timestamp features | `metadata.created_at`, `metadata.run_started_at` | Hour of day, day of week, weekend flag, UTC only |

`metadata.created_at` and `metadata.run_started_at` are both the start of the
run and are safe. `metadata.updated_at` is the *end* and is not (Section 3).

## 3. Inadmissible: post-execution fields

| Field | Path | Leakage class |
| --- | --- | --- |
| Conclusion | `metadata.conclusion` | This is the label |
| Status | `metadata.status` | Direct outcome encoding (always `completed` here) |
| Run end time | `metadata.updated_at` | Execution-dependent; `updated_at - created_at` is the duration |
| Parsed log insights | `log_insights[*]` | Execution-dependent in its entirety (see Section 4) |
| Log archive size | `total_logs_size` | Execution-dependent and strongly outcome-correlated |
| Log archive path | `logs_archive` | Execution-dependent |
| Step durations | `log_insights[*].steps[*].duration_sec` | Execution-dependent |
| Runner image version | `log_insights[*].image_version` | Resolved at runtime |
| Granted token permissions | `log_insights[*].token_permissions` | Printed by the runner at runtime |
| Artifact / rerun / cancel URLs | `metadata.artifacts_url`, `metadata.rerun_url`, `metadata.cancel_url` | Post-hoc existence signals |
| Previous attempt URL | `metadata.previous_attempt_url` | Populated only after a re-run exists |

`total_logs_size` deserves a specific warning: it is present for 90.7% of runs
and is trivially correlated with outcome, because failing jobs emit stack
traces while passing jobs emit little. It is the single most tempting and most
invalid feature in the dataset.

## 4. Inadmissible: two GitHub Actions-specific leakage classes

### 4.1 Fail-fast structural truncation

It is superficially reasonable to describe a workflow's *structure* — how many
jobs, how many steps, which actions, which runner OS, matrix breadth — and to
read that structure out of `log_insights`, since structure is declared in the
YAML and therefore knowable in advance. This is invalid.

GitHub Actions implements fail-fast: when one job in a matrix fails, its
concurrent siblings are cancelled, and when a step fails, the remaining steps
in that job never run. The step and job lists recovered from logs are
therefore **truncated by the outcome being predicted**. A failed run
systematically shows fewer steps, fewer jobs, and no late-stage steps such as
`upload-artifact` or a deployment action. "Number of steps" extracted from
logs is close to a proxy for the label.

The correct route to structural features is to parse `.github/workflows/<path>`
**as it existed at `metadata.head_sha`**, fetched from the Git history, never
from the logs. The declared structure is pre-execution; the observed structure
is not. Demonstrating the size of this gap empirically — train one model on
log-derived structure and one on YAML-derived structure, and report the
difference — is a self-contained, publishable result.

### 4.2 Repository snapshot aggregates

Every field under `repo.*` in `repositories.json.gz` is a **single snapshot
taken when the crawler visited the repository**, which for most runs is after
the run completed. Using them attributes October information to a July run.

| Field | Problem |
| --- | --- |
| `repo.stargazers`, `repo.forks`, `repo.watchers` | Monotonically growing counters measured after the fact |
| `repo.openIssues`, `repo.totalIssues`, `repo.openPullRequests`, `repo.totalPullRequests` | Change continuously; post-dated |
| `repo.commits`, `repo.branches`, `repo.releases`, `repo.contributors` | Post-dated counts |
| `repo.pushedAt`, `repo.updatedAt`, `repo.lastCommit`, `repo.lastCommitSHA` | Describe activity *after* most runs; `lastCommit` can postdate the run by months |
| `repo.codeLines`, `repo.blankLines`, `repo.commentLines`, `repo.size`, `repo.metrics` | Measured on the snapshot tree, not the tree at `head_sha` |
| `total_runs_90d`, `nb_runs`, `nb_workflows` | Window aggregates that include runs in the test set |
| `repo.isArchived` | Archival may postdate the run |

`total_runs_90d` is the worst offender: it was the criterion used to select
repositories into the dataset, so it encodes forward-looking activity for the
whole window and is an aggregate over the very runs being predicted.

A small number of `repo.*` fields are defensible because they are either fixed
or change slowly enough that the snapshot value is a close approximation of
the value at run time: `repo.createdAt` (fixed, gives repository age at run
time once differenced against `metadata.created_at`), `repo.defaultBranch`,
`repo.license`, `repo.mainLanguage`, `repo.hasWiki`, `repo.isFork`. Admit
these, and justify each one in a sentence.

Because that judgement is contestable, run the sensitivity analysis rather
than asserting the answer: fit the tabular models with and without the
slow-moving snapshot fields and report both. If the gap is large, the honest
conclusion is that snapshot features cannot be used.

## 5. Admissible with care: history features

These are legitimate — past outcomes are known at prediction time — but they
must be computed under two constraints.

1. **Strict causal ordering.** For a target run, aggregate only over runs whose
   `created_at` is strictly earlier. Never compute a per-repository or
   per-workflow failure rate over the whole dataset and join it back; that is
   future information leakage, and given the 18.4% between-repository standard
   deviation in failure rate it will dominate every other feature.
2. **Explicit missingness.** 25.5% of runs have no adjacent predecessor. Do not
   impute a global median into `previous_outcome`; carry an explicit
   "no history" indicator so the model can learn the cold-start regime, and
   report regime A results separately.

| Feature | Definition |
| --- | --- |
| Previous outcome (same workflow) | Outcome of the run with `run_number - 1`, when present |
| Consecutive prior failures | Length of the failure streak immediately preceding, capped at 4 by the sampling |
| Prior failure rate, same workflow | Over strictly earlier in-dataset runs only |
| Prior failure rate, same repository | Over strictly earlier runs of all the repository's workflows |
| Prior failure rate, same actor | Over strictly earlier runs triggered by that actor |
| Runs since last failure | Capped by the five-run window |
| Time since previous run | `created_at` difference, in hours |

Because at most four predecessors exist per workflow, every one of these is
truncated. State the cap wherever the features are described; a reader who
assumes TravisTorrent-style unbounded history will otherwise misread the
results.

## 6. Tier B: optional enrichment from the GitHub API

The dataset gives the commit SHA but not the diff. Diff-level features —
files changed, lines added and deleted, whether tests or the workflow file
itself were touched, configuration-only versus source changes — are strong
predictors in the just-in-time defect prediction literature and would
materially strengthen the work.

They require fetching, for each sampled run, the commit at `head_sha` and the
workflow YAML at that SHA. This carries real risks:

- **Availability decay.** The data was collected in 2023. Force-pushes, deleted
  branches, deleted repositories, and history rewrites mean a fraction of SHAs
  are no longer resolvable. The one published enrichment attempt on this
  dataset reported 87.4% completion; expect lower now.
- **Rate limits.** 5,000 authenticated REST calls per hour. Two calls per run
  makes full enrichment of 574k runs impractical; a stratified sample of
  40,000–60,000 runs is the realistic target.
- **Survivorship bias.** Repositories that still resolve are not a random
  sample of those that did in 2023.

Therefore: **pre-register Tier A as the primary analysis** so the dissertation
cannot fail on an external dependency, and treat Tier B as a secondary study
on a stratified subsample with the resolution rate reported. Measure the
resolution rate on a pilot sample of ~500 SHAs before committing to this
strand at all.

## 7. Reserved: logs as oracle, never as input

The 142 GB log archive is excluded from every feature set by construction, and
used for three evaluation purposes instead:

1. **Failure-cause labels.** Parse failed runs into categories — test failure,
   compilation error, dependency resolution, lint, timeout, infrastructure or
   network, out-of-memory — to support the multi-class extension and, more
   importantly, the explanation-faithfulness study.
2. **Explanation ground truth.** Check whether the tokens a model attributes
   its prediction to correspond to the cause the log reveals. This is what
   turns "the explanation is plausible" into "the explanation is correct", and
   it is only possible because this dataset ships logs for the same runs.
3. **Cost accounting.** `metadata.updated_at - metadata.created_at` gives run
   duration, which converts a prediction policy into minutes and currency
   saved. Used in the utility evaluation only; never as a feature.

Keeping this boundary explicit is what makes the design defensible: the one
genuinely novel asset of GHALogs is deliberately quarantined from the input
side.

## 8. Automated enforcement

Prose contracts are not enforcement. Implement the contract as code and run it
in CI:

- Maintain an explicit allowlist of admissible JSON paths. The feature builder
  reads the allowlist; any path not on it raises rather than being silently
  dropped.
- Assert that no feature column correlates with the label beyond a declared
  threshold (for example |r| > 0.9, or AUC > 0.95 for a single feature) without
  a written justification recorded in the test.
- Assert that for every history feature, the maximum `created_at` of the
  contributing rows is strictly less than the target row's `created_at`.
- Snapshot the resolved feature list into the artefact bundle for every
  experiment so that reported numbers are traceable to a feature set.
