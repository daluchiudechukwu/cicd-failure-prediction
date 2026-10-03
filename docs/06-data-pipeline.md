# Data pipeline: from GHALogs to a model-ready feature table

Three commands turn the published dataset into a leakage-audited feature
table. Every number quoted below is output from running them on the full
dataset, not an estimate.

```sh
export GHALOGS_SALT="$(openssl rand -hex 16)"   # keep this out of the repo

python -m ghalogs.pipeline ingest   --data-dir /path/to/ghalogs --out-dir data
python -m ghalogs.pipeline prepare  --out-dir data
python -m ghalogs.pipeline features --out-dir data

python scripts/audit_pipeline_output.py --features data/features.parquet
```

Measured runtime on 4 cores: ingest 2m01s, prepare 28s, features 14s. Peak
memory stays under 4 GB. Only the two metadata files are needed; the 142 GB
log archive is not touched.

## Module map

| Module | Responsibility |
| --- | --- |
| `src/ghalogs/config.py` | Every threshold and vocabulary, in one place, so the methodology chapter has a single citable source |
| `src/ghalogs/ingest.py` | Streaming JSON-lines to flattened Parquet |
| `src/ghalogs/quality.py` | Noise filtering with an auditable exclusion ledger |
| `src/ghalogs/textproc.py` | Unicode canonicalisation, PII redaction, pseudonymisation |
| `src/ghalogs/features.py` | 122 static (history-free) features |
| `src/ghalogs/history.py` | 10 causally ordered history features |
| `src/ghalogs/contract.py` | Executable enforcement of the feature contract |
| `src/ghalogs/pipeline.py` | Orchestration and CLI |
| `tests/test_pipeline.py` | 28 tests pinning the correctness-critical behaviour |

---

## Stage 1 — Ingestion

**What it does.** Reads `runs.json.gz` and `repositories.json.gz` one line at a
time, projects each nested record onto a flat Arrow schema, and appends it to
Parquet in 50,000-row batches.

**Why streaming.** `runs.json.gz` is 1.06 GB compressed and expands to roughly
8 GB of JSON text. Each record is deeply nested: 35 metadata keys, two
complete repository objects, and a `log_insights` array that can hold
thousands of parsed steps. `pd.read_json` on this is both unnecessary and
unreliable on a laptop. Streaming keeps peak memory proportional to the batch
size, and the resulting Parquet file is 68 MB — a 16× reduction, because
roughly 90% of each record is API URLs and avatar links that no model will
ever use.

**Why post-execution fields are still projected.** Fields such as
`updated_at`, `total_logs_size`, and the job count from `log_insights` are
carried through with an `audit_` prefix. They are needed to build the
exclusion ledger, to compute run duration for the cost analysis in Phase 8,
and to construct the deliberately-leaky comparison feature sets for RQ1. The
feature contract rejects any column whose name contains `audit_`, so they
cannot reach a model by accident.

**Why repository counters are dropped at source.** `flatten_repository` keeps
only seven fields: language, creation time, default branch, licence presence,
wiki flag, fork flag, and topic count. Stars, forks, watchers, commit counts,
line counts, and `total_runs_90d` are never read, because each is a single
snapshot taken when the crawler visited — which for most runs is after the run
finished. Omitting them at ingestion rather than filtering them later means
there is no intermediate artefact in which they exist to be accidentally used.

```python
def parse_timestamp(value: str | None) -> datetime | None:
    """GHALogs mixes two formats: run timestamps end in `Z` while
    repo.createdAt is naive. Both are UTC, so naive values are localised
    rather than discarded."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
```

That one function matters more than it looks: mixing naive and aware
timestamps raises `TypeError` on subtraction, which is how repository age is
computed.

---

## Stage 2 — Noise filtering

**What it does.** Reduces 580,641 ingested runs to the 567,814-run modelling
population, recording every exclusion in a ledger.

**Why a ledger rather than a boolean mask.** The ledger *is* the exclusion
table for the methodology chapter. A reader can see exactly how the population
was derived, and a reviewer can check that nothing convenient was quietly
dropped. `filter_ledger.csv` is written on every run.

Measured output:

```
exclusion ledger:
  ingested                      580,641
  duplicate run_id             -      0 ->  580,641  same run ingested more than once
  missing identity/timestamp   -      0 ->  580,641  cannot be ordered or grouped
  impossible timestamps        -     14 ->  580,627  run ended before it started
  conclusion=skipped           -  4,113 ->  576,514  matched no triggering condition
  conclusion=cancelled         -  1,713 ->  574,801  outcome unknown; selection bias
  conclusion=action_required   -    447 ->  574,354  blocked awaiting approval
  conclusion=startup_failure   -    375 ->  573,979  never started, usually invalid YAML
  null head_commit             -    133 ->  573,846  primary unstructured feature absent
  empty head_branch            -     19 ->  573,827  core pre-execution signal
  repository not selected      -  6,013 ->  567,814  outside the documented sampling frame
```

**Judgements worth defending in the viva.** Dropping 1,713 `cancelled` runs
introduces selection bias: a developer who cancels a run they expect to fail
removes a positive case. Dropping 375 `startup_failure` runs removes the one
failure mode that pre-execution state fully determines — invalid YAML — which
is arguably the easiest thing to predict, so keeping them would flatter the
model. Both are stated as threats to construct validity rather than buried.

The 6,013 runs belonging to repositories without `selected: true` are a
genuine finding: the published runs file contains runs from repositories
outside the sampling frame the dataset paper describes.

**Filters that are switches, not defaults.** `--drop-reruns` and
`--drop-dynamic` exist for the sensitivity analyses. Both default to off,
because 15,158 re-run attempts and 29,585 `dynamic`-event runs are real
traffic, and removing real traffic to make a metric look better is the thing
the whole protocol is designed to prevent. Their properties are measured
instead: the failure rate among re-runs is **28.51%**, nearly double the
15.87% baseline, which is what you would expect if people re-run things that
broke.

### The commit-group finding

`annotate_commit_groups` groups runs by `(repo, head_sha)`. This was added
after measurement, and it changes the project's design:

```
runs sharing a commit with another run:     396,625 (69.9%)
runs in commit groups with mixed outcomes:  126,410 (22.3%)
failures sharing a commit with a success:    43,012 (47.7% of failures)
irreducible error for a commit-text-only model: ~36,469 runs (6.42%)
```

One push typically starts several workflows. In 22.3% of runs, the same commit
passes one workflow and fails another. **47.7% of all failures share a commit
with a success**, so for nearly half the positive class, commit text alone
cannot possibly separate the cases — the difference is which workflow ran, not
what changed.

Two consequences:

1. **Workflow identity is a required input, not an optional extra.** A model
   reading only the commit message has a floor of 6.42% error on a task with a
   15.88% base rate. `features.workflow_features` and the field ordering in
   `textproc.build_transformer_input` both exist because of this number.
2. **It gives the dissertation a principled Bayes-error floor.** The
   literature reports accuracy as though 100% were attainable. This is a
   measured bound on what any commit-level model can achieve, and reporting it
   is one of the contributions in `docs/03-research-design.md`.

---

## Stage 3 — Text processing

**What it does.** Canonicalises Unicode, redacts personal data and
credentials, truncates to a budget, and replaces identifiers with salted
hashes.

**Why the order is canonicalise → redact → truncate.** This was corrected
twice, both times because an audit of real output caught a failure.

Redacting before canonicalising left 7 addresses in the corpus, because NFKC
folds fullwidth characters onto ASCII: `alice＠example.com` only becomes
matchable as an email *after* normalisation. Truncating before redacting would
mean an address near the end of a long message is removed from short records
and silently retained in long ones.

**Why the email pattern requires an alphabetic TLD.** A naive
`[\w.+-]+@[\w-]+(\.[\w-]+)+` matches the version references that saturate this
corpus — `actions/checkout@v3...v4`, `calcite-components@1.10.0`,
`python@3.11`. Redacting those would destroy exactly the dependency-bump
signal that makes `renovate/` branches (27.4% failure rate) predictive.
Requiring the final segment to be `[A-Za-z]{2,}` removes the false positives.

**Why identifier columns get a restricted redaction.** The first design left
branch and workflow names unredacted, on the grounds that a full pass would
rewrite `renovate/lodash-4.x` into noise. The output audit then found a
contributor's real email address used as a branch name in 9 runs. Identifier
columns are not exempt from privacy obligations just because they are usually
machine-generated. `redact_identifiers` applies addresses and credentials
only, leaving refs intact. It over-matches on exactly one workflow name in
567,814 rows, which is the right direction to err.

**Why markers are kept rather than deleted.** A redacted span becomes
`<EMAIL>`, `<URL>`, `<SHA>` rather than disappearing. "This message cites a
changelog URL" is plausible signal — dependency bumps do it constantly — while
the URL itself is not. 103,269 messages carry an `<EMAIL>` marker and 68,402 a
`<URL>` marker, and `static_msg_has_url_marker` is a feature. Keeping the
markers also makes token-level explanations readable in Phase 7.

**Why pseudonymisation rather than deletion.** Per-actor history features need
the same actor to map to the same token, but nothing needs the actor's
identity. A salted SHA-256 truncated to 16 hex characters gives both. The salt
lives in `$GHALOGS_SALT` and must not be committed: without it the mapping
cannot be reproduced, which is the point.

```python
def redact(text: str) -> str:
    """Ordered most specific first: a token inside a URL should be caught as a
    secret, and an email inside a URL should not be split across two rules."""
    text = SECRET_RE.sub(REDACTION_TOKENS["secret"], text)
    text = EMAIL_RE.sub(REDACTION_TOKENS["email"], text)
    text = URL_RE.sub(REDACTION_TOKENS["url"], text)
    text = MENTION_RE.sub(REDACTION_TOKENS["mention"], text)
    text = SHA_RE.sub(REDACTION_TOKENS["sha"], text)
    return text
```

**Transformer input assembly.** `build_transformer_input` concatenates the
fields with explicit labels, workflow identity first and commit message last:

```
workflow: Build and Test | event: pull_request | branch: CON-2277-add-subject |
title: CON-2277 Create new subject | message: ...
```

Workflow identity leads because of the 47.7% finding. The message goes last
because it is the only field that can be long, so truncation at the
tokeniser's maximum length removes the least important content rather than the
most.

---

## Stage 4 — Feature engineering

132 features in two blocks, kept physically separate so the RQ2 ablation is a
column selection rather than a re-run.

### Static features (122)

Computable from the trigger event and slow-moving repository attributes alone,
with no reference to any previous run. These are what remain available in the
25.5% of runs that have no history.

| Family | Count | Rationale |
| --- | --- | --- |
| Branch shape and keywords | 28 | Failure rates span 12.9% (`main`) to 27.4% (`renovate/`) |
| Commit-message shape and keywords | 43 | Shape only; semantics are left for the text models to add value on |
| Workflow identity | 26 | Load-bearing, per the 47.7% finding |
| Trigger context | 16 | Event type spans 2.5% (`dynamic`) to 20.9% (`workflow_dispatch`) |
| Repository context | 9 | Age at run time (raw and log), licence, wiki, topics, description length, language |

Two deliberate choices. Message features are *structural* — length, line
count, conventional-commit conformance, presence of an issue reference — not
semantic, so that the transformer models have somewhere real to improve rather
than re-encoding the same information. And keyword flags use substring rather
than token matching, because `fix` should fire on `hotfixes` and `bugfix/`, and
branch names rarely tokenise cleanly.

Repository age is derived by differencing a fixed creation timestamp against
the run's own timestamp, which is legitimate. No crawl-time counter appears.

### History features (10)

Legitimately pre-execution — whether the last run passed is a fact available
the instant the next run is queued — and the strongest signal available.
History alone reaches PR-AUC 0.677 against 0.336 for all 122 static features
combined. That strength is exactly why three mistakes are easy and each
silently invalidates results.

**1. Whole-dataset aggregation.** Computing a per-repository failure rate over
the whole table and joining it back leaks future outcomes into every row. With
an 18.4% between-repository standard deviation and 32.4% of repositories never
failing, such a feature would dominate the model and collapse in deployment.
Every aggregate is an expanding window implemented as a cumulative sum minus
the current row, which provably excludes the current observation:

```python
group = frame.groupby(keys, sort=False)
prior_count = group.cumcount()
prior_failures = group[FAILURE_LABEL].cumsum() - frame[FAILURE_LABEL]
```

**2. Imputing missing history.** 25.5% of runs have no adjacent predecessor.
Filling those with a global median tells the model that unknown history looks
average, which it does not: cold-start failure rate is 18.0% against 5.0% for
runs following a success. Missingness is carried as a `-1` sentinel plus a
boolean flag, so the model can learn cold start as its own regime.

**3. Treating a sampling gap as adjacency.** GHALogs keeps at most five runs
per workflow, so `run_number` gaps are common. A predecessor counts only when
its `run_number` is exactly one lower; run 200 is not the successor of run
150. Without this check the "previous outcome" feature would be months stale
and the cold-start regime would be understated. `test_non_adjacent_run_numbers_are_treated_as_cold_start`
pins the behaviour.

### The actor-history leak

The first implementation pooled each actor's history across all repositories.
That is a leak, and a subtle one worth describing in the dissertation because
no per-row temporal check can detect it.

Features are built once over the whole table, while evaluation groups by
repository. Dependabot triggers runs in thousands of repositories, so an
unscoped actor failure rate for a *training* row would be computed partly from
runs belonging to *test-fold* repositories. Every contributing run genuinely
is earlier in time, so a temporal assertion passes — but information has
crossed the group boundary.

Scoping to `(repo, actor)` keeps the feature inside the boundary. The
diagnostic was visible in the statistics: mean prior-run count fell from 1,531
to 12.5 once scoped. The unscoped variant is only safe if features are
recomputed inside each fold, which is the alternative if the pooled signal
turns out to matter.

These features are also a fairness hazard regardless of scope. A model that
flags a change because of who wrote it is a performance-management tool, not
an engineering one. They are included so RQ5 can examine them and reported in
their own ablation so the dissertation can argue against using them.

---

## Stage 5 — Contract enforcement

`docs/02-feature-contract.md` states in prose what a model may see. Prose is
not enforcement. `contract.py` turns it into assertions that run before every
experiment, so a leak is caught at the gate rather than discovered after the
models are trained.

**Mechanism 1 — prefix allowlist.** Only `static_*` and `hist_*` columns are
usable. Anything that is neither a feature nor an explicitly permitted
passenger raises. An allowlist is used rather than a denylist because a
denylist fails silently: the field you forget to ban becomes a feature.

**Mechanism 2 — named-field denial.** Explicit rejection of every known
leakage mechanism, matched as substrings so derived columns are caught too,
each annotated with its leakage class. This includes the two GitHub
Actions-specific mechanisms: `n_log_jobs` and `n_log_steps` are refused
because fail-fast truncates log-derived structure, and `total_runs_90d` is
refused because it was the dataset's own selection criterion.

**Mechanism 3 — statistical screening.** Any feature whose univariate ROC-AUC
against the label exceeds 0.95 is flagged. This catches leaks that neither
list anticipated. AUC is used rather than Pearson correlation because it is
invariant to monotone transformations and defined for binary and skewed
features, which dominate this feature set. The check is direction-agnostic: a
feature that perfectly predicts *success* is as much of a leak as one that
predicts failure.

The screen is a tripwire, not a proof, so tripping it demands a written
justification recorded in `SCREEN_EXEMPTIONS` rather than automatic removal.
The dictionary is empty, which means no exemption has been needed.

Measured output on the full dataset:

```
contract: 132 admissible features
  static_*: 122   hist_*: 10
  no feature exceeds the univariate AUC screen
```

The strongest single-feature correlation with the label is 0.520
(`hist_prev_failure_streak`), and the strongest static feature is 0.102
(`static_wf_kw_test`). Those are the magnitudes you expect from weak
individual signals combining into a usable model, not from a leak.

---

## The audit gate

`scripts/audit_pipeline_output.py` is the Phase 2 exit gate as a command. It
exits non-zero on failure, so it belongs in CI.

```
== 1. privacy ==    pass: no addresses, URLs, credentials, or plaintext identifiers
== 2. feature contract ==    pass: 132 admissible features
== 3. history causality ==   pass: every aggregate uses strictly earlier runs only
== 4. sanity ==
  B_prev_success   n= 356,433 (62.77%)  failure= 4.97%
  A_cold_start     n= 144,491 (25.45%)  failure=18.04%
  C_prev_failure   n=  66,890 (11.78%)  failure=69.23%
AUDIT PASSED
```

It earned its place: the two text-processing bugs and the actor-history leak
were all found by running it on real output, not by reasoning about the code.

## Tests

28 tests, each pinning a property whose silent failure would invalidate
results rather than chasing coverage:

- the contract refuses unrecognised columns and each named leaky field
- the AUC screen detects a planted leak given an innocuous name and a valid
  prefix, in both directions
- previous-outcome matches the actual preceding run, and a `run_number` gap is
  treated as cold start
- the first run in a group carries the sentinel, not a group average
- a run's own outcome never enters its own history aggregate
- the causality assertion fires when a whole-group aggregate is substituted
- actor history does not pool across repositories
- redaction removes addresses, URLs, mentions, SHAs and credentials, survives
  fullwidth obfuscation, and runs before truncation
- branch names keep their naming conventions while still losing addresses

## Effect on results

Re-running the feasibility experiment on the pipeline's features improves
every regime over the earlier ad-hoc feature set (PR-AUC):

| Regime | `static` | `hist` | `both` |
| --- | --- | --- | --- |
| All | 0.295 → **0.336** | 0.669 → **0.677** | 0.701 → **0.716** |
| A — cold start | 0.322 → **0.373** | 0.357 → **0.371** | 0.435 → **0.479** |
| B — previous run succeeded | 0.119 → **0.121** | 0.125 → **0.152** | 0.197 → **0.225** |
| C — previous run failed | 0.725 → **0.739** | 0.884 → **0.887** | 0.894 → **0.899** |

The full pre-execution model now reaches **PR-AUC 0.716 and ROC-AUC 0.898**
across unseen repositories, 4.51× the base rate. Static features alone reach
ROC-AUC 0.716, up from 0.682. The largest gains are in the two regimes that
matter: cold start is up 10% and new breakage up 14% relative.

## Next step

`docs/04-execution-plan.md` Phase 3 onward: materialise the split indices,
then the six models. The feature table and the contract are now stable, so
every model trains on identical inputs and the comparison isolates
architecture.
