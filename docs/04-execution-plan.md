# Execution plan

Eight phases, ordered by dependency. Each has an exit gate — a checkable
condition that must hold before the next phase starts. The gates matter more
than the ordering: most dissertations in this area fail at Phase 2 (a leak is
discovered after the models are trained) or Phase 5 (results beat nothing).

No calendar estimates are given; sequence and gate criteria are what the
supervisor should be reviewing against.

---

## Phase 0 — Protocol pre-registration

Write the analysis plan before touching a model: population and exclusions,
feature contract, splitting protocol, model list, primary metric, hypotheses,
and the statistical tests. Commit it, timestamp it, and have the supervisor
sign it off.

The reason is specific to this project. The headline comparison is against a
straw man that scores 90.97% accuracy. Without a pre-registered primary
metric, there is an overwhelming temptation to go metric-shopping after the
fact, and the work loses its credibility. `docs/03-research-design.md` is the
basis for this document; the primary metric decision is in Phase 5.

**Gate:** signed-off protocol in version control. Hypotheses H1–H7 recorded
with their directions.

---

## Phase 1 — Data acquisition and consolidation

Download `repositories.json.gz` (69 MB) and `runs.json.gz` (1.06 GB) from
Zenodo record 14796970. Do **not** download the 142 GB log archive yet; it is
needed only in Phase 7, and only for a sample.

Run `scripts/profile_dataset.py` and confirm it reproduces
`docs/01-dataset-profile.md`. If the numbers differ, the dataset version
changed and the profile must be regenerated before anything downstream.

Convert both files to Parquet, partitioned by repository, with one row per
run. JSON lines at this size are workable in a streaming pass but unusable for
iterative experimentation; Parquet plus DuckDB gives interactive queries on a
laptop. Keep the raw `.gz` files immutable and treat Parquet as derived.

Apply the population filter: keep `metadata.conclusion in {success, failure}`,
yielding 573,993 rows. Record the count of every exclusion in a table that
goes straight into the dissertation's methodology chapter.

**Gate:** 573,993 rows in Parquet; exclusion table written; profiler output
matches the committed profile.

---

## Phase 2 — Feature construction under the contract

Implement `docs/02-feature-contract.md` as code, not as documentation.

Build three feature groups, kept physically separate so ablations are trivial:

- `text_*` — the unstructured trigger-time fields (commit message, branch,
  display title, workflow name, repository description, topics).
- `struct_*` — structured trigger-time context (event, actor type, bot flag,
  fork flags, timestamp features, PR-present flag, run attempt).
- `hist_*` — causally ordered history features with explicit missingness
  indicators.

Non-negotiable implementation requirements:

1. **Allowlist, not denylist.** The builder reads an explicit list of
   admissible JSON paths and raises on anything else. A denylist guarantees
   that the next field you forget about becomes a leak.
2. **Causal history joins.** Every `hist_*` aggregate is computed over rows
   with strictly smaller `created_at`. Given the 18.4% between-repository
   standard deviation in failure rate, a careless whole-dataset group-by on
   repository will produce a near-perfect feature and a worthless model.
3. **Missingness carried, not imputed.** 25.5% of runs have no predecessor.
   Encode "no history" as a flag; do not fill with a median.
4. **A test suite that fails the build.** Assert no single feature exceeds a
   declared single-feature AUC threshold without written justification; assert
   the temporal ordering of every history aggregate; assert no path outside
   the allowlist reached the frame.

Also construct, deliberately, the **leaky comparison feature sets** needed for
RQ1: one adding log-derived structural features from `log_insights`, one
adding snapshot aggregates from `repo.*`, one adding `total_logs_size` and
duration. These exist solely to quantify inflation and must be named so they
can never be confused with the clean set (`features_leaky_structural`, etc.).

**Gate:** leakage test suite green on the clean set; the same suite *fails* on
each leaky set, demonstrating it has teeth. Feature counts and definitions
tabulated.

---

## Phase 3 — Splitting protocol

Three split strategies, with a clear primary.

**Primary — repository-grouped, 5-fold.** No repository appears in both train
and test. This is the cross-project setting and the only one that answers "can
this work on a project the model has never seen". It is mandatory here because
32.4% of repositories never fail and the model can otherwise score well by
memorising repository identity.

**Secondary — temporal.** Train on runs before a cutoff, test after. Use this
as a robustness check only, and state the caveat from
`docs/01-dataset-profile.md`, Section 10: because only the five most recent
runs per workflow were captured and capture proceeded over time, the two
halves are not exchangeable. A chronological cut here is weaker evidence than
it would be on a longitudinal dataset.

**Tertiary — within-project online validation.** For repositories with enough
runs, walk forward run by run. This mirrors how a deployed model would be used
and is the right setting for the cost analysis.

Hold out a validation set from the training folds for model selection and
threshold choice. **The test fold is touched once, at the end.** The plan calls
for selecting the best deep or transformer model and then optimising it; if
that selection is made on test data, every subsequent number is invalid.

**Gate:** split indices materialised, versioned, and shared by all models. No
model-specific splits.

---

## Phase 4 — Baselines before models

Establish these first and carry them through every results table.

| Baseline | Definition | Known value |
| --- | --- | --- |
| Majority class | Always predict success | 84.12% accuracy, 0 failure recall |
| Base rate | Predict 0.1588 for everything | Calibrated, zero discrimination |
| **Straw man** | Copy the previous run's outcome | 90.97% accuracy, 0.708 failure F1 (regimes B+C) |
| Per-repository prior | Causal prior failure rate for the repository | To be measured |
| Keyword rules | Hand-written regex over branch name and commit message | To be measured; branch buckets alone span 12.9%–27.4% |

The keyword baseline is the one most often omitted and the most important for
RQ2. If a fine-tuned transformer over commit messages and branch names cannot
beat ten regexes, the representation learning contributed nothing, and that is
a finding worth reporting.

**Gate:** all five baselines evaluated on the primary split, per regime, with
repository-level bootstrap intervals. These numbers are frozen as the
comparison targets.

---

## Phase 5 — Model development

### Metric decision, made now

Accuracy is near-useless at a 15.88% base rate and is beaten by a heuristic.
Primary metric: **failure-class PR-AUC (average precision)**, with **MCC** as
the primary threshold-dependent summary. Report alongside, always: ROC-AUC,
failure-class precision/recall/F1 at the chosen threshold, Brier score,
expected calibration error, and accuracy (for comparability with prior work
only). Every metric reported per regime as well as pooled.

### The six models

Chosen so that each pair genuinely differs in inductive bias, rather than
being two variations of the same thing.

**Classical (2)**
- *Logistic regression* with L2, on `struct_* + hist_*` plus TF-IDF character
  and word n-grams over the text fields. The fully transparent control
  condition, and a surprisingly strong text baseline.
- *Gradient boosting* (LightGBM or XGBoost) on the same features. This is the
  real competitor. On short text with strong tabular signal, boosting over
  TF-IDF and engineered features frequently beats fine-tuned encoders, and the
  dissertation is stronger for having tested that honestly. Consider adding
  Random Forest as a third, purely to align with the published GHALogs
  comparison point.

**Deep (2)**
- *Embedding MLP over tabular inputs* — learned entity embeddings for
  high-cardinality categoricals (event, language, actor bucket, workflow name
  tokens) feeding a multilayer perceptron. Tests whether deep tabular
  modelling beats boosting on the same inputs.
- *BiLSTM with attention over the text sequence* — trained from scratch or over
  static embeddings. This is the pre-transformer text model; its attention
  weights also give a cheap attribution signal for Phase 7. Choosing an LSTM
  here keeps continuity with the DL-CIBuild line of work, which applied LSTMs
  to CI build prediction.

**Transformers (2)**
- *A code-pretrained encoder* — CodeBERT or UniXcoder. Pre-trained on paired
  natural language and code, which is exactly the register of commit messages
  and branch names. 512-token window, which comfortably covers the text fields
  (commit message p95 is 591 characters).
- *A long-context modern encoder* — ModernBERT-base, or Longformer as a
  fallback. The long window matters only if the Tier B workflow-YAML strand
  succeeds, since YAML files exceed 512 tokens. If Tier B is dropped, replace
  this with a second short-context encoder of a different pre-training
  lineage (for example RoBERTa) so the comparison still contrasts code-domain
  against general-domain pre-training.

For each transformer, build three input conditions to serve RQ2: text-only,
structured-only (a tabular model, for reference), and **late fusion** — the
encoder's pooled representation concatenated with the scaled structured and
history features, passed through an MLP head. Late fusion is preferred over
serialising numbers into the prompt, which wastes context and tokenises badly.

### Imbalance handling

Use class weighting in the loss, or focal loss, as the default. Avoid SMOTE:
it is incoherent on text and on grouped temporal data, and synthesising
minority examples across repository boundaries manufactures runs that could
not exist. If SMOTE is reported at all, report it as a sensitivity analysis
with the objection stated.

### Compute

Short sequences keep this tractable. A 256-token encoder over ~574k rows for
3 epochs is a single-GPU job; one A100 or equivalent will do, and Colab Pro or
an institutional HPC allocation is sufficient. If compute is constrained,
train the transformers on a stratified subsample (150k–200k runs) and report
the sample size honestly rather than silently shrinking the data. LoRA or
other parameter-efficient fine-tuning cuts memory substantially and is worth
including as an efficiency result in its own right.

**Gate:** all six models trained on identical splits and features, evaluated
on validation only. Test folds untouched. A results table against the Phase 4
baselines, per regime.

---

## Phase 6 — Optimisation of the selected model

Select the best deep or transformer model **on validation PR-AUC**, record the
selection and the margin, then optimise along these axes:

1. Hyperparameters — learning rate, schedule, warmup, batch size, weight
   decay, dropout, number of frozen layers. Use Bayesian search (Optuna) over
   random search; log every trial.
2. Input construction — which text fields to concatenate and in what order,
   separator tokens, truncation strategy, maximum length.
3. Fusion — late concatenation versus gated fusion versus a cross-attention
   head over structured features.
4. Imbalance — class weights versus focal loss, and the loss's effect on
   calibration, which is usually degraded by reweighting.
5. Parameter-efficient fine-tuning — LoRA rank and target modules.
6. Calibration — Platt scaling, isotonic regression, or beta calibration
   fitted on validation data. Treat this as part of the model, not an
   afterthought: CI triage work shows post-hoc calibration substantially
   reduces fixed-threshold decision cost even when ranking metrics barely
   move.

**The seed discipline.** Repeat the final configuration across at least five
seeds and report mean and standard deviation. An improvement smaller than the
seed standard deviation is not an improvement. Reporting one lucky run as an
optimisation gain is the most common way this phase goes wrong.

**Gate:** optimised configuration frozen, calibrated, with seed variance
quantified, and the honest statement of how much of the gain survives it.

---

## Phase 7 — Explainability

### 7a. Global and local explanations

- Global: TreeSHAP for gradient boosting, permutation importance for all
  models, ALE plots for the main continuous features (preferred over partial
  dependence, which misleads under correlated features — and history features
  here are strongly correlated).
- Local: SHAP for individual predictions; LIME as a cross-check, with
  disagreement between the two reported rather than hidden.
- Token-level for the deep and transformer models: Integrated Gradients and
  token-level SHAP as the primary methods. Attention rollout may be included
  as a secondary view, with the standard caveat that attention weights are not
  explanations.
- Counterfactuals (DiCE or similar) restricted to *actionable* features.
  "Rename your branch" and "split this commit" are actionable; "make your
  repository older" is not. Separating actionable from non-actionable
  attributions is what makes the framework useful rather than merely
  transparent.

### 7b. Faithfulness evaluation

Plausible explanations are not necessarily faithful ones. Measure:

- **Comprehensiveness** — prediction change when the top-attributed tokens are
  removed.
- **Sufficiency** — prediction change when only the top-attributed tokens are
  kept.
- **AOPC** over progressive token deletion.
- **Stability** — attribution similarity under semantically neutral
  perturbations (rewording a commit message, changing capitalisation).

### 7c. Ground-truth cause alignment — the key experiment

This is where the dataset's logs earn their place, and it is the single most
novel component of the project.

1. Download the log archive, or the subset covering a stratified sample of
   2,000–3,000 failed runs. `log_insights` is already present for 89.2% of
   runs and helps locate the failing job and step without full-text parsing.
2. Classify each sampled failure into a cause category — test failure,
   compilation error, dependency resolution, lint or format, timeout,
   infrastructure or network, out-of-memory, configuration. Build the
   classifier from regular expressions over the failing step's output, then
   **manually validate a stratified subsample of 300–400 runs with a second
   annotator and report Cohen's kappa.** A supervisor or peer can serve as the
   second annotator; the agreement figure is what makes the labels citable.
3. For each sampled run, test whether the model's top-attributed tokens
   correspond to the identified cause. For example: does a dependency-resolution
   failure on a `renovate/` branch attribute to the branch token and the
   lockfile mention in the commit message, or does it attribute to the
   repository name?
4. Report alignment against a random-attribution baseline and, ideally, against
   human annotators asked to predict the cause from the same metadata. Rate of
   agreement above chance but below human is the expected and publishable
   result.

### 7d. Cost of transparency

Plot accuracy or PR-AUC against an interpretability ordering (logistic
regression → shallow tree → gradient boosting with SHAP → optimised
transformer). This is the quantitative answer to the question the original
project framing raised, and it supports whichever recommendation the data
actually warrants.

**Gate:** faithfulness metrics computed; cause-alignment study complete with
inter-annotator agreement reported; trade-off curve produced.

---

## Phase 8 — Deployment analysis, threats, and write-up

### Cost model

Convert predictions into decisions. GitHub Actions bills per minute, and
`metadata.updated_at - metadata.created_at` gives each run's duration — a
legitimate use of post-execution data, because it enters the utility
calculation and never the feature set.

Define the asymmetric costs: a missed failure wastes the full run duration; a
false alarm costs developer attention and, if it causes a skipped or deferred
build, risks delaying a legitimate change. Sweep the decision threshold and
report expected cost, with and without calibration. Then state plainly the
threshold range, if any, in which acting on predictions is net-positive. A
negative answer here is a legitimate and valuable result.

### Statistical rigour

- Bootstrap confidence intervals **over repositories**, not over runs. Runs are
  nested in workflows nested in repositories; run-level bootstrapping
  understates variance badly.
- Friedman test across folds for the six-model comparison, with Nemenyi
  post-hoc; Wilcoxon signed-rank for pairwise comparisons.
- Cliff's delta for effect sizes. A statistically significant 0.3pp difference
  on 574k rows is not a practically meaningful one.
- Holm or Bonferroni correction for the family of comparisons, and say how
  many comparisons are in the family.

### Threats to validity

Write these from the profile rather than from a template:

- *Construct* — `cancelled` runs excluded introduces selection bias; binary
  labels collapse distinct failure modes; flakiness means some labels are not
  deterministic functions of pre-execution state (2.65% rerun rate, with
  roughly two thirds of reruns flaky in the literature), setting an
  irreducible error floor.
- *Internal* — shallow history (at most four predecessors per workflow);
  `repo.*` snapshot timing; residual leakage risk from the fields in
  `docs/02-feature-contract.md`, Section 4.
- *External* — public repositories with 100+ stars and 30+ runs per 90 days
  only; GitHub Actions only; 4.5-month window; no enterprise or private code.
- *Conclusion* — large n makes trivial differences significant, hence the
  effect-size requirement; multiple comparisons across six models, three
  input conditions, and three regimes require correction.

### Ethics and data protection

This needs doing early, not at the end. Commit messages, author names, and
email addresses are personal data under UK GDPR, and GHALogs ships all three.

- Pseudonymise actor logins and author emails with a salted hash; keep the salt
  out of the repository.
- Do not publish derived datasets that contain raw author identifiers, and do
  not quote commit messages verbatim in the dissertation without checking for
  personal information.
- If a developer survey is used for explanation evaluation, institutional
  ethics approval is required before any recruitment. Start that application
  during Phase 1 if the survey is wanted, since approval timelines are outside
  your control.
- Discuss the fairness implication of actor-identity features explicitly: a
  model that flags builds based on who authored them is a performance-
  management tool, not an engineering tool, and the dissertation should say so.

### Artefacts

Reproducibility package: feature contract as code, leakage test suite, dataset
profiler, split indices, model configurations, trained weights or a recipe to
reproduce them, and the pre-registered protocol with any deviations recorded
and explained.

**Gate:** cost analysis complete; threats written; ethics position documented;
artefact bundle archived with a DOI.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Models fail to beat the 90.97% straw man | High | Severe if framed as outcome prediction | Reframe around regimes A and B per `docs/03-research-design.md`, Section 1; the regime decomposition is a contribution on its own |
| A leak is found after models are trained | Medium | Severe — invalidates results | Phase 2 gate with an executable audit that must fail on the leaky sets |
| Tier B enrichment unavailable (2023 SHAs no longer resolvable) | Medium-high | Moderate | Pre-register Tier A as primary; pilot 500 SHAs before committing; report the resolution rate |
| GPU capacity insufficient for 574k rows | Medium | Moderate | Short sequences, LoRA, or an honestly reported stratified subsample |
| 142 GB log download impractical | Medium | Moderate | Use `log_insights` for most of the cause labelling; download logs only for the 2–3k sampled failures |
| Cause-label quality too poor for the alignment study | Medium | Moderate | Regex plus manual validation with a second annotator and reported kappa; narrow to the cause categories that annotate reliably |
| Transformers lose to gradient boosting | Medium-high | Low | This is a finding, not a failure. Pre-register it as a possible outcome so it reads as a result rather than an excuse |
| Scope creep across six models, three input conditions, three regimes, and an XAI study | High | Moderate | RQ4 and RQ6 are explicitly the trimmable depth; RQ1–RQ3 and RQ5 are the core |

## Mapping to dissertation chapters

| Chapter | Source |
| --- | --- |
| Introduction, aim, objectives | `docs/03-research-design.md` §1, §5 |
| Literature review and gaps | `docs/03-research-design.md` §2 |
| Research questions and hypotheses | `docs/03-research-design.md` §3, §4 |
| Dataset | `docs/01-dataset-profile.md` |
| Methodology — features and leakage | `docs/02-feature-contract.md`, Phases 2–3 |
| Methodology — models and evaluation | Phases 4–6 |
| Results | Phases 4–6 outputs |
| Explainability | Phase 7 |
| Discussion, deployment, threats | Phase 8 |
| Conclusion and future work | `docs/03-research-design.md` §6, §8 |
