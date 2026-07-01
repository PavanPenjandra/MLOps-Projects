# MLOps for Data Scientists: An ML-Native Path to Production
### A 40-Hour Course for Experienced Data Scientists Transitioning into ML Engineering

**Format:** 8 sessions × 5 hours (or 5-day bootcamp × 8 hours/day)
**Prerequisites:** Working Python, pandas, scikit-learn/PyTorch fluency, comfort with the CLI and git basics
**Explicitly out of scope:** Docker, Kubernetes, Terraform, Ansible, Jenkins, GitHub Actions, cloud networking/IAM

---

## How This Course Is Different

Most "MLOps" courses are secretly DevOps courses wearing an ML costume — half the syllabus disappears into YAML, container registries, and cluster networking before a student ever touches a model artifact. This course inverts that. Every tool taught here was built *by and for* ML practitioners, speaks the language of experiments, datasets, features, and models (not pods and services), and can be run, debugged, and reasoned about entirely from a Python/CLI workflow a Data Scientist already has muscle memory for. The goal is for a student to walk away owning the full ML-native lifecycle — validate → version → track → register → store features → serve → monitor — before they ever need to learn how to write a Dockerfile.

---

## Session 1 — Data Validation & Quality Gates
**Core Concept:** "Garbage in, garbage out" is the single most expensive failure mode in production ML, and it's silent. This session reframes data validation as a CI-style gate that runs *before* training or inference, not an ad-hoc `df.describe()` habit.

**Topics:**
- Why schema drift and silent data quality failures are the #1 cause of production model degradation
- Declarative vs. assertion-based validation philosophies
- Great Expectations: Expectation Suites, Data Context, Checkpoints, Data Docs
- Pandera: schema-as-code, DataFrame-level and column-level checks, decorators for function I/O validation

**Hands-On Lab:**
1. Profile a raw tabular dataset and auto-generate a baseline Great Expectations suite
2. Hand-author custom expectations (null thresholds, categorical value sets, statistical ranges, referential checks across columns)
3. Wrap the same logic in Pandera as a reusable `DataFrameSchema`, and decorate a feature-engineering function with `@pa.check_types`
4. Build a Checkpoint that fails a (mock) pipeline run and generates a human-readable Data Docs HTML report

**Trade-off Discussion:** *Great Expectations vs. Pandera — when to use which?*
Great Expectations is heavier, more declarative, and produces shareable documentation/audit trails — a good fit for data engineering teams and regulated environments where validation needs to be visible to non-engineers. Pandera is lightweight, Pythonic, type-hint-native, and ideal for validating function inputs/outputs inside an ML codebase where you want validation to live next to the code it protects. Many production teams use Pandera inside the codebase and Great Expectations at the pipeline/warehouse boundary.

---

## Session 2 — Data & Pipeline Versioning with DVC
**Core Concept:** Git tracks code; it does not — and should not — track 50GB of training data. This session establishes data and pipeline versioning as the foundation that makes every later experiment reproducible.

**Topics:**
- Why "which CSV trained this model?" is an unanswerable question on most DS teams
- DVC architecture: `.dvc` files, remote storage, content-addressable caching
- DVC pipelines (`dvc.yaml`) as a lightweight, ML-native alternative to a workflow orchestrator
- DVC Experiments: tracking parameter sweeps without polluting git history

**Hands-On Lab:**
1. Initialize DVC in a git repo, add a dataset, push to a local/remote DVC store
2. Build a multi-stage `dvc.yaml` pipeline (validate → preprocess → train → evaluate) with explicit deps/outs
3. Run `dvc repro` and observe DAG-aware caching (only changed stages re-run)
4. Use `dvc exp run` to sweep hyperparameters and compare experiment results in a table without creating git branches per run

**Trade-off Discussion:** *DVC Pipelines vs. a full orchestrator (Airflow/Prefect) — when is DVC enough?*
DVC pipelines are sufficient when the entire ML workflow lives on one machine or one job runner and the primary need is reproducibility and caching, not scheduling, retries across distributed infra, or complex branching logic. Once you need cross-team scheduling, SLA-based retries, or fan-out across hundreds of parallel jobs, a dedicated orchestrator becomes necessary — but that's an infrastructure decision layered *on top of*, not instead of, DVC's versioning guarantees.

---

## Session 3 — Experiment Tracking with MLflow
**Core Concept:** A spreadsheet of hyperparameters and accuracy scores does not scale past a handful of experiments. This session builds the habit of treating every training run as a first-class, queryable artifact.

**Topics:**
- The anatomy of an MLflow run: params, metrics, tags, artifacts
- MLflow Tracking Server architecture (local file store vs. remote backend store + artifact store)
- Autologging vs. manual instrumentation
- Comparing runs programmatically and visually

**Hands-On Lab:**
1. Stand up a local MLflow Tracking Server with a SQLite backend store
2. Instrument a scikit-learn and a PyTorch training script with `mlflow.autolog()` and manual `log_param`/`log_metric`/`log_artifact` calls
3. Run a 20-point hyperparameter sweep, log each as a child run under a parent run
4. Use the MLflow UI and the `MlflowClient` Python API to programmatically pull the best run by a chosen metric

**Trade-off Discussion:** *MLflow Tracking vs. DVC Experiments — when to use which?*
DVC Experiments excel when reproducibility is paramount — every experiment is tied to an exact, versioned snapshot of data, code, and pipeline DAG, making "rerun this exact result" trivial. MLflow Tracking excels at the *exploration* phase — rich querying, comparison UIs, and metric visualization across hundreds of runs, but without DVC's strict data-lineage guarantee. In practice: use DVC to guarantee any experiment is reproducible from scratch, use MLflow to actually browse, compare, and communicate results to stakeholders. They are complementary, not competing.

---

## Session 4 — Feature Stores & Engineering Orchestration with Feast
**Core Concept:** Training/serving skew — where a feature is computed one way offline and a subtly different way online — silently destroys more production models than any algorithmic mistake. A feature store is the ML-native answer.

**Topics:**
- The training/serving skew problem, concretely demonstrated
- Feast architecture: Feature Views, Entities, Data Sources, Online vs. Offline Store
- Point-in-time correctness ("time-travel joins") and why it matters for avoiding label leakage
- Feature freshness and materialization

**Hands-On Lab:**
1. Define entities and Feature Views in a `feature_store.yaml` + Python feature repo
2. Generate a point-in-time-correct training dataset via `get_historical_features`
3. Materialize features to an online store (local SQLite/Redis-compatible) and serve them via `get_online_features`
4. Deliberately reproduce training/serving skew by bypassing Feast, then fix it by routing both paths through the same Feature View

**Trade-off Discussion:** *Feast vs. hand-rolled feature pipelines — when does a feature store earn its complexity?*
A feature store is overkill for a single model with a handful of static features computed once at training time. It earns its keep once features are *reused across multiple models*, computed by *different teams*, or require *low-latency online serving* with guaranteed consistency against the offline training view. The break-even point is usually somewhere around "more than one model in production" or "any model needing sub-100ms feature lookups."

---

## Session 5 — Model Registry & Lifecycle Management with MLflow
**Core Concept:** A trained model sitting in an S3 bucket with a UUID filename is not a managed asset. The Model Registry turns "the model" into a governed entity with versions, stages, and an audit trail.

**Topics:**
- Registering a model from a tracked run; semantic versioning of models
- Stage transitions: None → Staging → Production → Archived, and approval workflows
- Model signatures and input examples for contract enforcement
- Loading registered models generically via `mlflow.pyfunc`

**Hands-On Lab:**
1. Register the best run from Session 3's sweep as a new Model Registry entry
2. Add a model signature (inferred from training data) and an input example
3. Transition the model through Staging → Production via the API, attaching a transition comment/description
4. Load the Production-stage model purely by name/stage alias (no hardcoded run ID) using `mlflow.pyfunc.load_model`

**Trade-off Discussion:** *Registry-managed staging vs. ad-hoc "prod_model_v2_final.pkl" file conventions — what does formal governance actually buy you?*
File-naming conventions break down the moment more than one person touches the model lifecycle: there's no enforced transition history, no signature contract, no single source of truth for "what is currently in production." A registry costs you a small amount of process overhead (you must register and promote deliberately) in exchange for auditability, rollback-by-stage-alias, and the ability to decouple "which code trained this" from "which artifact is currently serving traffic."

---

## Session 6 — ML-Native Model Serving with BentoML / Triton + **Capstone Kickoff (Part 1 of 3)**
**Core Concept:** Serving a model doesn't require container orchestration to be production-grade. BentoML and Triton give you batching, versioning, and multi-framework serving as Python-native or config-native concerns.

**Topics:**
- BentoML's `Service`/`Runner` abstraction; building a "Bento" as a versioned, deployable serving unit
- Adaptive batching and async inference for throughput
- Triton Inference Server as a polyglot, high-performance alternative for GPU-heavy or multi-framework (ONNX/TensorRT/PyTorch) serving
- Choosing between them based on latency, framework diversity, and team skillset

**Hands-On Lab:**
1. Wrap the Session 5 Production-stage MLflow model in a BentoML `Service` with a REST endpoint
2. Add input validation (reusing the Pandera schema from Session 1) directly in the serving path
3. Build and version a Bento; serve it locally with `bentoml serve` and load-test it with adaptive batching enabled
4. (Stretch) Export the same model to ONNX and serve it through a local Triton instance, comparing latency/throughput

**Trade-off Discussion:** *BentoML vs. Triton — when do you need each?*
BentoML is the right default for Python-centric teams serving scikit-learn/PyTorch/XGBoost models who want serving logic, pre/post-processing, and batching expressed as ordinary Python — fastest path from "trained model" to "API." Triton is the right choice when you need maximum throughput on GPUs, must serve multiple frameworks (TensorRT/ONNX/PyTorch) behind one server, or are serving at a scale where its C++-level performance optimizations matter more than developer convenience.

### 🏗️ Capstone Project — Part 1: Foundation
Starting now, students build a single incremental pipeline that will be extended through Sessions 7 and 8. **No Dockerfiles, no Kubernetes manifests — every step is a local Python/CLI workflow.**

*Part 1 deliverables (this session):*
- A Pandera/Great Expectations validation gate on the chosen capstone dataset
- The dataset and a `dvc.yaml` pipeline versioned in DVC
- An MLflow-tracked baseline training run, registered to the Model Registry at "Staging"
- A working BentoML service wrapping the Staging model

---

## Session 7 — Monitoring & Drift Detection with Evidently AI / WhyLabs + **Capstone Part 2 of 3**
**Core Concept:** Deploying a model is the midpoint of its lifecycle, not the end. This session builds the skill of detecting *when* a production model has silently stopped reflecting reality.

**Topics:**
- Data drift vs. concept drift vs. label drift — distinct problems requiring distinct detection
- Evidently AI: Reports, Test Suites, and the Drift Detection algorithms (PSI, KS-test, Wasserstein distance) underneath them
- WhyLabs/whylogs: lightweight statistical profiling designed for streaming/high-volume data where full dataset snapshots aren't feasible
- Designing alerting thresholds that don't cry wolf

**Hands-On Lab:**
1. Generate an Evidently Data Drift Report comparing the capstone's training reference data against a simulated "production" batch with injected drift
2. Build an Evidently Test Suite that mirrors the Session 1 validation gates, but for live inference data
3. Log whylogs statistical profiles for the same data and compare the workflow/footprint against Evidently's full-dataset approach
4. Wire the drift report's pass/fail signal back into the capstone pipeline as an automated retraining trigger condition

**Trade-off Discussion:** *Evidently AI vs. WhyLabs/whylogs — when do you need full reference-data comparison vs. lightweight streaming profiles?*
Evidently is built around comparing two datasets (reference vs. current) and is ideal for batch monitoring workflows where you can afford to hold both datasets in memory and want rich, visual, statistically rigorous reports. whylogs/WhyLabs is built for high-volume or streaming contexts — it computes compact statistical "sketches" on the fly without retaining raw data, trading some analytical depth for the ability to monitor at scale or in privacy-sensitive settings where retaining raw production data isn't an option.

### 🏗️ Capstone Project — Part 2: Productionizing
*Part 2 deliverables (this session):*
- A monitoring layer (Evidently Test Suite) running against the BentoML service's logged inference data
- A documented drift-triggered retraining condition that, if breached, re-runs the DVC pipeline and logs a new MLflow run
- The new candidate model registered at "Staging," with its metrics compared against current "Production" via the MLflow Registry API

---

## Session 8 — Capstone Integration, End-to-End Walkthrough & Wrap-Up
**Core Concept:** This session has no new tool. It is entirely dedicated to forcing every framework from Sessions 1–7 to operate together as one coherent, ML-native lifecycle, and to building the judgment to explain *why* each handoff exists.

**Topics:**
- Tracing one record's journey: raw data → Great Expectations/Pandera gate → DVC-versioned dataset → MLflow-tracked training run → Feast-served features → MLflow Registry promotion → BentoML serving → Evidently-monitored inference → drift-triggered retraining loop
- Common integration failure points between these tools and how to debug them
- What "production-ready" means in an ML-native (non-container) context, and its honest limits

**Hands-On Lab / Capstone Finale:**
1. Run the full capstone pipeline end-to-end from a single entry script, with each stage's pass/fail gate enforced
2. Promote the Part 2 candidate model to "Production" via the Registry, and demonstrate instant rollback to the prior Production version using only registry aliases (no redeploy, no rebuild)
3. Present a 10-minute walkthrough: architecture diagram of the full pipeline, the trade-off decisions made at each stage, and what would need to change to operate this at 10x scale

**Trade-off Discussion (capstone-wide):** *Where does this ML-native stack stop, and where does real infrastructure begin?*
This entire pipeline can run reproducibly on a single machine or CI runner — which is precisely its strength for a Data Scientist's first production system. But it deliberately stops short of solving multi-service deployment, horizontal autoscaling under load, network security boundaries, and secrets management, all of which are the explicit job of the DevOps layer this course intentionally excluded.

### 🏗️ Capstone Project — Part 3 (Final): The Full Loop
*Final deliverables:*
- One reproducible pipeline script demonstrating the complete Validate → Version → Track → Register → Feature-Store → Serve → Monitor → Retrain loop
- A short architecture write-up naming every tool's role and citing the trade-off discussion that justified its choice over alternatives
- A live demo of rollback via Model Registry stage aliases

---

## Stack Rationale

This course deliberately assembles MLflow, DVC, Great Expectations/Pandera, Feast, Evidently AI/WhyLabs, and BentoML/Triton instead of a Docker/Kubernetes/Terraform-centric stack because every one of these tools encodes ML-specific concepts — experiment runs, dataset versions, feature point-in-time correctness, model signatures, data drift — directly into its API, meaning a Data Scientist can reason about production concerns using the same mental models they already use for modeling, rather than first becoming a systems engineer. This keeps the course's 40 hours entirely focused on the *ML* lifecycle rather than spending half the time on YAML and cluster networking that a platform/DevOps team typically owns anyway. That said, this is intentionally a partial map: students finishing this course will still need to separately learn containerization and orchestration (Docker/Kubernetes) for scaling serving horizontally and isolating dependencies, infrastructure-as-code (Terraform) for provisioning the cloud resources these tools ultimately run on, CI/CD automation (GitHub Actions/Jenkins) for fully hands-off deployment pipelines, and cloud-specific security concerns (IAM, VPCs, secrets management) for operating any of this safely outside a laptop or single trusted server — those gaps are the natural Phase 2 of the ML Engineering transition, and are excluded here by design, not oversight.
