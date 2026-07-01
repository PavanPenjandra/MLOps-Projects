# Complete Notes — Session 3, 4 & Session 5
### MLOps for Data Scientists: An ML-Native Path to Production

---

# SESSION 3 — Experiment Tracking with MLflow
**Duration:** 5 hours | **Tool:** MLflow Tracking

---

## 3.1 Why This Session Exists

By the end of Session 2 you could reproduce any prior pipeline run byte-for-byte. That's the reproducibility guarantee. But reproducibility alone doesn't answer the *comparison* question: which of the 20 runs you've done this week is actually best? What combination of preprocessing choices and hyperparameters produced it? Did anyone else on the team already try the thing you're about to try?

In most DS teams, the answer to these questions lives in a shared Google Sheet, a folder of CSVs named `metrics_final_v3_real.csv`, or someone's memory. That breaks down fast. **Experiment Tracking** turns every training run into a queryable, versioned, first-class record — with parameters, metrics, artifacts, environment info, and a human-readable name — stored somewhere everyone can see.

MLflow is the dominant open-source tool for this, largely because it:
- Requires almost no setup to start (local file store, zero infrastructure)
- Has first-class integrations with scikit-learn, PyTorch, XGBoost, Keras, LightGBM via `autolog`
- Provides a built-in comparison UI
- Scales to a remote backend (PostgreSQL + S3) when team size demands it — without changing any application code

The most important mindset shift: **a run is not just a model file. It is a snapshot of the exact context that produced that model.** Parameters, metrics, code version, library versions, data reference, and the output artifact — all together. If any of those is missing, the run is partially blind.

---

## 3.2 Core Concepts

**Run:** the fundamental unit. Every execution of a training script creates one run, capturing:
- `params` — the hyperparameters and configuration choices for this run (immutable once logged)
- `metrics` — numeric measurements that can evolve over time within a run (e.g., loss per epoch)
- `tags` — free-form string labels (team name, dataset version, git commit hash)
- `artifacts` — any file produced by the run: model pickle, confusion matrix PNG, feature importance CSV

**Experiment:** a named container that groups related runs. Think of it as a project or problem definition — e.g., `"churn-prediction-rf-baseline"` or `"churn-prediction-feature-ablation"`.

**Run ID:** a unique identifier assigned by MLflow. This is your primary handle for retrieving any run's artifacts or metadata later.

**Tracking Server:** where run data is persisted. Three tiers:
1. **Local file store** — `mlruns/` directory, zero config, good for solo work
2. **Local SQLite + artifact folder** — structured queries via SQL, still no server process needed
3. **Remote (PostgreSQL backend + S3 artifact store)** — team-scale, requires infra — out of scope for this course

**MLflow UI:** a local web app that visualizes runs in a table/chart view. Launch with `mlflow ui`.

---

## 3.3 Architecture Diagram (Text)

```
Your Training Script
        │
        │  mlflow.log_param("lr", 0.01)
        │  mlflow.log_metric("accuracy", 0.91)
        │  mlflow.log_artifact("model.pkl")
        ▼
  MLflow Tracking Client (Python library)
        │
        ▼
  Tracking Server  ──────────────────────────────────────────────┐
  (local file store / SQLite)                                    │
        │                                                         │
   ┌────┴──────────────┐                              Artifact Store
   │  Params & Metrics  │                           (local ./mlruns/)
   │  (structured data) │
   └───────────────────┘
```

---

## 3.4 Setting Up

```bash
pip install mlflow scikit-learn pandas
```

Start with the simplest possible backend — local file store. No config required:

```python
import mlflow
# By default, mlflow writes to ./mlruns/ in the current directory
# No setup needed. The tracking URI is "file:./mlruns" implicitly.
```

To use a SQLite backend (enables SQL queries against run metadata):

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns_artifacts \
  --host 0.0.0.0 --port 5000
```

Then in your Python code, point the client at this server:

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

To open the UI (either mode):

```bash
mlflow ui   # opens http://localhost:5000
```

---

## 3.5 Instrumenting a Training Script — Manual Logging

The explicit API teaches you what every piece of metadata *means* before relying on autolog:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

# --- Load data (this is the DVC-versioned file from Session 2) ---
df = pd.read_csv("data/train.csv")
X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define hyperparameters ---
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 5,
    "random_state": 42,
}

# --- Create or reuse an experiment ---
mlflow.set_experiment("churn-prediction-baseline")

# --- Open a run context ---
with mlflow.start_run(run_name="rf-max_depth-8") as run:
    print(f"Run ID: {run.info.run_id}")

    # Log every hyperparameter
    mlflow.log_params(params)

    # Tag with data lineage and git context
    mlflow.set_tags({
        "dataset_version": "Q2-2024",
        "dvc_commit": "a1b2c3d",    # tie back to the Session 2 dvc.lock commit
        "team": "ml-platform",
    })

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    # Log metrics
    mlflow.log_metrics({
        "val_accuracy": accuracy_score(y_val, preds),
        "val_f1": f1_score(y_val, preds),
        "val_roc_auc": roc_auc_score(y_val, proba),
    })

    # Log the model as a run artifact (scikit-learn flavor)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log any other artifacts — plots, CSVs, etc.
    import json
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump({"roc_auc": roc_auc_score(y_val, proba)}, f)
    mlflow.log_artifact("metrics/eval_metrics.json")
```

---

## 3.6 Autologging — Letting MLflow Do the Heavy Lifting

Once you understand what manual logging does, `autolog` saves you from writing it for well-known libraries:

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("churn-prediction-autolog")

# One line enables param/metric/model logging for the entire script
mlflow.sklearn.autolog(
    log_input_examples=True,       # saves a sample of X_train as reference
    log_model_signatures=True,     # infers input/output schema automatically
    log_post_training_metrics=True # logs validation metrics after .fit()
)

with mlflow.start_run(run_name="rf-autolog"):
    model = RandomForestClassifier(n_estimators=200, max_depth=8)
    model.fit(X_train, y_train)    # MLflow captures everything from here
```

For PyTorch, autolog captures per-epoch train/val loss curves automatically:

```python
import mlflow.pytorch

mlflow.pytorch.autolog()

with mlflow.start_run(run_name="pytorch-mlp-v1"):
    model = MyMLPModel()
    trainer.fit(model, train_loader, val_loader)
```

**When to use manual vs. autolog:** use autolog for fast exploration; add explicit `log_params` / `log_metrics` for anything the framework doesn't capture automatically (business metrics, custom eval sets, data lineage tags). In production code, prefer explicit logging — it makes the code self-documenting and doesn't depend on autolog's inference being correct.

---

## 3.7 Parent & Child Runs — Organizing Hyperparameter Sweeps

Without nesting, a 20-run sweep creates 20 flat, hard-to-navigate runs. Parent/child nesting groups them cleanly:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

mlflow.set_experiment("churn-prediction-sweep")

param_grid = [
    {"n_estimators": 100, "max_depth": 4},
    {"n_estimators": 200, "max_depth": 4},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 200, "max_depth": 8},
    {"n_estimators": 400, "max_depth": 8},
]

# Parent run — represents the whole sweep
with mlflow.start_run(run_name="sweep-depth-vs-trees") as parent:
    mlflow.set_tag("sweep_type", "grid")

    for params in param_grid:
        # Child run — represents one configuration
        with mlflow.start_run(run_name=f"rf-d{params['max_depth']}-n{params['n_estimators']}",
                              nested=True):
            mlflow.log_params(params)
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            mlflow.log_metric("val_roc_auc", auc)
            mlflow.sklearn.log_model(model, artifact_path="model")
```

In the UI, the parent run collapses/expands to show its children, making it trivial to spot the best child at a glance.

---

## 3.8 Querying Runs Programmatically (MlflowClient API)

The UI is for humans. For automation — CI pipelines, comparison scripts, promotion triggers — use the Python client:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Retrieve the best run by a metric (higher is better → order="DESC")
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="tags.sweep_type = 'grid'",
    order_by=["metrics.val_roc_auc DESC"],
    max_results=1,
)

best_run = runs[0]
print(f"Best Run ID  : {best_run.info.run_id}")
print(f"Best ROC-AUC : {best_run.data.metrics['val_roc_auc']:.4f}")
print(f"Params       : {best_run.data.params}")
```

This is the pattern that feeds into Session 5: you'll programmatically identify the best run here, then register it to the Model Registry — no human needs to look at the UI to decide which model to promote.

---

## 3.9 Logging Epoch-Level Metrics (Training Curves)

For iterative models (PyTorch, Keras, gradient boosting), log one metric per step:

```python
with mlflow.start_run(run_name="gbm-training-curve"):
    import lightgbm as lgb
    import mlflow.lightgbm

    mlflow.lightgbm.autolog()

    callbacks = [
        lgb.log_evaluation(period=10),
        lgb.record_evaluation({})
    ]

    model = lgb.train(
        params={"num_leaves": 63, "learning_rate": 0.05, "objective": "binary"},
        train_set=lgb.Dataset(X_train, y_train),
        valid_sets=[lgb.Dataset(X_val, y_val)],
        num_boost_round=200,
        callbacks=callbacks,
    )
    # Autolog captures per-round train/val loss — plotted as curves in the UI
```

For manual step logging:

```python
with mlflow.start_run():
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss   = evaluate(model, val_loader)
        # step= makes the UI plot these as a curve, not overwrite the same value
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss",   val_loss,   step=epoch)
```

---

## 3.10 Connecting MLflow Back to DVC (Data Lineage Tag Pattern)

MLflow doesn't natively know about your DVC-versioned data. The lightweight bridge is a tag:

```python
import subprocess

def get_dvc_data_hash(dvc_file: str) -> str:
    """Read the md5 hash from a .dvc file for lineage tracking."""
    import yaml
    with open(dvc_file) as f:
        dvc_meta = yaml.safe_load(f)
    return dvc_meta["outs"][0]["md5"]

with mlflow.start_run():
    mlflow.set_tags({
        "data.raw.md5":    get_dvc_data_hash("data/customers_raw.csv.dvc"),
        "data.train.md5":  get_dvc_data_hash("data/train.csv.dvc"),
        "dvc.pipeline.commit": subprocess.getoutput("git rev-parse HEAD"),
    })
    # ... rest of training
```

Now every MLflow run is *pinned* to the exact DVC-versioned dataset that produced it. A future investigator can take any run's tags, `git checkout` the commit, and `dvc checkout` to recover the exact data — and then `dvc repro` to reproduce the model.

---

## 3.11 Lab Walkthrough (Session 3)

1. Stand up a local MLflow Tracking Server with a SQLite backend.
2. Instrument the Session 2 `train.py` script with both manual logging (params, metrics, artifacts, data lineage tags) and `mlflow.sklearn.autolog()` — compare what each approach captures.
3. Run a 6-child sweep over `max_depth` × `n_estimators` under a single parent run.
4. Browse the MLflow UI: compare runs in the parallel coordinates plot, identify the best child run.
5. Use `MlflowClient.search_runs()` to programmatically pull the best run's ID and save it to a file for Session 5's registration step.
6. Add epoch-step metric logging to a PyTorch or LightGBM training loop; observe the resulting training-curve chart in the UI.
7. Add the DVC data-lineage tag pattern, verify that the MLflow run card links back to the exact git commit.

---

## 3.12 Trade-off Discussion — Recap

**MLflow Tracking vs. DVC Experiments — when to use which?**
DVC Experiments are tied to a full pipeline snapshot — data version, code, params, and outputs are all locked together. If reproducibility and "rerun this exactly" matter most, DVC wins. MLflow Tracking excels at the exploration and communication phase — rich UI, metric curves, cross-run comparison, queryable by anyone on the team, and easily shared with non-engineers as a report. In practice: use DVC to make any run reproducible; use MLflow to discover which run you actually want.

---

## 3.13 Common Pitfalls

- **Logging metrics without a step parameter for iterative models.** Each `log_metric("val_loss", 0.4)` *overwrites* the previous value unless you supply `step=epoch`. The training curve becomes a single dot.
- **Using `run_name` as your primary reference.** Run names are not unique — only `run_id` is. Use the ID when referencing runs in code.
- **Logging inside the run context vs. outside.** Any `log_metric` called outside a `with mlflow.start_run()` block goes to the *default experiment*, silently — easy to lose data this way.
- **Not tagging data lineage.** Six months later you won't remember which dataset a run used. Tag it from the start.
- **Treating autolog as a substitute for understanding.** Autolog is convenient, but it logs things you may not understand (e.g., feature importances for every tree depth variant). Know what it captures, and audit the run card before trusting it.

---
---

# SESSION 4 — Feature Stores & Engineering Orchestration with Feast
**Duration:** 5 hours | **Tool:** Feast

---

## 4.1 Why This Session Exists

Imagine two separate systems in your organization:

**Offline (training):** a data engineer built a Spark pipeline that computes "rolling 30-day average spend per customer" and writes it to a Parquet file every night. Your training script reads that file and joins it to labels by customer ID.

**Online (inference):** at prediction time, a FastAPI service receives a `customer_id`, needs the "rolling 30-day average spend" feature immediately, and computes it differently — maybe using a quick SQL query over a Postgres table that doesn't run on the same schedule as Spark.

Those two compute paths **will diverge.** Different logic, different data freshness, rounding differences, timezone bugs. The model trains on one definition of the feature and runs on another. This is **training/serving skew**, and it silently corrupts the predictions of more production models than any algorithmic mistake.

A **Feature Store** is the infrastructure solution: define the feature computation *once*, and both the offline training path and the online serving path retrieve it from the same definition. The computation is identical; only the storage backend (a fast key-value store for online, a slow columnar store for offline) differs.

Feast is the leading open-source feature store. It is notably ML-native: its entire API is expressed as Python definitions and CLI commands — no Dockerfile, no Kubernetes manifest in scope for this course.

---

## 4.2 Core Concepts

**Entity:** the primary object your features describe — typically a customer, product, driver, user. Defined by a join key (e.g., `customer_id`).

**Feature View:** a named group of features computed from a specific data source, associated with an entity. This is the central definition — what Feast guarantees is consistent between offline and online retrieval.

**Data Source:** where Feast reads raw feature data from — a local Parquet file, a file on S3, BigQuery, Redshift, etc. For this course: local Parquet/CSV files.

**Offline Store:** Feast's read layer for training — retrieves historical feature values with *point-in-time correctness*.

**Online Store:** Feast's low-latency serving layer — a key-value store (local SQLite in this course; Redis/DynamoDB in production) populated via materialization.

**Materialization:** the explicit operation of reading features from the offline store and writing them into the online store, up to a specified point in time.

**Point-in-Time Correctness (PIT Join):** when generating training data, Feast ensures that the feature value retrieved for each label corresponds to the value *as it existed at the label's event timestamp* — not the "latest as of today" value. This is what prevents **label leakage**.

---

## 4.3 Why Point-in-Time Correctness Matters (Concrete Example)

Suppose you're predicting churn for customers as of 2024-03-15. Customer A churned on 2024-04-01. Their account-closure date is now in the database. If you join "naively" by customer ID without respecting time, you'd train the model on features that include post-churn account info — information that was *not available* at prediction time on March 15. The model learns to predict from the future.

Feast's `get_historical_features` automatically performs a **point-in-time join**: for each row in your label set (customer_id + event_timestamp), it retrieves only feature values that existed *before or at* that timestamp. The label leakage problem disappears by construction.

---

## 4.4 Setting Up a Feast Feature Repository

```bash
pip install feast
feast init churn_feature_repo
cd churn_feature_repo
```

This creates a `feature_store.yaml` and a `features/` directory with example definitions. Let's replace them with domain-specific definitions:

```yaml
# feature_store.yaml
project: churn_mlops_course
registry: data/registry.db          # local SQLite registry
provider: local                      # local provider — no cloud account needed
online_store:
  type: sqlite
  path: data/online_store.db
```

---

## 4.5 Defining Entities and Feature Views

```python
# features/customer_features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# Step 1: Define the Entity
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="A unique customer in the system.",
)

# Step 2: Define a Data Source (local Parquet for this course)
customer_stats_source = FileSource(
    path="data/customer_stats.parquet",   # your DVC-versioned dataset
    timestamp_field="event_timestamp",     # Feast uses this for PIT joins
    created_timestamp_column="created",
)

# Step 3: Define a Feature View
customer_stats_fv = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=7),          # how long a feature value is considered fresh
    schema=[
        Field(name="monthly_spend",       dtype=Float32),
        Field(name="rolling_30d_spend",   dtype=Float32),
        Field(name="tenure_days",         dtype=Int64),
        Field(name="subscription_tier",   dtype=String),
        Field(name="support_tickets_30d", dtype=Int64),
    ],
    source=customer_stats_source,
)
```

Apply the definitions to the registry:

```bash
feast apply    # validates definitions and registers them in the local registry
```

---

## 4.6 Generating a Training Dataset (Offline — PIT Join)

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path=".")

# Your label set: the entity keys + event timestamps you have labels for
# This is the anchor for the point-in-time join
entity_df = pd.DataFrame({
    "customer_id": [1001, 1002, 1003, 1004, 1005],
    "event_timestamp": [
        datetime(2024, 3, 15),
        datetime(2024, 2, 28),
        datetime(2024, 3, 1),
        datetime(2024, 3, 10),
        datetime(2024, 3, 15),
    ],
    "churn_label": [0, 1, 0, 1, 0],   # your labels — included for convenience
})

# Retrieve historical features with PIT correctness
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_stats:monthly_spend",
        "customer_stats:rolling_30d_spend",
        "customer_stats:tenure_days",
        "customer_stats:subscription_tier",
        "customer_stats:support_tickets_30d",
    ],
).to_df()

print(training_df.head())
# customer_id | event_timestamp | churn_label | monthly_spend | rolling_30d_spend | ...
# Each row's feature values are from BEFORE that row's event_timestamp
```

This DataFrame is what you then feed into the Session 3 MLflow training run — and *this specific call* is your guarantee that the offline features are computed from the same `FeatureView` definition as the online path.

---

## 4.7 Materializing to the Online Store

```bash
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

Or from Python:

```python
from feast import FeatureStore
from datetime import datetime

store = FeatureStore(repo_path=".")
store.materialize_incremental(end_date=datetime.utcnow())
```

This reads from the offline source (Parquet) and writes the *latest* feature values per entity key into the online store (SQLite in this course). Incremental materialization only processes records newer than the last run.

---

## 4.8 Serving Features Online (Inference Path)

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# At inference time: customer IDs arrive in the request
entity_rows = [
    {"customer_id": 1001},
    {"customer_id": 1004},
]

online_features = store.get_online_features(
    features=[
        "customer_stats:monthly_spend",
        "customer_stats:rolling_30d_spend",
        "customer_stats:tenure_days",
        "customer_stats:subscription_tier",
        "customer_stats:support_tickets_30d",
    ],
    entity_rows=entity_rows,
).to_dict()

print(online_features)
# Same feature names, same computation logic as the offline path.
# training/serving skew eliminated by construction.
```

---

## 4.9 Deliberately Reproducing Training/Serving Skew — Then Fixing It

This lab step forces the *conceptual understanding*, not just the API:

**Breaking version (skew present):**

```python
# Offline (training) — spend normalized to monthly rate in preprocessing
df["spend_feature"] = df["raw_spend"] / df["days_in_period"] * 30

# Online (inference) — someone computed it differently in the API endpoint
# (raw_spend summed directly without normalization — a subtle bug)
spend_feature = raw_spend_last_30d   # ← different definition!
```

This is the skew: both paths produce a column called `spend_feature`, but they mean different things. The model trains on one and predicts on the other.

**Fixed version (Feast enforces consistency):**

```python
# Only ONE definition exists — in the FeatureView
# Both training (get_historical_features) and inference (get_online_features)
# call the SAME FeatureView — the computation is guaranteed identical.
```

---

## 4.10 Feature Freshness and TTL

The `ttl` field in a `FeatureView` defines how long a feature value is considered valid:

```python
customer_stats_fv = FeatureView(
    name="customer_stats",
    ttl=timedelta(days=7),   # A feature older than 7 days at query time is considered stale
    ...
)
```

If materialization hasn't run in more than 7 days, online feature retrieval will return `None` for stale features and log a warning. This prevents serving outdated, stale features silently — another class of production skew.

---

## 4.11 Full Integration: Feast → MLflow

```python
import mlflow
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

mlflow.set_experiment("churn-feast-features")

with mlflow.start_run(run_name="feast-integrated-rf"):

    # Log the feature definitions used — so the run is traceable to the FeatureView version
    mlflow.set_tags({
        "feast.feature_view": "customer_stats",
        "feast.features": str([
            "monthly_spend", "rolling_30d_spend",
            "tenure_days", "subscription_tier", "support_tickets_30d"
        ]),
        "feast.repo_commit": subprocess.getoutput("git rev-parse HEAD"),
    })

    # Pull training features via Feast (PIT-correct)
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[...],
    ).to_df()

    mlflow.log_param("feature_count", len(training_df.columns) - 2)

    # Train as normal
    model = RandomForestClassifier(n_estimators=200, max_depth=8)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
```

Now the MLflow run card carries the exact FeatureView reference. A future engineer can look at this run and know *exactly* which feature definitions produced it.

---

## 4.12 Lab Walkthrough (Session 4)

1. Initialize a Feast feature repo with `feast init`, replace the scaffold with a custom `customer` entity and `customer_stats` Feature View pointing at the DVC-versioned Parquet from Session 2.
2. `feast apply` — read the output to understand what was registered.
3. Generate a PIT-correct training dataset using `get_historical_features` and compare the timestamps of retrieved feature values against the entity_df's event timestamps to confirm PIT behavior.
4. **Deliberately introduce skew**: compute a feature one way in a notebook and a different way in a FastAPI handler. Observe the model's prediction distribution shift.
5. **Fix it with Feast**: route both paths through `get_historical_features` (training) and `get_online_features` (inference) from the same Feature View. Confirm predictions are now consistent.
6. `feast materialize-incremental` — inspect the SQLite online store to verify materialized values.
7. Wrap `get_online_features` in a `predict(customer_ids)` function that will be the basis of the Session 6 BentoML serving endpoint.

---

## 4.13 Trade-off Discussion — Recap

**Feast vs. hand-rolled feature pipelines — when does a feature store earn its complexity?**
Feast adds real complexity: a registry, two store backends, a materialization job, and an explicit deployment step (`feast apply`). That overhead is not worth it for a single model with a handful of simple features computed once and never reused. The break-even is roughly:
- More than one model sharing the same features (without a store, each model team re-derives them independently → divergence guaranteed over time)
- Any feature that needs to be served with sub-100ms latency at inference time
- Features computed by a different team than the one training the model
- Regulated environments where "which exact feature values were used for this prediction" must be auditable

---

## 4.14 Common Pitfalls

- **Forgetting `event_timestamp` in the feature data source.** Without it, Feast can't perform the PIT join and falls back to a simple join — label leakage risk reappears silently.
- **Not running `feast apply` after changing a Feature View.** The registry won't reflect your edits; you'll be serving stale definitions.
- **Materializing to online store only once.** If you update features offline but don't re-materialize, the online store serves stale data. Treat `feast materialize-incremental` as a scheduled job, not a one-time step.
- **Using TTL too aggressively.** A very short TTL (e.g., 1 hour) will cause most online feature retrievals to return `None` if materialization isn't frequent enough — silent nulls in serving.
- **Mismatching feature names between `get_historical_features` and `get_online_features`.** Both calls must reference the same `"feature_view:feature_name"` strings — typos here are a live skew bug.

---
---

# SESSION 5 — Model Registry & Lifecycle Management with MLflow
**Duration:** 5 hours | **Tool:** MLflow Model Registry

---

## 5.1 Why This Session Exists

At the end of Session 3, you had the best run's ID saved to a file. At the end of Session 4, that model was trained on Feast-sourced, PIT-correct features. But the model artifact itself still lives at a path like:

```
mlruns/1/a1b2c3d4e5f6/artifacts/model/
```

That path is:
- Opaque (what does `a1b2c3d4e5f6` mean?)
- Not versioned by intent (it's a run ID, not a model version)
- Not stage-aware (there's no way to know if it's "in staging" or "in production" from the path alone)
- Not rollback-friendly (to "go back" you have to remember a run ID and remember where it was)

The **Model Registry** solves all of this. It turns a raw run artifact into a named, versioned, stage-managed asset with an audit trail. The key mental shift: a *run* is a training event. A *registered model version* is a deployment candidate. They are different lifecycle objects that happen to be linked.

---

## 5.2 Core Concepts

**Registered Model:** a named entity in the registry (e.g., `"churn-prediction-rf"`). Think of it as the *product* — independent of any specific version.

**Model Version:** every time you register a new artifact under a registered model name, a new numeric version is created (v1, v2, v3...). Versions are immutable — you can't overwrite v2.

**Stage (Lifecycle Stage):** each version occupies one stage:
- `None` → freshly registered, not yet evaluated
- `Staging` → promoted for validation/shadow testing
- `Production` → live, actively serving
- `Archived` → retired, no longer serving (but preserved for audit)

**Model Alias:** a mutable, human-readable pointer to a specific version (e.g., `"champion"` → v3, `"challenger"` → v5). Aliases are the recommended way to decouple "what is serving" from hardcoded version numbers in serving code.

**Model Signature:** a schema contract (input column names/types, output names/types) embedded in the registered artifact. Prevents serving a model with the wrong input shape.

**Input Example:** a sample row of valid input embedded in the artifact — used by MLflow to auto-generate REST API docs and verify signatures at load time.

---

## 5.3 Architecture: Run → Registry → Serving

```
Training Script (Session 3)
        │  mlflow.sklearn.log_model(model, "model")
        ▼
   MLflow Run Artifact
        │
        │  mlflow.register_model(run_uri, "churn-prediction-rf")
        ▼
   Model Registry
   ┌─────────────────────────────────────────────────────┐
   │  Registered Model: "churn-prediction-rf"            │
   │                                                     │
   │  Version 1 → Stage: Archived                        │
   │  Version 2 → Stage: Archived                        │
   │  Version 3 → Stage: Production  ← alias "champion"  │
   │  Version 4 → Stage: Staging     ← alias "challenger" │
   └─────────────────────────────────────────────────────┘
        │
        │  mlflow.pyfunc.load_model("models:/churn-prediction-rf@champion")
        ▼
   Serving Code (Session 6 BentoML endpoint)
   — no hardcoded run ID, no hardcoded path
```

---

## 5.4 Registering a Model

### Option A — Register Directly in the Training Run

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("churn-prediction-baseline")

with mlflow.start_run(run_name="rf-final-candidate") as run:
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # Log model WITH signature and input example — both become part of the registry entry
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:3]

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="churn-prediction-rf",  # registers automatically
    )
```

### Option B — Register a Prior Run After the Fact

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Identify the best run (from Session 3's search_runs call)
best_run_id = "a1b2c3d4e5f6"

result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="churn-prediction-rf",
)
print(f"Registered as Version: {result.version}")
```

Option B is more common in mature pipelines: training is decoupled from promotion — the training script just logs artifacts, and a separate "promotion script" decides whether to register.

---

## 5.5 Model Signatures — Enforcing the Input Contract

A signature defines what the model expects, checked at serving time:

```python
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# Automatic inference from data (preferred)
signature = infer_signature(X_train, model.predict_proba(X_train))

# Manual definition (when explicit control is needed)
signature = ModelSignature(
    inputs=Schema([
        ColSpec("double", "monthly_spend"),
        ColSpec("double", "rolling_30d_spend"),
        ColSpec("long",   "tenure_days"),
        ColSpec("string", "subscription_tier"),
        ColSpec("long",   "support_tickets_30d"),
    ]),
    outputs=Schema([
        ColSpec("double", "churn_probability_class_0"),
        ColSpec("double", "churn_probability_class_1"),
    ]),
)
```

A model logged without a signature is a model that can be called with anything at serving time — the kind of error that only surfaces in production at 2 AM.

---

## 5.6 Stage Transitions — Promoting Through the Lifecycle

### Via the Python API (preferred for automation):

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "churn-prediction-rf"

# Promote Version 4 from None → Staging
client.transition_model_version_stage(
    name=model_name,
    version=4,
    stage="Staging",
    archive_existing_versions=False,   # don't auto-archive current staging
    description="Promoting candidate after passing offline eval suite.",
)

# After shadow testing or A/B validation — move to Production
client.transition_model_version_stage(
    name=model_name,
    version=4,
    stage="Production",
    archive_existing_versions=True,   # archive the previous Production version
    description="Promoting v4 — ROC-AUC improved from 0.87 to 0.91 on hold-out set.",
)
```

### Via Aliases (Recommended Modern Pattern — MLflow ≥ 2.0):

```python
# Assign a human-readable alias — the serving code never needs to change
client.set_registered_model_alias(model_name, "champion", version=4)

# Test a challenger without touching production
client.set_registered_model_alias(model_name, "challenger", version=5)

# Rollback: point champion to v3 with ONE line, zero redeploy
client.set_registered_model_alias(model_name, "champion", version=3)
```

Stage names are the traditional mechanism; aliases are more flexible and are the direction MLflow is moving. This course uses **both**: stages for human-readable lifecycle state, aliases for serving pointer management.

---

## 5.7 Adding Transition Notes & Audit Trail

Every transition is recorded. Add human-readable context:

```python
client.update_model_version(
    name=model_name,
    version=4,
    description=(
        "Trained on Q2-2024 Feast features (customer_stats v3). "
        "Passed data validation gates (Pandera + Great Expectations). "
        "ROC-AUC: 0.913 on 2024-03-15 hold-out. "
        "Approved by: ml-platform-team."
    ),
)
```

The audit trail is automatically maintained — every stage transition is timestamped and attributed.

---

## 5.8 Loading a Registered Model for Inference

The power of the registry: serving code never references a run ID or a filesystem path:

```python
import mlflow.pyfunc

# Load by stage (classic)
model = mlflow.pyfunc.load_model("models:/churn-prediction-rf/Production")

# Load by alias (preferred — decoupled from hardcoded stage names)
model = mlflow.pyfunc.load_model("models:/churn-prediction-rf@champion")

# Both return a generic pyfunc wrapper — works regardless of which flavor
# (sklearn, PyTorch, LightGBM, XGBoost) was registered
predictions = model.predict(X_new)
```

The generic `pyfunc` flavor is the abstraction that makes serving code framework-agnostic. Session 6's BentoML service will call this exact pattern — it loads whatever model is currently aliased as `"champion"` without knowing whether it's sklearn or PyTorch.

---

## 5.9 Programmatic Comparison — Challenger vs. Champion

This is the decision gate before a Production promotion:

```python
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from sklearn.metrics import roc_auc_score

client = MlflowClient()
model_name = "churn-prediction-rf"

# Load champion and challenger
champion   = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
challenger = mlflow.pyfunc.load_model(f"models:/{model_name}@challenger")

# Evaluate on a shared hold-out set
champion_auc   = roc_auc_score(y_holdout, champion.predict(X_holdout))
challenger_auc = roc_auc_score(y_holdout, challenger.predict(X_holdout))

print(f"Champion AUC   : {champion_auc:.4f}")
print(f"Challenger AUC : {challenger_auc:.4f}")

THRESHOLD_IMPROVEMENT = 0.005  # challenger must improve by at least 0.5%

if challenger_auc > champion_auc + THRESHOLD_IMPROVEMENT:
    challenger_version = client.get_model_version_by_alias(model_name, "challenger").version
    client.set_registered_model_alias(model_name, "champion", version=challenger_version)
    print(f"✅ Challenger promoted to champion (v{challenger_version})")
else:
    print("❌ Challenger did not exceed threshold. Champion retained.")
```

This is the complete automation loop that, in Session 7's capstone work, gets triggered by a drift alert from Evidently — a fully automated champion/challenger promotion without any human clicking the MLflow UI.

---

## 5.10 Rollback — The Key Production Safety Property

```python
# Scenario: v5 was just promoted to champion, but monitoring shows degraded performance.
# Rollback in one API call:

client.set_registered_model_alias(model_name, "champion", version=4)

# Serving code is already loading via alias — it will pick up v4 on the next load.
# No redeploy. No Dockerfile rebuild. No Kubernetes rollout.
# Time-to-rollback: seconds.
```

This is the production safety guarantee that makes the Model Registry worth the process overhead. Without it, "rollback" means locating a run ID, finding its artifact path, reconfiguring the serving endpoint, and redeploying — a process that takes minutes to hours and requires someone with access to the serving infrastructure.

---

## 5.11 Full Pipeline: DVC → Feast → MLflow Tracking → Registry

Putting Sessions 2–5 together in one orchestrating script:

```python
import subprocess
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# --- 0. Run DVC pipeline (validation + preprocessing) ---
subprocess.run(["dvc", "repro"], check=True)

# --- 1. Load Feast-sourced training features ---
store = FeatureStore(repo_path=".")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_stats:monthly_spend",
              "customer_stats:rolling_30d_spend",
              "customer_stats:tenure_days",
              "customer_stats:support_tickets_30d"],
).to_df()

X_train = training_df.drop(["customer_id", "event_timestamp", "churn_label"], axis=1)
y_train = training_df["churn_label"]

# --- 2. Train with MLflow Tracking ---
mlflow.set_experiment("churn-full-pipeline")

with mlflow.start_run(run_name="full-pipeline-run") as run:
    mlflow.set_tags({
        "feast.feature_view": "customer_stats",
        "dvc.commit": subprocess.getoutput("git rev-parse HEAD"),
    })

    params = {"n_estimators": 200, "max_depth": 8}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    mlflow.log_metric("val_roc_auc", auc)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:3],
        registered_model_name="churn-prediction-rf",
    )
    registered_version = MlflowClient().get_latest_versions("churn-prediction-rf")[0].version

# --- 3. Promote to Staging ---
client = MlflowClient()
client.transition_model_version_stage(
    name="churn-prediction-rf",
    version=registered_version,
    stage="Staging",
)
client.set_registered_model_alias("churn-prediction-rf", "challenger", registered_version)
print(f"✅ v{registered_version} promoted to Staging as 'challenger'")
```

---

## 5.12 Lab Walkthrough (Session 5)

1. Register the best Session 3 run artifact to the Model Registry using `mlflow.register_model`, adding a signature inferred from the Feast training data.
2. Add a description documenting the training conditions (dataset version, Feast feature view, evaluation metric).
3. Transition the version through `None → Staging → Production` via the Python API, not the UI — write a helper function `promote(name, version, stage, description)` that is reusable in the capstone.
4. Assign aliases (`"champion"` for the current Production version, `"challenger"` for a freshly registered candidate).
5. Write the challenger vs. champion comparison script — run it and verify the promotion/no-promotion logic works correctly with fabricated challenger metrics.
6. Demonstrate rollback: promote v5 to champion, deliberately degrade its hold-out metric, run the comparison script, watch v4 be re-aliased as champion in one API call.
7. Verify that the BentoML load-by-alias pattern (from Session 6's preview) works: `mlflow.pyfunc.load_model("models:/churn-prediction-rf@champion")` returns a working model.

---

## 5.13 Trade-off Discussion — Recap

**Registry-managed staging vs. ad-hoc file conventions — what does formal governance actually buy you?**
File-naming conventions (`prod_model_v2_final_real.pkl`) break down the moment a second person touches the model lifecycle, because there's no enforced transition log, no signature contract at the file boundary, and no canonical answer to "what is currently serving traffic?" A registry costs a small amount of deliberate process (you must register and promote explicitly) and returns auditability, rollback-by-alias in seconds, signature enforcement at load time, and a single source of truth that both the training team and the serving team read from.

---

## 5.14 Common Pitfalls

- **Registering models without a signature.** At serving time, any input shape is accepted — the first indication of a mismatch is a runtime crash in production, not a clear validation error.
- **Using hardcoded version numbers in serving code** (`load_model(".../version/3")`). Version numbers increment every registration — v3 today is v7 in two weeks. Use aliases; they insulate serving code from version churn.
- **Skipping `archive_existing_versions=True`** when promoting to Production. Your registry fills up with multiple "Production" versions — the canonical state of "what is live" becomes ambiguous.
- **Using stage transitions as the sole promotion mechanism (without descriptions).** Six months from now, you will not remember why v4 was promoted over v3. Always add `update_model_version(description=...)` at every transition.
- **Conflating the training run with the registered model version.** They are separate lifecycle objects. A run can exist without being registered (most will). A registered version always traces back to a run (its provenance).

---

## What Connects to Session 6

You now have a named, version-managed, aliased model in the registry with a signature that enforces its input contract. Session 6 wraps this artifact in a **BentoML Service** that exposes it as a local REST API — and the serving code will load it *by alias*, not by run ID or file path, which means the serving layer is already compatible with the Session 7 rollback demo without any code change.

---

## Cross-Session Integration Map (Sessions 1–5)

```
Session 1 (Pandera/GE)
      │  validate_or_halt(df) called as Stage 0
      ▼
Session 2 (DVC)
      │  data versioned → dvc.yaml pipeline → dvc.lock reproducibility
      │  dvc exp run sweep → best params selected
      ▼
Session 4 (Feast)
      │  PIT-correct training features via get_historical_features()
      │  Same FeatureView definition → online serving via get_online_features()
      ▼
Session 3 (MLflow Tracking)
      │  every run logged with Feast tags + DVC commit hash
      │  MlflowClient.search_runs() → best run_id
      ▼
Session 5 (MLflow Registry)
      │  best run registered → versioned → signed → aliased
      │  champion/challenger comparison → auto-promotion logic
      ▼
  Session 6 → BentoML serving (loads via alias)
  Session 7 → Evidently monitoring (triggers retraining loop)
  Session 8 → Full capstone integration
```
