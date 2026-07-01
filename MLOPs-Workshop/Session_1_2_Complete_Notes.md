# Complete Notes — Session 1 & Session 2
### MLOps for Data Scientists: An ML-Native Path to Production

---

# SESSION 1 — Data Validation & Quality Gates
**Duration:** 5 hours | **Tools:** Great Expectations, Pandera

## 1.1 Why This Session Exists

Ask any ML Engineer with production scars what actually broke their pipeline last quarter, and the honest answer is rarely "the model was wrong." It's almost always one of:

- A source system silently changed a column's units (dollars → cents)
- A categorical column gained a new, unseen value
- An upstream job started passing 30% nulls where there used to be 0%
- A join key duplicated and quietly doubled the row count

None of these are *modeling* problems. They are **data contract violations**, and they fail silently because nothing in a typical DS workflow asserts what "valid data" means — code just runs on whatever DataFrame it's handed. The fix is not "be more careful." The fix is to make the data contract executable: write down what valid data looks like, and have the pipeline refuse to proceed if reality doesn't match.

This is the single most under-taught skill in the transition from Data Scientist to ML Engineer, because in notebooks, a silent bad value just produces a slightly-off chart you never look at twice. In production, it trains a model on garbage for three weeks before anyone notices.

## 1.2 Two Philosophies of Validation

| | **Great Expectations (GE)** | **Pandera** |
|---|---|---|
| Style | Declarative, suite-based, designed to be read by non-engineers | Pythonic, schema-as-code, type-hint native |
| Where it lives | Pipeline/warehouse boundary, often its own service | Inside your codebase, next to the functions it protects |
| Output | Rich HTML "Data Docs" reports, shareable artifacts | Exceptions / validated DataFrames, lightweight |
| Best for | Data engineering teams, audit trails, cross-team contracts | ML codebases, function I/O validation, fast iteration |

You will learn both, because real teams typically use **both at different layers**: GE at the ingestion/warehouse boundary where the report needs to be human-readable, Pandera inside the model training/serving code where validation needs to be fast, typed, and co-located with logic.

## 1.3 Great Expectations — Core Concepts

**Expectation:** a single, testable claim about your data — e.g. "`age` is never null," "`country` is one of these 12 values," "`price` is between 0 and 100000."

**Expectation Suite:** a named collection of Expectations bound to a dataset — your data contract.

**Data Context:** the GE project root; holds configuration for where data lives, where Expectation Suites are stored, and where validation results go.

**Checkpoint:** a reusable, runnable bundle that says "validate *this* batch of data against *this* suite, and *do this* with the result" (raise an error, write a report, send a Slack alert, etc.). This is the unit you actually wire into a pipeline.

**Data Docs:** auto-generated, human-readable HTML reports showing what passed, what failed, and why — the artifact you hand to a non-engineer stakeholder asking "is this dataset OK to train on?"

### 1.3.1 Setting Up

```bash
pip install great_expectations pandas
great_expectations init   # scaffolds a GE project (gx/ directory)
```

### 1.3.2 Auto-Profiling a Baseline Suite

Rather than hand-writing 40 expectations from scratch, GE can profile a known-good dataset and propose a starting suite:

```python
import great_expectations as gx

context = gx.get_context()

# Connect to a pandas source
validator = context.sources.add_pandas("training_data").read_csv(
    "data/customers_clean.csv"
)

# Auto-generate expectations from the data's observed shape
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_unique("customer_id")
validator.expect_column_values_to_be_between("age", min_value=18, max_value=100)
validator.expect_column_values_to_be_in_set(
    "subscription_tier", ["free", "pro", "enterprise"]
)
validator.expect_column_mean_to_be_between("monthly_spend", min_value=0, max_value=500)

validator.save_expectation_suite(discard_failed_expectations=False)
```

### 1.3.3 Authoring Custom, Business-Aware Expectations

Auto-profiling gives you *statistical* expectations from a sample. It cannot know your *business rules*. Always layer these in by hand:

```python
# Referential / cross-column logic GE's profiler can't infer on its own
validator.expect_column_pair_values_a_to_be_greater_than_b(
    column_A="signup_date", column_B="last_login_date", or_equal=True
)

# Null-rate threshold rather than a hard "never null" rule —
# realistic for upstream systems with some expected missingness
validator.expect_column_values_to_not_be_null(
    "email", mostly=0.98   # allow up to 2% nulls before failing
)

validator.expect_table_row_count_to_be_between(min_value=1000, max_value=5_000_000)
```

### 1.3.4 Checkpoints — Wiring Validation Into a Pipeline

```python
checkpoint = context.add_or_update_checkpoint(
    name="training_data_checkpoint",
    validator=validator,
)

result = checkpoint.run()

if not result["success"]:
    context.build_data_docs()
    raise ValueError(
        "Data validation FAILED — see Data Docs for details. Halting pipeline."
    )
```

This is the pattern that turns validation from "a notebook cell I sometimes run" into "a gate the pipeline cannot get past." Note this is *pure Python/CLI* — no container, no orchestrator required to get this guarantee.

### 1.3.5 Generating & Reading Data Docs

```bash
great_expectations docs build
```

Opens a local HTML report showing, per expectation: pass/fail, observed value vs. expected, and a sample of offending rows. This is what you hand to a stakeholder, or attach to a PR, as proof a dataset is fit to train on.

---

## 1.4 Pandera — Core Concepts

Pandera expresses the same idea — "assert this DataFrame matches a contract" — but as ordinary Python objects that live in your codebase and integrate with type hints.

### 1.4.1 Defining a Schema

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check

customer_schema = DataFrameSchema(
    {
        "customer_id": Column(int, Check.greater_than(0), unique=True),
        "age": Column(int, Check.in_range(18, 100)),
        "subscription_tier": Column(
            str, Check.isin(["free", "pro", "enterprise"])
        ),
        "monthly_spend": Column(
            float, Check.greater_than_or_equal_to(0), nullable=False
        ),
        "email": Column(str, nullable=True),
    },
    strict=True,        # reject any column not declared in the schema
    coerce=True,         # attempt dtype coercion before validating
)
```

### 1.4.2 Validating a DataFrame

```python
import pandas as pd

df = pd.read_csv("data/customers_clean.csv")

try:
    validated_df = customer_schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as err:
    print(err.failure_cases)   # a DataFrame of every row/column that failed
    raise
```

`lazy=True` is the key production setting: it collects **every** violation across the whole DataFrame before raising, rather than stopping at the first error — essential for getting a complete picture in one run instead of a slow fix-one-error-at-a-time loop.

### 1.4.3 Decorating Functions — Validation at the Code Boundary

This is Pandera's signature advantage over GE: you can enforce a schema directly on function inputs/outputs using type hints, so validation travels with the code itself.

```python
from pandera.typing import DataFrame, Series
import pandera as pa

class RawCustomerSchema(pa.DataFrameModel):
    customer_id: Series[int] = pa.Field(unique=True, gt=0)
    signup_date: Series[str]
    monthly_spend: Series[float] = pa.Field(ge=0)

class FeaturizedCustomerSchema(pa.DataFrameModel):
    customer_id: Series[int] = pa.Field(unique=True, gt=0)
    spend_zscore: Series[float]
    tenure_days: Series[int] = pa.Field(ge=0)

@pa.check_types
def engineer_features(
    df: DataFrame[RawCustomerSchema],
) -> DataFrame[FeaturizedCustomerSchema]:
    df["spend_zscore"] = (df.monthly_spend - df.monthly_spend.mean()) / df.monthly_spend.std()
    df["tenure_days"] = (pd.Timestamp.now() - pd.to_datetime(df.signup_date)).dt.days
    return df[["customer_id", "spend_zscore", "tenure_days"]]
```

If `engineer_features` ever receives malformed input, or somehow produces malformed output (a refactor introduces a bug), this raises **immediately at the function boundary** — the failure is pinned to the exact transformation that caused it, not discovered three steps later when the model silently underperforms.

---

## 1.5 Lab Walkthrough (Session 1)

**Goal:** end-to-end validation gate on a raw dataset, in both GE and Pandera, with a deliberate failure demonstrated.

1. Load a raw CSV with intentionally injected issues (a few nulls, an out-of-range age, an unseen category).
2. Auto-profile a GE suite on the *clean* reference version of the dataset.
3. Run the GE Checkpoint against the *dirty* version → observe and read the Data Docs failure report.
4. Build the equivalent Pandera `DataFrameModel`.
5. Validate the dirty DataFrame with `lazy=True` and inspect `err.failure_cases`.
6. Decorate a feature-engineering function with `@pa.check_types` and watch it reject malformed output after a deliberately introduced bug (e.g., forgetting to drop nulls before a calculation).
7. Wrap steps 2–3 in a function `validate_or_halt(df)` that raises `SystemExit` on failure — this is the gate function reused in Session 2's DVC pipeline.

## 1.6 Trade-off Discussion — Recap

**Great Expectations vs. Pandera — when to use which?**
GE is heavier and declarative, and it produces shareable Data Docs — best where validation needs to be visible and auditable to non-engineers (data eng teams, regulated/compliance environments). Pandera is lightweight, Pythonic, and lives next to your code via type hints — best for validating function inputs/outputs inside an ML codebase where speed of iteration matters more than a polished report. Most mature teams run **both**: GE at the data warehouse/ingestion boundary, Pandera inside the feature engineering and training code.

## 1.7 Common Pitfalls

- **Over-constraining early.** Don't write `expect_column_values_to_be_between` with the *exact* min/max of your current sample — leave headroom or you'll get false-positive failures the moment legitimate new data arrives.
- **Validating after the damage is done.** Validation that runs *after* training (just to log metrics) doesn't protect you. The gate must block before the next stage runs.
- **Ignoring `mostly=`.** Real-world data almost never satisfies a 100% rule. Use tolerance thresholds (`mostly=0.98`) deliberately, and document why.
- **Schema drift in the schema itself.** When the upstream data legitimately changes shape, update the schema in a reviewed PR — don't silently loosen a check to make a build pass.

---
---

# SESSION 2 — Data & Pipeline Versioning with DVC
**Duration:** 5 hours | **Tool:** DVC (Data Version Control)

## 2.1 Why This Session Exists

Git is excellent at versioning code and catastrophic at versioning data. Commit a 5GB Parquet file to git and you'll bloat the repo permanently, slow every clone to a crawl, and still have no good way to track *which* version of the data a given model was trained on. Yet "which exact dataset produced this model?" is one of the most common questions an ML Engineer is asked — during an incident, an audit, or simply trying to reproduce a colleague's result.

DVC solves this by keeping git as the system of record for *pointers* (small `.dvc` metadata files), while the actual data bytes live in a separate, content-addressable storage layer (local disk, S3, GCS, or any remote — note we use **local/file-based remotes only** in this course, no cloud provisioning). The result: `git checkout` a commit, run `dvc checkout`, and your working directory's *data* snapshots back to exactly the state it was in at that commit — alongside the code.

## 2.2 Core Concepts

**`.dvc` file:** a small, git-trackable text file that points to the actual data's location in the DVC cache via a content hash (MD5). This is what you commit to git — never the raw data itself.

**DVC Cache / Remote:** content-addressable storage where actual data files live, keyed by hash. "Remote" here just means "a storage location DVC knows about" — for this course, that's a local directory, no cloud account required.

**`dvc.yaml` (Pipeline):** declares your workflow as a DAG of stages, each with explicit `deps` (inputs) and `outs` (outputs). DVC tracks the hash of every dep/out and only re-runs a stage if something it depends on actually changed.

**`dvc.lock`:** auto-generated, records the exact hashes of every dep/out for the last successful pipeline run — this is what makes a pipeline run byte-for-byte reproducible.

**DVC Experiments (`dvc exp`):** a lightweight way to run many pipeline variations (e.g., a hyperparameter sweep) *without* creating a git commit or branch per run — experiments live in a temporary workspace and you only commit/promote the one(s) that matter.

## 2.3 Setting Up

```bash
pip install dvc
git init
dvc init               # creates .dvc/ — DVC's project metadata, git-tracked
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Configure a LOCAL remote (no cloud account/provisioning needed for this course)
dvc remote add -d localstore /mnt/dvc-storage
git add .dvc/config
git commit -m "Configure local DVC remote"
```

## 2.4 Versioning a Dataset (Manual `dvc add`)

```bash
dvc add data/customers_raw.csv
```

This produces `data/customers_raw.csv.dvc` (commit this to git) and moves the actual file into the DVC cache, replacing the working-copy file with a lightweight link to the cache.

```bash
git add data/customers_raw.csv.dvc data/.gitignore
git commit -m "Version raw customer dataset v1"
dvc push   # uploads the actual bytes to the configured remote
```

Now imagine the data updates:

```bash
# customers_raw.csv changes upstream...
dvc add data/customers_raw.csv     # hash changes, .dvc file updates
git add data/customers_raw.csv.dvc
git commit -m "Update raw dataset — Q2 refresh"
dvc push
```

To go back to the original version:

```bash
git checkout <v1-commit-hash> -- data/customers_raw.csv.dvc
dvc checkout
```

Your working directory now has the *exact* v1 data back — this is the "which CSV trained this model" question, answered definitively.

## 2.5 Building a Multi-Stage Pipeline (`dvc.yaml`)

Rather than `dvc add`-ing files manually at each step, define the whole workflow as a DAG. This is where Session 1's validation gate plugs in as Stage 0.

```yaml
# dvc.yaml
stages:
  validate:
    cmd: python src/validate.py data/customers_raw.csv
    deps:
      - src/validate.py
      - data/customers_raw.csv
    outs:
      - data/validation_passed.flag

  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/customers_raw.csv
      - data/validation_passed.flag
    params:
      - preprocess.test_size
    outs:
      - data/train.csv
      - data/test.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/train.csv
    params:
      - train.n_estimators
      - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/test.csv
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

```yaml
# params.yaml — the single source of truth for tunable values
preprocess:
  test_size: 0.2
train:
  n_estimators: 200
  max_depth: 8
```

`validate.py` is literally the `validate_or_halt(df)` function built at the end of Session 1's lab — Session 1's output becomes Session 2's Stage 0, and the pipeline physically cannot proceed to `preprocess` if validation fails, because `preprocess` depends on `validation_passed.flag`, which is only written on success.

## 2.6 Running & Caching the Pipeline

```bash
dvc repro
```

`dvc repro` walks the DAG and re-runs **only the stages whose deps/params actually changed** since the last successful run — recorded in `dvc.lock`. Change `train.max_depth` in `params.yaml` and rerun: `validate` and `preprocess` are skipped (cached, untouched), only `train` and `evaluate` re-execute. This DAG-aware caching is most of what an orchestrator gives you, without needing one.

```bash
git add dvc.yaml dvc.lock params.yaml
git commit -m "Pipeline run: max_depth=8"
dvc push
```

## 2.7 DVC Experiments — Sweeping Without Git Noise

```bash
dvc exp run --queue -S train.max_depth=4
dvc exp run --queue -S train.max_depth=8
dvc exp run --queue -S train.max_depth=12
dvc exp run --queue -S train.n_estimators=400 -S train.max_depth=8
dvc exp run --run-all          # executes everything queued
```

```bash
dvc exp show       # tabular comparison of every experiment's params + metrics
```

Each `dvc exp run` reproduces the pipeline in an isolated workspace — no commit, no branch, no checkout disruption to your actual working copy. Only once you find the winner do you promote it:

```bash
dvc exp apply exp-a1b2c   # bring that experiment's results into your workspace
git add dvc.lock params.yaml
git commit -m "Promote max_depth=8, n_estimators=400 as new baseline"
```

## 2.8 Lab Walkthrough (Session 2)

1. `dvc init` and configure a local remote (no cloud account).
2. `dvc add` the raw dataset from Session 1, commit, `dvc push`.
3. Build the 4-stage `dvc.yaml` pipeline above, with `validate.py` calling the Session 1 `validate_or_halt` gate as Stage 0.
4. Run `dvc repro` — confirm all 4 stages execute on first run.
5. Change only `params.yaml`'s `max_depth`, rerun `dvc repro` — confirm `validate`/`preprocess` are skipped (DAG-aware caching) and only `train`/`evaluate` execute.
6. Deliberately corrupt the dataset (re-introduce a Session 1 validation failure) and rerun — confirm the pipeline halts at Stage 0 and never reaches `train`.
7. Queue a 6-run `dvc exp` sweep over `max_depth` × `n_estimators`, run with `--run-all`, compare with `dvc exp show`, and `dvc exp apply` the winner.
8. `git checkout` an earlier commit + `dvc checkout` to prove the exact prior dataset/model state restores.

## 2.9 Trade-off Discussion — Recap

**DVC Pipelines vs. a full orchestrator (Airflow/Prefect) — when is DVC enough?**
DVC pipelines are sufficient when the whole workflow runs on one machine or one job runner and the goal is reproducibility + caching, not cross-team scheduling, distributed retries, or complex fan-out/fan-in across hundreds of parallel jobs. The moment you need SLA-based scheduling or orchestration across many machines, you add a dedicated orchestrator — but it sits *on top of* DVC's versioning guarantees, it doesn't replace them.

## 2.10 Common Pitfalls

- **Committing data to git "just this once."** It compounds — every future clone pays for it forever. Always `dvc add`, never `git add` on raw data.
- **Forgetting `dvc push`.** A `.dvc` file commit without a push means the *pointer* is in git but the *bytes* aren't in the remote — anyone else's `dvc checkout` will fail.
- **Putting tunable values inline in scripts instead of `params.yaml`.** This breaks `dvc exp run -S` sweeps and DAG-aware re-run detection for parameter changes.
- **Skipping the validation stage dependency.** If `preprocess` doesn't explicitly depend on `validation_passed.flag`, DVC has no way to know it must wait — the DAG only enforces what you declare.

---

## What Connects to Session 3

Every `dvc exp run` you just queued produced metrics — right now, your only view into them is `dvc exp show`'s table. Session 3 introduces **MLflow Tracking**, which takes over as the rich, queryable, UI-driven layer for comparing runs — while DVC continues to guarantee that any of those runs is reproducible byte-for-byte from versioned data. You are not replacing what you just built; you're about to make it browsable.
