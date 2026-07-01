# Complete Notes — Session 6 & Session 7
### MLOps for Data Scientists: An ML-Native Path to Production

---

# SESSION 6 — ML-Native Model Serving with BentoML & Triton Inference Server
**Duration:** 5 hours | **Tools:** BentoML, Triton Inference Server (intro) | **Capstone Part 1 kickoff**

---

## 6.1 Why This Session Exists

By the end of Session 5 you had a named, versioned, aliased model in the MLflow Registry with an enforced input signature. That model is still not doing anything useful — it's an artifact on disk. Serving is the step that turns it into a live API a downstream system can actually call.

Most "how to deploy an ML model" tutorials immediately reach for Flask, FastAPI, Docker, and Kubernetes. That stack works, but it treats the model as just another web application, forcing you to own all the serving infrastructure that has nothing to do with ML: container image builds, health checks, container registries, and orchestration manifests. Worse, it ignores concerns that are *entirely ML-specific*: adaptive batching (grouping individual requests into batches to saturate a GPU), multi-framework support (your team uses both sklearn and PyTorch in the same service), model runner isolation (a heavy model inference shouldn't block the pre/post-processing logic on the same thread), and serving-side input validation against a schema.

**BentoML** is built specifically to solve these problems at the Python layer — *before* any containerization is involved. A BentoML `Service` is a Python object you run locally the same way you'd run a FastAPI app, with none of the container machinery. **Triton Inference Server** is the GPU-optimized, polyglot alternative for teams with ONNX/TensorRT/PyTorch workloads at scale. Both are taught here; the decision framework between them is a key trade-off discussion.

No Dockerfiles are written in this session. No Kubernetes manifests. Serving is treated as a Python-native concern.

---

## 6.2 BentoML — Core Concepts

**Service:** the top-level BentoML object. Analogous to a FastAPI `app`. Exposes one or more API endpoints and wires them to one or more Runners.

**Runner:** a BentoML abstraction that wraps a model (or any callable) and runs it in an isolated process, enabling adaptive batching and non-blocking async inference. The service's web-facing logic and the model's compute logic are deliberately separated into different processes.

**Bento:** a versioned, self-contained build artifact produced by `bentoml build`. Contains the service code, model artifacts, dependencies, and a `bentofile.yaml` — the ML-native equivalent of a build artefact. Think of it as a "shipping unit."

**Model Store:** BentoML's local model registry (`~/.bentoml/models/`). Separate from the MLflow Registry. In this course, we bridge the two: the MLflow Registry is the *source of truth* for model lifecycle; BentoML's store is where the artifact lands immediately before serving.

**Input/Output Descriptor:** defines what the BentoML endpoint accepts and returns — Pandas DataFrame, NumPy array, JSON, raw bytes, image, etc. These descriptors drive automatic API documentation and input deserialization.

**Adaptive Batching:** BentoML can automatically group multiple concurrent single-item requests into one batched model call — critical for GPU utilization. Configured per-Runner, not per-request.

---

## 6.3 Architecture: MLflow Registry → BentoML Service

```
MLflow Model Registry
   "churn-prediction-rf@champion"
              │
              │  mlflow.pyfunc.load_model() at startup
              ▼
   BentoML Runner
   (isolated process — model inference only)
              │
              │  runner.predict.async_run(batch)
              ▼
   BentoML Service
   (web process — HTTP handling, pre/post processing, Pandera validation)
              │
              │  HTTP POST /predict  { "customer_id": 1001, ... }
              ▼
   Client (downstream system / load test / capstone monitoring layer)
```

---

## 6.4 Setting Up

```bash
pip install bentoml mlflow scikit-learn pandera
```

---

## 6.5 Saving a Model to BentoML's Store

BentoML has its own local model store. The first step is loading the champion from MLflow and saving it into BentoML's store. This is a one-time bootstrap that you'd re-run whenever a new champion is promoted in the registry:

```python
# scripts/save_model_to_bentoml.py
import mlflow.sklearn
import bentoml

MODEL_NAME    = "churn-prediction-rf"
MODEL_ALIAS   = "champion"

# Load from MLflow Registry by alias — no hardcoded run ID
sklearn_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)

# Save into BentoML's local store
saved_model = bentoml.sklearn.save_model(
    name="churn_rf_champion",
    model=sklearn_model,
    signatures={
        "predict":        {"batchable": True, "batch_dim": 0},
        "predict_proba":  {"batchable": True, "batch_dim": 0},
    },
    metadata={
        "mlflow_model_name":  MODEL_NAME,
        "mlflow_model_alias": MODEL_ALIAS,
        "description": "Churn prediction RandomForest — Q2-2024 Feast features",
    },
)

print(f"Saved to BentoML store: {saved_model.tag}")
# → churn_rf_champion:a1b2c3d4e5f6
```

---

## 6.6 Building a BentoML Service

```python
# service.py
import bentoml
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from bentoml.io import PandasDataFrame, NumpyNdarray
import numpy as np

# --- Step 1: Reuse the Pandera schema from Session 1 as the serving-side gate ---
class ChurnInputSchema(pa.DataFrameModel):
    monthly_spend:       Series[float] = pa.Field(ge=0)
    rolling_30d_spend:   Series[float] = pa.Field(ge=0)
    tenure_days:         Series[int]   = pa.Field(ge=0)
    subscription_tier:   Series[str]   = pa.Field(isin=["free", "pro", "enterprise"])
    support_tickets_30d: Series[int]   = pa.Field(ge=0)

    class Config:
        coerce = True

# --- Step 2: Create a Runner from the saved BentoML model ---
churn_runner = bentoml.sklearn.get("churn_rf_champion:latest").to_runner()

# --- Step 3: Define the Service, binding it to the Runner ---
svc = bentoml.Service(
    name="churn_prediction_service",
    runners=[churn_runner],
)

# --- Step 4: Define an API endpoint ---
@svc.api(
    input=PandasDataFrame(
        orient="records",          # expects JSON list of row dicts
        dtype={
            "monthly_spend":       float,
            "rolling_30d_spend":   float,
            "tenure_days":         int,
            "subscription_tier":   str,
            "support_tickets_30d": int,
        },
        apply_column_names=True,
    ),
    output=NumpyNdarray(dtype="float32"),
    route="/predict",
)
@pa.check_types   # Session 1's decorator — validates input BEFORE model sees it
async def predict(input_df: DataFrame[ChurnInputSchema]) -> np.ndarray:
    """
    Returns churn probability (class 1) for each input row.
    Input validated against ChurnInputSchema at the serving boundary.
    """
    # async_run sends to the Runner's isolated process — non-blocking
    proba = await churn_runner.predict_proba.async_run(input_df)
    return proba[:, 1].astype("float32")   # return P(churn=1) only
```

---

## 6.7 Running the Service Locally

```bash
bentoml serve service:svc --reload
# → Service running on http://localhost:3000
# → API docs auto-generated at http://localhost:3000/docs
```

Test with `curl`:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"monthly_spend": 120.5, "rolling_30d_spend": 95.0,
     "tenure_days": 450, "subscription_tier": "pro",
     "support_tickets_30d": 2},
    {"monthly_spend": 15.0, "rolling_30d_spend": 12.0,
     "tenure_days": 30, "subscription_tier": "free",
     "support_tickets_30d": 7}
  ]'
# → [0.08432, 0.79231]
```

Or from Python:

```python
import requests
import json

payload = [
    {
        "monthly_spend": 120.5,
        "rolling_30d_spend": 95.0,
        "tenure_days": 450,
        "subscription_tier": "pro",
        "support_tickets_30d": 2,
    }
]

response = requests.post(
    "http://localhost:3000/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
)
print(response.json())
```

---

## 6.8 Adaptive Batching Configuration

When multiple concurrent requests arrive, BentoML's Runner can group them into a single batched model call — critical for throughput on CPU and essential for GPU saturation:

```python
# Configured when saving the model to BentoML's store
saved_model = bentoml.sklearn.save_model(
    name="churn_rf_champion",
    model=sklearn_model,
    signatures={
        "predict_proba": {
            "batchable":     True,
            "batch_dim":     0,         # stack along axis 0 (rows)
            "max_batch_size": 100,      # max rows per merged batch
            "max_latency_ms": 100,      # wait at most 100ms before flushing a partial batch
        },
    },
)
```

With this config, BentoML will accumulate concurrent single-row requests and merge them into a single `predict_proba(batch)` call, dramatically improving throughput under load — without any change to the Service code or the endpoint contract.

---

## 6.9 Logging Inference Data for Session 7 Monitoring

This is the forward-integration step: every prediction the service makes gets logged so the Session 7 monitoring layer has real inference data to analyze for drift.

```python
# service.py (extended)
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

INFERENCE_LOG_PATH = Path("data/inference_log.jsonl")

@svc.api(
    input=PandasDataFrame(orient="records"),
    output=NumpyNdarray(dtype="float32"),
    route="/predict",
)
@pa.check_types
async def predict(input_df: DataFrame[ChurnInputSchema]) -> np.ndarray:
    proba = await churn_runner.predict_proba.async_run(input_df)
    result = proba[:, 1].astype("float32")

    # Log inference data for drift monitoring (Session 7)
    log_rows = input_df.copy()
    log_rows["churn_probability"] = result
    log_rows["prediction_ts"]     = datetime.utcnow().isoformat()

    with open(INFERENCE_LOG_PATH, "a") as f:
        for record in log_rows.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")

    return result
```

This JSONL file is what Session 7's Evidently monitoring will read to build the "current" distribution and compare it against the training "reference" distribution.

---

## 6.10 Building & Versioning a Bento

Once the service works locally, you build it into a versioned Bento artifact:

```yaml
# bentofile.yaml
service: "service:svc"
description: "Churn Prediction Service — Q2-2024 Feast Features"
labels:
  owner: ml-platform
  stage: staging
include:
  - "service.py"
  - "schemas/*.py"
python:
  packages:
    - scikit-learn
    - pandera
    - mlflow
    - feast
```

```bash
bentoml build
# → Successfully built Bento(tag="churn_prediction_service:v1a2b3c4")

bentoml list
# Shows all local bentos with their tags

bentoml serve churn_prediction_service:v1a2b3c4
# Serve the immutable, versioned artifact — not the live code
```

---

## 6.11 Load Testing with Adaptive Batching

```python
# scripts/load_test.py
import requests
import concurrent.futures
import time
import json

ENDPOINT = "http://localhost:3000/predict"
SAMPLE_PAYLOAD = [
    {
        "monthly_spend": 120.5,
        "rolling_30d_spend": 95.0,
        "tenure_days": 450,
        "subscription_tier": "pro",
        "support_tickets_30d": 2,
    }
]

def single_request(_):
    start = time.time()
    r = requests.post(ENDPOINT,
                      headers={"Content-Type": "application/json"},
                      data=json.dumps(SAMPLE_PAYLOAD))
    return time.time() - start, r.status_code

# Fire 50 concurrent single-row requests
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = list(executor.map(single_request, range(50)))

latencies   = [r[0] for r in results]
status_codes = [r[1] for r in results]
print(f"Avg latency : {sum(latencies)/len(latencies)*1000:.1f}ms")
print(f"P99 latency : {sorted(latencies)[int(0.99*len(latencies))]*1000:.1f}ms")
print(f"Success rate: {status_codes.count(200)/len(status_codes)*100:.1f}%")
```

Run this once *without* adaptive batching enabled and once *with* — the difference in throughput and P99 latency demonstrates concretely why adaptive batching matters for production serving.

---

## 6.12 Triton Inference Server — Concepts & When to Use It

Triton is NVIDIA's open-source inference server. Where BentoML is Python-native and developer-friendly, Triton is performance-optimized and framework-agnostic, built in C++, accepting ONNX, TensorRT, PyTorch TorchScript, TensorFlow SavedModel, and OpenVINO models from a common model repository format.

**Model Repository Layout:**

```
model_repository/
└── churn_rf_onnx/
    ├── config.pbtxt          # model configuration — Triton's native format
    └── 1/
        └── model.onnx        # the model artifact
```

**Model Config (`config.pbtxt`):**

```protobuf
name: "churn_rf_onnx"
backend: "onnxruntime"
max_batch_size: 128

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 5 ]    # 5 features per row
  }
]

output [
  {
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 2 ]    # P(churn=0), P(churn=1)
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 32, 64 ]
  max_queue_delay_microseconds: 100000   # 100ms queue wait
}
```

**Exporting a scikit-learn model to ONNX:**

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import mlflow.sklearn

# Load champion from MLflow
model = mlflow.sklearn.load_model("models:/churn-prediction-rf@champion")

# Convert to ONNX
initial_type = [("input", FloatTensorType([None, 5]))]   # None = dynamic batch size
onnx_model   = convert_sklearn(model, initial_types=initial_type)

# Save to Triton model repository
import os
os.makedirs("model_repository/churn_rf_onnx/1", exist_ok=True)
with open("model_repository/churn_rf_onnx/1/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model written to Triton repository")
```

**Querying Triton (HTTP client):**

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Build input tensor
input_data  = np.array([[120.5, 95.0, 450, 1.0, 2]], dtype=np.float32)
# (subscription_tier encoded to float — Triton handles numeric types only)

inputs  = [httpclient.InferInput("input", input_data.shape, "FP32")]
outputs = [httpclient.InferRequestedOutput("probabilities")]
inputs[0].set_data_from_numpy(input_data)

result = client.infer(model_name="churn_rf_onnx", inputs=inputs, outputs=outputs)
proba  = result.as_numpy("probabilities")
print(f"P(churn=1): {proba[0][1]:.4f}")
```

---

## 6.13 BentoML vs. Triton — Decision Framework

| Dimension | BentoML | Triton |
|---|---|---|
| **Language** | Python-native — pre/post processing as ordinary Python functions | Config-native — `config.pbtxt` + binary model formats |
| **Framework support** | sklearn, PyTorch, XGBoost, Keras, LightGBM, HuggingFace, custom Python | ONNX, TensorRT, PyTorch TorchScript, TensorFlow, OpenVINO |
| **GPU optimization** | Batching + async, not C++-level GPU ops | Dynamic batching with CUDA-native execution — maximum GPU utilization |
| **Pre/post processing** | First-class Python in the Service object | Separate BLS (Business Logic Scripting) layer — added complexity |
| **Multi-model serving** | Multiple Runners per Service | Model ensemble config natively supported |
| **Setup complexity** | `pip install bentoml` → `bentoml serve` | Binary server process, model repo structure, protobuf config |
| **Best for** | Python-centric teams, sklearn/PyTorch, fast iteration, mixed ML+logic endpoints | High-throughput GPU workloads, multi-framework serving, latency-sensitive LLM/vision pipelines |

**Rule of thumb:** BentoML is almost always the right starting point. Graduate to Triton when GPU utilization matters (you're leaving throughput on the table with Python-layer batching), when you need to serve ONNX/TensorRT artifacts from multiple frameworks behind one server, or when sub-10ms latency becomes a hard requirement.

---

## 6.14 Lab Walkthrough (Session 6)

1. Run `scripts/save_model_to_bentoml.py` — load the `@champion` from the MLflow Registry and save it into BentoML's local store.
2. Build `service.py` with the `ChurnInputSchema` Pandera validation (reusing Session 1's schema object), a `predict` endpoint, and the inference log writer.
3. `bentoml serve service:svc --reload` — test via `curl` and the auto-generated Swagger UI at `/docs`.
4. Send a request with a deliberately invalid payload (e.g., `"subscription_tier": "platinum"`) — confirm Pandera raises a `SchemaError` and the service returns a 422 before the model is ever called.
5. Enable adaptive batching in `save_model_to_bentoml.py` (`batchable=True`, `max_latency_ms=100`), rebuild, and run the load test — compare P99 latency with batching on vs. off.
6. `bentoml build` — verify the Bento is versioned and immutable.
7. **(Stretch):** Export the champion to ONNX, write the Triton `config.pbtxt`, verify the ONNX model accepts a batch of 5 rows and returns probabilities.

---

## 6.15 Trade-off Discussion — Recap

**BentoML vs. Triton — when do you need each?**
BentoML is the right default: Python-native, fast to iterate on, excellent for mixed ML+business-logic endpoints, and requires no understanding of protobuf configs or binary model formats. Triton is the right choice when maximum GPU throughput matters more than developer convenience — it eliminates Python overhead from the hot inference path, handles multi-framework ensembles natively, and scales to thousands of concurrent GPU requests with CUDA-native execution. Most teams start with BentoML and graduate specific high-load endpoints to Triton when profiling shows Python is the bottleneck.

---

## 6.16 Common Pitfalls

- **Loading the model inside the endpoint function (not at startup).** Loading a model on every request is catastrophically slow — it must be loaded once, in the Runner, at service startup.
- **Not pinning the model alias when saving to BentoML's store.** If you always save as `"latest"`, you can't tell which MLflow champion version a given Bento was built from. Store the alias and run ID in the model metadata.
- **Skipping Pandera validation at the serving boundary.** The model was trained on clean features — if corrupted data reaches it, it predicts silently and wrongly. Validation at the serving layer is not redundant with Session 1's pipeline gate; it's a separate, serving-time guarantee.
- **Using synchronous `runner.run()` for high-concurrency endpoints.** Under load, synchronous calls block the event loop. Always use `await runner.method.async_run()` in async endpoint functions.
- **Forgetting to flush the inference log.** If the log file isn't flushed/rotated, Session 7's monitoring will read an incomplete or outdated window.

---

## 🏗️ CAPSTONE PROJECT — Part 1: Foundation
### Building the ML-Native Serving Foundation (No Dockerfiles, No Kubernetes)

**Context:** The Capstone builds a single, incrementally extended pipeline across Sessions 6, 7, and 8. Part 1 establishes all serving-layer components that Parts 2 and 3 will monitor, automate, and integrate. Choose your own dataset — the running example uses a churn prediction problem, but any tabular binary/multi-class classification dataset is valid.

---

### Capstone Part 1 — Deliverable 1: Data Validation Gate

Ensure Session 1's validation gate is production-ready and integrated into the pipeline:

```python
# src/validate.py
import great_expectations as gx
import pandera as pa
from pandera.typing import DataFrame, Series
import pandas as pd
import sys

class RawDataSchema(pa.DataFrameModel):
    """Pandera schema — used both here and imported by the BentoML service."""
    customer_id:         Series[int]   = pa.Field(unique=True, gt=0)
    monthly_spend:       Series[float] = pa.Field(ge=0)
    rolling_30d_spend:   Series[float] = pa.Field(ge=0)
    tenure_days:         Series[int]   = pa.Field(ge=0)
    subscription_tier:   Series[str]   = pa.Field(isin=["free", "pro", "enterprise"])
    support_tickets_30d: Series[int]   = pa.Field(ge=0)
    churn_label:         Series[int]   = pa.Field(isin=[0, 1])

def validate_or_halt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    try:
        validated = RawDataSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("❌ Pandera validation FAILED:")
        print(err.failure_cases.to_string())
        sys.exit(1)

    # Great Expectations Checkpoint (pre-configured suite)
    context = gx.get_context()
    result  = context.checkpoints.get("capstone_checkpoint").run(
        batch_parameters={"path": path}
    )
    if not result["success"]:
        context.build_data_docs()
        print("❌ GE Checkpoint FAILED — see Data Docs")
        sys.exit(1)

    print(f"✅ Validation passed — {len(validated)} rows, {len(validated.columns)} columns")
    open("data/validation_passed.flag", "w").close()
    return validated

if __name__ == "__main__":
    validate_or_halt(sys.argv[1])
```

---

### Capstone Part 1 — Deliverable 2: DVC Pipeline

```yaml
# dvc.yaml
stages:
  validate:
    cmd: python src/validate.py data/capstone_raw.csv
    deps:
      - src/validate.py
      - data/capstone_raw.csv
    outs:
      - data/validation_passed.flag

  featurize:
    cmd: python src/featurize.py
    deps:
      - src/featurize.py
      - data/capstone_raw.csv
      - data/validation_passed.flag
      - feast_repo/feature_store.yaml
    params:
      - featurize.feast_repo_path
    outs:
      - data/training_features.parquet

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/training_features.parquet
    params:
      - train.n_estimators
      - train.max_depth
      - train.random_state
    outs:
      - models/capstone_model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/capstone_model.pkl
      - data/training_features.parquet
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

```bash
dvc repro        # runs the full pipeline, gates enforced by DAG deps
git add dvc.yaml dvc.lock params.yaml
git commit -m "Capstone Part 1: DVC pipeline with validation gate"
dvc push
```

---

### Capstone Part 1 — Deliverable 3: MLflow-Tracked Training Run + Registry

```python
# src/train.py
import mlflow
import mlflow.sklearn
import subprocess
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from mlflow.tracking import MlflowClient

# Load params from DVC params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

training_df   = pd.read_parquet("data/training_features.parquet")
X_train       = training_df.drop(["customer_id", "event_timestamp", "churn_label"], axis=1)
y_train       = training_df["churn_label"]
X_val, y_val  = X_train[-200:], y_train[-200:]
X_train, y_train = X_train[:-200], y_train[:-200]

mlflow.set_experiment("capstone-churn-prediction")

with mlflow.start_run(run_name=f"capstone-rf-d{params['max_depth']}") as run:
    mlflow.log_params(params)
    mlflow.set_tags({
        "feast.feature_view":  "customer_stats",
        "dvc.commit":          subprocess.getoutput("git rev-parse HEAD"),
        "pipeline.stage":      "capstone-part-1",
    })

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    val_f1  = f1_score(y_val, model.predict(X_val))
    mlflow.log_metrics({"val_roc_auc": val_auc, "val_f1": val_f1})

    import json
    with open("metrics/eval_metrics.json", "w") as mf:
        json.dump({"roc_auc": val_auc, "f1": val_f1}, mf)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:3],
        registered_model_name="capstone-churn-rf",
    )

    # Promote to Staging immediately
    client  = MlflowClient()
    version = client.get_latest_versions("capstone-churn-rf", stages=["None"])[0].version
    client.transition_model_version_stage("capstone-churn-rf", version, "Staging")
    client.set_registered_model_alias("capstone-churn-rf", "challenger", version)
    print(f"✅ v{version} registered and promoted to Staging as 'challenger'")
```

---

### Capstone Part 1 — Deliverable 4: Working BentoML Service

The BentoML `service.py` from Section 6.6 is the final Part 1 deliverable. Verify these four things before calling Part 1 complete:

```bash
# 1. Service runs and accepts requests
bentoml serve service:svc
curl -X POST http://localhost:3000/predict -H "Content-Type: application/json" \
  -d '[{"monthly_spend": 120.5, "rolling_30d_spend": 95.0,
         "tenure_days": 450, "subscription_tier": "pro",
         "support_tickets_30d": 2}]'

# 2. Invalid input is rejected at serving boundary (not inside the model)
curl -X POST http://localhost:3000/predict -H "Content-Type: application/json" \
  -d '[{"monthly_spend": -50, "subscription_tier": "diamond"}]'
# → 422 Unprocessable Entity (Pandera SchemaError)

# 3. Inference log is being written
tail -f data/inference_log.jsonl

# 4. Bento build produces a versioned artifact
bentoml build && bentoml list
```

**Part 1 is complete** when all four checks pass, dvc.lock is committed, the MLflow Registry shows version 1 at Staging with alias `challenger`, and the inference log file exists.

---
---

# SESSION 7 — Monitoring & Drift Detection with Evidently AI & WhyLabs
**Duration:** 5 hours | **Tools:** Evidently AI, whylogs / WhyLabs | **Capstone Part 2**

---

## 7.1 Why This Session Exists

Deploying a model is not the end of the ML lifecycle — it's the beginning of the hardest part. The world keeps changing. Customer behavior shifts, upstream data pipelines evolve, feature distributions drift, and the relationship between features and labels changes over time (concept drift). None of this announces itself; it just quietly erodes model performance until a business stakeholder asks why the churn prediction model is performing so poorly.

The gap between "model deployed" and "model performing as expected" is the monitoring layer. In most teams, monitoring means: (a) tracking business KPIs and noticing they got worse, then (b) blaming the model. That's post-hoc debugging, not proactive monitoring. The goal of this session is to build a monitoring layer that catches drift *before* it causes visible business impact — and automatically triggers the retraining loop.

Two tools serve this need in complementary ways:
- **Evidently AI** — a Python library for computing rich statistical comparison reports between a reference dataset (training) and a current dataset (recent inference data). Produces HTML reports and programmatic Test Suites with pass/fail signals.
- **whylogs / WhyLabs** — a statistical profiling library that produces compact "sketches" of data distributions without retaining raw data. Designed for streaming and high-volume settings where holding full datasets in memory is not feasible.

---

## 7.2 Core Monitoring Concepts

Before the tools: the vocabulary of what you are monitoring.

**Data Drift:** the statistical distribution of input features has changed between training time and inference time. The model has not changed, but the data it now receives is different from the data it was trained on. Example: average customer tenure shifts from 300 days to 150 days because a new marketing campaign acquired younger users.

**Concept Drift:** the relationship between features and the target label has changed. The same feature values now predict a different outcome. Example: features that predicted churn in Q1 no longer predict churn in Q3 because competitor pricing changed. This is harder to detect — it requires ground-truth labels, which often arrive with a lag.

**Label Drift:** the distribution of the predicted (or actual) target variable has shifted. Example: the model is now predicting "churn" for 40% of customers when historically it predicted 15% — possibly indicating data drift upstream or a genuine behavioral change.

**Prediction Drift:** the model's output distribution has shifted even if no individual prediction seems wrong. Example: the model starts assigning very high churn probabilities (>0.9) to almost everyone — a sign that the input feature distribution has moved out of the training manifold.

**Reference Dataset:** the "baseline" — usually a representative sample of the training data, or the first month of production data after go-live. All drift metrics are comparisons *against this reference*.

**Current Dataset:** a window of recent production inference data — the thing being compared to the reference.

---

## 7.3 Evidently AI — Core Concepts

**Report:** a collection of one or more Metrics computed over reference vs. current data. Produces a rich HTML visualization — designed for human inspection.

**Test Suite:** a collection of one or more Tests — each Test is a pass/fail assertion over a metric (e.g., "dataset drift share < 20%"). Designed for programmatic pipeline gates.

**Metric:** the unit of computation — e.g., `DataDriftTable` (per-column drift statistics), `DatasetDriftMetric` (overall drift summary), `ClassificationQualityMetric` (model performance), `DataQualityMetrics` (nulls, ranges, value distributions).

**Preset:** a convenience bundle of related Metrics — e.g., `DataDriftPreset()` bundles everything needed for a comprehensive drift analysis in one line.

**Drift Detection Algorithms** (selectable per column type):
- **PSI (Population Stability Index):** industry standard from credit risk modeling. Values: <0.1 = no drift, 0.1–0.2 = moderate drift, >0.2 = significant drift.
- **Kolmogorov-Smirnov (KS) test:** non-parametric test for continuous distributions. Evidently's default for numeric columns.
- **Wasserstein Distance (Earth Mover's Distance):** measures the "effort" to move one distribution into the other. Intuitive and scale-invariant.
- **Jensen-Shannon Divergence:** symmetric, bounded [0,1] divergence between two distributions. Common for categorical columns.
- **Chi-Squared test:** standard test for categorical distribution shifts.

---

## 7.4 Setting Up Evidently

```bash
pip install evidently
```

---

## 7.5 Building a Data Drift Report

```python
# src/monitoring/drift_report.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
)

# --- Reference: sample from training data ---
reference_df = pd.read_parquet("data/training_features.parquet")
reference_df = reference_df.drop(
    ["customer_id", "event_timestamp", "churn_label"], axis=1
).sample(1000, random_state=42)

# --- Current: recent inference data from BentoML's log ---
inference_df = pd.read_json("data/inference_log.jsonl", lines=True)
current_df   = inference_df.drop(
    ["churn_probability", "prediction_ts"], axis=1
).tail(500)   # last 500 predictions as the monitoring window

# --- Build Report ---
drift_report = Report(metrics=[
    DataDriftPreset(),           # per-column drift table + summary
    DataQualityPreset(),         # nulls, ranges, value distributions
    ColumnDriftMetric(column_name="monthly_spend"),     # drill down on specific column
    ColumnDriftMetric(column_name="rolling_30d_spend"),
    ColumnDriftMetric(column_name="tenure_days"),
])

drift_report.run(reference_data=reference_df, current_data=current_df)

# Save as HTML for human inspection
drift_report.save_html("reports/drift_report.html")
print("✅ Drift report saved to reports/drift_report.html")

# Extract the JSON results for programmatic use
report_dict = drift_report.as_dict()
drift_share = report_dict["metrics"][0]["result"]["share_of_drifted_columns"]
print(f"Share of drifted columns: {drift_share:.1%}")
```

---

## 7.6 Building a Test Suite — The Pipeline Gate

A Report is for humans. A Test Suite is for pipelines: it produces a single pass/fail signal and a structured JSON result.

```python
# src/monitoring/drift_test_suite.py
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
    TestColumnValueMean,
    TestColumnValueMin,
    TestColumnValueMax,
    TestShareOfMissingValues,
)

reference_df = pd.read_parquet("data/training_features.parquet") \
                 .drop(["customer_id", "event_timestamp", "churn_label"], axis=1) \
                 .sample(1000, random_state=42)

inference_df = pd.read_json("data/inference_log.jsonl", lines=True)
current_df   = inference_df.drop(["churn_probability", "prediction_ts"], axis=1).tail(500)

# --- Define Tests as assertions ---
test_suite = TestSuite(tests=[
    # Overall drift gate: fail if >20% of columns are drifted
    TestShareOfDriftedColumns(lt=0.20),

    # Per-column drift tests (PSI for spend columns — financial industry standard)
    TestColumnDrift(column_name="monthly_spend",     stattest="psi", threshold=0.2),
    TestColumnDrift(column_name="rolling_30d_spend", stattest="psi", threshold=0.2),
    TestColumnDrift(column_name="tenure_days",       stattest="ks",  threshold=0.05),

    # Data quality tests on the serving window
    TestShareOfMissingValues(lt=0.02),

    # Range sanity checks — catch upstream ETL bugs
    TestColumnValueMin(column_name="monthly_spend",  gt=0),
    TestColumnValueMean(column_name="tenure_days",   gt=60),
])

test_suite.run(reference_data=reference_df, current_data=current_df)
test_suite.save_html("reports/test_suite_report.html")

# Programmatic pass/fail
results    = test_suite.as_dict()
all_passed = results["summary"]["all_passed"]
failed     = [t for t in results["tests"] if t["status"] == "FAIL"]

print(f"Test Suite: {'✅ PASSED' if all_passed else '❌ FAILED'}")
for f in failed:
    print(f"  FAILED: {f['name']} — {f['description']}")
```

---

## 7.7 Injecting Synthetic Drift for Lab Use

Production drift happens over weeks. The lab demonstrates it in minutes by corrupting a subset of the inference window:

```python
# scripts/inject_drift.py
import pandas as pd
import numpy as np
import json

inference_df = pd.read_json("data/inference_log.jsonl", lines=True)

# Simulate a drift scenario:
# - New marketing campaign attracted users with much lower tenure
# - Spend distribution has shifted upward (premium tier push)
drifted_df = inference_df.copy()
drifted_df["tenure_days"]      = np.random.randint(10, 60,  size=len(drifted_df))
drifted_df["monthly_spend"]    = np.random.uniform(200, 500, size=len(drifted_df))
drifted_df["rolling_30d_spend"]= drifted_df["monthly_spend"] * np.random.uniform(0.8, 1.2, size=len(drifted_df))

with open("data/inference_log_drifted.jsonl", "w") as f:
    for record in drifted_df.to_dict(orient="records"):
        f.write(json.dumps(record) + "\n")

print(f"✅ Drifted inference log written: {len(drifted_df)} rows")
```

Run both the Report and Test Suite against the drifted log — the Test Suite should now fail on `TestColumnDrift` for `tenure_days` and `monthly_spend`, and `TestShareOfDriftedColumns` should exceed 20%.

---

## 7.8 Automated Retraining Trigger

This is the integration point between Evidently and the capstone's DVC pipeline:

```python
# src/monitoring/trigger_retrain.py
import subprocess
import sys
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns, TestColumnDrift

def check_drift_and_maybe_retrain(reference_path: str,
                                   current_log_path: str,
                                   drift_threshold: float = 0.20) -> bool:
    reference_df = pd.read_parquet(reference_path) \
                     .drop(["customer_id", "event_timestamp", "churn_label"], axis=1) \
                     .sample(1000, random_state=42)
    current_df   = pd.read_json(current_log_path, lines=True) \
                     .drop(["churn_probability", "prediction_ts"], axis=1) \
                     .tail(500)

    test_suite = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=drift_threshold),
    ])
    test_suite.run(reference_data=reference_df, current_data=current_df)
    passed = test_suite.as_dict()["summary"]["all_passed"]

    if not passed:
        print("⚠️  Drift threshold exceeded — triggering retraining pipeline")
        subprocess.run(["dvc", "repro", "--force"], check=True)
        print("✅ Retraining complete — new model version queued for registration")
        return True   # retrain was triggered
    else:
        print("✅ No significant drift detected — current champion retained")
        return False  # no retrain needed

if __name__ == "__main__":
    triggered = check_drift_and_maybe_retrain(
        reference_path="data/training_features.parquet",
        current_log_path="data/inference_log.jsonl",
    )
    sys.exit(0 if not triggered else 2)   # exit 2 = retrain triggered (for CI consumption)
```

---

## 7.9 Model Performance Monitoring (When Labels Arrive)

Data drift monitoring is leading-indicator monitoring — it doesn't require labels. When ground-truth labels become available (e.g., with a lag of 30 days for churn), add model performance monitoring:

```python
# src/monitoring/performance_report.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import ClassificationQualityMetric

# Assume labels have been joined back to inference records after 30-day lag
labeled_current = pd.read_parquet("data/labeled_inference_window.parquet")
labeled_ref     = pd.read_parquet("data/training_features.parquet").sample(1000)

# Both DataFrames must have a 'target' and 'prediction' column
# Evidently's ClassificationPreset expects these names
labeled_current = labeled_current.rename(columns={
    "churn_label":       "target",
    "churn_probability": "prediction",
})
labeled_ref = labeled_ref.rename(columns={
    "churn_label": "target"
})
labeled_ref["prediction"] = 0.5   # placeholder — no inference data for reference

perf_report = Report(metrics=[
    ClassificationPreset(),
    ClassificationQualityMetric(),
])
perf_report.run(reference_data=labeled_ref, current_data=labeled_current)
perf_report.save_html("reports/performance_report.html")
```

---

## 7.10 whylogs / WhyLabs — Lightweight Streaming Profiles

Evidently requires two complete datasets in memory. whylogs computes a statistical *profile* — a compact sketch — of a dataset without storing the raw data. This makes it suitable for streaming pipelines and privacy-sensitive environments.

```bash
pip install whylogs
```

**Profiling the training reference:**

```python
import whylogs as why
import pandas as pd

reference_df = pd.read_parquet("data/training_features.parquet") \
                 .drop(["customer_id", "event_timestamp", "churn_label"], axis=1)

# Log the reference dataset as a whylogs profile
with why.logger(dataset_timestamp=...) as logger:
    logger.log(reference_df)
    reference_profile = logger.get_profile()

reference_profile.writer("local").write(dest="profiles/reference_profile")
print("✅ Reference profile saved")
```

**Profiling the current serving window:**

```python
import whylogs as why
import pandas as pd

current_df = pd.read_json("data/inference_log.jsonl", lines=True) \
               .drop(["churn_probability", "prediction_ts"], axis=1) \
               .tail(500)

with why.logger() as logger:
    logger.log(current_df)
    current_profile = logger.get_profile()

current_profile.writer("local").write(dest="profiles/current_profile")
```

**Comparing profiles:**

```python
from whylogs.viz import NotebookProfileVisualizer
from whylogs.core.view.dataset_profile_view import DatasetProfileView

reference_view = DatasetProfileView.read("profiles/reference_profile")
current_view   = DatasetProfileView.read("profiles/current_profile")

viz = NotebookProfileVisualizer()
viz.set_profiles(
    target_profile_view=current_view,
    reference_profile_view=reference_view,
)
viz.summary_drift_report()   # renders in a Jupyter notebook
```

**Key difference from Evidently:** whylogs profiles contain statistical sketches (histograms, quantile approximations, cardinality estimates) but not raw rows. The profile for a 10M-row dataset is a few KB — Evidently would require storing and comparing both 10M-row datasets.

---

## 7.11 Evidently vs. WhyLogs — Side-by-Side Comparison

| Dimension | Evidently AI | whylogs / WhyLabs |
|---|---|---|
| **Data model** | Two full DataFrames (reference + current) in memory | Compact statistical sketches — raw data not retained |
| **Output** | Rich HTML reports, Test Suite JSON | Profile objects, Jupyter viz, WhyLabs platform |
| **Best monitoring mode** | Batch — daily/weekly window comparison | Streaming / high-volume — profile every hour or per-request |
| **Statistical tests** | KS, PSI, Wasserstein, Chi-squared, Jensen-Shannon | Hellinger distance, Kolmogorov-Smirnov (via sketches) |
| **Privacy** | Requires holding raw production data | Profiles contain no raw values — GDPR-friendly |
| **Setup** | Pure Python, zero infra | Pure Python; optional WhyLabs cloud for dashboard |
| **Best for** | Batch ML systems, rich offline drift analysis, pipeline gates | Streaming ML, high-volume APIs, privacy-constrained environments |

**Choosing between them:**
- If your serving volume fits in memory (~100k rows/day) and you want rich, detailed drift reports with visual HTML output for stakeholders → **Evidently**.
- If you're processing millions of events/day, running in a streaming pipeline, or can't store raw inference data for privacy/compliance reasons → **whylogs**.
- For the capstone (batch, moderate volume, needs a pipeline gate) → **Evidently Test Suite**.

---

## 7.12 Lab Walkthrough (Session 7)

1. Generate 200 "clean" inference records by calling the Part 1 BentoML service with realistic inputs and writing them to `inference_log.jsonl`.
2. Run `drift_report.py` against the clean log — confirm all columns pass, inspect the HTML report, understand what PSI and KS values look like for non-drifted data.
3. Run `inject_drift.py` to generate a drifted log with shifted `tenure_days` and `monthly_spend` distributions.
4. Rerun `drift_report.py` against the drifted log — visually compare the distributions in the HTML report.
5. Run `drift_test_suite.py` — confirm it **fails** on `TestColumnDrift` for the drifted columns; read the structured JSON output to understand the failure message format.
6. Run `trigger_retrain.py` — confirm it prints "Drift threshold exceeded" and invokes `dvc repro`.
7. Profile the training data with whylogs, profile the drifted serving window, and compare profiles using the `NotebookProfileVisualizer` — observe which columns show the largest Hellinger distance.
8. Restore the clean log and confirm `trigger_retrain.py` now prints "No significant drift detected."

---

## 7.13 Trade-off Discussion — Recap

**Evidently AI vs. whylogs/WhyLabs — when do you need full reference-data comparison vs. lightweight streaming profiles?**
Evidently is the better choice when you can afford to hold both datasets in memory and want rich, statistically rigorous, visual reports — it's the monitoring tool for data scientists who need to *explain* drift to stakeholders. whylogs is the better choice when volume, latency, or privacy constraints prevent holding raw data — a 10M-row production serving stream can't be held in a Pandas DataFrame, but its statistical signature can be cheaply profiled in a few KB and compared against a reference sketch. Most production teams end up using both: whylogs for always-on, low-overhead streaming monitoring, and Evidently for scheduled deep-dive batch analyses.

---

## 7.14 Common Pitfalls

- **Using too small a current window.** Statistical drift tests have low power on small samples — a 50-row window will produce noisy, unreliable drift signals. Use at minimum 300–500 rows; more is better.
- **Drifting on the reference dataset itself.** If your reference is a biased sample of training data (e.g., only Q1 customers), any natural seasonality will appear as "drift" even when nothing is wrong. Use a broad, representative reference sample.
- **Setting thresholds too tight.** A PSI threshold of 0.05 on a noisy feature will fire constantly. Calibrate thresholds against known-good periods first — understand what PSI values look like in normal, non-drifted operation before tuning alert levels.
- **Not logging the prediction (output) in addition to inputs.** Input feature drift is leading-indicator monitoring. Prediction drift (the model's output distribution) is often the fastest signal that something is wrong. Log both.
- **Treating drift as synonymous with "model is wrong."** Drift is a *signal*, not a diagnosis. A drifted feature might mean the model needs retraining, or it might mean an upstream ETL pipeline changed behavior harmlessly. Always pair drift alerts with a human-readable description of what changed and by how much.
- **Retraining on the drifted data without re-validating.** When `trigger_retrain.py` fires `dvc repro`, the pipeline starts at Stage 0 — the Pandera/GE validation gate. If the drifted data is also *invalid* (not just statistically different), the pipeline halts at validation, not at training. This is the intended behavior.

---

## 🏗️ CAPSTONE PROJECT — Part 2: The Monitoring & Retrain Loop

---

### Capstone Part 2 — Deliverable 1: Evidently Test Suite Monitoring Layer

The `drift_test_suite.py` from Section 7.6 is adapted to read directly from the BentoML inference log produced by Part 1's service. Ensure these specific tests are present:

```python
test_suite = TestSuite(tests=[
    TestShareOfDriftedColumns(lt=0.20),
    TestColumnDrift(column_name="monthly_spend",
                    stattest="psi", threshold=0.2),
    TestColumnDrift(column_name="rolling_30d_spend",
                    stattest="psi", threshold=0.2),
    TestColumnDrift(column_name="tenure_days",
                    stattest="ks",  threshold=0.05),
    TestShareOfMissingValues(lt=0.02),
])
```

---

### Capstone Part 2 — Deliverable 2: Drift-Triggered DVC Retraining

The `trigger_retrain.py` script is the wiring between monitoring and the DVC pipeline. **Verify this specific end-to-end sequence works:**

```bash
# Step 1: Confirm clean inference log passes monitoring
python src/monitoring/trigger_retrain.py
# → ✅ No significant drift detected — current champion retained

# Step 2: Inject drift into the log
python scripts/inject_drift.py

# Step 3: Confirm drifted log triggers retraining
python src/monitoring/trigger_retrain.py
# → ⚠️  Drift threshold exceeded — triggering retraining pipeline
# → [DVC runs: validate → featurize → train → evaluate]
# → ✅ Retraining complete — new model version queued for registration
```

---

### Capstone Part 2 — Deliverable 3: Challenger Registration & Comparison

After `dvc repro` completes, the retrained model must be registered to the MLflow Registry and compared programmatically against the current champion. Add this block at the end of `trigger_retrain.py`:

```python
# src/monitoring/trigger_retrain.py (extended)
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from sklearn.metrics import roc_auc_score
import pandas as pd

def register_and_compare_challenger(model_name: str,
                                     holdout_path: str,
                                     improvement_threshold: float = 0.005):
    client = MlflowClient()

    # Fetch the latest run from the capstone experiment (just completed by DVC)
    runs = client.search_runs(
        experiment_ids=client.get_experiment_by_name(
            "capstone-churn-prediction"
        ).experiment_id,
        order_by=["start_time DESC"],
        max_results=1,
    )
    latest_run_id = runs[0].info.run_id

    # Register as a new challenger version
    new_version = mlflow.register_model(
        f"runs:/{latest_run_id}/model", model_name
    ).version
    client.transition_model_version_stage(model_name, new_version, "Staging")
    client.set_registered_model_alias(model_name, "challenger", new_version)
    print(f"✅ Retrained model registered as v{new_version} (challenger)")

    # Load holdout data for comparison
    holdout_df = pd.read_parquet(holdout_path)
    X_holdout  = holdout_df.drop(
        ["customer_id", "event_timestamp", "churn_label"], axis=1
    )
    y_holdout  = holdout_df["churn_label"]

    champion   = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    challenger = mlflow.pyfunc.load_model(f"models:/{model_name}@challenger")

    champ_auc = roc_auc_score(y_holdout,
                               champion.predict(X_holdout))
    chal_auc  = roc_auc_score(y_holdout,
                               challenger.predict(X_holdout))

    print(f"Champion  AUC: {champ_auc:.4f}")
    print(f"Challenger AUC: {chal_auc:.4f}")

    if chal_auc > champ_auc + improvement_threshold:
        client.set_registered_model_alias(model_name, "champion", new_version)
        client.transition_model_version_stage(model_name, new_version, "Production")
        print(f"🏆 Challenger v{new_version} promoted to champion!")
    else:
        print("⛔ Challenger did not exceed improvement threshold. Champion retained.")
```

---

### Capstone Part 2 — Full Integrated Loop (End-to-End Script)

```python
# run_monitoring_loop.py
"""
Capstone Part 2: Full monitoring and optional retraining loop.
Run this on a schedule (or manually) to check for drift and promote
a new champion if retraining improves performance.
No Dockerfiles. No Kubernetes. Pure Python/CLI.
"""

import sys
from src.monitoring.trigger_retrain import check_drift_and_maybe_retrain
from src.monitoring.trigger_retrain import register_and_compare_challenger

MODEL_NAME     = "capstone-churn-rf"
REFERENCE_PATH = "data/training_features.parquet"
CURRENT_LOG    = "data/inference_log.jsonl"
HOLDOUT_PATH   = "data/holdout.parquet"

print("=" * 60)
print("CAPSTONE MONITORING LOOP — Starting")
print("=" * 60)

retrain_triggered = check_drift_and_maybe_retrain(
    reference_path=REFERENCE_PATH,
    current_log_path=CURRENT_LOG,
    drift_threshold=0.20,
)

if retrain_triggered:
    register_and_compare_challenger(
        model_name=MODEL_NAME,
        holdout_path=HOLDOUT_PATH,
        improvement_threshold=0.005,
    )
else:
    print("No action required — monitoring loop complete.")

print("=" * 60)
print("CAPSTONE MONITORING LOOP — Done")
print("=" * 60)
```

---

### Capstone Part 2 — Verification Checklist

Before calling Part 2 complete, verify all of the following:

```
[ ] Evidently Test Suite runs against the inference log and produces HTML + JSON output
[ ] Clean inference log → Test Suite PASSES → no retraining triggered
[ ] Drifted inference log → Test Suite FAILS → dvc repro fires automatically
[ ] After dvc repro: new MLflow run appears in the capstone experiment
[ ] New run is registered to the Registry as "challenger" at Staging
[ ] Champion vs. challenger comparison runs and prints both AUC values
[ ] If challenger is better: alias "champion" moves to the new version
[ ] If challenger is not better: alias "champion" is unchanged
[ ] run_monitoring_loop.py executes all steps above from a single entry point
```

**Part 2 is complete** when every item above is checked and the full loop can be demonstrated in a single `python run_monitoring_loop.py` invocation.

---

## Cross-Session Integration Map (Sessions 1–7)

```
Session 1 (Pandera/GE)
      │  validate_or_halt() → Stage 0 gate in DVC
      │  ChurnInputSchema → reused in BentoML service.py
      ▼
Session 2 (DVC)
      │  dvc.yaml 4-stage pipeline (validate → featurize → train → evaluate)
      │  dvc exp run sweeps → best params → dvc.lock commit
      ▼
Session 4 (Feast)
      │  get_historical_features() → PIT-correct training data
      │  get_online_features() → serving path (same FeatureView)
      ▼
Session 3 (MLflow Tracking)
      │  every training run logged with Feast tags + DVC commit hash
      │  MlflowClient.search_runs() → best run_id identified
      ▼
Session 5 (MLflow Registry)
      │  best run registered → signed → versioned → aliased champion/challenger
      │  register_and_compare_challenger() → alias-based promotion/rollback
      ▼
Session 6 (BentoML)
      │  loads model by alias "champion" (no hardcoded run ID)
      │  Pandera ChurnInputSchema enforced at serving boundary
      │  inference_log.jsonl written for every prediction
      ▼
Session 7 (Evidently + whylogs)
      │  Evidently Test Suite: reference training data vs. inference_log window
      │  Pass → no action / Fail → dvc repro → retrain → register challenger
      │  whylogs profiles for streaming/privacy-safe alternative
      ▼
Session 8 → Full capstone integration + rollback demo + architecture presentation
```
