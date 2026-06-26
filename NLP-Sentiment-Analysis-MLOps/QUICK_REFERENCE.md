# 🚀 Quick Reference - Multi-Language Text Classifier

## One-Minute Setup

```bash
# Navigate to project
cd c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps

# Run the demo
make demo

# View results in MLflow
make mlflow-ui
```

## Core Components Cheat Sheet

### 1️⃣ Multi-Language Classification
```python
from src.models.multilanguage_classifier import MultiLanguageClassifier

classifier = MultiLanguageClassifier(device="cpu")

# Single text
result = classifier.classify("I love this!")
print(f"{result.language_flag} {result.label} ({result.confidence:.1%})")
# 🇺🇸 positive (98.5%)

# Batch
results = classifier.classify_batch(["Text 1", "Text 2"])
```

**Supported Languages:** 🇺🇸 🇪🇸 🇫🇷 🇩🇪 🇮🇹 🇵🇹 🇷🇺 🇨🇳 🇯🇵 🇰🇷

---

### 2️⃣ Data Quality Validation
```python
from src.utils.data_quality_validator import DataQualityValidator

validator = DataQualityValidator()

# Full validation
result = validator.run_full_validation(texts, labels)

# Print report
validator.print_validation_report()
```

**Checks:** Nulls, Length, Encoding, Duplicates, Language, Labels, Special Chars

---

### 3️⃣ Hyperparameter Optimization
```python
from src.models.hyperparameter_optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(n_trials=50)
optimizer.create_study(sampler='tpe')

def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-5)
    return validation_score

best_params = optimizer.optimize(objective)
```

**Samplers:** `tpe` (Bayesian), `random`, `grid`

---

### 4️⃣ Model Registry
```python
from src.models.mlflow_registry import ModelRegistry

registry = ModelRegistry()

# Register
v = registry.register_model("my-model", "runs:/abc/model")

# Promote
registry.transition_model("my-model", v, "Production")

# Load
model = registry.load_model("my-model", stage="Production")
```

**Stages:** Development → Staging → Production → Archived

---

### 5️⃣ Canary Deployment
```python
from src.models.mlflow_registry import CanaryDeploymentManager

deploy = CanaryDeploymentManager(registry)

# Create plan
plan = deploy.create_canary_deployment("model", "v2")

# Progress stages
deploy.promote_canary_stage("model", "1")

# Rollback if needed
deploy.rollback_deployment("model", "1")
```

**Rollout:** 5% (5m) → 25% (10m) → 50% (20m) → 100%

---

### 6️⃣ Full Pipeline
```python
from src.pipeline.orchestrator import TrainingPipeline

pipeline = TrainingPipeline()

results = pipeline.run_full_pipeline(
    train_texts=texts,
    train_labels=labels,
    validate=True,
    optimize=True,
    register_model=True,
    model_name="my-classifier"
)

pipeline.save_results(results)
```

**Stages:** Validate → Analyze → Optimize → Register → Deploy

---

### 7️⃣ Vibecoding Logger
```python
from src.utils.vibecoding_logger import setup_vibecoding_logger

logger = setup_vibecoding_logger(__name__)

logger.info("Training started")     # ✨ INFO
logger.warning("Low accuracy")      # ⚠️ WARNING
logger.error("Training failed")     # ❌ ERROR

# Visualizations
visualizer = PipelineVisualizer()
visualizer.print_training_start()
visualizer.print_pipeline_stage("Training", 1, 5)
```

---

## Common Make Commands

```bash
make demo              # 🎬 Run everything (START HERE!)
make pipeline          # 🚀 Full training pipeline
make validate          # ✅ Data quality checks
make optimize          # ⚡ Hyperparameter tuning
make mlflow-ui         # 📊 View experiments
make serve             # 🌐 Start API (port 8001)
make clean             # 🗑️ Clean cache
```

---

## Configuration (config.yaml)

```yaml
# Model
model: distilbert-base-uncased

# Training
epochs: 1
batch_size: 16
learning_rate: 2.0e-5
max_length: 256

# Device
device: cpu
use_cuda: false

# Optuna
optuna:
  n_trials: 50
  sampler: tpe

# MLflow
mlflow:
  experiment_name: sentiment-analysis
  tracking_uri: http://localhost:5000
```

---

## File Structure

```
src/
├── models/
│   ├── multilanguage_classifier.py    ← Classifications
│   ├── hyperparameter_optimizer.py    ← Tuning
│   └── mlflow_registry.py             ← Registry & Deploy
├── pipeline/
│   └── orchestrator.py                ← Full Pipeline
├── utils/
│   ├── vibecoding_logger.py           ← Visualizations
│   └── data_quality_validator.py      ← Validation

scripts/
└── demo_pipeline.py                   ← RUN THIS!
```

---

## Output Example

```
================================================================
🔍 DATA QUALITY VALIDATION DEMO
================================================================
✅ null_values: PASSED
✅ text_length: PASSED
✅ encoding: PASSED
✅ duplicate_texts: PASSED
✅ special_characters: PASSED
✅ label_distribution: PASSED

==============================================================
🌍 MULTI-LANGUAGE CLASSIFICATION DEMO
==============================================================
🇺🇸 English      | I love this product!         | positive (98.5%)
🇪🇸 Spanish      | ¡Me encanta este producto!   | positive (97.2%)
🇫🇷 French       | J'aime ce produit!           | positive (96.8%)

==============================================================
✅ Pipeline Complete
==============================================================
Duration: 45.3s
Run ID: 20240115_143022
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model too slow | Use `distilbert-base-uncased` |
| Out of memory | Reduce `batch_size` to 8 |
| Port 8001 taken | Change `port:` in config.yaml |
| Language not detected | Manually set `language='en'` |
| Missing dependencies | Run `make install` |

---

## Performance

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| DistilBERT | ⚡⚡⚡ | ⭐⭐⭐⭐ | 268MB |
| BERT | ⚡⚡ | ⭐⭐⭐⭐⭐ | 440MB |
| XLM-RoBERTa | ⚡ | ⭐⭐⭐⭐ | 1.4GB |

---

## 🎯 Recommended Workflow

1. **Start:**
   ```bash
   make demo
   ```

2. **View Results:**
   ```bash
   make mlflow-ui
   ```

3. **Load Data:**
   ```python
   texts = load_your_data()
   labels = load_your_labels()
   ```

4. **Run Pipeline:**
   ```python
   pipeline = TrainingPipeline()
   results = pipeline.run_full_pipeline(texts, labels)
   ```

5. **Check Models:**
   ```bash
   make mlflow-ui
   ```

6. **Deploy:**
   ```python
   registry.transition_model(name, version, "Production")
   ```

---

## 📚 Documentation Links

- **Full Docs:** [SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md)
- **Implementation:** [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **Features:** [MULTILANGLANGUAGE_CLASSIFIER_README.md](MULTILANGLANGUAGE_CLASSIFIER_README.md)

---

## ✅ Status

✅ Production Ready  
✅ All Components Working  
✅ Demo Available  
✅ Documentation Complete  

**Start with:** `make demo`
