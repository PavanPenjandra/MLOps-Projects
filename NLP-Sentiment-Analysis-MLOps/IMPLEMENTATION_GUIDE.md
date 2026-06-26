# Multi-Language Text Classifier - Implementation Guide

## 🎯 Quick Start

### 1. Verify Installation

All required packages should be installed:
```bash
pip list | grep -E "transformers|torch|optuna|mlflow|langdetect"
```

### 2. Run Demo

```bash
cd c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps
python scripts/demo_pipeline.py
```

Expected output:
```
##########################################################################
# MULTILINGUAL TEXT CLASSIFIER - COMPLETE SYSTEM DEMO
##########################################################################

================================================================
🔍 DATA QUALITY VALIDATION DEMO
================================================================
...
✅ ALL DEMOS COMPLETED SUCCESSFULLY!
```

### 3. Access MLflow UI

```bash
mlflow ui
```

Visit: http://localhost:5000

## 📋 System Components Breakdown

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Pipeline                          │
│                   orchestrator.py                           │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐        ┌──────────┐      ┌──────────┐
   │ Data    │        │Language  │      │MLflow    │
   │Quality  │        │Classify  │      │Registry  │
   │Validator│        │          │      │          │
   └─────────┘        └──────────┘      └──────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌──────────┐       ┌──────────┐      ┌──────────┐
   │Optuna    │       │Vibecoding│      │Canary    │
   │Optimizer │       │Logger    │      │Deploy    │
   └──────────┘       └──────────┘      └──────────┘
```

### Module Dependency Graph

```
orchestrator.py
  ├─ multilanguage_classifier.py
  │   ├─ torch
  │   ├─ transformers
  │   └─ langdetect
  ├─ hyperparameter_optimizer.py
  │   ├─ optuna
  │   └─ plotly
  ├─ mlflow_registry.py
  │   └─ mlflow
  ├─ vibecoding_logger.py (no external deps)
  └─ data_quality_validator.py
      ├─ langdetect (optional)
      └─ great_expectations (optional)
```

## 🔧 Configuration Guide

### config.yaml - Key Settings

**For Local Development:**
```yaml
local_mode:
  enabled: true
  use_sample_data: true
  max_samples: 500

device: cpu
use_cuda: false
batch_size: 16
epochs: 1
```

**For Production:**
```yaml
local_mode:
  enabled: false

device: cuda
use_cuda: true
batch_size: 32
epochs: 3
learning_rate: 2.0e-5
```

**For Different Languages:**
```yaml
model: xlm-roberta-base  # Supports more languages
languages: [en, es, fr, de, it, pt, ru, zh, ja, ko]
```

## 📊 Data Format Requirements

### Input Data

**Single Prediction:**
```python
{
    "text": "I love this product!",
    "language": "en"  # optional, auto-detected
}
```

**Batch Processing:**
```python
{
    "texts": [
        "I love this product!",
        "¡Me encanta este producto!",
        "J'aime ce produit!"
    ]
}
```

**Training Data:**
```python
texts = [
    "I absolutely love this product!",
    "This is terrible and I hate it.",
    "The quality is decent but could be better.",
]

labels = [
    "positive",
    "negative",
    "neutral"
]
```

### Output Format

```python
{
    'text': 'I love this product!',
    'language': 'en',
    'language_flag': '🇺🇸',
    'label': 'positive',
    'confidence': 0.9847,
    'probabilities': {
        'positive': 0.9847,
        'negative': 0.0102,
        'neutral': 0.0051
    },
    'model_used': 'distilbert-base-uncased'
}
```

## 🚀 Pipeline Stages Explained

### Stage 1: Data Validation
```
Input: Raw training texts and labels
Process:
  ✓ Check null/empty values
  ✓ Validate text length
  ✓ Verify UTF-8 encoding
  ✓ Detect duplicates
  ✓ Analyze language distribution
  ✓ Check label balance
Output: Validation report, pass/fail status
```

### Stage 2: Multi-Language Analysis
```
Input: Validated texts
Process:
  ✓ Auto-detect language for each text
  ✓ Map to language-specific model
  ✓ Classify with appropriate model
  ✓ Compute language statistics
Output: Predictions with language flags, distribution stats
```

### Stage 3: Hyperparameter Optimization
```
Input: Training configuration
Process:
  ✓ Create Optuna study with TPE sampler
  ✓ Generate parameter combinations
  ✓ Train model with each combination
  ✓ Evaluate and track results
  ✓ Identify best parameters
Output: Best hyperparameters, optimization history
```

### Stage 4: Model Registration
```
Input: Trained model artifacts
Process:
  ✓ Register in MLflow Model Registry
  ✓ Tag with metadata
  ✓ Transition to Development stage
  ✓ Create model version
Output: Model version, registry entry
```

### Stage 5: Canary Deployment
```
Input: Registered model
Process:
  ✓ Create deployment plan (5% → 25% → 50% → 100%)
  ✓ Set traffic allocation
  ✓ Define rollout schedule
  ✓ Setup monitoring
Output: Deployment plan, promotion schedule
```

## 🎯 Usage Examples

### Example 1: Simple Classification

```python
from src.models.multilanguage_classifier import MultiLanguageClassifier

# Initialize classifier
classifier = MultiLanguageClassifier(device="cpu")

# Classify single text
result = classifier.classify("I love this product!")
print(f"{result.language_flag} {result.label}: {result.confidence:.2%}")
# Output: 🇺🇸 positive: 98.47%
```

### Example 2: Batch Processing

```python
texts = [
    "I love this!",
    "¡Me encanta!",
    "J'aime ça!",
]

results = classifier.classify_batch(texts)

for result in results:
    print(f"{result.language_flag} {result.text[:20]:20s} → {result.label}")
```

### Example 3: Data Quality Check

```python
from src.utils.data_quality_validator import DataQualityValidator

validator = DataQualityValidator()

texts = ["Text 1", "Text 2", ""]
labels = ["positive", "negative", "positive"]

# Run validation
validation = validator.run_full_validation(texts, labels)

if validation['failed_checks'] == 0:
    print("✅ Data is valid!")
else:
    print(f"⚠️ Found {validation['failed_checks']} issues")
```

### Example 4: Hyperparameter Optimization

```python
from src.models.hyperparameter_optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(n_trials=20)
optimizer.create_study(sampler='tpe', direction='maximize')

def objective(trial):
    # Your training logic here
    return validation_accuracy

best_params = optimizer.optimize(objective)
print(f"Best learning rate: {best_params['learning_rate']}")
print(f"Best batch size: {best_params['batch_size']}")
```

### Example 5: Model Registry

```python
from src.models.mlflow_registry import ModelRegistry

registry = ModelRegistry()

# Register model
version = registry.register_model(
    model_name="sentiment-classifier",
    model_uri="runs:/abc123/model",
    description="Multilingual sentiment classifier"
)

# Promote to production
registry.transition_model("sentiment-classifier", version, "Production")

# Load production model
model = registry.load_model("sentiment-classifier", stage="Production")
```

### Example 6: Complete Pipeline

```python
from src.pipeline.orchestrator import TrainingPipeline

pipeline = TrainingPipeline()

results = pipeline.run_full_pipeline(
    train_texts=training_texts,
    train_labels=training_labels,
    validate=True,
    optimize=True,
    register_model=True,
    model_name="my-classifier"
)

print(f"Pipeline completed in {results['duration_seconds']:.1f}s")
print(f"Run ID: {results['run_id']}")
```

## 📈 Monitoring

### MLflow Tracking

**View experiments:**
```bash
mlflow ui --port 5000
```

**Logged metrics:**
- Training loss/accuracy
- Validation metrics
- Per-language performance
- Hyperparameter values
- Inference time

### Data Quality Tracking

**Check validation reports:**
```python
validator = DataQualityValidator()
validation = validator.run_full_validation(texts, labels)
validator.print_validation_report()
```

### Model Performance

**Compare model versions:**
```python
registry = ModelRegistry()
comparison = registry.compare_models(
    "sentiment-classifier",
    version1="1",
    version2="2"
)
```

## ⚙️ Performance Tuning

### For Speed:
```yaml
model: distilbert-base-uncased  # Smaller model
batch_size: 32                   # Larger batches
max_length: 128                  # Shorter sequences
epochs: 1                        # Fewer epochs
device: cuda                     # GPU acceleration
```

### For Accuracy:
```yaml
model: xlm-roberta-large         # Larger model
batch_size: 8                    # Smaller batches
max_length: 512                  # Full sequences
epochs: 3                        # More epochs
learning_rate: 2.0e-5            # Lower LR for stability
warmup_steps: 500                # Gradual warmup
```

### For Multi-Language:
```yaml
model: xlm-roberta-base          # Multilingual
languages: [en, es, fr, de, it, pt, ru, zh, ja, ko]
min_samples_per_language: 50     # Data quality check
```

## 🐛 Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# OR set environment variable
import os
os.environ['LOGLEVEL'] = 'DEBUG'
```

### Troubleshoot Common Issues

**Issue: Model downloading slowly**
```python
# Set cache directory
import os
os.environ['HF_HOME'] = './models_cache'
```

**Issue: Out of memory**
```yaml
batch_size: 8  # Reduce batch size
max_length: 128  # Reduce sequence length
device: cpu  # Use CPU instead
```

**Issue: Language detection failing**
```python
# Manually specify language
result = classifier.classify(text, language='en')
```

## 📚 API Documentation Reference

See [SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md) for complete API reference.

## ✅ Testing

### Run Tests
```bash
pytest tests/
```

### Run Demo
```bash
python scripts/demo_pipeline.py
```

### Individual Component Tests

```python
# Test data quality
python -c "from scripts.demo_pipeline import demo_data_quality; demo_data_quality()"

# Test multi-language
python -c "from scripts.demo_pipeline import demo_multilanguage; demo_multilanguage()"

# Test optimization
python -c "from scripts.demo_pipeline import demo_hyperparameter_optimization; demo_hyperparameter_optimization()"
```

## 🔗 Integration Points

### With FastAPI Server
```python
# scripts/inference_server.py
from src.models.multilanguage_classifier import MultiLanguageClassifier

app = FastAPI()
classifier = MultiLanguageClassifier()

@app.post("/predict")
async def predict(text: str):
    result = classifier.classify(text)
    return result
```

### With MLflow
```python
import mlflow

mlflow.start_run()
mlflow.log_metric("accuracy", 0.95)
mlflow.log_params(best_params)
mlflow.end_run()
```

### With DVC
```bash
# Track data versions
dvc add data/training_data.csv

# Track model versions
dvc add models/sentiment_model
```

## 🎓 Learning Path

1. **Beginner**: Run demo_pipeline.py
2. **Intermediate**: Use individual components (classifier, optimizer)
3. **Advanced**: Customize training pipeline for your use case
4. **Expert**: Implement custom objective functions, add new languages

## 📞 Support

**Common Questions:**

Q: How do I add a new language?
A: Add to LANGUAGE_FLAGS and LANGUAGE_MODELS in multilanguage_classifier.py

Q: How do I use a custom model?
A: Set `model` in config.yaml to HuggingFace model ID

Q: How do I track metrics in MLflow?
A: Use mlflow.log_metric() inside your objective function

Q: How do I enable GPU acceleration?
A: Set `use_cuda: true` in config.yaml and install pytorch-gpu

## 🚀 Next Steps

1. ✅ Run demo_pipeline.py
2. ✅ Check MLflow UI (http://localhost:5000)
3. ✅ Load your own data
4. ✅ Customize config.yaml
5. ✅ Run full_pipeline with your data
6. ✅ Deploy with canary strategy
7. ✅ Monitor performance
