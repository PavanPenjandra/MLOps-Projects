# 🌍 Multi-Language Text Classifier - MLOps System

Enterprise-grade text classification system with support for **10+ languages**, **automated hyperparameter tuning**, **MLflow model registry**, **canary deployments**, and **vibecoding-styled visualizations**.

## ✨ Features

### 🎯 Core Capabilities
- ✅ **Multi-Language Support**: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean
- ✅ **Automatic Language Detection**: langdetect integration for automatic language identification
- ✅ **Multi-Model Architecture**: Language-specific optimized models
- ✅ **Batch Processing**: Efficient processing of multiple texts
- ✅ **JSON Export**: Results export for downstream processing

### ⚙️ MLOps Features
- ✅ **Data Quality Validation**: Comprehensive checks (nulls, length, encoding, duplicates, language distribution)
- ✅ **Hyperparameter Optimization**: Optuna-based Bayesian search with TPE sampler
- ✅ **MLflow Integration**: Experiment tracking and model registry
- ✅ **Model Versioning**: Stage management (Dev → Staging → Production)
- ✅ **Canary Deployments**: Progressive rollout with automatic rollback
- ✅ **Monitoring Thresholds**: Production metrics tracking

### 🎨 Developer Experience
- ✅ **Vibecoding Logger**: Colored logs with emoji indicators and ASCII art
- ✅ **Loading Animations**: Visual progress indicators (dots, bar, line, arrow)
- ✅ **Rainbow Progress Bars**: Colorful progress tracking
- ✅ **Pipeline Visualizations**: ASCII diagrams for training pipelines
- ✅ **Heatmap Displays**: Terminal-based confusion matrix visualization
- ✅ **Language Flags**: Emoji flags for each language (🇺🇸 🇪🇸 🇫🇷 🇩🇪 🇮🇹 🇵🇹 🇷🇺 🇨🇳 🇯🇵 🇰🇷)

## 🚀 Quick Start

### 1. Setup
```bash
# Clone/navigate to project
cd c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps

# Install dependencies (already done)
make install

# Setup local environment
make local-setup
```

### 2. Run Demo
```bash
make demo
```

This runs the complete system demo showing:
- ✅ Data quality validation
- ✅ Multi-language classification
- ✅ Hyperparameter optimization
- ✅ Visual components
- ✅ Full training pipeline

### 3. View MLflow
```bash
make mlflow-ui
```
Visit: http://localhost:5000

### 4. Run Full Pipeline
```bash
make pipeline
```

## 📁 Project Structure

```
src/
├── models/
│   ├── multilanguage_classifier.py      # Multi-language classification
│   ├── hyperparameter_optimizer.py      # Optuna-based tuning
│   ├── mlflow_registry.py              # Model registry & canary deploy
│   ├── train.py                        # Training logic
│   └── evaluate.py                     # Model evaluation
├── pipeline/
│   ├── orchestrator.py                 # End-to-end pipeline
│   └── __init__.py
├── utils/
│   ├── vibecoding_logger.py            # Colored logging & visualization
│   ├── data_quality_validator.py       # Data validation checks
│   ├── utils.py                        # Helper functions
│   └── validate_data.py                # Data validation
├── serving/
│   └── inference.py                    # Model inference
├── features/
│   └── preprocess.py                   # Text preprocessing
├── data/
│   └── load_data.py                    # Data loading
└── app/
    └── main.py                         # FastAPI application

scripts/
├── demo_pipeline.py                    # Complete system demo
├── train_pipeline.py                   # Training script
└── example_inference.py                # Inference examples

notebooks/
├── exploration.ipynb                   # Data exploration
├── model_comparison.ipynb              # Model evaluation
└── deployment_analysis.ipynb           # Deployment metrics

docs/
├── SYSTEM_DOCUMENTATION.md             # Complete API reference
├── IMPLEMENTATION_GUIDE.md             # Implementation guide
└── (other docs)
```

## 🎯 Usage Examples

### Example 1: Simple Classification
```python
from src.models.multilanguage_classifier import MultiLanguageClassifier

classifier = MultiLanguageClassifier(device="cpu")
result = classifier.classify("I love this product!")

print(f"{result.language_flag} {result.label}: {result.confidence:.2%}")
# Output: 🇺🇸 positive: 98.47%
```

### Example 2: Batch Processing
```python
texts = [
    "I love this!",        # English
    "¡Me encanta!",       # Spanish
    "J'aime ça!",         # French
]

results = classifier.classify_batch(texts)

for result in results:
    print(f"{result.language_flag} {result.text:20s} → {result.label}")
```

### Example 3: Data Quality
```python
from src.utils.data_quality_validator import DataQualityValidator

validator = DataQualityValidator()

texts = ["Text 1", "Text 2", ""]
labels = ["positive", "negative", "positive"]

validation = validator.run_full_validation(texts, labels)
validator.print_validation_report()
```

### Example 4: Full Pipeline
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

pipeline.save_results(results)
```

## 🔧 Configuration

### config.yaml - Key Settings

```yaml
# Model
model: distilbert-base-uncased
models:
  multilingual: xlm-roberta-base
  english: distilbert-base-uncased

# Training
epochs: 1
batch_size: 16
learning_rate: 2.0e-5
max_length: 256

# Device
device: cpu
use_cuda: false

# Local Mode
local_mode:
  enabled: true
  use_sample_data: true
  max_samples: 500

# MLflow
mlflow:
  experiment_name: sentiment-analysis
  tracking_uri: http://localhost:5000

# Optuna
optuna:
  n_trials: 50
  sampler: tpe
  direction: maximize
```

## 📊 Available Commands

```bash
# Core Commands
make install          # Install dependencies
make test             # Run tests
make train            # Train model
make serve            # Start API server (port 8001)
make lint             # Lint code
make format           # Format with black

# MLOps Pipeline Commands
make demo             # Run complete system demo
make pipeline         # Run full training pipeline
make validate         # Run data quality validation
make optimize         # Run hyperparameter optimization
make mlflow-ui        # Start MLflow UI

# Utilities
make local-setup      # Setup for local development
make clean            # Clean up cache files
```

## 📈 System Components

### 1. **MultiLanguageClassifier** (`src/models/multilanguage_classifier.py`)
Automatic language detection and multi-language text classification.

**Supported Languages:**
- 🇺🇸 English
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇮🇹 Italian
- 🇵🇹 Portuguese
- 🇷🇺 Russian
- 🇨🇳 Chinese
- 🇯🇵 Japanese
- 🇰🇷 Korean

### 2. **HyperparameterOptimizer** (`src/models/hyperparameter_optimizer.py`)
Optuna-based automated hyperparameter tuning with Bayesian optimization.

**Features:**
- TPE (Tree-structured Parzen Estimator) sampler
- Random and Grid search alternatives
- Parameter importance visualization
- Study persistence (save/load)
- Preset search spaces

### 3. **ModelRegistry** (`src/models/mlflow_registry.py`)
MLflow Model Registry integration for production lifecycle management.

**Stages:**
- 🏗️ Development
- 🧪 Staging
- 🚀 Production
- 📦 Archived

### 4. **CanaryDeploymentManager** (`src/models/mlflow_registry.py`)
Progressive rollout strategy with automatic rollback.

**Default Rollout:**
- Stage 1: 5% traffic (5 min)
- Stage 2: 25% traffic (10 min)
- Stage 3: 50% traffic (20 min)
- Stage 4: 100% traffic

### 5. **DataQualityValidator** (`src/utils/data_quality_validator.py`)
Comprehensive data validation and quality checks.

**Checks:**
- Null/empty values
- Text length constraints
- UTF-8 encoding
- Duplicate detection
- Language distribution
- Label balance
- Special characters

### 6. **VibecodeLogger** (`src/utils/vibecoding_logger.py`)
Vibecoding-styled logging with colors, emojis, ASCII art, and animations.

**Features:**
- ANSI color codes
- Emoji indicators
- Loading animations
- Rainbow progress bars
- ASCII pipeline diagrams
- Confusion matrix heatmaps

### 7. **TrainingPipeline** (`src/pipeline/orchestrator.py`)
End-to-end orchestration of all components.

**Stages:**
1. Data Validation
2. Multi-Language Analysis
3. Hyperparameter Optimization
4. Model Registration
5. Canary Deployment Setup

## 🎓 Learning Path

### Beginner
```bash
# Run demo to see everything working
make demo

# Check MLflow UI
make mlflow-ui
```

### Intermediate
```python
# Use individual components
classifier = MultiLanguageClassifier()
result = classifier.classify("Your text")

validator = DataQualityValidator()
validator.run_full_validation(texts, labels)
```

### Advanced
```python
# Customize training pipeline
pipeline = TrainingPipeline(config_path="custom_config.yaml")
results = pipeline.run_full_pipeline(...)
```

### Expert
```python
# Implement custom objective functions
def my_objective(trial):
    # Custom hyperparameter tuning
    return score

optimizer.optimize(my_objective, custom_search_space)
```

## 📊 Monitoring & Metrics

### MLflow UI
```bash
make mlflow-ui
# Visit: http://localhost:5000
```

**Tracked Metrics:**
- Training loss/accuracy
- Validation metrics
- Per-language performance
- Hyperparameter values
- Inference time

### Data Quality
```python
validator.print_validation_report()
```

### Model Performance
```python
registry.compare_models("model-name", "v1", "v2")
```

## 🐛 Troubleshooting

### Model Download Issues
```python
import os
os.environ['HF_HOME'] = './models_cache'
```

### Out of Memory
```yaml
# config.yaml
batch_size: 8
max_length: 128
device: cpu
```

### Language Detection Failing
```python
# Manually specify language
result = classifier.classify(text, language='en')
```

### Port Binding Error
```yaml
# config.yaml
port: 8001  # Change port if needed
```

## 📚 Documentation

- **[SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md)** - Complete API reference and system overview
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Step-by-step implementation guide
- **[LOCAL_MODE_SETUP.md](LOCAL_MODE_SETUP.md)** - Local development setup
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture

## 🔗 Integration

### With FastAPI
```python
from src.models.multilanguage_classifier import MultiLanguageClassifier

app = FastAPI()
classifier = MultiLanguageClassifier()

@app.post("/predict")
async def predict(text: str):
    return classifier.classify(text)
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
dvc add data/
dvc add models/
```

## 🎯 Next Steps

1. ✅ Run `make demo` to see the system in action
2. ✅ Check `make mlflow-ui` to view experiments
3. ✅ Load your own data
4. ✅ Customize `config.yaml`
5. ✅ Run `make pipeline` with your data
6. ✅ Deploy with canary strategy
7. ✅ Monitor performance

## 📞 Support

**Frequently Asked Questions:**

**Q: How do I add a new language?**
A: Add to `LANGUAGE_FLAGS` and `LANGUAGE_MODELS` in `multilanguage_classifier.py`

**Q: How do I use a different model?**
A: Set `model` in `config.yaml` to any HuggingFace model ID

**Q: How do I enable GPU?**
A: Set `use_cuda: true` in `config.yaml`

**Q: How do I track custom metrics?**
A: Use `mlflow.log_metric()` in your training code

## 🎉 Success!

You now have a production-ready multi-language text classifier system with:
- ✅ Multi-language support
- ✅ Automated hyperparameter tuning
- ✅ MLflow experiment tracking
- ✅ Model registry management
- ✅ Canary deployment strategy
- ✅ Data quality validation
- ✅ Vibecoding visualizations

**Start with:** `make demo`

---

**System Version:** 1.0  
**Last Updated:** 2024  
**Status:** ✅ Production Ready
