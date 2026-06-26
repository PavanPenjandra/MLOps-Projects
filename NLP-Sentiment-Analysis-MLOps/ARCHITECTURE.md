# NLP MLOps Project - Architecture & Implementation Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS Cloud Deployment                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐         ┌──────────────┐      ┌──────────────┐   │
│  │   S3 Bucket  │◄────────│ CloudFormation│─────►│   ECR Repo   │   │
│  │  (Models)    │         │   (Stack)    │      │   (Images)   │   │
│  └──────────────┘         └──────────────┘      └──────────────┘   │
│         │                        │                       │           │
│         ▼                        ▼                       ▼           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              SageMaker Endpoint (ml.m5.large)                │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  Sentiment Analysis Model (BERT-base-uncased)          │  │  │
│  │  │  - Inference Server (FastAPI)                          │  │  │
│  │  │  - Request Handler                                     │  │  │
│  │  │  - Response Formatter                                  │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                           │
└─────────┼───────────────────────────────────────────────────────────┘
          │
    ┌─────▼─────────────┐
    │  REST API Clients │
    │  - Web Apps       │
    │  - Mobile Apps    │
    │  - Services       │
    └───────────────────┘
```

## Component Architecture

### 1. Data Layer (`src/data/`)

**Components**:
- `DataLoader`: Multi-source data loading
  - HuggingFace datasets
  - S3 CSV files
  - Local CSV files
- `create_train_test_split()`: Stratified splitting

**Data Flow**:
```
Raw Data (HF/S3/Local) 
    ▼
DataLoader
    ▼
DataFrame (pandas)
    ▼
Train/Test Split
```

### 2. Feature Engineering Layer (`src/features/`)

**Components**:
- `TextPreprocessor`:
  - Text cleaning (lowercase, remove URLs/emails)
  - Tokenization
  - Stop word removal
  
- `FeatureEngineer`:
  - BERT tokenization
  - Label encoding
  - Text statistics
  
- `DataValidator`:
  - Quality checks
  - Class balance analysis
  - Text validation

**Preprocessing Pipeline**:
```
Raw Text
    ▼
[Clean] → Remove URLs, special chars, normalize
    ▼
[Tokenize] → Word/Subword tokenization
    ▼
[Encode] → BERT token IDs
    ▼
[Pad/Truncate] → Fixed sequence length
    ▼
Ready for Model
```

### 3. Model Training Layer (`src/models/`)

**Components**:
- `SentimentAnalysisTrainer`:
  - Training loop with validation
  - Gradient clipping
  - Learning rate scheduling
  - Model checkpointing
  
- `ModelEvaluator`:
  - Multi-metric evaluation
  - Confusion matrix
  - Performance threshold checking

**Training Loop**:
```
Forward Pass: Input → Model → Logits
    ▼
Calculate Loss
    ▼
Backward Pass: Gradients
    ▼
Optimizer Update: Weights
    ▼
Scheduler Update: Learning Rate
    ▼
Validation: Metrics Computation
    ▼
MLflow Logging
```

### 4. Serving Layer (`src/serving/`)

**Components**:
- `SentimentAnalysisInference`:
  - Model loading
  - Batch inference
  - Confidence scoring
  - Prediction explanation

**Inference Pipeline**:
```
Input Text(s)
    ▼
Tokenization
    ▼
Batch Processing
    ▼
Model Forward Pass
    ▼
Softmax → Probabilities
    ▼
Class Selection & Confidence
    ▼
Output Formatting
```

### 5. API Layer (`src/app/`)

**FastAPI Endpoints**:
```
GET  /health              → Health check
POST /predict             → Single prediction
POST /predict_batch       → Batch predictions (max 100)
POST /explain             → Prediction explanation
GET  /model-info          → Model metadata
```

**Request/Response Lifecycle**:
```
HTTP Request
    ▼
Request Validation (Pydantic)
    ▼
Model Inference
    ▼
Response Formatting
    ▼
HTTP Response (JSON)
```

## Technology Stack Details

### Core ML Stack
| Component | Purpose | Version |
|-----------|---------|---------|
| PyTorch | Deep Learning | 2.0.1 |
| Transformers | Pre-trained Models | 4.31.0 |
| scikit-learn | ML Utilities | 1.3.0 |
| pandas | Data Processing | 2.0.3 |

### Serving Stack
| Component | Purpose | Version |
|-----------|---------|---------|
| FastAPI | Web Framework | 0.100.0 |
| Uvicorn | ASGI Server | 0.23.2 |
| Pydantic | Data Validation | 2.0.2 |

### MLOps Stack
| Component | Purpose | Version |
|-----------|---------|---------|
| MLflow | Experiment Tracking | 2.5.0 |
| boto3 | AWS SDK | 1.28.0 |
| Docker | Containerization | - |
| pytest | Testing | 7.4.0 |

### Cloud Stack
| Service | Purpose | Type |
|---------|---------|------|
| ECR | Container Registry | Managed |
| SageMaker | ML Hosting | Managed |
| S3 | Storage | Object Storage |
| CloudFormation | Infrastructure | IaC |
| CloudWatch | Monitoring | Observability |

## Deployment Architecture

### Local Development
```
Docker Compose
├── MLflow Server (port 5000)
├── FastAPI Server (port 8000)
└── Volume Mounts
    ├── Models
    ├── Data
    └── MLflow DB
```

### AWS Production
```
CloudFormation Stack
├── ECR Repository
│   └── Docker Image
├── SageMaker Model
│   └── Model Artifacts (S3)
├── SageMaker Endpoint
│   ├── ml.m5.large Instance
│   ├── Auto-scaling Policy
│   └── CloudWatch Metrics
└── S3 Bucket
    ├── Model Artifacts
    ├── Training Data
    └── Inference Logs
```

## CI/CD Pipeline

### GitHub Actions Workflow
```
Push to main
    ▼
┌─────────────────────────────┐
│ Test Job                    │
│ - Run pytest                │
│ - Code linting              │
│ - Coverage reports          │
└─────────────────────────────┘
    ▼ (on success)
┌─────────────────────────────┐
│ Build Job                   │
│ - Docker build              │
│ - Tag image                 │
│ - Push to ECR               │
└─────────────────────────────┘
    ▼ (on success)
┌─────────────────────────────┐
│ Deploy Job                  │
│ - Upload model to S3        │
│ - CloudFormation deploy     │
│ - SageMaker endpoint update │
└─────────────────────────────┘
    ▼
Production Ready
```

## Data Flow Diagrams

### Training Pipeline
```
HuggingFace / S3 / Local
    │
    ▼
DataLoader.load_*()
    │
    ▼
pandas DataFrame
    │
    ├─→ Train Set (80%)
    │   ├─→ TextPreprocessor.clean_text()
    │   ├─→ FeatureEngineer.tokenize_batch()
    │   ├─→ PyTorch Dataset
    │   └─→ DataLoader (batches)
    │       │
    │       ▼
    │   SentimentAnalysisTrainer.train()
    │       │
    │       ├─→ Forward Pass
    │       ├─→ Loss Calculation
    │       ├─→ Backward Pass
    │       ├─→ Optimizer Step
    │       ├─→ Scheduler Step
    │       │
    │       └─→ MLflow.log_metrics()
    │
    └─→ Test Set (20%)
        ├─→ Same Preprocessing
        ├─→ PyTorch Dataset
        └─→ DataLoader
            │
            ▼
        SentimentAnalysisTrainer.evaluate()
            │
            └─→ ModelEvaluator.evaluate_predictions()
                │
                └─→ Metrics Report
```

### Inference Pipeline (Production)
```
Client Request (JSON)
    │
    ▼
FastAPI /predict endpoint
    │
    ├─→ Pydantic Validation
    │
    ▼
SentimentAnalysisInference.predict_single()
    │
    ├─→ AutoTokenizer.tokenize()
    │
    ├─→ move tensors to GPU/CPU
    │
    ├─→ model.forward() [no_grad]
    │
    ├─→ softmax → probabilities
    │
    ├─→ argmax → predicted class
    │
    └─→ format result
        │
        ▼
PredictionResponse (Pydantic)
    │
    ▼
JSON Response
    │
    ▼
Client
```

## Model Architecture

### BERT-based Sentiment Classification
```
Input: Text Sequence
    │
    ▼
┌────────────────────────────────────┐
│ Tokenizer (WordPiece)              │
│ - Input: "I love this!"            │
│ - Output: [101, 1045, 2572, ...]   │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ Embedding Layer                    │
│ - Token Embeddings (768-dim)       │
│ - Positional Embeddings            │
│ - Segment Embeddings               │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ BERT Encoder (12 layers)           │
│ - Multi-head Self-Attention        │
│ - Feed Forward Network             │
│ - Layer Normalization              │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ [CLS] Token Representation         │
│ (768-dimensional vector)           │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ Classification Head                │
│ - Linear (768 → 256)               │
│ - ReLU                             │
│ - Dropout                          │
│ - Linear (256 → 2)                 │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ Softmax                            │
│ Output: [pos_score, neg_score]     │
└────────────────────────────────────┘
    │
    ▼
Prediction: positive or negative
```

## Scalability Considerations

### Horizontal Scaling
- **Auto-scaling**: SageMaker auto-scaling policies
- **Load Balancing**: Multiple endpoints with round-robin
- **Caching**: Redis for frequently requested predictions

### Vertical Scaling
- **Larger Instances**: ml.p3.2xlarge for GPU acceleration
- **Batch Size**: Increase for better throughput
- **Model Optimization**: Quantization, distillation

### Performance Optimization
- **Faster Model**: DistilBERT (40% faster, 98% accuracy)
- **Quantization**: 4x speedup with INT8
- **ONNX Export**: Framework-agnostic optimization
- **Caching Layer**: Redis for hot predictions

## Monitoring & Observability

### Metrics to Track
- **Latency**: p50, p95, p99
- **Throughput**: Requests/sec
- **Errors**: 4xx, 5xx rates
- **Model Quality**: Accuracy drift, prediction distribution

### Logging Strategy
```python
# Structured logging in JSON
logger.info({
    'event': 'prediction',
    'text_length': len(text),
    'predicted_label': prediction,
    'confidence': confidence,
    'latency_ms': latency,
    'timestamp': datetime.utcnow()
})
```

### CloudWatch Integration
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='SentimentAnalysis',
    MetricData=[
        {
            'MetricName': 'PredictionLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds'
        }
    ]
)
```

## Security Best Practices

1. **Input Validation**: Pydantic models for all inputs
2. **Rate Limiting**: Implement request throttling
3. **Authentication**: Add API keys or OAuth2
4. **Model Storage**: Encrypt model artifacts in S3
5. **Network**: Use VPC endpoints for AWS services
6. **Logging**: Sanitize logs, never log sensitive data

## Cost Optimization

| Component | Cost Driver | Optimization |
|-----------|------------|--------------|
| SageMaker | Instance hours | Use smaller instances, auto-scaling |
| S3 | Storage + requests | Lifecycle policies, compression |
| ECR | Storage | Delete old images regularly |
| Data Transfer | Egress charges | Use VPC endpoints, cache |

---

**Project Completion**: ✅ All components created and documented
