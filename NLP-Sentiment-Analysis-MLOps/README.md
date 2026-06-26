# NLP Sentiment Analysis - MLOps Project

A production-ready MLOps project for sentiment analysis using transformers, designed for deployment on AWS.

## Project Structure

```
├── src/
│   ├── app/                 # FastAPI serving application
│   ├── data/                # Data loading and management
│   ├── features/            # Feature engineering and preprocessing
│   ├── models/              # Model training
│   ├── serving/             # Model inference
│   └── utils/               # Utility functions
├── tests/                   # Unit and integration tests
├── scripts/                 # Training and utility scripts
├── aws/                     # AWS deployment configurations
├── notebooks/               # Jupyter notebooks for exploration
├── config.yaml              # Configuration file
└── dockerfile               # Docker containerization
```

## Features

- **Data Pipeline**: Load data from HuggingFace, S3, or local CSV
- **NLP Model**: Fine-tuned BERT for sentiment classification
- **MLflow Integration**: Experiment tracking and model versioning
- **FastAPI Serving**: REST API for model inference
- **Docker Support**: Containerized deployment
- **AWS Integration**: SageMaker, ECR, CloudFormation templates
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Unit Tests**: Comprehensive test coverage

## Installation

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd NLP-Sentiment-Analysis-MLOps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- Data source (HuggingFace, local, S3)
- Model architecture (BERT variants)
- Training hyperparameters
- AWS settings

```yaml
data:
  source: "huggingface"
  dataset_name: "imdb"
  
training:
  model_name: "bert-base-uncased"
  num_epochs: 3
  batch_size: 32
  learning_rate: 2.0e-5
```

## Usage

### Training

```bash
# Train model with configuration
python scripts/train_pipeline.py --config config.yaml

# View MLflow UI
mlflow ui
```

### Serving

```bash
# Start FastAPI server
python -m uvicorn src.app.main:app --reload

# API endpoints:
# - POST /predict - Single prediction
# - POST /predict_batch - Batch predictions
# - POST /explain - Prediction explanation
# - GET /health - Health check
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Docker Deployment

```bash
# Build image
docker build -t nlp-sentiment:latest .

# Run container
docker run -p 8000:8000 -e MODEL_PATH=/app/models/sentiment-model nlp-sentiment:latest

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

## AWS Deployment

### Prerequisites
- AWS Account with appropriate IAM permissions
- ECR repository created
- SageMaker IAM role
- AWS CLI configured

### Deploy Steps

```bash
# 1. Build and push Docker image to ECR
docker build -t nlp-sentiment:latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag nlp-sentiment:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/nlp-sentiment:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/nlp-sentiment:latest

# 2. Upload model to S3
aws s3 cp models/sentiment-model s3://nlp-sentiment-models/models/ --recursive

# 3. Deploy using CloudFormation
python aws/deploy.py \
  --region us-east-1 \
  --model-dir models/sentiment-model \
  --image-uri <account-id>.dkr.ecr.us-east-1.amazonaws.com/nlp-sentiment:latest
```

### SageMaker Deployment

```python
import boto3
from sagemaker.model import Model

# Create SageMaker model
model = Model(
    image_uri="<ecr-image-uri>",
    model_data="s3://nlp-sentiment-models/models/",
    role="<sagemaker-role-arn>",
    name="sentiment-analysis-model"
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

# Make prediction
result = predictor.predict({"text": "Great product!"})
```

## CI/CD Pipeline

GitHub Actions automatically:
1. Runs tests on push/PR
2. Builds Docker image on main branch
3. Pushes to ECR
4. Deploys to AWS SageMaker

Required secrets:
- `AWS_ROLE_TO_ASSUME`: OIDC role ARN
- `AWS_ACCOUNT_ID`: AWS account ID

## API Examples

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Response:
```json
{
  "text": "I love this product!",
  "label": "positive",
  "confidence": 0.98,
  "probabilities": {
    "positive": 0.98,
    "negative": 0.02
  }
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great quality!",
      "Terrible experience",
      "Average product"
    ]
  }'
```

## Model Information

- **Architecture**: BERT-base-uncased
- **Task**: Binary sentiment classification (positive/negative)
- **Training Data**: IMDB movie reviews
- **Accuracy**: ~92% on test set
- **Model Size**: ~440MB

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 92.3% |
| Precision | 91.8% |
| Recall | 92.7% |
| F1-Score | 92.2% |

## Monitoring

MLflow tracks:
- Training metrics (loss, accuracy)
- Evaluation metrics (F1, precision, recall)
- Model parameters and hyperparameters
- Model artifacts and versions

Access MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Troubleshooting

### Out of Memory

Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # from 32
```

### Slow Training

Use a smaller model:
```yaml
training:
  model_name: "distilbert-base-uncased"
```

### AWS Deployment Issues

Check CloudFormation events:
```bash
aws cloudformation describe-stack-events --stack-name sentiment-analysis-stack
```

## Contributing

1. Create feature branch
2. Make changes and add tests
3. Run `pytest` and linting
4. Create pull request

## License

MIT License

## Support

For issues and questions, create a GitHub issue or contact the maintainers.
