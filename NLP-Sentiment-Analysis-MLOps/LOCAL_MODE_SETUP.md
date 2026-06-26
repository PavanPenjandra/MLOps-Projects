# Local Mode Setup Guide

This guide provides instructions for running the NLP Sentiment Analysis project in **local mode** without AWS dependencies or GPU requirements.

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment (if not already activated)
.\venv\Scripts\Activate.ps1

# Install dependencies for local mode
pip install -r requirements.txt
```

### 2. Create Required Directories

```bash
mkdir -p data models logs
```

### 3. Configure for Local Mode

The project is pre-configured for local mode in `config.yaml`:

- **Data Source**: HuggingFace IMDB dataset
- **Model**: DistilBERT (faster, lighter than BERT)
- **Epochs**: 1 (quick training)
- **Batch Size**: 16 (reduced for local machines)
- **Sample Size**: 500 (reduced dataset)
- **GPU**: Disabled by default

## Using Make Commands

### Setup for Local Development
```bash
make local-setup
```

### Train Model
```bash
make train
```
This will:
- Download IMDB dataset (500 samples)
- Fine-tune DistilBERT for sentiment classification
- Save model to `models/` directory
- Track metrics with MLflow (local storage)

### Run API Server
```bash
make serve
```
The API will be available at: `http://localhost:8001`

### Run Tests
```bash
make test
```

### Run Linting
```bash
make lint
```

## Direct Commands

### Train with Python
```bash
python scripts/train_pipeline.py --config config.yaml
```

### Start API Server
```bash
python -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8001
```

### View MLflow Dashboard
```bash
mlflow ui
```
Access at: `http://localhost:5000`

## Configuration Options

Edit `config.yaml` to customize:

### Adjust Dataset Size
```yaml
data:
  sample_size: 1000  # Increase for more data
```

### Use More Epochs
```yaml
training:
  num_epochs: 3  # More training iterations
```

### Use Larger Model (if you have GPU)
```yaml
training:
  model_name: "bert-base-uncased"
  use_cuda: true
```

### Reduce Batch Size (if out of memory)
```yaml
training:
  batch_size: 8  # Lower batch size
```

## Troubleshooting

### Out of Memory Error
Reduce batch size and sample size in `config.yaml`:
```yaml
training:
  batch_size: 8
data:
  sample_size: 100
```

### Slow Performance
1. Reduce sample size
2. Use fewer epochs
3. Consider using a smaller max_length:
```yaml
training:
  max_length: 128
```

### Model Download Issues
The model is downloaded from HuggingFace on first run. Ensure you have internet connection and ~500MB free disk space.

### Port Already in Use
Change the port in Makefile or use:
```bash
python -m uvicorn src.app.main:app --reload --port 8002
```

## Project Structure in Local Mode

```
├── data/                    # Downloaded datasets
├── models/                  # Saved models
├── logs/                    # Training logs
├── mlruns/                  # MLflow artifacts (local)
├── config.yaml              # Local mode configuration
├── requirements.txt         # Dependencies (AWS optional)
└── scripts/
    └── train_pipeline.py    # Training script
```

## API Usage (Local Mode)

### Health Check
```bash
curl http://localhost:8001/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8001/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great quality!",
      "Terrible experience",
      "Average product"
    ]
  }'
```

## Switching to Production Mode

To switch from local to production mode with AWS:

1. Update `config.yaml`:
   ```yaml
   local_mode:
     enabled: false
   aws:
     enabled: true
   training:
     model_name: "bert-base-uncased"
     num_epochs: 3
     batch_size: 32
   ```

2. Install AWS dependencies:
   ```bash
   pip install boto3>=1.26.0
   ```

3. Configure AWS credentials:
   ```bash
   aws configure
   ```

4. Deploy to AWS:
   ```bash
   python aws/deploy.py --region us-east-1
   ```

## Performance Notes

**Local Mode Training Time** (approximate):
- With CPU only: 10-20 minutes per epoch
- With GPU (if available): 2-5 minutes per epoch

**Local Mode System Requirements**:
- RAM: 8GB minimum (16GB recommended)
- Disk: 2GB for models and data
- CPU: Modern processor (i5+/Ryzen 5+)
- Internet: Required for first download only

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Support

For issues specific to local mode, check:
1. Virtual environment is activated
2. All dependencies are installed: `pip list`
3. Disk space available: `df -h`
4. Port 8001 is not in use: `netstat -ano | findstr 8001`
