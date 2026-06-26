# NLP Sentiment Analysis MLOps - Beginner's Guide

## What is This Project?

**Sentiment Analysis** means determining if a piece of text is positive, negative, or neutral.

**Example:**
- Text: "I love this product!" → **Positive** ✅
- Text: "This is terrible" → **Negative** ❌
- Text: "It's okay" → **Neutral** 😐

This project automates this process using AI/Machine Learning and is ready for production deployment.

---

## 🎯 Project Goals

```
Goal: Build an AI system that reads reviews and automatically classifies them as positive/negative
```

**Real-world uses:**
- 📱 Social media: Analyze tweets/comments
- 🛍️ E-commerce: Rate customer reviews automatically
- 📧 Support tickets: Identify urgent complaints
- 📰 News analysis: Track sentiment trends

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ANALYSIS PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   RAW DATA  │
                              │  (Reviews)  │
                              └──────┬──────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   DATA LOADING & CLEANING        │
                    │  - Download from HuggingFace     │
                    │  - Load from local CSV           │
                    │  - Load from AWS S3 bucket       │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   TEXT PREPROCESSING             │
                    │  - Remove special characters     │
                    │  - Convert to lowercase          │
                    │  - Tokenization                  │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   MODEL TRAINING                 │
                    │  - Use BERT model                │
                    │  - Learn patterns for sentiment  │
                    │  - Optimize weights              │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   MODEL EVALUATION               │
                    │  - Test on unseen data           │
                    │  - Calculate accuracy (92.3%)    │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   SAVE & DEPLOY                  │
                    │  - Save trained model            │
                    │  - Package in Docker container   │
                    │  - Deploy to AWS SageMaker       │
                    └────────────────┬─────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │   MAKE PREDICTIONS               │
                    │  - API accepts new reviews       │
                    │  - Returns sentiment label       │
                    │  - Confidence score              │
                    └─────────────────────────────────┘
```

---

## 📁 Project Structure Explained

```
NLP-Sentiment-Analysis-MLOps/
│
├── src/                          # 🔧 Core application code
│   ├── app/
│   │   └── main.py              # FastAPI web server (receives predictions)
│   │
│   ├── data/
│   │   └── load_data.py         # Load data from HF, S3, or local files
│   │
│   ├── features/
│   │   └── preprocess.py        # Clean & prepare text data
│   │
│   ├── models/
│   │   ├── train.py             # Train the sentiment model
│   │   ├── evaluate.py          # Test model accuracy
│   │   └── tune.py              # Find best parameters
│   │
│   ├── serving/
│   │   └── inference.py         # Use trained model for predictions
│   │
│   └── utils/
│       ├── utils.py             # Helper functions
│       └── validate_data.py     # Check data quality
│
├── scripts/
│   └── train_pipeline.py        # 🚀 Main script to train everything
│
├── tests/
│   ├── test_pipeline.py         # Test training code
│   └── test_api.py              # Test API server
│
├── aws/                          # ☁️ Cloud deployment files
│   ├── cloudformation_template.json  # Describes AWS setup
│   ├── deploy.py                # Deploy to AWS with Python
│   └── deploy.sh                # Deploy to AWS with Bash
│
├── config.yaml                   # ⚙️ Configuration settings
├── requirements.txt              # 📦 Python packages needed
├── dockerfile                    # 🐳 Docker container setup
└── Makefile                      # Shortcut commands

```

**What each folder does:**
- `src/` = Main code (the brain of the project)
- `scripts/` = Run the full training pipeline
- `tests/` = Check if code works correctly
- `aws/` = Deploy to cloud
- `config.yaml` = Settings you can change

---

## 🔄 Data Flow: From Text to Prediction

```
                    User Input
                        │
                        ▼
                  "I love it!"
                        │
                        ▼
        ┌──────────────────────────────────┐
        │  1️⃣ TEXT PREPROCESSING           │
        │  - Remove punctuation             │
        │  - Convert: "I LOVE IT!" → "i love it"
        │  - Tokenize: ["i", "love", "it"]│
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  2️⃣ CONVERT TO NUMBERS           │
        │  BERT Tokenizer converts words    │
        │  to numbers computer understands  │
        │  ["i", "love", "it"]             │
        │       ↓↓↓                         │
        │  [101, 1045, 2572, 2009, 102]    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  3️⃣ PASS THROUGH BERT MODEL      │
        │  Neural network learned from      │
        │  examples during training         │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  4️⃣ MODEL OUTPUT                 │
        │  Two numbers representing         │
        │  confidence for each class:       │
        │  Positive: 0.95                  │
        │  Negative: 0.05                  │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  5️⃣ FINAL PREDICTION             │
        │  Result: POSITIVE ✅             │
        │  Confidence: 95%                 │
        └─────────────────────────────────┘
```

---

## 🧠 What is BERT? (The AI Model)

```
Simple Analogy:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You learn sentiment analysis by:
1. Reading thousands of examples
2. Understanding patterns (negative words, positive words, context)
3. Practice recognizing new examples

BERT does the same thing:
1. Reads millions of text examples (trained by Google)
2. Learns patterns about language
3. We fine-tune it for sentiment analysis
4. It recognizes sentiment in new reviews

BERT = Bidirectional Encoder Representations from Transformers
(You don't need to understand the fancy name - just know it's a powerful AI)

┌─────────────────────────────┐
│   Input Text: "I love it!"  │
└────────┬────────────────────┘
         │
         ▼
    ┌─────────────────────────┐
    │  BERT Neural Network    │
    │  (Learns from data)     │
    │                         │
    │ Layer 1: Recognize      │
    │ words & context         │
    │                         │
    │ Layer 2: Understand     │
    │ relationships           │
    │                         │
    │ Layer 3: Extract        │
    │ sentiment signals       │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  Classification Head    │
    │  (Decides positive/neg) │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  Output: POSITIVE ✅    │
    │  Score: 0.98           │
    └─────────────────────────┘
```

---

## 📊 Training Process Explained

```
┌─────────────────────────────────────────────────────────┐
│              HOW THE MODEL LEARNS                        │
└─────────────────────────────────────────────────────────┘

BEFORE TRAINING:
  Model is untrained (random guesses)
  "I hate this" → Predicts: POSITIVE ❌ (Wrong!)

TRAINING PROCESS (Epoch = One pass through all data):

Epoch 1:
  ┌──────────────────────────────────────────┐
  │ Review: "I hate this" (Label: NEGATIVE)  │
  │ Model predicts: POSITIVE ❌              │
  │ Error: 0.85 (Very wrong)                 │
  │ → Adjust weights to be less wrong        │
  └──────────────────────────────────────────┘
  
  ┌──────────────────────────────────────────┐
  │ Review: "Love it!" (Label: POSITIVE)     │
  │ Model predicts: NEGATIVE ❌              │
  │ Error: 0.92 (Very wrong)                 │
  │ → Adjust weights to be less wrong        │
  └──────────────────────────────────────────┘
  
  Progress: 50,000 more reviews...
  Epoch 1 Accuracy: 60%

Epoch 2:
  Model is now better informed
  Progress: 50,000 reviews...
  Epoch 2 Accuracy: 75%

Epoch 3:
  Model has learned a lot
  Progress: 50,000 reviews...
  Epoch 3 Accuracy: 92.3% ✅

AFTER TRAINING:
  Model is now expert at sentiment classification
  "I hate this" → Predicts: NEGATIVE ✅ (Correct!)
```

---

## 🚀 How It Works: Training vs. Prediction

### Training Phase (Learning)
```
┌─────────────────────────────────────┐
│      TRAINING (One-time setup)      │
└─────────────────────────────────────┘

$ python scripts/train_pipeline.py --config config.yaml

├─ Load 25,000 IMDB reviews
├─ Clean text data
├─ Split into train (80%) & test (20%)
├─ Train BERT model for 3 epochs
├─ Evaluate on test data
│  └─ Accuracy: 92.3% ✅
├─ Save model to disk
│  └─ models/sentiment-model/
└─ Monitor with MLflow dashboard
   └─ http://localhost:5000

⏱️ Time: ~30-60 minutes on GPU
```

### Prediction Phase (Using the trained model)
```
┌─────────────────────────────────────┐
│   PREDICTION (Fast, recurring)      │
└─────────────────────────────────────┘

$ python -m uvicorn src.app.main:app --reload --port 8000

Server starts listening on http://localhost:8000

User sends request:
POST /predict
{
  "text": "This product is amazing!"
}
                    │
                    ▼
        Load trained model
        Run prediction
                    │
                    ▼
Response sent back:
{
  "label": "positive",
  "confidence": 0.97,
  "probabilities": {
    "positive": 0.97,
    "negative": 0.03
  }
}

⏱️ Time: ~50ms per prediction
```

---

## 🌐 API Endpoints (How to Use)

```
The API is like a waiter:
You ask (POST a request) → Server answers (sends prediction)

┌───────────────────────────────────────────────────────┐
│              API ENDPOINTS                            │
└───────────────────────────────────────────────────────┘

1️⃣ CHECK IF SERVER IS ALIVE
   GET /health
   
   Response: {"status": "healthy"} ✅

2️⃣ PREDICT SENTIMENT FOR ONE TEXT
   POST /predict
   Body: {"text": "I love this!"}
   
   Response:
   {
     "text": "I love this!",
     "label": "positive",
     "confidence": 0.98,
     "probabilities": {
       "positive": 0.98,
       "negative": 0.02
     }
   }

3️⃣ PREDICT FOR MULTIPLE TEXTS
   POST /predict_batch
   Body: {"texts": ["Good!", "Bad", "Neutral"]}
   
   Response: Array of 3 predictions

4️⃣ GET MODEL INFO
   GET /model-info
   
   Response: {"model": "bert-base-uncased", ...}

5️⃣ EXPLAIN PREDICTION (Why positive?)
   POST /explain
   Body: {"text": "Amazing product!"}
   
   Response: {
     "label": "positive",
     "explanation": "Words: 'amazing' has high positive weight"
   }
```

---

## 🐳 Deployment: From Your Computer to AWS

```
┌─────────────────────────────────────────────────────────┐
│             DEPLOYMENT STAGES                           │
└─────────────────────────────────────────────────────────┘

STAGE 1: Local Development (Your Computer)
┌──────────────────────────┐
│  Train Model             │
│  Test API locally        │
│  Debug issues            │
└──────────────┬───────────┘
               │
STAGE 2: Containerization (Docker)
┌──────────────────────────┐
│  Package everything      │
│  Create Docker image     │
│  Test in container       │
└──────────────┬───────────┘
               │
STAGE 3: Cloud Deployment (AWS)
┌──────────────────────────┐
│  Upload image to ECR     │
│  (Elastic Container      │
│   Registry)              │
└──────────────┬───────────┘
               │
STAGE 4: Scale (SageMaker)
┌──────────────────────────┐
│  Deploy on AWS SageMaker │
│  Create endpoint         │
│  Auto-scaling available  │
└──────────────┬───────────┘
               │
STAGE 5: Monitoring
┌──────────────────────────┐
│  Track performance       │
│  Monitor errors          │
│  View logs               │
└──────────────────────────┘

RESULT: Your model is accessible 24/7 from anywhere! ☁️
```

---

## 📋 Configuration: What You Can Change

Edit `config.yaml` to customize:

```yaml
# Data Source
data:
  source: "huggingface"          # Where to get data
  dataset_name: "imdb"           # Which dataset
  
# Training Settings
training:
  model_name: "bert-base-uncased" # AI model type
  num_epochs: 3                   # How many times to loop through data
  batch_size: 32                  # How many reviews at a time
  learning_rate: 2.0e-5          # How fast it learns

# AWS Settings
aws:
  region: "us-east-1"            # Which AWS region
  s3_bucket: "nlp-sentiment-models" # Where to store models
```

**Common tweaks:**
- Slow training? → Reduce batch_size to 16
- GPU out of memory? → Reduce batch_size to 8
- Need faster results? → Use "distilbert-base-uncased" model

---

## 🔧 Quick Commands

```bash
# Setup (one time)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Training (takes ~1 hour)
python scripts/train_pipeline.py --config config.yaml

# Start API server (for predictions)
make serve
# or
python -m uvicorn src.app.main:app --reload --port 8000

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'

# Monitor training with MLflow
mlflow ui
# Visit http://localhost:5000

# Run tests
pytest tests/ -v

# Deploy to AWS
python aws/deploy.py --region us-east-1

# Clean up
make clean
```

---

## 📊 Key Metrics Explained

```
Accuracy: 92.3%
  └─ Out of 100 reviews, model got 92 correct ✅

F1-Score: 92.2%
  └─ Balances precision & recall
  └─ Good for imbalanced datasets

Precision: How many predicted positives were actually positive?
  └─ 90% = Of 100 predicted as positive, 90 really were ✅

Recall: How many actual positives did we find?
  └─ 94% = Of 100 really positive reviews, we found 94 ✅

Inference Time: 50ms per sample
  └─ Takes 50 milliseconds to predict sentiment
  └─ Fast enough for real-time use! ⚡
```

---

## 🎓 Beginner Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ResolutionImpossible` error | Conflicting package versions | `pip install -r requirements.txt --upgrade` |
| "Cannot import from data.load_data" | Python path issue | Make sure to run from project root: `cd NLP-Sentiment-Analysis-MLOps` |
| CUDA out of memory | GPU too small for batch size | Reduce batch_size in config.yaml to 8 or 16 |
| Slow training | Using CPU instead of GPU | Make sure CUDA/GPU drivers are installed |
| API won't start | Port 8000 already in use | Use different port: `--port 8001` |
| "Model not found" | Model didn't train successfully | Run `python scripts/train_pipeline.py --config config.yaml` first |

---

## 💡 Real-World Example: From Text to Decision

```
┌──────────────────────────────────────────────┐
│  REAL-WORLD USE CASE: E-commerce Platform   │
└──────────────────────────────────────────────┘

Customer writes review:
  "This product is terrible! Broke after 1 day. 
   Worst purchase ever. Seller didn't help."

                    │ (Via API)
                    ▼
              ┌─────────────┐
              │ Our System  │
              │ Analyzes    │
              └──────┬──────┘
                     │
                     ▼
    ┌──────────────────────────────────┐
    │ Result: NEGATIVE (Confidence: 99%)
    │ Automatic Actions:               │
    │ ✓ Flag for quality team          │
    │ ✓ Offer refund automatically     │
    │ ✓ Remove from featured reviews   │
    │ ✓ Notify seller                  │
    └──────────────────────────────────┘

Alternative positive review:
  "Excellent quality! Arrived quickly.
   Exactly as described. Highly recommend!"

                    │ (Via API)
                    ▼
              ┌─────────────┐
              │ Our System  │
              │ Analyzes    │
              └──────┬──────┘
                     │
                     ▼
    ┌──────────────────────────────────┐
    │ Result: POSITIVE (Confidence: 98%)
    │ Automatic Actions:               │
    │ ✓ Feature on homepage            │
    │ ✓ Increase seller rating         │
    │ ✓ Show in recommendations        │
    │ ✓ Boost seller sales             │
    └──────────────────────────────────┘

Result: Better customer experience + More sales! 📈
```

---

## 🎯 Next Steps for Learning

1. **Understand the basics** (You just did! ✅)
2. **Run training**: `python scripts/train_pipeline.py --config config.yaml`
3. **Start the server**: `make serve`
4. **Test predictions**: Use curl or Python requests
5. **Explore the code**: Read through src/ files
6. **Try modifications**: Change config.yaml values
7. **Deploy to AWS**: Follow aws/README.md

---

## 📚 Helpful Resources

- [Hugging Face Documentation](https://huggingface.co/docs) - Learn about transformers
- [PyTorch Documentation](https://pytorch.org/docs) - Deep learning framework
- [FastAPI Guide](https://fastapi.tiangolo.com) - Building the API
- [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/) - Cloud deployment
- [MLflow Docs](https://mlflow.org/docs) - Experiment tracking

---

## ❓ FAQ

**Q: Do I need a GPU?**  
A: No, but training will be slow on CPU. GPU makes it ~10x faster.

**Q: Can I use different models?**  
A: Yes! Change `model_name` in config.yaml to any HuggingFace model.

**Q: How accurate is it?**  
A: 92.3% on IMDB reviews. Real-world accuracy depends on your data.

**Q: Can I use it for other languages?**  
A: Yes! Use multilingual models like "xlm-roberta-base".

**Q: What if I have custom data?**  
A: Put CSV file in data/ folder and modify config.yaml to use it.

---

**Congratulations! You now understand the whole project! 🎉**

Next step: Train the model and make your first prediction! 🚀
