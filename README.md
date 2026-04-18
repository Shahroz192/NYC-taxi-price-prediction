# NYC Taxi Price Prediction

End-to-end ML pipeline for predicting NYC taxi fares using machine learning with MLflow experiment tracking and FastAPI serving.

## 🚕 Project Overview

| Component | Description |
|-----------|-------------|
| **Model** | Machine Learning model for taxi fare prediction |
| **Framework** | scikit-learn, Pandas, NumPy |
| **Experiment Tracking** | MLflow |
| **API** | FastAPI + Uvicorn |
| **Monitoring** | Evidently AI |

## 📁 Project Structure

```
NYC-taxi-price-prediction/
├── src/
│   ├── app.py          # FastAPI application
│   ├── train.py        # Model training script
│   ├── preprocess.py   # Data preprocessing
│   ├── retrain.py      # Model retraining pipeline
│   ├── test_model.py   # Model evaluation
│   └── utils.py        # Utility functions
├── models/             # Trained model artifacts
├── mlruns/             # MLflow tracking logs
├── mlartifacts/        # MLflow artifacts
├── tests/              # Unit tests
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── config.yaml         # Configuration
```

## 🛠️ Tech Stack

- **Python** ≥3.8
- **ML**: scikit-learn, Pandas, NumPy
- **Experiment Tracking**: MLflow
- **API**: FastAPI, Uvicorn
- **Monitoring**: Evidently
- **Testing**: pytest

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Shahroz192/NYC-taxi-price-prediction.git
cd NYC-taxi-price-prediction

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Start API
python src/app.py
```

## 📊 Features

- Data preprocessing pipeline
- Model training with hyperparameter tuning
- MLflow experiment tracking
- REST API for predictions
- Model monitoring with Evidently
- Docker containerization

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get fare prediction |
| `/health` | GET | Health check |
| `/model-info` | GET | Model metadata |

## 📈 Model Details

- **Input Features**: Trip distance, pickup/dropoff location, time of day, passenger count
- **Target**: Fare amount
- **Evaluation**: RMSE, MAE, R² score

## 🐳 Docker

```bash
docker build -t taxi-price-prediction .
docker run -p 8000:8000 taxi-price-prediction
```

## 📝 License

MIT License

---

*Built as part of ML engineering portfolio*