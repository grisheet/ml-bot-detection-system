# ML Bot Detection System

A machine learning-powered bot detection system that classifies web traffic as **bot** or **human** using behavioral and network-based features.

## Features

- **Multi-feature Input**: Request frequency, user-agent patterns, mouse movement/timing, and IP behavior
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and Neural Network
- **Full Pipeline**: Data generation → Preprocessing → Feature engineering → Training → Evaluation
- **REST API**: FastAPI inference endpoint with real-time bot probability scoring
- **Visualization Dashboard**: Interactive Streamlit dashboard for monitoring and analysis
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC

## Project Structure

```
ml-bot-detection-system/
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   └── sample_data.csv        # Sample generated dataset
├── src/
│   ├── preprocess.py          # Data preprocessing and feature engineering
│   ├── train.py               # Model training and evaluation
│   └── inference.py           # Inference engine
├── api/
│   └── app.py                 # FastAPI REST API
├── dashboard/
│   └── dashboard.py           # Streamlit visualization dashboard
├── models/                    # Saved trained models (auto-generated)
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/grisheet/ml-bot-detection-system.git
cd ml-bot-detection-system
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data

```bash
python data/generate_data.py
```

This creates `data/sample_data.csv` with 10,000 labeled samples.

### 2. Train Models

```bash
python src/train.py
```

Trains Logistic Regression, Random Forest, XGBoost, and Neural Network models. Saves the best model to `models/best_model.pkl`.

### 3. Run Inference

```bash
python src/inference.py
```

### 4. Start the API

```bash
uvicorn api.app:app --reload --port 8000
```

API endpoints:
- `GET /` — Health check
- `POST /predict` — Single prediction
- `POST /predict/batch` — Batch predictions
- `GET /model/info` — Model metadata

### 5. Launch Dashboard

```bash
streamlit run dashboard/dashboard.py
```

## API Usage Example

```python
import requests

data = {
    "request_rate": 15.2,
    "request_interval_std": 0.05,
    "ua_bot_score": 0.9,
    "ua_entropy": 2.1,
    "mouse_movement_score": 0.1,
    "click_interval_std": 0.02,
    "ip_request_count": 500,
    "ip_unique_paths": 3,
    "session_duration": 10.0,
    "error_rate": 0.3,
    "payload_size_avg": 512.0,
    "time_of_day": 3
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
# {"classification": "bot", "bot_probability": 0.97, "confidence": "high"}
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~88% | ~87% | ~89% | ~88% |
| Random Forest | ~95% | ~95% | ~95% | ~95% |
| XGBoost | ~96% | ~96% | ~96% | ~96% |
| Neural Network | ~94% | ~94% | ~94% | ~94% |

*Results on synthetic data — actual performance varies with real-world data.*

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `request_rate` | Requests per second |
| `request_interval_std` | Standard deviation of request intervals |
| `ua_bot_score` | Probability the user-agent is a bot |
| `ua_entropy` | Entropy of user-agent string |
| `mouse_movement_score` | Naturalness score of mouse movement (0=bot, 1=human) |
| `click_interval_std` | Std dev of click intervals |
| `ip_request_count` | Total requests from IP in session window |
| `ip_unique_paths` | Unique URL paths accessed by IP |
| `session_duration` | Session length in seconds |
| `error_rate` | HTTP error rate (4xx/5xx) |
| `payload_size_avg` | Average request payload size |
| `time_of_day` | Hour of day (0-23) |

## Tech Stack

- **ML**: scikit-learn, XGBoost, PyTorch
- **API**: FastAPI, Uvicorn
- **Dashboard**: Streamlit, Plotly
- **Data**: pandas, NumPy
- **Serialization**: joblib

## Production Scaling

- **Redis caching** for IP-based feature aggregation
- **Kafka** for real-time event streaming
- **Docker** containerization for deployment
- **Model versioning** with MLflow
- **A/B testing** framework for model comparison

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Built as an advanced ML portfolio project demonstrating end-to-end machine learning system design.
