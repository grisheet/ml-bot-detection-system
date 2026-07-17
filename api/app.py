"""
FastAPI REST API for ML Bot Detection System
Provides endpoints for single and batch bot classification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Bot Detection API",
    description="Detect whether web traffic requests come from bots or humans.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load the detector to avoid import errors on cold start
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        try:
            from src.inference import BotDetector
            _detector = BotDetector()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e) + " — run python src/train.py first.",
            )
    return _detector


# ---- Pydantic Schemas ----

class RequestFeatures(BaseModel):
    request_rate: float = Field(..., ge=0, description="Requests per second")
    request_interval_std: float = Field(..., ge=0, description="Std dev of request intervals")
    ua_bot_score: float = Field(..., ge=0, le=1, description="User-agent bot probability (0-1)")
    ua_entropy: float = Field(..., ge=0, description="User-agent string entropy")
    mouse_movement_score: float = Field(..., ge=0, le=1, description="Mouse movement naturalness (0=bot, 1=human)")
    click_interval_std: float = Field(..., ge=0, description="Std dev of click intervals")
    ip_request_count: float = Field(..., ge=0, description="Total requests from IP")
    ip_unique_paths: float = Field(..., ge=0, description="Unique paths accessed by IP")
    session_duration: float = Field(..., ge=0, description="Session duration in seconds")
    error_rate: float = Field(..., ge=0, le=1, description="HTTP error rate (0-1)")
    payload_size_avg: float = Field(..., ge=0, description="Average payload size in bytes")
    time_of_day: float = Field(..., ge=0, le=23, description="Hour of day (0-23)")

    class Config:
        json_schema_extra = {
            "example": {
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
                "time_of_day": 3,
            }
        }


class PredictionResponse(BaseModel):
    classification: str
    bot_probability: float
    confidence: str
    latency_ms: float


class BatchRequest(BaseModel):
    requests: List[RequestFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_latency_ms: float
    bot_count: int
    human_count: int


class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    threshold: float
    status: str


# ---- Endpoints ----

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "ML Bot Detection API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(features: RequestFeatures):
    """Classify a single request as bot or human."""
    detector = get_detector()
    t0 = time.time()
    result = detector.predict_single(features.dict())
    latency = (time.time() - t0) * 1000
    return PredictionResponse(
        classification=result["classification"],
        bot_probability=result["bot_probability"],
        confidence=result["confidence"],
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(batch: BatchRequest):
    """Classify a batch of requests."""
    if len(batch.requests) > 1000:
        raise HTTPException(status_code=400, detail="Max batch size is 1000.")

    detector = get_detector()
    t0 = time.time()

    import pandas as pd
    df = pd.DataFrame([r.dict() for r in batch.requests])
    results = detector.predict(df)

    total_latency = (time.time() - t0) * 1000
    predictions = [
        PredictionResponse(
            classification=r["classification"],
            bot_probability=r["bot_probability"],
            confidence=r["confidence"],
            latency_ms=round(total_latency / len(results), 2),
        )
        for r in results
    ]

    bot_count = sum(1 for r in results if r["classification"] == "bot")
    return BatchResponse(
        predictions=predictions,
        total_latency_ms=round(total_latency, 2),
        bot_count=bot_count,
        human_count=len(results) - bot_count,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def model_info():
    """Return metadata about the loaded model."""
    detector = get_detector()
    model_type = type(detector.model).__name__
    return ModelInfo(
        model_type=model_type,
        features=detector.feature_cols,
        threshold=detector.threshold,
        status="loaded",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
