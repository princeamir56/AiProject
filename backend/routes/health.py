"""Health and utility routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime, timedelta

from ..database import get_db
from ..models import Prediction, MLModel
from ..schemas import HealthResponse, StatsResponse, DropdownOptions
from ..ml_service import get_ml_service, MLService
from ..config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service)
):
    """Health check endpoint"""
    # Check database
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        database=db_status,
        model_loaded=ml_service.is_model_loaded()
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service)
):
    """Get application statistics"""
    # Total predictions
    total = db.query(func.count(Prediction.id)).scalar() or 0
    
    # Price stats
    avg_price = db.query(func.avg(Prediction.predicted_price)).scalar()
    min_price = db.query(func.min(Prediction.predicted_price)).scalar()
    max_price = db.query(func.max(Prediction.predicted_price)).scalar()
    
    # Today's predictions
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(func.count(Prediction.id)).filter(
        Prediction.created_at >= today
    ).scalar() or 0
    
    return StatsResponse(
        total_predictions=total,
        avg_predicted_price=float(avg_price) if avg_price else None,
        min_predicted_price=float(min_price) if min_price else None,
        max_predicted_price=float(max_price) if max_price else None,
        predictions_today=today_count,
        active_model_version=ml_service.get_model_version()
    )


@router.get("/options", response_model=DropdownOptions)
async def get_form_options(
    ml_service: MLService = Depends(get_ml_service)
):
    """Get dropdown options for the prediction form"""
    return ml_service.get_dropdown_options()
