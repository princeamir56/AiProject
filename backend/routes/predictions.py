"""Prediction routes"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from datetime import datetime, timedelta

from ..database import get_db
from ..models import Prediction
from ..schemas import (
    PredictionRequest, PredictionResponse, PredictionListResponse,
    PredictionHistory, HouseFeatures
)
from ..ml_service import get_ml_service, MLService

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
async def create_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service)
):
    """Create a new prediction"""
    try:
        # Convert features to dict
        features_dict = request.features.model_dump()
        
        # Make prediction
        predicted_price = ml_service.predict(features_dict)
        
        # Save to database
        prediction = Prediction(
            input_features=features_dict,
            predicted_price=predicted_price,
            model_version=ml_service.get_model_version(),
            session_id=request.session_id
        )
        
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return PredictionResponse(
            id=prediction.id,
            predicted_price=prediction.predicted_price,
            formatted_price=f"${prediction.predicted_price:,.2f}",
            model_version=prediction.model_version,
            input_features=prediction.input_features,
            created_at=prediction.created_at
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("", response_model=PredictionListResponse)
async def list_predictions(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all predictions with pagination"""
    offset = (page - 1) * per_page
    
    # Get total count
    total = db.query(func.count(Prediction.id)).scalar()
    
    # Get paginated results
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(per_page)
        .all()
    )
    
    return PredictionListResponse(
        predictions=[
            PredictionHistory(
                id=p.id,
                predicted_price=p.predicted_price,
                input_features=p.input_features,
                model_version=p.model_version,
                created_at=p.created_at
            )
            for p in predictions
        ],
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific prediction by ID"""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return PredictionResponse(
        id=prediction.id,
        predicted_price=prediction.predicted_price,
        formatted_price=f"${prediction.predicted_price:,.2f}",
        model_version=prediction.model_version,
        input_features=prediction.input_features,
        created_at=prediction.created_at
    )


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """Delete a prediction"""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    db.delete(prediction)
    db.commit()
    
    return {"message": "Prediction deleted successfully"}
