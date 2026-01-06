"""ML Model routes"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional

from ..database import get_db
from ..models import MLModel
from ..schemas import (
    MLModelInfo, MLModelListResponse, TrainRequest, TrainResponse
)
from ..ml_service import get_ml_service, MLService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=MLModelListResponse)
async def list_models(
    ml_service: MLService = Depends(get_ml_service),
    db: Session = Depends(get_db)
):
    """List all available ML models"""
    # Get models from file system
    file_models = ml_service.get_available_models()
    
    # Get active model version
    active_version = ml_service.get_model_version()
    
    models = [
        MLModelInfo(
            id=i,
            name=m.get('name', 'Unknown'),
            version=m.get('version', 'Unknown'),
            metrics=m.get('metrics'),
            is_active=(m.get('version') == active_version),
            created_at=m.get('created_at')
        )
        for i, m in enumerate(file_models, 1)
    ]
    
    return MLModelListResponse(
        models=models,
        active_model=active_version
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    ml_service: MLService = Depends(get_ml_service),
    db: Session = Depends(get_db)
):
    """Train a new model"""
    try:
        # Train the model
        result = ml_service.train(request.model_name)
        
        metadata = result['metadata']
        metrics = result['metrics']
        
        # Save to database
        ml_model = MLModel(
            name=metadata['model_name'],
            version=metadata['version'],
            model_path=metadata['model_path'],
            metrics=metrics,
            is_active=True,
            description=request.description
        )
        
        # Deactivate other models
        db.query(MLModel).update({MLModel.is_active: False})
        
        db.add(ml_model)
        db.commit()
        db.refresh(ml_model)
        
        return TrainResponse(
            success=True,
            message="Model trained successfully",
            model_info=MLModelInfo(
                id=ml_model.id,
                name=ml_model.name,
                version=ml_model.version,
                metrics=ml_model.metrics,
                is_active=ml_model.is_active,
                created_at=ml_model.created_at
            ),
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/{version}/activate")
async def activate_model(
    version: str,
    ml_service: MLService = Depends(get_ml_service),
    db: Session = Depends(get_db)
):
    """Activate a specific model version"""
    try:
        success = ml_service.load_model_by_version(version)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found")
        
        # Update database
        db.query(MLModel).update({MLModel.is_active: False})
        db.query(MLModel).filter(MLModel.version == version).update({MLModel.is_active: True})
        db.commit()
        
        return {"message": f"Model version {version} activated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation failed: {str(e)}")


@router.get("/metrics")
async def get_model_metrics(
    ml_service: MLService = Depends(get_ml_service)
):
    """Get current model metrics"""
    metrics = ml_service.get_model_metrics()
    version = ml_service.get_model_version()
    
    return {
        "version": version,
        "metrics": metrics,
        "is_loaded": ml_service.is_model_loaded()
    }


@router.get("/feature-importance")
async def get_feature_importance(
    ml_service: MLService = Depends(get_ml_service)
):
    """Get feature importance from the model"""
    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        importance = ml_service.model.get_feature_importance()
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
