"""Database Models"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.sql import func
from .database import Base


class Prediction(Base):
    """Prediction history model"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Input features (stored as JSON)
    input_features = Column(JSON, nullable=False)
    
    # Prediction result
    predicted_price = Column(Float, nullable=False)
    
    # Model info
    model_version = Column(String(50), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Optional user tracking
    user_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)


class MLModel(Base):
    """ML Model registry"""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model identification
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False, unique=True)
    
    # Model file path
    model_path = Column(String(500), nullable=False)
    
    # Metrics (stored as JSON)
    metrics = Column(JSON, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False)
    
    # Description
    description = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
