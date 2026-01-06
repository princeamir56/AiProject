"""Pydantic Schemas for request/response validation"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


# ==================== Prediction Schemas ====================

class HouseFeatures(BaseModel):
    """House features for prediction - simplified version"""
    
    # Quality ratings (1-10)
    OverallQual: int = Field(..., ge=1, le=10, description="Overall quality (1-10)")
    OverallCond: int = Field(..., ge=1, le=10, description="Overall condition (1-10)")
    
    # Size features
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Year house was built")
    TotalBsmtSF: float = Field(..., ge=0, description="Total basement area in sq ft")
    GrLivArea: float = Field(..., ge=0, description="Above ground living area in sq ft")
    
    # Rooms
    FullBath: int = Field(..., ge=0, le=10, description="Number of full bathrooms")
    HalfBath: int = Field(..., ge=0, le=10, description="Number of half bathrooms")
    BedroomAbvGr: int = Field(..., ge=0, le=20, description="Number of bedrooms above ground")
    TotRmsAbvGrd: int = Field(..., ge=0, le=30, description="Total rooms above ground")
    
    # Garage
    GarageCars: int = Field(..., ge=0, le=10, description="Garage car capacity")
    GarageArea: float = Field(..., ge=0, description="Garage area in sq ft")
    
    # Categorical features
    MSZoning: str = Field(..., description="Zoning classification")
    Neighborhood: str = Field(..., description="Physical location")
    BldgType: str = Field(..., description="Type of dwelling")
    HouseStyle: str = Field(..., description="Style of dwelling")
    CentralAir: str = Field(..., pattern="^[YN]$", description="Central air conditioning (Y/N)")
    KitchenQual: str = Field(..., description="Kitchen quality")
    
    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2005,
                "TotalBsmtSF": 1000,
                "GrLivArea": 1500,
                "FullBath": 2,
                "HalfBath": 1,
                "BedroomAbvGr": 3,
                "TotRmsAbvGrd": 7,
                "GarageCars": 2,
                "GarageArea": 500,
                "MSZoning": "RL",
                "Neighborhood": "CollgCr",
                "BldgType": "1Fam",
                "HouseStyle": "2Story",
                "CentralAir": "Y",
                "KitchenQual": "Gd"
            }
        }


class PredictionRequest(BaseModel):
    """Request for prediction"""
    features: HouseFeatures
    session_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response for prediction"""
    id: int
    predicted_price: float
    formatted_price: str
    model_version: Optional[str]
    input_features: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True


class PredictionHistory(BaseModel):
    """Prediction history item"""
    id: int
    predicted_price: float
    input_features: Dict[str, Any]
    model_version: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    """List of predictions"""
    predictions: List[PredictionHistory]
    total: int
    page: int
    per_page: int


# ==================== ML Model Schemas ====================

class MLModelInfo(BaseModel):
    """ML Model information"""
    id: int
    name: str
    version: str
    metrics: Optional[Dict[str, Any]] = None
    is_active: bool = False
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class MLModelListResponse(BaseModel):
    """List of ML models"""
    models: List[MLModelInfo]
    active_model: Optional[str]


class TrainRequest(BaseModel):
    """Request to train a new model"""
    model_name: Optional[str] = None
    description: Optional[str] = None


class TrainResponse(BaseModel):
    """Response after training"""
    success: bool
    message: str
    model_info: Optional[MLModelInfo]
    metrics: Optional[Dict[str, float]]


# ==================== Dropdown Options Schemas ====================

class DropdownOptions(BaseModel):
    """Dropdown options for the form"""
    MSZoning: List[Dict[str, str]]
    Neighborhood: List[Dict[str, str]]
    BldgType: List[Dict[str, str]]
    HouseStyle: List[Dict[str, str]]
    KitchenQual: List[Dict[str, str]]


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database: str
    model_loaded: bool


# ==================== Statistics Schemas ====================

class StatsResponse(BaseModel):
    """Statistics response"""
    total_predictions: int
    avg_predicted_price: Optional[float]
    min_predicted_price: Optional[float]
    max_predicted_price: Optional[float]
    predictions_today: int
    active_model_version: Optional[str]
