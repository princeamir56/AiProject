"""Backend Configuration with SQLite support"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    APP_NAME: str = "House Price Predictor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database settings - Default to SQLite for easy setup
    DATABASE_URL: str = "sqlite:///./house_prices.db"
    
    # CORS settings
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ML_DIR: Path = BASE_DIR / "ml"
    MODELS_DIR: Path = ML_DIR / "models"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings"""
    return Settings()


settings = get_settings()
