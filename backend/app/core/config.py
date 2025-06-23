from pydantic import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Basic app settings
    app_name: str = "Phytocognix API"
    app_version: str = "1.0.0"
    debug: bool = False
    api_version: str = "v1"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS settings
    allowed_origins: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://*.streamlit.app",
        "https://phytocognix.streamlit.app",
        "https://*.onrender.com"
    ]
    
    # File upload settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    
    # Model settings
    model_path: str = "ml_models/plant_disease_model.h5"
    confidence_threshold: float = 0.7
    
    # Weather API (optional)
    weather_api_key: Optional[str] = None
    weather_api_url: str = "http://api.openweathermap.org/data/2.5/weather"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("ml_models", exist_ok=True)