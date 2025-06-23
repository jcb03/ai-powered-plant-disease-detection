"""
Main FastAPI application for Plant Disease Detection API.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import time
import os
from pathlib import Path

from .core.config import settings
from .api.routes import router
from .models.prediction import predictor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app - FORCE docs to be enabled
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered plant disease detection system for farmers and gardeners",
    debug=True,  # FORCE debug mode for now
    docs_url="/docs",  # ALWAYS enable docs
    redoc_url="/redoc"  # ALWAYS enable redoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add trusted host middleware for security (only in production)
#if not settings.debug and hasattr(settings, 'debug') and not settings.debug:
#    app.add_middleware(
#        TrustedHostMiddleware,
#        allowed_hosts=["localhost", "127.0.0.1", "*.herokuapp.com", "*.render.com"]
#    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

# Include API routes - FIXED: Use explicit version
app.include_router(router, prefix="/api/v1", tags=["Plant Disease Detection"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "api_version": "v1",  # EXPLICIT version
        "docs_url": "/docs",
        "model_status": "loaded" if predictor.model_loaded else "not_loaded",
        "supported_classes": len(predictor.class_names),
        "timestamp": time.time(),
        "available_endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict", 
            "debug_prediction": "/api/v1/debug-prediction",
            "test_preprocessing": "/api/v1/test-preprocessing",
            "model_debug": "/api/v1/model/debug",
            "classes": "/api/v1/classes",
            "model_info": "/api/v1/model/info"
        }
    }

# Health check endpoint (duplicate for load balancers)
@app.get("/health")
async def health():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": predictor.model_loaded
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: True (forced for development)")
    logger.info(f"Model loaded: {predictor.model_loaded}")
    logger.info(f"Supported classes: {len(predictor.class_names)}")
    logger.info(f"API docs available at: /docs")
    logger.info(f"API endpoints available at: /api/v1/")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("ml_models", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down application")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # EXPLICIT host
        port=8000,       # EXPLICIT port
        reload=True,     # FORCE reload for development
        log_level="info" # EXPLICIT log level
    )
