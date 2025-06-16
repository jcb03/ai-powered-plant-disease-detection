"""
API routes for plant disease detection.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from pathlib import Path
import time

from ..models.prediction import predictor
from ..utils.image_processing import validate_plant_image, extract_image_metadata, enhance_image_quality
from ..utils.disease_info import get_diseases_by_crop, search_diseases
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if predictor.model_loaded else "not_loaded"
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_status": model_status,
        "version": settings.api_version,
        "app_name": settings.app_name,
        "supported_classes": len(predictor.class_names)
    }

@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    enhance_image: bool = Form(default=True),
    include_metadata: bool = Form(default=False)
):
    """
    Predict plant disease from uploaded image.
    
    Args:
        file: Uploaded image file
        enhance_image: Whether to enhance image quality before prediction
        include_metadata: Whether to include image metadata in response
    
    Returns:
        Prediction results with disease information
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(settings.allowed_file_types)}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Validate file size
        if len(image_bytes) > settings.max_file_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {settings.max_file_size // (1024*1024)}MB"
            )
        
        # Validate if it's a plant image
        is_valid, validation_message = validate_plant_image(image_bytes)
        
        # Enhance image quality if requested
        if enhance_image:
            try:
                image_bytes = enhance_image_quality(image_bytes)
            except Exception as e:
                logger.warning(f"Image enhancement failed: {str(e)}")
                # Continue with original image if enhancement fails
        
        # Make prediction
        result = predictor.predict(image_bytes)
        
        # Add file information
        result.update({
            "filename": file.filename,
            "file_size": len(image_bytes),
            "content_type": file.content_type,
            "image_enhanced": enhance_image,
            "validation": {
                "is_valid": is_valid,
                "message": validation_message
            }
        })
        
        # Add metadata if requested
        if include_metadata:
            try:
                metadata = extract_image_metadata(image_bytes)
                result["image_metadata"] = metadata
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {str(e)}")
                result["image_metadata"] = {"error": str(e)}
        
        # Add warning for invalid images
        if not is_valid:
            result["warning"] = validation_message
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch")
async def batch_predict(
    files: List[UploadFile] = File(...),
    enhance_images: bool = Form(default=True),
    max_files: int = Query(default=10, le=20)
):
    """
    Predict diseases for multiple images.
    
    Args:
        files: List of uploaded image files
        enhance_images: Whether to enhance image quality before prediction
        max_files: Maximum number of files to process
    
    Returns:
        List of prediction results
    """
    if len(files) > max_files:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Maximum {max_files} files allowed per batch"
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                results.append({
                    "batch_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Read and validate file
            image_bytes = await file.read()
            
            if len(image_bytes) > settings.max_file_size:
                results.append({
                    "batch_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": "File too large"
                })
                continue
            
            # Validate plant image
            is_valid, validation_message = validate_plant_image(image_bytes)
            
            # Enhance image if requested
            if enhance_images:
                try:
                    image_bytes = enhance_image_quality(image_bytes)
                except Exception as e:
                    logger.warning(f"Enhancement failed for {file.filename}: {str(e)}")
            
            # Make prediction
            result = predictor.predict(image_bytes)
            result.update({
                "batch_index": i,
                "filename": file.filename,
                "file_size": len(image_bytes),
                "validation": {
                    "is_valid": is_valid,
                    "message": validation_message
                }
            })
            
            if not is_valid:
                result["warning"] = validation_message
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Batch prediction failed for file {i}: {str(e)}")
            results.append({
                "batch_index": i,
                "filename": file.filename if file else f"file_{i}",
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful_predictions": sum(1 for r in results if r.get("success", True)),
        "failed_predictions": sum(1 for r in results if not r.get("success", True))
    }

@router.get("/classes")
async def get_supported_classes():
    """Get list of supported plant disease classes."""
    return {
        "classes": predictor.class_names,
        "total_classes": len(predictor.class_names),
        "crops": predictor.get_supported_crops(),
        "model_loaded": predictor.model_loaded
    }

@router.get("/model/info")
async def get_model_info():
    """Get detailed model information."""
    return predictor.get_model_summary()

@router.get("/diseases/crop/{crop_name}")
async def get_diseases_by_crop_name(crop_name: str):
    """Get all diseases for a specific crop."""
    diseases = get_diseases_by_crop(crop_name)
    
    if not diseases:
        raise HTTPException(
            status_code=404, 
            detail=f"No diseases found for crop: {crop_name}"
        )
    
    return {
        "crop": crop_name,
        "diseases": diseases,
        "total_diseases": len(diseases)
    }

@router.get("/diseases/search")
async def search_diseases_endpoint(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(default=10, le=50, description="Maximum results to return")
):
    """Search diseases by name, symptoms, or crop."""
    results = search_diseases(q)
    
    # Limit results
    if limit < len(results):
        results = results[:limit]
    
    return {
        "query": q,
        "results": results,
        "total_found": len(results)
    }

@router.get("/stats")
async def get_api_stats():
    """Get API usage statistics."""
    return {
        "model_status": "loaded" if predictor.model_loaded else "not_loaded",
        "supported_classes": len(predictor.class_names),
        "supported_crops": len(predictor.get_supported_crops()),
        "confidence_threshold": settings.confidence_threshold,
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
        "allowed_file_types": settings.allowed_file_types,
        "tensorflow_version": predictor.model_info.get("version", "unknown") if predictor.model_info else "unknown"
    }

@router.post("/feedback")
async def submit_feedback(
    prediction_id: str = Form(...),
    actual_disease: str = Form(...),
    feedback_type: str = Form(...),  # correct, incorrect, partial
    comments: Optional[str] = Form(default=None)
):
    """Submit feedback on prediction results."""
    # In a real application, you would store this in a database
    # For now, we'll just log it
    
    feedback_data = {
        "prediction_id": prediction_id,
        "actual_disease": actual_disease,
        "feedback_type": feedback_type,
        "comments": comments,
        "timestamp": time.time()
    }
    
    logger.info(f"Feedback received: {feedback_data}")
    
    return {
        "message": "Feedback submitted successfully",
        "feedback_id": f"fb_{int(time.time())}",
        "status": "received"
    }
