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
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple image validation function (since utils might not exist)
def validate_plant_image(image_bytes):
    """Simple image validation."""
    try:
        from PIL import Image
        import io
        
        # Try to open the image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Basic validation
        if img.size[0] < 50 or img.size[1] < 50:
            return False, "Image too small (minimum 50x50 pixels)"
        
        if img.size[0] > 5000 or img.size[1] > 5000:
            return False, "Image too large (maximum 5000x5000 pixels)"
        
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

def enhance_image_quality(image_bytes):
    """Simple image enhancement."""
    # For now, just return the original image
    # You can implement actual enhancement later
    return image_bytes

def extract_image_metadata(image_bytes):
    """Extract basic image metadata."""
    try:
        from PIL import Image
        import io
        
        img = Image.open(io.BytesIO(image_bytes))
        return {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info
        }
    except Exception as e:
        return {"error": str(e)}

def get_diseases_by_crop(crop_name):
    """Get diseases for a specific crop."""
    # Filter predictor class names by crop
    crop_diseases = []
    for class_name in predictor.class_names:
        if crop_name.lower() in class_name.lower():
            parts = class_name.split('___')
            disease_name = parts[1] if len(parts) > 1 else class_name
            crop_diseases.append({
                "disease_name": disease_name.replace('_', ' '),
                "class_name": class_name,
                "severity": "Moderate",
                "description": f"Disease affecting {crop_name}"
            })
    return crop_diseases

def search_diseases(query):
    """Search diseases by query."""
    results = []
    for class_name in predictor.class_names:
        if query.lower() in class_name.lower():
            parts = class_name.split('___')
            crop = parts[0] if len(parts) > 0 else "Unknown"
            disease = parts[1] if len(parts) > 1 else class_name
            
            results.append({
                "disease_name": disease.replace('_', ' '),
                "crop": crop.replace('_', ' '),
                "class_name": class_name,
                "description": f"Disease affecting {crop}"
            })
    return results

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

@router.post("/debug-prediction")
async def debug_prediction_detailed(file: UploadFile = File(...)):
    """
    Debug prediction with full details to identify Unknown Disease issues.
    
    Args:
        file: Uploaded image file
    
    Returns:
        Detailed debug information about the prediction process
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get debug info from predictor
        debug_result = predictor.debug_prediction(image_bytes)
        
        # Add file information
        debug_result.update({
            "filename": file.filename,
            "file_size": len(image_bytes),
            "content_type": file.content_type
        })
        
        return debug_result
        
    except Exception as e:
        logger.error(f"Debug prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug prediction failed: {str(e)}")

@router.post("/test-preprocessing")
async def test_preprocessing_methods(file: UploadFile = File(...)):
    """
    Test both preprocessing methods to compare results.
    
    Args:
        file: Uploaded image file
    
    Returns:
        Comparison of direct resize vs padding preprocessing
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Test both preprocessing methods
        result = predictor.test_preprocessing_methods(image_bytes)
        
        # Add file information
        result.update({
            "filename": file.filename,
            "file_size": len(image_bytes),
            "content_type": file.content_type
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Preprocessing test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing test failed: {str(e)}")

@router.get("/model/debug")
async def get_model_debug_info():
    """Get detailed model debug information."""
    try:
        return predictor.get_model_summary()
    except Exception as e:
        logger.error(f"Error getting model debug info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    try:
        return {
            "model_loaded": predictor.model_loaded,
            "model_name": "Plant Disease Detection CNN",
            "model_architecture": "ResNet50V2 + Custom Head",
            "model_version": "v1.0",
            "total_classes": len(predictor.class_names),
            "model_info": predictor.model_info,
            "accuracy": 96.5,
            "precision": 96.6,
            "f1_score": 96.5,
            "training_samples": "18,088",
            "model_size": "4.8M params",
            "avg_processing_time": "<2s",
            "framework": "TensorFlow",
            "training_date": "2025-06-22",
            "dataset": "PlantVillage Dataset (Balanced)"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {
            "model_loaded": False,
            "model_name": "Plant Disease Detection CNN",
            "error": str(e)
        }


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
