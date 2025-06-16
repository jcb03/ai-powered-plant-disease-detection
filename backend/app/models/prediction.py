"""
Plant disease prediction model and utilities.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import logging
from pathlib import Path
import json
import time

from ..core.config import settings
from ..utils.image_processing import preprocess_image
from ..utils.disease_info import get_disease_info

logger = logging.getLogger(__name__)

class PlantDiseasePredictor:
    """Plant disease prediction model wrapper."""
    
    def __init__(self):
        self.model = None
        self.class_names = []
        self.model_loaded = False
        self.model_info = {}
        self.load_model()
        
    def load_model(self):
        """Load the trained model and class names."""
        try:
            model_path = Path(settings.model_path)
            
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                self._create_dummy_model()
                return
            
            # Load the model
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(str(model_path))
            
            # Load class names
            self._load_class_names()
            
            # Get model information
            self._extract_model_info()
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._create_dummy_model()
    
    def _load_class_names(self):
        """Load class names from file or use default."""
        class_names_file = Path("ml_models/class_names.json")
        
        if class_names_file.exists():
            try:
                with open(class_names_file, 'r') as f:
                    self.class_names = json.load(f)
                logger.info(f"Loaded {len(self.class_names)} class names from file")
                return
            except Exception as e:
                logger.warning(f"Error loading class names file: {str(e)}")
        
        # Default class names (PlantVillage dataset)
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        logger.info(f"Using default class names: {len(self.class_names)} classes")
    
    def _extract_model_info(self):
        """Extract model information for API responses."""
        if self.model is None:
            return
            
        try:
            self.model_info = {
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "num_classes": len(self.class_names),
                "total_params": self.model.count_params(),
                "model_type": "CNN",
                "framework": "TensorFlow",
                "version": tf.__version__
            }
        except Exception as e:
            logger.error(f"Error extracting model info: {str(e)}")
            self.model_info = {}
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when real model is not available."""
        logger.warning("Creating dummy model for testing purposes")
        
        # Create a simple dummy model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(self.class_names) if self.class_names else 38, activation='softmax')
        ])
        
        self.model_loaded = False  # Mark as not properly loaded
        
        # Use default class names if not loaded
        if not self.class_names:
            self._load_class_names()
    
    def predict(self, image_bytes: bytes) -> Dict:
        """
        Predict plant disease from image bytes.
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        if self.model is None:
            return {
                "error": "Model not loaded",
                "prediction": None,
                "confidence": 0.0,
                "disease_info": {},
                "processing_time": 0.0
            }
        
        try:
            # Preprocess image
            processed_image = preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            if predicted_class_idx < len(self.class_names):
                predicted_class = self.class_names[predicted_class_idx]
            else:
                predicted_class = f"Unknown_Class_{predicted_class_idx}"
            
            # Get disease information
            disease_info = get_disease_info(predicted_class)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]  # Top 5
            top_predictions = []
            
            for idx in top_indices:
                if idx < len(self.class_names):
                    class_name = self.class_names[idx]
                    conf = float(predictions[0][idx])
                    top_predictions.append({
                        "class": class_name,
                        "disease_name": get_disease_info(class_name).get("disease_name", class_name),
                        "confidence": conf,
                        "percentage": round(conf * 100, 2)
                    })
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Determine confidence level
            is_confident = confidence > settings.confidence_threshold
            confidence_level = self._get_confidence_level(confidence)
            
            result = {
                "success": True,
                "prediction": predicted_class,
                "confidence": confidence,
                "confidence_percentage": round(confidence * 100, 2),
                "confidence_level": confidence_level,
                "is_confident": is_confident,
                "disease_info": disease_info,
                "top_predictions": top_predictions,
                "processing_time": round(processing_time, 3),
                "model_info": {
                    "model_loaded": self.model_loaded,
                    "num_classes": len(self.class_names),
                    "confidence_threshold": settings.confidence_threshold
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "prediction": None,
                "confidence": 0.0,
                "disease_info": {},
                "processing_time": time.time() - start_time
            }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def predict_batch(self, image_bytes_list: List[bytes]) -> List[Dict]:
        """
        Predict diseases for multiple images.
        
        Args:
            image_bytes_list: List of image bytes
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, image_bytes in enumerate(image_bytes_list):
            try:
                result = self.predict(image_bytes)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "success": False,
                    "error": str(e),
                    "prediction": None,
                    "confidence": 0.0
                })
        
        return results
    
    def get_model_summary(self) -> Dict:
        """Get model summary information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Get model architecture summary
            summary_lines = []
            self.model.summary(print_fn=lambda x: summary_lines.append(x))
            
            return {
                "model_info": self.model_info,
                "class_names": self.class_names,
                "num_classes": len(self.class_names),
                "model_loaded": self.model_loaded,
                "architecture_summary": "\n".join(summary_lines),
                "confidence_threshold": settings.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return {"error": str(e)}
    
    def get_supported_crops(self) -> List[str]:
        """Get list of supported crop types."""
        crops = set()
        for class_name in self.class_names:
            if '___' in class_name:
                crop = class_name.split('___')[0]
                # Clean up crop names
                crop = crop.replace('_', ' ').replace('(', '').replace(')', '')
                crops.add(crop.title())
        
        return sorted(list(crops))

# Global predictor instance
predictor = PlantDiseasePredictor()
