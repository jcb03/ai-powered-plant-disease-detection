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
from ..utils.image_processing import preprocess_image, preprocess_image_with_padding
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
            
            # Load class names FIRST
            self._load_class_names()
            
            # CRITICAL: Test model immediately after loading
            self._test_model_loading()
            
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
                
                # DEBUG: Print first few class names
                logger.info(f"First 5 classes: {self.class_names[:5]}")
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
    
    def _test_model_loading(self):
        """Test model immediately after loading to catch issues early."""
        if self.model is None:
            logger.error("Model is None!")
            return
        
        try:
            # Test with dummy input
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_predictions = self.model.predict(dummy_input, verbose=0)
            
            logger.info(f"Model test successful!")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            logger.info(f"Test prediction shape: {test_predictions.shape}")
            logger.info(f"Number of classes in model: {test_predictions.shape[1]}")
            logger.info(f"Number of class names loaded: {len(self.class_names)}")
            
            # CRITICAL CHECK: Verify class count matches
            if test_predictions.shape[1] != len(self.class_names):
                logger.error(f"MISMATCH: Model outputs {test_predictions.shape[1]} classes but we have {len(self.class_names)} class names!")
                logger.error("This will cause 'Unknown Disease' predictions!")
                
                # Try to fix by truncating or padding class names
                if test_predictions.shape[1] < len(self.class_names):
                    logger.warning(f"Truncating class names from {len(self.class_names)} to {test_predictions.shape[1]}")
                    self.class_names = self.class_names[:test_predictions.shape[1]]
                else:
                    logger.warning(f"Model has more classes than names. Adding placeholder names.")
                    for i in range(len(self.class_names), test_predictions.shape[1]):
                        self.class_names.append(f"Unknown_Class_{i}")
            
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
    
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
        
        # Use default class names if not loaded
        if not self.class_names:
            self._load_class_names()
        
        # Create a simple dummy model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model_loaded = False  # Mark as not properly loaded

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
                "prediction": "Unknown Disease",
                "confidence": 0.0,
                "disease_info": {},
                "processing_time": 0.0
            }
        
        try:
            # CRITICAL: Use the correct preprocessing function
            processed_image = preprocess_image(image_bytes, target_size=(224, 224))
            
            # DEBUG: Log preprocessing info
            logger.debug(f"Processed image shape: {processed_image.shape}")
            logger.debug(f"Processed image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # DEBUG: Log prediction info
            logger.debug(f"Predicted class index: {predicted_class_idx}")
            logger.debug(f"Confidence: {confidence:.3f}")
            logger.debug(f"Available classes: {len(self.class_names)}")
            
            # Get class name with bounds checking
            if predicted_class_idx < len(self.class_names):
                predicted_class = self.class_names[predicted_class_idx]
            else:
                logger.error(f"Predicted class index {predicted_class_idx} is out of bounds for {len(self.class_names)} classes!")
                predicted_class = f"Unknown_Class_{predicted_class_idx}"
            
            # Get disease information
            disease_info = get_disease_info(predicted_class)
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]
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
                },
                # DEBUG INFO
                "debug_info": {
                    "predicted_index": int(predicted_class_idx),
                    "total_classes": len(self.class_names),
                    "preprocessing_shape": processed_image.shape,
                    "prediction_shape": predictions.shape
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "prediction": "Unknown Disease",
                "confidence": 0.0,
                "disease_info": {},
                "processing_time": time.time() - start_time
            }
    
    def test_preprocessing_methods(self, image_bytes: bytes) -> Dict:
        """Test both preprocessing methods to see which works better."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Method 1: Direct resize (should match training)
            processed_direct = preprocess_image(image_bytes)
            predictions_direct = self.model.predict(processed_direct, verbose=0)
            
            # Method 2: With padding (current method)
            processed_padding = preprocess_image_with_padding(image_bytes)
            predictions_padding = self.model.predict(processed_padding, verbose=0)
            
            # Compare results
            direct_class_idx = np.argmax(predictions_direct[0])
            padding_class_idx = np.argmax(predictions_padding[0])
            
            direct_conf = float(predictions_direct[0][direct_class_idx])
            padding_conf = float(predictions_padding[0][padding_class_idx])
            
            direct_class = self.class_names[direct_class_idx] if direct_class_idx < len(self.class_names) else f"Unknown_{direct_class_idx}"
            padding_class = self.class_names[padding_class_idx] if padding_class_idx < len(self.class_names) else f"Unknown_{padding_class_idx}"
            
            logger.info(f"Direct resize: {direct_class} ({direct_conf:.3f})")
            logger.info(f"With padding: {padding_class} ({padding_conf:.3f})")
            
            return {
                "direct_resize": {
                    "class": direct_class,
                    "confidence": direct_conf,
                    "class_index": int(direct_class_idx)
                },
                "with_padding": {
                    "class": padding_class,
                    "confidence": padding_conf,
                    "class_index": int(padding_class_idx)
                },
                "recommendation": "direct_resize" if direct_conf > padding_conf else "with_padding"
            }
            
        except Exception as e:
            logger.error(f"Preprocessing test failed: {str(e)}")
            return {"error": str(e)}
    
    def debug_prediction(self, image_bytes: bytes) -> Dict:
        """Debug prediction with detailed information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Test preprocessing
            processed_image = preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get detailed info
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get top 10 predictions for debugging
            top_indices = np.argsort(predictions[0])[-10:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                if idx < len(self.class_names):
                    class_name = self.class_names[idx]
                    conf = float(predictions[0][idx])
                    top_predictions.append({
                        "index": int(idx),
                        "class": class_name,
                        "confidence": conf,
                        "percentage": round(conf * 100, 2)
                    })
            
            return {
                "model_info": {
                    "input_shape": self.model.input_shape,
                    "output_shape": self.model.output_shape,
                    "total_classes": len(self.class_names)
                },
                "preprocessing_info": {
                    "processed_shape": processed_image.shape,
                    "processed_range": [float(processed_image.min()), float(processed_image.max())],
                    "processed_mean": float(processed_image.mean())
                },
                "prediction_info": {
                    "prediction_shape": predictions.shape,
                    "predicted_index": int(predicted_class_idx),
                    "predicted_class": self.class_names[predicted_class_idx] if predicted_class_idx < len(self.class_names) else f"Unknown_{predicted_class_idx}",
                    "confidence": confidence
                },
                "top_predictions": top_predictions
            }
            
        except Exception as e:
            logger.error(f"Debug prediction failed: {str(e)}")
            return {"error": str(e)}
    
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
                    "prediction": "Unknown Disease",
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
