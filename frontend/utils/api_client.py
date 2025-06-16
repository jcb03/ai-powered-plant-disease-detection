"""
API client for communicating with the backend service.
"""

import requests
import streamlit as st
from typing import Dict, List, Optional
import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class APIClient:
    """Client for Plant Disease Detection API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
    
    def health_check(self) -> Dict:
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return {"healthy": True, **response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"healthy": False, "error": str(e)}
    
    def predict_disease(self, image_file, enhance_image: bool = True, include_metadata: bool = False) -> Dict:
        """Send image to API for disease prediction."""
        try:
            if hasattr(image_file, 'read'):
                files = {"file": image_file}
            else:
                files = {"file": ("image.jpg", image_file, "image/jpeg")}
            
            data = {
                "enhance_image": enhance_image,
                "include_metadata": include_metadata
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/predict",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Prediction request failed: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}
    
    def predict_batch(self, image_files: List, enhance_images: bool = True) -> Dict:
        """Send multiple images for batch prediction."""
        try:
            files = [("files", file) for file in image_files]
            data = {"enhance_images": enhance_images}
            
            response = self.session.post(
                f"{self.base_url}/api/v1/predict/batch",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return {"error": f"Batch prediction failed: {str(e)}"}
    
    def get_supported_classes(self) -> Dict:
        """Get list of supported disease classes."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/classes")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get classes: {str(e)}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict:
        """Get detailed model information."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}
    
    def get_diseases_by_crop(self, crop_name: str) -> Dict:
        """Get diseases for a specific crop."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/diseases/crop/{crop_name}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get diseases for crop {crop_name}: {str(e)}")
            return {"error": str(e)}
    
    def search_diseases(self, query: str) -> Dict:
        """Search diseases by query."""
        try:
            params = {"q": query}
            response = self.session.get(f"{self.base_url}/api/v1/diseases/search", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Disease search failed: {str(e)}")
            return {"error": str(e)}
    
    def get_api_stats(self) -> Dict:
        """Get API statistics."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get API stats: {str(e)}")
            return {"error": str(e)}

@st.cache_resource
def get_api_client() -> APIClient:
    """Get configured API client with caching."""
    api_url = st.secrets.get("API_URL", "http://localhost:8000")
    return APIClient(api_url)
