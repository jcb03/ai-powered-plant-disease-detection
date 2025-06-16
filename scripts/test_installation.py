"""
Test installation and functionality of Plant Disease Detection System.
"""

import sys
import subprocess
import time
import requests
import json
from pathlib import Path
import logging
from PIL import Image
import io
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstallationTester:
    """Test installation and functionality."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:8501"
        
    def test_imports(self):
        """Test if all required packages can be imported."""
        logger.info("Testing package imports...")
        
        # Backend imports
        backend_imports = [
            "fastapi",
            "uvicorn", 
            "tensorflow",
            "numpy",
            "pillow",
            "opencv-python"
        ]
        
        # Frontend imports
        frontend_imports = [
            "streamlit",
            "plotly",
            "requests",
            "pandas"
        ]
        
        all_imports = backend_imports + frontend_imports
        failed_imports = []
        
        for package in all_imports:
            try:
                # Handle special cases
                if package == "opencv-python":
                    import cv2
                elif package == "pillow":
                    from PIL import Image
                else:
                    __import__(package)
                logger.info(f"✅ {package}")
            except ImportError as e:
                logger.error(f"❌ {package}: {e}")
                failed_imports.append(package)
        
        if failed_imports:
            logger.error(f"❌ Failed imports: {failed_imports}")
            return False
        
        logger.info("✅ All imports successful!")
        return True
    
    def test_model_loading(self):
        """Test if the model can be loaded."""
        logger.info("Testing model loading...")
        
        try:
            import tensorflow as tf
            
            model_path = self.project_root / "backend" / "ml_models" / "plant_disease_model.h5"
            
            if not model_path.exists():
                logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Load model
            model = tf.keras.models.load_model(str(model_path))
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Input shape: {model.input_shape}")
            logger.info(f"   Output shape: {model.output_shape}")
            logger.info(f"   Parameters: {model.count_params():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def test_backend_startup(self):
        """Test if backend can start up."""
        logger.info("Testing backend startup...")
        
        try:
            # Start backend in background
            backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"
            ], cwd=self.project_root / "backend", 
               stdout=subprocess.PIPE, 
               stderr=subprocess.PIPE)
            
            # Wait for startup
            logger.info("Waiting for backend to start...")
            time.sleep(10)
            
            # Test health endpoint
            try:
                response = requests.get(f"{self.backend_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Backend started successfully!")
                    logger.info(f"   Health check: {response.json()}")
                    
                    # Test API endpoints
                    self.test_api_endpoints()
                    
                    backend_process.terminate()
                    return True
                else:
                    logger.error(f"❌ Backend health check failed: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Backend connection failed: {e}")
            
            backend_process.terminate()
            return False
            
        except Exception as e:
            logger.error(f"❌ Backend startup failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints."""
        logger.info("Testing API endpoints...")
        
        endpoints = [
            ("/", "Root endpoint"),
            ("/health", "Health check"),
            ("/api/v1/classes", "Supported classes"),
            ("/api/v1/stats", "API statistics")
        ]
        
        for endpoint, description in endpoints:
            try:
                response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {description}: OK")
                else:
                    logger.warning(f"⚠️ {description}: {response.status_code}")
            except Exception as e:
                logger.error(f"❌ {description}: {e}")
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint with sample image."""
        logger.info("Testing prediction endpoint...")
        
        try:
            # Create a sample image
            sample_image = Image.new('RGB', (224, 224), color='green')
            img_bytes = io.BytesIO()
            sample_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Test prediction
            files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
            response = requests.post(
                f"{self.backend_url}/api/v1/predict",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Prediction endpoint working!")
                logger.info(f"   Prediction: {result.get('prediction', 'N/A')}")
                logger.info(f"   Confidence: {result.get('confidence', 0):.1%}")
                return True
            else:
                logger.error(f"❌ Prediction failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Prediction test failed: {e}")
            return False
    
    def test_frontend_startup(self):
        """Test if frontend can start up."""
        logger.info("Testing frontend startup...")
        
        try:
            # Start frontend in background
            frontend_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "app.py", 
                "--server.port", "8501", "--server.headless", "true"
            ], cwd=self.project_root / "frontend",
               stdout=subprocess.PIPE,
               stderr=subprocess.PIPE)
            
            # Wait for startup
            logger.info("Waiting for frontend to start...")
            time.sleep(15)
            
            # Test frontend accessibility
            try:
                response = requests.get(self.frontend_url, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Frontend started successfully!")
                    frontend_process.terminate()
                    return True
                else:
                    logger.error(f"❌ Frontend not accessible: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Frontend connection failed: {e}")
            
            frontend_process.terminate()
            return False
            
        except Exception as e:
            logger.error(f"❌ Frontend startup failed: {e}")
            return False
    
    def test_file_structure(self):
        """Test if all required files exist."""
        logger.info("Testing file structure...")
        
        required_files = [
            "backend/app/main.py",
            "backend/app/models/prediction.py",
            "backend/app/utils/image_processing.py",
            "backend/app/utils/disease_info.py",
            "backend/requirements.txt",
            "frontend/app.py",
            "frontend/utils/api_client.py",
            "frontend/utils/ui_components.py",
            "frontend/requirements.txt",
            "model_training/train_model.py",
            "model_training/data_preprocessing.py",
            "scripts/setup.py"
        ]
        
        missing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path}")
            else:
                logger.error(f"❌ {file_path}")
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Missing files: {missing_files}")
            return False
        
        logger.info("✅ All required files present!")
        return True
    
    def test_docker_setup(self):
        """Test Docker configuration."""
        logger.info("Testing Docker setup...")
        
        docker_files = [
            "backend/Dockerfile",
            "frontend/Dockerfile",
            "docker-compose.yml"
        ]
        
        missing_docker_files = []
        
        for file_path in docker_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path}")
            else:
                logger.error(f"❌ {file_path}")
                missing_docker_files.append(file_path)
        
        if missing_docker_files:
            logger.warning(f"⚠️ Missing Docker files: {missing_docker_files}")
            return False
        
        # Test if Docker is available
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Docker available: {result.stdout.strip()}")
                return True
            else:
                logger.warning("⚠️ Docker not available")
                return False
        except FileNotFoundError:
            logger.warning("⚠️ Docker not installed")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive installation test."""
        logger.info("🧪 Running comprehensive installation test...")
        logger.info("=" * 60)
        
        tests = [
            ("File Structure", self.test_file_structure),
            ("Package Imports", self.test_imports),
            ("Model Loading", self.test_model_loading),
            ("Docker Setup", self.test_docker_setup),
            # Note: Backend and frontend tests require manual startup
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running {test_name} test...")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name:20} {status}")
        
        logger.info("-" * 60)
        logger.info(f"TOTAL: {passed}/{total} tests passed ({passed/total:.1%})")
        
        if passed == total:
            logger.info("🎉 All tests passed! Installation is successful.")
            logger.info("\nNext steps:")
            logger.info("1. Start backend: cd backend && python -m uvicorn app.main:app --reload")
            logger.info("2. Start frontend: cd frontend && streamlit run app.py")
            return True
        else:
            logger.error("❌ Some tests failed. Please check the issues above.")
            return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Plant Disease Detection System installation')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--backend', action='store_true', help='Test backend startup')
    parser.add_argument('--frontend', action='store_true', help='Test frontend startup')
    parser.add_argument('--prediction', action='store_true', help='Test prediction endpoint')
    
    args = parser.parse_args()
    
    tester = InstallationTester()
    
    if args.quick:
        # Quick tests
        success = (
            tester.test_file_structure() and
            tester.test_imports() and
            tester.test_model_loading()
        )
    elif args.backend:
        success = tester.test_backend_startup()
    elif args.frontend:
        success = tester.test_frontend_startup()
    elif args.prediction:
        success = tester.test_prediction_endpoint()
    else:
        # Comprehensive test
        success = tester.run_comprehensive_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
