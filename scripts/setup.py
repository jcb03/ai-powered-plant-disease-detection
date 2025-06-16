"""
Setup script for Plant Disease Detection System.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import urllib.request
import zipfile
import json
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemSetup:
    """System setup and installation utilities."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.python_version = sys.version_info
        self.platform = platform.system()
        
    def check_system_requirements(self):
        """Check system requirements."""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check available disk space
        if not self.check_disk_space():
            return False
        
        # Check internet connection
        if not self.check_internet_connection():
            return False
        
        logger.info("✅ All system requirements met!")
        return True
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        if self.python_version.major != 3 or self.python_version.minor < 11:
            logger.error(f"❌ Python 3.11+ required. Current version: {self.python_version.major}.{self.python_version.minor}")
            return False
        logger.info(f"✅ Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        return True
    
    def check_disk_space(self, required_gb=10):
        """Check available disk space."""
        try:
            if self.platform == "Windows":
                import shutil
                free_bytes = shutil.disk_usage(self.project_root).free
            else:
                stat = os.statvfs(self.project_root)
                free_bytes = stat.f_bavail * stat.f_frsize
            
            free_gb = free_bytes / (1024**3)
            
            if free_gb < required_gb:
                logger.error(f"❌ Insufficient disk space. Required: {required_gb}GB, Available: {free_gb:.1f}GB")
                return False
            
            logger.info(f"✅ Disk space: {free_gb:.1f}GB available")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not check disk space: {e}")
            return True
    
    def check_internet_connection(self):
        """Check internet connectivity."""
        try:
            urllib.request.urlopen('https://www.google.com', timeout=5)
            logger.info("✅ Internet connection available")
            return True
        except Exception:
            logger.error("❌ No internet connection. Some features may not work.")
            return False
    
    def setup_directories(self):
        """Create necessary directories."""
        logger.info("Setting up project directories...")
        
        directories = [
            "backend/ml_models",
            "backend/logs",
            "backend/temp",
            "frontend/assets",
            "model_training/outputs",
            "model_training/datasets",
            "scripts/logs",
            "temp",
            "data"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
    
    def install_dependencies(self):
        """Install Python dependencies for all components."""
        logger.info("Installing dependencies...")
        
        components = [
            ("backend", "Backend API"),
            ("frontend", "Frontend UI"),
            ("model_training", "Model Training")
        ]
        
        for component, description in components:
            logger.info(f"Installing {description} dependencies...")
            
            requirements_file = self.project_root / component / "requirements.txt"
            
            if not requirements_file.exists():
                logger.warning(f"⚠️ Requirements file not found: {requirements_file}")
                continue
            
            if not self.run_command(
                f"pip install -r {requirements_file}",
                f"Installing {description} dependencies"
            ):
                logger.error(f"❌ Failed to install {description} dependencies")
                return False
        
        logger.info("✅ All dependencies installed successfully!")
        return True
    
    def create_sample_model(self):
        """Create a sample model for testing."""
        logger.info("Creating sample model for testing...")
        
        model_script = '''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from pathlib import Path

# Create a simple model for testing
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(38, activation='softmax')  # 38 classes for PlantVillage
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save the model
model_dir = Path("backend/ml_models")
model_dir.mkdir(parents=True, exist_ok=True)
model.save(str(model_dir / "plant_disease_model.h5"))

# Create class names
class_names = [
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

# Save class names
with open(model_dir / "class_names.json", 'w') as f:
    json.dump(class_names, f, indent=2)

print("Sample model created successfully!")
'''
        
        # Write and execute the script
        script_path = self.project_root / "temp" / "create_sample_model.py"
        with open(script_path, 'w') as f:
            f.write(model_script)
        
        if self.run_command(f"python {script_path}", "Creating sample model"):
            logger.info("✅ Sample model created successfully!")
            script_path.unlink()  # Clean up
            return True
        else:
            logger.error("❌ Failed to create sample model")
            return False
    
    def create_env_files(self):
        """Create environment configuration files."""
        logger.info("Creating environment configuration files...")
        
        # Backend .env file
        backend_env = '''# Backend Configuration
DEBUG=true
API_VERSION=v1
CONFIDENCE_THRESHOLD=0.7

# CORS Settings
ALLOWED_ORIGINS=["http://localhost:8501", "http://127.0.0.1:8501"]

# Model Settings
MODEL_PATH=ml_models/plant_disease_model.h5

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Security
SECRET_KEY=your-secret-key-change-this-in-production
'''
        
        backend_env_path = self.project_root / "backend" / ".env"
        with open(backend_env_path, 'w') as f:
            f.write(backend_env)
        
        # Frontend secrets
        frontend_secrets = '''# Streamlit Secrets
API_URL = "http://localhost:8000"
'''
        
        streamlit_dir = self.project_root / "frontend" / ".streamlit"
        streamlit_dir.mkdir(exist_ok=True)
        
        secrets_path = streamlit_dir / "secrets.toml"
        with open(secrets_path, 'w') as f:
            f.write(frontend_secrets)
        
        logger.info("✅ Environment files created!")
    
    def run_command(self, command, description=""):
        """Run a command and handle errors."""
        logger.info(f"Running: {description or command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            if result.stdout:
                logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            return False
    
    def test_installation(self):
        """Test the installation."""
        logger.info("Testing installation...")
        
        # Test backend imports
        test_backend = '''
try:
    import fastapi
    import tensorflow as tf
    import uvicorn
    print("✅ Backend dependencies OK")
except ImportError as e:
    print(f"❌ Backend import error: {e}")
    exit(1)
'''
        
        # Test frontend imports
        test_frontend = '''
try:
    import streamlit
    import plotly
    import requests
    print("✅ Frontend dependencies OK")
except ImportError as e:
    print(f"❌ Frontend import error: {e}")
    exit(1)
'''
        
        # Write test scripts
        backend_test_path = self.project_root / "temp" / "test_backend.py"
        frontend_test_path = self.project_root / "temp" / "test_frontend.py"
        
        with open(backend_test_path, 'w') as f:
            f.write(test_backend)
        
        with open(frontend_test_path, 'w') as f:
            f.write(test_frontend)
        
        # Run tests
        backend_ok = self.run_command(f"python {backend_test_path}", "Testing backend")
        frontend_ok = self.run_command(f"python {frontend_test_path}", "Testing frontend")
        
        # Clean up
        backend_test_path.unlink()
        frontend_test_path.unlink()
        
        if backend_ok and frontend_ok:
            logger.info("✅ Installation test passed!")
            return True
        else:
            logger.error("❌ Installation test failed!")
            return False
    
    def setup_complete(self):
        """Complete setup process."""
        logger.info("🎉 Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start the backend server:")
        logger.info("   cd backend && python -m uvicorn app.main:app --reload")
        logger.info("\n2. Start the frontend (in another terminal):")
        logger.info("   cd frontend && streamlit run app.py")
        logger.info("\n3. Open your browser to: http://localhost:8501")
        logger.info("\n4. To train a model:")
        logger.info("   cd model_training && python train_model.py --data_dir datasets/your_data")

def main():
    """Main setup function."""
    print("🌱 Plant Disease Detection System Setup")
    print("=" * 50)
    
    setup = SystemSetup()
    
    try:
        # Check system requirements
        if not setup.check_system_requirements():
            logger.error("❌ System requirements not met. Please fix the issues and try again.")
            return False
        
        # Setup directories
        setup.setup_directories()
        
        # Install dependencies
        if not setup.install_dependencies():
            logger.error("❌ Failed to install dependencies.")
            return False
        
        # Create sample model
        if not setup.create_sample_model():
            logger.error("❌ Failed to create sample model.")
            return False
        
        # Create environment files
        setup.create_env_files()
        
        # Test installation
        if not setup.test_installation():
            logger.error("❌ Installation test failed.")
            return False
        
        # Setup complete
        setup.setup_complete()
        return True
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Setup interrupted by user.")
        return False
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
