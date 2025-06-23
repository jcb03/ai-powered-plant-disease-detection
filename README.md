<div align="center">

# 🌱 AI-Powered Plant Disease Detection

### *Empowering Farmers with Intelligent Crop Health Solutions*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-96.5%25-brightgreen.svg)](https://github.com/jcb03/ai-powered-plant-disease-detection)

[🚀 Live Demo](https://your-demo-link.streamlit.app) • [📖 Documentation](https://github.com/jcb03/ai-powered-plant-disease-detection/wiki) • [🐛 Report Bug](https://github.com/jcb03/ai-powered-plant-disease-detection/issues) • [💡 Request Feature](https://github.com/jcb03/ai-powered-plant-disease-detection/issues)

</div>

---

## 🌟 Overview

**AI-Powered Plant Disease Detection** is a cutting-edge machine learning solution that helps farmers and agricultural professionals identify plant diseases instantly through image analysis. Built with state-of-the-art deep learning techniques, this system achieves **96.5% accuracy** across 38 different plant disease classes.

### 🎯 Problem Statement

Farmers worldwide lose **billions of dollars annually** due to:
- Undetected crop diseases
- Lack of access to agricultural experts
- Delayed diagnosis leading to crop failure
- Limited knowledge about treatment options

### 💡 Our Solution

A comprehensive AI system that provides:
- **Instant disease diagnosis** from plant leaf images
- **Treatment recommendations** with organic alternatives
- **Prevention strategies** to avoid future outbreaks
- **Mobile-friendly interface** accessible anywhere

---

## ✨ Key Features

<div align="center">

| 🔬 **AI-Powered Detection** | 📱 **Mobile-Friendly** | 🌿 **Comprehensive Database** |
|:---:|:---:|:---:|
| 96.5% accuracy across 38 diseases | Responsive web interface | 14 crop types supported |
| ResNet50V2 + Custom CNN | Camera integration | Detailed treatment guides |
| Real-time predictions | Offline capability | Organic solutions included |

</div>

### 🚀 Core Capabilities

- **🔍 Instant Disease Detection**: Upload or capture plant images for immediate analysis
- **🎯 High Accuracy**: 96.5% validation accuracy with 96.6% precision
- **🌾 Multi-Crop Support**: Covers 14 major crops including tomatoes, apples, grapes, corn, and more
- **💊 Treatment Recommendations**: Detailed treatment plans with organic alternatives
- **📊 Batch Processing**: Analyze multiple images simultaneously
- **📈 Analytics Dashboard**: Track prediction history and performance metrics
- **🔄 Real-time Processing**: Get results in under 2 seconds

---

## 🏗️ System Architecture

<div align="center">

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 📱 Frontend │────│ 🔗 API │────│ 🚀 Backend │
│ Streamlit │ │ Gateway │ │ FastAPI │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│
┌─────────────────┐ ┌─────────────────┐
│ 🧠 ML Model │ │ 📊 Disease │
│ ResNet50V2 │ │ Database │
└─────────────────┘ └─────────────────┘


</div>

### 🛠️ Technology Stack

| **Component** | **Technology** | **Purpose** |
|---------------|----------------|-------------|
| **Frontend** | Streamlit | Interactive web interface |
| **Backend** | FastAPI | RESTful API services |
| **ML Framework** | TensorFlow 2.12 | Deep learning model |
| **Model Architecture** | ResNet50V2 + Custom Head | Image classification |
| **Image Processing** | PIL, OpenCV | Image preprocessing |
| **Deployment** | Docker, Render | Containerization & hosting |

---

## 📊 Model Performance

<div align="center">

### 🎯 Accuracy Metrics

| **Metric** | **Score** | **Description** |
|------------|-----------|-----------------|
| **Validation Accuracy** | **96.5%** | Overall model accuracy |
| **Precision** | **96.6%** | Macro average precision |
| **Recall** | **96.5%** | Macro average recall |
| **F1-Score** | **96.5%** | Harmonic mean of precision and recall |

### 📈 Performance by Crop Category

| **Category** | **Accuracy** | **Crops Included** |
|--------------|--------------|-------------------|
| **Fruits** | 97.2% | Apple, Grape, Peach, Cherry, Orange, Strawberry, Blueberry, Raspberry |
| **Vegetables** | 96.8% | Tomato, Potato, Bell Pepper, Squash |
| **Field Crops** | 95.1% | Corn (Maize), Soybean |

</div>

---

## 🚀 Quick Start

### 📋 Prerequisites

- Python 3.11+
- Git
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for training)

### ⚡ Installation

1. **Clone the repository**

git clone https://github.com/jcb03/ai-powered-plant-disease-detection.git
cd ai-powered-plant-disease-detection


2. **Set up virtual environment**

Using uv (recommended)
uv venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

Or using pip
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate


3. **Install dependencies**

Backend dependencies
cd backend
pip install -r requirements.txt

Frontend dependencies
cd ../frontend
pip install -r requirements.txt


4. **Download the trained model**

Download pre-trained model (link in releases)

wget https://github.com/jcb03/ai-powered-plant-disease-detection/releases/download/v1.0/plant_disease_model.h5
mv plant_disease_model.h5 backend/ml_models/


### 🏃‍♂️ Running the Application(Locally)

1. **Start the Backend API**

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


2. **Start the Frontend** (in a new terminal)

cd frontend
streamlit run app.py --server.port 8501


3. **Access the Application**
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

---

## 🐳 Docker Deployment

### Quick Deploy with Docker Compose

Clone and navigate to project
git clone https://github.com/jcb03/ai-powered-plant-disease-detection.git
cd ai-powered-plant-disease-detection

Build and run with Docker Compose
docker-compose up --build


### Individual Container Deployment

Build backend
docker build -t plant-disease-backend ./backend

Build frontend
docker build -t plant-disease-frontend ./frontend

Run backend
docker run -p 8000:8000 plant-disease-backend

Run frontend
docker run -p 8501:8501 plant-disease-frontend


---

## 📚 Supported Diseases

<details>
<summary><b>🍎 Fruits (8 crops, 12 diseases)</b></summary>

- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Grape**: Black Rot, Esca (Black Measles), Leaf Blight, Healthy
- **Peach**: Bacterial Spot, Healthy
- **Orange**: Citrus Greening (HLB)
- **Cherry**: Powdery Mildew, Healthy
- **Strawberry**: Leaf Scorch, Healthy
- **Blueberry**: Healthy
- **Raspberry**: Healthy

</details>

<details>
<summary><b>🥬 Vegetables (4 crops, 11 diseases)</b></summary>

- **Tomato**: Late Blight, Early Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Bell Pepper**: Bacterial Spot, Healthy
- **Squash**: Powdery Mildew

</details>

<details>
<summary><b>🌾 Field Crops (2 crops, 5 diseases)</b></summary>

- **Corn (Maize)**: Northern Leaf Blight, Common Rust, Gray Leaf Spot, Healthy
- **Soybean**: Healthy

</details>

---

## 🔬 Model Training

### 📊 Dataset Information

- **Source**: PlantVillage Dataset (Balanced)
- **Total Images**: 21,280 images
- **Training Set**: 18,088 images (85%)
- **Validation Set**: 3,192 images (15%)
- **Classes**: 38 disease categories
- **Image Size**: 224x224 pixels

### 🏋️‍♂️ Training Process

cd model_training

Train the model
python train_model.py --data_dir datasets/processed/train
--epochs 40
--batch_size 32
--model_type resnet

Monitor training progress
tensorboard --logdir outputs/logs


### 📈 Training Results

- **Training Duration**: ~2.5 hours
- **Final Training Accuracy**: 98.7%
- **Final Validation Accuracy**: 96.5%
- **Model Size**: 4.8M parameters
- **Training Hardware**: NVIDIA RTX 3080

---

## 🔧 API Reference

### 🚀 Main Endpoints

| **Endpoint** | **Method** | **Description** |
|--------------|------------|-----------------|
| `/api/v1/predict` | POST | Predict disease from image |
| `/api/v1/predict/batch` | POST | Batch prediction for multiple images |
| `/api/v1/classes` | GET | Get supported disease classes |
| `/api/v1/model/info` | GET | Get model information |
| `/api/v1/health` | GET | Health check |

### 📝 Example Usage

import requests

Single prediction
files = {'file': open('plant_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/api/v1/predict', files=files)
result = response.json()

print(f"Disease: {result['disease_info']['disease_name']}")
print(f"Confidence: {result['confidence']:.2%}")


---

## 📁 Project Structure

ai-powered-plant-disease-detection/
├── 📁 backend/ # FastAPI backend
│ ├── 📁 app/
│ │ ├── 📁 api/ # API routes
│ │ ├── 📁 core/ # Configuration
│ │ ├── 📁 models/ # ML models
│ │ └── 📁 utils/ # Utilities
│ ├── 📁 ml_models/ # Trained models
│ └── 📄 requirements.txt
├── 📁 frontend/ # Streamlit frontend
│ ├── 📁 utils/ # UI components
│ ├── 📄 app.py # Main app
│ └── 📄 requirements.txt
├── 📁 model_training/ # Training scripts
│ ├── 📄 train_model.py # Training script
│ └── 📁 outputs/ # Model outputs
├── 📁 datasets/ # Training data
├── 🐳 docker-compose.yml # Docker setup
├── 📄 README.md # This file
└── 📄 LICENSE # MIT License


---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🐛 Bug Reports
- Use the [issue tracker](https://github.com/jcb03/ai-powered-plant-disease-detection/issues)
- Include detailed reproduction steps
- Provide system information

### 💡 Feature Requests
- Suggest new features via [issues](https://github.com/jcb03/ai-powered-plant-disease-detection/issues)
- Explain the use case and benefits
- Consider implementation complexity

### 🔧 Development

1. **Fork the repository**
2. **Create a feature branch**

git checkout -b feature/amazing-feature

3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**

git commit -m "Add amazing feature"

6. **Push to the branch**

git push origin feature/amazing-feature

7. **Open a Pull Request**

---

## 🚀 Deployment

### 🌐 Render Deployment

1. **Backend Deployment**
- Connect your GitHub repository to Render
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

2. **Frontend Deployment**
- Deploy as a Streamlit app on Streamlit Cloud
- Connect to your GitHub repository
- Set main file path: `frontend/app.py`

### ☁️ Cloud Deployment Options

- **Heroku**: Use provided Procfile
- **AWS**: Deploy using EC2 or ECS
- **Google Cloud**: Use Cloud Run or Compute Engine
- **Azure**: Deploy using Container Instances

---

## 📈 Performance Optimization

### 🔧 Model Optimization
- **Quantization**: Reduce model size by 75%
- **TensorRT**: GPU acceleration for inference
- **ONNX**: Cross-platform model deployment

### ⚡ API Optimization
- **Caching**: Redis for prediction caching
- **Load Balancing**: Multiple API instances
- **CDN**: Static asset delivery

---

## 🔒 Security

- **Input Validation**: File type and size restrictions
- **Rate Limiting**: API request throttling
- **CORS**: Configured for secure cross-origin requests
- **HTTPS**: SSL/TLS encryption in production

---

## 🧪 Testing

Run backend tests
cd backend
pytest tests/

Run frontend tests
cd frontend
pytest tests/

Run integration tests
pytest integration_tests/


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PlantVillage** for providing the comprehensive dataset
- **TensorFlow** team for the excellent deep learning framework
- **Streamlit** for the intuitive frontend framework
- **FastAPI** for the high-performance backend framework
- **Agricultural Research Community** for domain expertise

---

## 📞 Contact & Support

<div align="center">

### 👨‍💻 Created by **Jai Chaudhary**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/your-profile)
[![Microsoft Learn](https://img.shields.io/badge/Microsoft_Learn-258ffa?style=for-the-badge&logo=microsoft&logoColor=white)](https://learn.microsoft.com/en-us/users/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jcb03)

### 📧 Get in Touch

- **Email**: your.email@example.com
- **Issues**: [GitHub Issues](https://github.com/jcb03/ai-powered-plant-disease-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jcb03/ai-powered-plant-disease-detection/discussions)

</div>

---

<div align="center">

### 🌟 Star this repository if you found it helpful!

**Made with ❤️ for the farming community**

[![GitHub stars](https://img.shields.io/github/stars/jcb03/ai-powered-plant-disease-detection.svg?style=social&label=Star)](https://github.com/jcb03/ai-powered-plant-disease-detection)
[![GitHub forks](https://img.shields.io/github/forks/jcb03/ai-powered-plant-disease-detection.svg?style=social&label=Fork)](https://github.com/jcb03/ai-powered-plant-disease-detection/fork)
[![GitHub watchers](https://img.shields.io/github/watchers/jcb03/ai-powered-plant-disease-detection.svg?style=social&label=Watch)](https://github.com/jcb03/ai-powered-plant-disease-detection)

</div>
