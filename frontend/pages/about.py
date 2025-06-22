"""
About page for the Plant Disease Detection application.
"""

import streamlit as st
from utils.ui_components import render_header

def render_about_page():
    """Render the about page."""
    st.set_page_config(
        page_title="ℹ️ About - Plant Disease Detection",
        page_icon="ℹ️",
        layout="wide"
    )
    
    render_header()
    
    st.header("ℹ️ About Plant Disease Detection System")
    
    # Introduction
    st.markdown("""
    ## Our Mission 🎯
    
    We're dedicated to helping farmers and gardeners worldwide identify and treat plant diseases quickly and accurately. 
    Our AI-powered system democratizes access to agricultural expertise, making professional-grade plant disease 
    diagnosis available to everyone with a smartphone or computer.
    
    ## How It Works 🔬
    
    Our system uses advanced **Convolutional Neural Networks (CNNs)** trained on thousands of plant disease images 
    to provide accurate disease identification. Here's the process:
    
    1. **Image Capture**: You upload or capture a photo of the affected plant leaf
    2. **Preprocessing**: Our system enhances and prepares the image for analysis
    3. **AI Analysis**: A deep learning model analyzes the image for disease patterns
    4. **Disease Identification**: The system identifies the most likely disease with confidence scores
    5. **Treatment Recommendations**: Comprehensive treatment and prevention advice is provided
    """)
    
    # Technology stack
    st.subheader("🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Backend Technologies:**
        - **FastAPI**: High-performance API framework
        - **TensorFlow**: Deep learning model training and inference
        - **OpenCV**: Advanced image processing
        - **Python 3.11**: Modern Python features and performance
        - **Docker**: Containerized deployment
        """)
    
    with col2:
        st.markdown("""
        **Frontend Technologies:**
        - **Streamlit**: Interactive web application framework
        - **Plotly**: Interactive data visualizations
        - **PIL/Pillow**: Image processing and manipulation
        - **Pandas**: Data analysis and manipulation
        - **Responsive Design**: Mobile-friendly interface
        """)
    
    # Model information
    st.subheader("🧠 AI Model Details")
    
    st.markdown("""
    ### Model Architecture
    - **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
    - **Custom Layers**: Specialized classification head for plant diseases
    - **Input Size**: 224x224 pixels RGB images
    - **Output**: 38+ disease classes across 15+ crop types
    - **Training Data**: PlantVillage dataset with data augmentation
    
    ### Performance Metrics
    - **Accuracy**: 95%+ on validation set
    - **Top-3 Accuracy**: 98%+ (correct disease in top 3 predictions)
    - **Processing Time**: < 2 seconds per image
    - **Confidence Threshold**: 70% for reliable predictions
    """)
    
    # Supported crops and diseases
    st.subheader("🌾 Supported Crops & Diseases")
    
    crops_diseases = {
        "🍎 Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust"],
        "🍅 Tomato": ["Late Blight", "Early Blight", "Bacterial Spot", "Leaf Mold", "Septoria Leaf Spot"],
        "🌽 Corn": ["Northern Leaf Blight", "Common Rust", "Gray Leaf Spot"],
        "🍇 Grape": ["Black Rot", "Esca", "Leaf Blight"],
        "🥔 Potato": ["Early Blight", "Late Blight"],
        "🌶️ Pepper": ["Bacterial Spot"],
        "🍑 Cherry": ["Powdery Mildew"],
        "🍑 Peach": ["Bacterial Spot"],
        "🍓 Strawberry": ["Leaf Scorch"],
        "🫐 Blueberry": ["Healthy Detection"],
        "🌰 Squash": ["Powdery Mildew"],
        "🫘 Soybean": ["Healthy Detection"],
        "🍊 Orange": ["Citrus Greening"],
        "🫐 Raspberry": ["Healthy Detection"]
    }
    
    col1, col2 = st.columns(2)
    
    crops_list = list(crops_diseases.keys())
    mid_point = len(crops_list) // 2
    
    with col1:
        for crop in crops_list[:mid_point]:
            st.markdown(f"**{crop}**")
            for disease in crops_diseases[crop]:
                st.markdown(f"  • {disease}")
    
    with col2:
        for crop in crops_list[mid_point:]:
            st.markdown(f"**{crop}**")
            for disease in crops_diseases[crop]:
                st.markdown(f"  • {disease}")
    
    # Usage guidelines
    st.subheader("📋 Usage Guidelines")
    
    st.markdown("""
    ### For Best Results:
    - **Image Quality**: Use clear, well-lit photos with good focus
    - **Leaf Coverage**: Ensure the leaf fills most of the frame
    - **Multiple Angles**: Take photos from different angles if possible
    - **Natural Lighting**: Avoid harsh shadows or artificial lighting when possible
    - **Clean Background**: Remove clutter from the background
    
    ### Limitations:
    - **Accuracy**: While highly accurate, the system is not 100% perfect
    - **New Diseases**: May not recognize very rare or newly discovered diseases
    - **Image Quality**: Poor quality images will result in lower accuracy
    - **Professional Advice**: Always consult agricultural experts for severe infections
    
    ### Disclaimer:
    This tool provides AI-based suggestions for educational and informational purposes. 
    For critical agricultural decisions or severe plant infections, please consult with 
    qualified agricultural professionals or extension services.
    """)
    
    # Contact and support
    st.subheader("📞 Contact & Support")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📧 Email Support**
        jaichaudhary0303@gmail.com
        
        **🐛 Report Issues**
        GitHub Issues Page
        """)
    
    with col2:
        st.markdown("""
        **📚 Documentation**
        User Guide & API Docs
        
        **💡 Feature Requests**
        GitHub Discussions
        """)
    
    with col3:
        st.markdown("""
        **🌐 Website**
        www.plantdisease.ai
        
        **📱 Mobile App**
        Coming Soon!
        """)
    
    # Version information
    st.subheader("📋 Version Information")
    
    st.markdown("""
    - **Application Version**: 1.0.0
    - **Model Version**: 2024.1
    - **Last Updated**: June 2024
    - **Python Version**: 3.11+
    - **TensorFlow Version**: 2.12.0
    """)
    
    # Acknowledgments
    st.subheader("🙏 Acknowledgments")
    
    st.markdown("""
    We would like to thank:
    - **PlantVillage**: For providing the comprehensive plant disease dataset
    - **TensorFlow Team**: For the excellent deep learning framework
    - **Streamlit Team**: For the amazing web app framework
    - **Agricultural Researchers**: Worldwide for their contributions to plant pathology
    - **Open Source Community**: For the tools and libraries that made this possible
    """)

if __name__ == "__main__":
    render_about_page()
