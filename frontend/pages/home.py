"""
Home page for the Plant Disease Detection application.
"""

import streamlit as st
from utils.api_client import get_api_client
from utils.ui_components import render_header, render_sidebar

def render_home_page():
    """Render the home page."""
    st.set_page_config(
        page_title="🌱 Plant Disease Detection - Home",
        page_icon="🌱",
        layout="wide"
    )
    
    render_header()
    
    # Welcome section
    st.markdown("""
    ## Welcome to AI-Powered Plant Disease Detection! 🌱
    
    Our advanced machine learning system helps farmers and gardeners identify plant diseases quickly and accurately.
    Simply upload a photo of your plant leaf, and get instant diagnosis with treatment recommendations.
    
    ### Key Features:
    - 🔬 **AI-Powered Analysis**: Advanced CNN model trained on thousands of plant images
    - 📱 **Mobile Friendly**: Works on any device with camera support
    - 🌾 **Multiple Crops**: Supports detection for 15+ crop types
    - 💊 **Treatment Advice**: Get specific treatment and prevention recommendations
    - 📊 **Batch Processing**: Analyze multiple images at once
    - 🌿 **Organic Options**: Includes organic treatment alternatives
    
    ### How It Works:
    1. **Capture or Upload** a clear photo of the affected plant leaf
    2. **AI Analysis** processes the image using deep learning
    3. **Get Results** with disease identification and confidence score
    4. **Follow Treatment** recommendations provided by our system
    
    ### Supported Crops:
    Apple, Tomato, Corn, Grape, Potato, Pepper, Cherry, Peach, Strawberry, and more!
    """)
    
    # Quick start section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📸 Quick Start
        Ready to analyze your plants?
        """)
        if st.button("Start Disease Detection", type="primary"):
            st.switch_page("app.py")
    
    with col2:
        st.markdown("""
        ### 📚 Learn More
        Explore our disease database
        """)
        if st.button("Browse Disease Database"):
            st.switch_page("app.py")
    
    with col3:
        st.markdown("""
        ### 📊 View Analytics
        See your prediction history
        """)
        if st.button("View Analytics"):
            st.switch_page("app.py")
    
    # Statistics section
    st.markdown("---")
    st.subheader("📊 System Statistics")
    
    api_client = get_api_client()
    stats = api_client.get_api_stats()
    
    if "error" not in stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Disease Classes", stats.get('supported_classes', 0))
        
        with col2:
            st.metric("Supported Crops", stats.get('supported_crops', 0))
        
        with col3:
            st.metric("Max File Size", f"{stats.get('max_file_size_mb', 0)}MB")
        
        with col4:
            model_status = "✅ Online" if stats.get('model_status') == 'loaded' else "❌ Offline"
            st.metric("Model Status", model_status)

if __name__ == "__main__":
    render_home_page()
