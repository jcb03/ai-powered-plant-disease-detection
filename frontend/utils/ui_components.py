"""
UI components and utilities for the Streamlit frontend.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any
import time

def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🌱 AI-Powered Plant Disease Detection</h1>
        <p>Upload or capture a photo of your plant leaf to get instant disease diagnosis and treatment recommendations</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(api_client) -> Dict:
    """Render sidebar with instructions and API status."""
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        ### How to Use:
        1. **Take a clear photo** of the affected plant leaf
        2. **Ensure good lighting** and focus
        3. **Upload the image** using the options provided
        4. **Get instant results** with treatment recommendations
        
        ### 📱 Tips for Best Results:
        - Use natural lighting when possible
        - Fill the frame with the leaf
        - Avoid blurry or dark images
        - Include both healthy and affected areas if visible
        - Take photos from multiple angles for better accuracy
        """)
        
        st.header("🔧 System Status")
        with st.spinner("Checking API status..."):
            health_status = api_client.health_check()
        
        if not health_status.get('healthy', False):
            st.error(f"❌ API Offline")
            st.error(f"Error: {health_status.get('error', 'Unknown error')}")
        else:
            st.success("✅ API Online")
            model_status = health_status.get('model_status', 'unknown')
            st.info(f"Model: {model_status}")
            
            if 'supported_classes' in health_status:
                st.metric("Supported Classes", health_status['supported_classes'])
        
        st.header("📊 Quick Stats")
        try:
            stats = api_client.get_api_stats()
            if "error" not in stats:
                st.metric("Supported Crops", stats.get('supported_crops', 0))
                st.metric("Disease Classes", stats.get('supported_classes', 0))
                st.metric("Max File Size", f"{stats.get('max_file_size_mb', 0)}MB")
        except Exception as e:
            st.warning("Could not load stats")
        
        return health_status

def render_prediction_results(result: Dict, confidence_threshold: float = 0.7):
    """Render prediction results in a formatted card."""
    prediction = result.get('prediction', 'Unknown')
    confidence = result.get('confidence', 0.0)
    disease_info = result.get('disease_info', {})
    is_confident = confidence >= confidence_threshold
    
    if 'healthy' in prediction.lower():
        st.success(f"✅ **{disease_info.get('disease_name', 'Healthy Plant')}**")
    else:
        if is_confident:
            st.warning(f"⚠️ **{disease_info.get('disease_name', prediction)}** detected")
        else:
            st.error(f"🔍 **{disease_info.get('disease_name', prediction)}** (Low confidence)")
    
    st.metric("Confidence", f"{confidence:.1%}")
    return result

def render_confidence_gauge(confidence: float, threshold: float = 0.7) -> go.Figure:
    """Create a confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        delta = {'reference': threshold * 100},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen" if confidence >= threshold else "orange"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, threshold * 100], 'color': "yellow"},
                {'range': [threshold * 100, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def render_disease_info(disease_info: Dict):
    """Render detailed disease information."""
    if not disease_info:
        return
    
    st.subheader(f"🦠 {disease_info.get('disease_name', 'Unknown Disease')}")
    
    if disease_info.get('scientific_name'):
        st.write(f"**Scientific Name:** {disease_info['scientific_name']}")
    
    if disease_info.get('crop'):
        st.write(f"**Affected Crop:** {disease_info['crop']}")
    
    if disease_info.get('severity'):
        severity_colors = {
            'None': '🟢',
            'Low': '🟡',
            'Moderate': '🟠', 
            'High': '🔴',
            'Very High': '🔴'
        }
        severity_icon = severity_colors.get(disease_info['severity'], '⚪')
        st.write(f"**Severity:** {severity_icon} {disease_info['severity']}")
    
    if disease_info.get('description'):
        st.write(f"**Description:** {disease_info['description']}")
    
    if disease_info.get('symptoms'):
        st.subheader("🔍 Symptoms")
        for symptom in disease_info['symptoms']:
            st.write(f"• {symptom}")
    
    if disease_info.get('treatment'):
        st.subheader("💊 Treatment")
        for i, treatment in enumerate(disease_info['treatment'], 1):
            st.write(f"{i}. {treatment}")
    
    if disease_info.get('prevention'):
        st.subheader("🛡️ Prevention")
        for prevention in disease_info['prevention']:
            st.write(f"• {prevention}")
    
    if disease_info.get('organic_treatment'):
        st.subheader("🌿 Organic Treatment Options")
        for treatment in disease_info['organic_treatment']:
            st.write(f"• {treatment}")

def render_image_preview(image, caption: str = "Uploaded Image"):
    """Render image preview with metadata."""
    st.image(image, caption=caption, use_column_width=True)
    
    if hasattr(image, 'size'):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Width", f"{image.size[0]}px")
        with col2:
            st.metric("Height", f"{image.size[1]}px")

def create_processing_animation():
    """Create a processing animation placeholder."""
    placeholder = st.empty()
    
    for i in range(3):
        placeholder.text(f"🔬 Analyzing image{'.' * (i + 1)}")
        time.sleep(0.5)
    
    return placeholder

def render_error_message(error: str, error_type: str = "Error"):
    """Render formatted error message."""
    st.error(f"❌ **{error_type}:** {error}")
    
    if "connection" in error.lower():
        st.info("💡 **Troubleshooting Tips:**")
        st.write("• Check if the backend server is running")
        st.write("• Verify the API URL in configuration")
        st.write("• Check your internet connection")

def render_success_message(message: str):
    """Render formatted success message."""
    st.success(f"✅ {message}")

def render_warning_message(message: str):
    """Render formatted warning message."""
    st.warning(f"⚠️ {message}")

def render_info_message(message: str):
    """Render formatted info message."""
    st.info(f"ℹ️ {message}")
