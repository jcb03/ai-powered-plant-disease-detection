"""
Streamlit frontend for Plant Disease Detection System.
"""

import streamlit as st
import requests
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import json
from pathlib import Path
import base64

# Import custom utilities
from utils.api_client import get_api_client
from utils.ui_components import (
    render_header, render_sidebar, render_prediction_results,
    render_disease_info, render_confidence_gauge, render_image_preview
)

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/plant-disease-detection',
        'Report a bug': 'https://github.com/your-repo/plant-disease-detection/issues',
        'About': "AI-powered plant disease detection system for farmers and gardeners."
    }
)

def load_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .disease-info {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .healthy-info {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .treatment-section {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    
    .warning-section {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .stButton > button {
        width: 100%;
        background: #4CAF50;
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background: #45a049;
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        .main-header p {
            font-size: 0.9rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if 'api_client' not in st.session_state:
        st.session_state.api_client = get_api_client()
    
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None

def main():
    """Main application function."""
    load_css()
    initialize_session_state()
    
    render_header()
    api_status = render_sidebar(st.session_state.api_client)
    
    if not api_status.get('healthy', False):
        st.error("🚨 **API Connection Failed**")
        st.error(f"Unable to connect to the backend API: {api_status.get('error', 'Unknown error')}")
        st.info("Please ensure the backend server is running and accessible.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Disease Detection", "📊 Batch Analysis", "📚 Disease Database", "📈 Analytics"])
    
    with tab1:
        render_disease_detection_tab()
    
    with tab2:
        render_batch_analysis_tab()
    
    with tab3:
        render_disease_database_tab()
    
    with tab4:
        render_analytics_tab()

def render_disease_detection_tab():
    """Render the main disease detection interface."""
    st.header("🔍 Plant Disease Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Plant Image")
        
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload from device", "📷 Take photo with camera"],
            horizontal=True
        )
        
        uploaded_file = None
        
        if input_method == "📁 Upload from device":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Supported formats: PNG, JPG, JPEG (Max 10MB)"
            )
        
        elif input_method == "📷 Take photo with camera":
            uploaded_file = st.camera_input("Take a picture of the plant leaf")
        
        with st.expander("⚙️ Advanced Options"):
            enhance_image = st.checkbox("Enhance image quality", value=True, help="Apply image enhancement for better results")
            include_metadata = st.checkbox("Include image metadata", value=False, help="Include technical image information")
            confidence_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.7, 0.1, help="Minimum confidence for reliable predictions")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.info(f"📊 Image size: {image.size[0]}x{image.size[1]} pixels")
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Analyze Plant Disease", type="primary"):
                with st.spinner("🔬 Analyzing image... This may take a few seconds"):
                    uploaded_file.seek(0)
                    
                    result = st.session_state.api_client.predict_disease(
                        uploaded_file, 
                        enhance_image=enhance_image,
                        include_metadata=include_metadata
                    )
                    
                    if "error" in result:
                        st.error(f"❌ Analysis failed: {result['error']}")
                    else:
                        st.session_state.current_prediction = result
                        
                        st.session_state.prediction_history.append({
                            "timestamp": time.time(),
                            "filename": uploaded_file.name,
                            "result": result
                        })
                        
                        display_prediction_results(result, confidence_threshold)
        else:
            st.info("👆 Please upload or capture an image to start analysis")

def render_batch_analysis_tab():
    """Render batch analysis interface."""
    st.header("📊 Batch Analysis")
    st.write("Upload multiple images for batch disease detection.")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload up to 10 images for batch processing"
    )
    
    if uploaded_files:
        st.write(f"📁 Selected {len(uploaded_files)} files")
        
        cols = st.columns(min(len(uploaded_files), 4))
        for i, file in enumerate(uploaded_files[:4]):
            with cols[i]:
                image = Image.open(file)
                st.image(image, caption=file.name, use_column_width=True)
        
        if len(uploaded_files) > 4:
            st.info(f"... and {len(uploaded_files) - 4} more files")
        
        if st.button("🚀 Analyze All Images", type="primary"):
            if len(uploaded_files) > 10:
                st.error("❌ Maximum 10 files allowed for batch processing")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            batch_results = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                file.seek(0)
                result = st.session_state.api_client.predict_disease(file)
                result['filename'] = file.name
                batch_results.append(result)
            
            status_text.text("✅ Batch processing completed!")
            display_batch_results(batch_results, results_container)

def render_disease_database_tab():
    """Render disease database interface."""
    st.header("📚 Disease Database")
    st.write("Explore information about plant diseases our AI can detect.")
    
    classes_info = st.session_state.api_client.get_supported_classes()
    
    if "error" in classes_info:
        st.error(f"Failed to load disease database: {classes_info['error']}")
        return
    
    search_query = st.text_input("🔍 Search diseases, crops, or symptoms:", placeholder="e.g., apple scab, tomato blight")
    
    if search_query:
        search_results = st.session_state.api_client.search_diseases(search_query)
        if search_results and "results" in search_results:
            st.write(f"Found {len(search_results['results'])} results for '{search_query}':")
            display_disease_search_results(search_results['results'])
    
    st.subheader("🌾 Browse by Crop")
    crops = classes_info.get('crops', [])
    
    if crops:
        selected_crop = st.selectbox("Select a crop:", ['All'] + crops)
        
        if selected_crop != 'All':
            crop_diseases = st.session_state.api_client.get_diseases_by_crop(selected_crop)
            if crop_diseases and "diseases" in crop_diseases:
                display_crop_diseases(crop_diseases['diseases'])
    
    st.subheader("📊 Database Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Disease Classes", classes_info.get('total_classes', 0))
    
    with col2:
        st.metric("Supported Crops", len(crops))
    
    with col3:
        st.metric("Model Status", "✅ Loaded" if classes_info.get('model_loaded') else "❌ Not Loaded")

def render_analytics_tab():
    """Render analytics and statistics."""
    st.header("📈 Analytics & Statistics")
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Start by analyzing some images!")
        return
    
    df = pd.DataFrame([
        {
            'timestamp': pd.to_datetime(pred['timestamp'], unit='s'),
            'filename': pred['filename'],
            'prediction': pred['result'].get('prediction', 'Unknown'),
            'confidence': pred['result'].get('confidence', 0),
            'is_healthy': 'healthy' in pred['result'].get('prediction', '').lower(),
            'processing_time': pred['result'].get('processing_time', 0)
        }
        for pred in st.session_state.prediction_history
    ])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    
    with col2:
        healthy_count = df['is_healthy'].sum()
        st.metric("Healthy Plants", healthy_count)
    
    with col3:
        diseased_count = len(df) - healthy_count
        st.metric("Diseased Plants", diseased_count)
    
    with col4:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_time = px.line(df, x='timestamp', y='confidence', 
                          title='Prediction Confidence Over Time',
                          labels={'confidence': 'Confidence Score', 'timestamp': 'Time'})
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        disease_counts = df['prediction'].value_counts().head(10)
        fig_dist = px.bar(x=disease_counts.values, y=disease_counts.index, 
                         orientation='h', title='Top 10 Detected Diseases')
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.subheader("⚡ Performance Metrics")
    avg_processing_time = df['processing_time'].mean()
    st.metric("Average Processing Time", f"{avg_processing_time:.2f}s")
    
    fig_perf = px.histogram(df, x='processing_time', title='Processing Time Distribution',
                           labels={'processing_time': 'Processing Time (seconds)'})
    st.plotly_chart(fig_perf, use_container_width=True)

def display_prediction_results(result, confidence_threshold):
    """Display prediction results in a formatted way."""
    prediction = result.get('prediction', 'Unknown')
    confidence = result.get('confidence', 0.0)
    disease_info = result.get('disease_info', {})
    is_confident = confidence >= confidence_threshold
    
    if 'healthy' in prediction.lower():
        st.markdown(f"""
        <div class="healthy-info">
            <h3>✅ Good News!</h3>
            <h4>{disease_info.get('disease_name', 'Healthy Plant')}</h4>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence_color = "#4CAF50" if is_confident else "#ff9800"
        st.markdown(f"""
        <div class="disease-info">
            <h3>🔍 Disease Detected</h3>
            <h4>{disease_info.get('disease_name', prediction)}</h4>
            <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{confidence:.1%}</span></p>
            {f"<p><em>⚠️ Low confidence - consider retaking the photo</em></p>" if not is_confident else ""}
        </div>
        """, unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4CAF50" if is_confident else "#ff9800"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcdd2"},
                {'range': [50, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#e8f5e8"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence_threshold * 100
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    if disease_info.get('description'):
        st.markdown(f"**📝 Description:** {disease_info['description']}")
    
    if disease_info.get('symptoms'):
        st.markdown("**🔍 Symptoms:**")
        for symptom in disease_info['symptoms']:
            st.markdown(f"• {symptom}")
    
    if disease_info.get('treatment'):
        st.markdown("""
        <div class="treatment-section">
            <h4>💊 Treatment Recommendations:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, treatment in enumerate(disease_info['treatment'], 1):
            st.markdown(f"{i}. {treatment}")
    
    if disease_info.get('prevention'):
        st.markdown("**🛡️ Prevention Tips:**")
        for tip in disease_info['prevention']:
            st.markdown(f"• {tip}")
    
    if result.get('top_predictions'):
        with st.expander("📊 Alternative Possibilities"):
            st.markdown("**Top 5 Predictions:**")
            for i, pred in enumerate(result['top_predictions'], 1):
                disease_name = pred.get('disease_name', pred['class'])
                st.markdown(f"{i}. **{disease_name}**: {pred['confidence']:.1%}")
    
    if not is_confident:
        st.markdown("""
        <div class="warning-section">
            <h4>⚠️ Low Confidence Detection</h4>
            <p>The model is not very confident about this prediction. Consider:</p>
            <ul>
                <li>Taking a clearer, well-lit photo</li>
                <li>Ensuring the leaf fills most of the frame</li>
                <li>Trying a different angle or lighting condition</li>
                <li>Consulting with a local agricultural expert</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_batch_results(results, container):
    """Display batch processing results."""
    with container:
        st.subheader("📊 Batch Results Summary")
        
        successful = sum(1 for r in results if r.get('success', True))
        failed = len(results) - successful
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(results))
        with col2:
            st.metric("Successful", successful)
        with col3:
            st.metric("Failed", failed)
        
        st.subheader("Individual Results")
        
        for i, result in enumerate(results):
            with st.expander(f"📁 {result.get('filename', f'File {i+1}')}"):
                if result.get('success', True) and 'error' not in result:
                    prediction = result.get('prediction', 'Unknown')
                    confidence = result.get('confidence', 0)
                    disease_name = result.get('disease_info', {}).get('disease_name', prediction)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Disease:** {disease_name}")
                        st.write(f"**Confidence:** {confidence:.1%}")
                    
                    with col2:
                        if 'healthy' in prediction.lower():
                            st.success("✅ Healthy")
                        else:
                            st.warning("⚠️ Disease Detected")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

def display_disease_search_results(results):
    """Display disease search results."""
    for result in results:
        with st.expander(f"🦠 {result.get('disease_name', 'Unknown Disease')} - {result.get('crop', 'Unknown Crop')}"):
            st.write(f"**Scientific Name:** {result.get('scientific_name', 'N/A')}")
            st.write(f"**Severity:** {result.get('severity', 'Unknown')}")
            st.write(f"**Description:** {result.get('description', 'No description available')}")
            
            if result.get('symptoms'):
                st.write("**Symptoms:**")
                for symptom in result['symptoms'][:3]:
                    st.write(f"• {symptom}")

def display_crop_diseases(diseases):
    """Display diseases for a specific crop."""
    for disease in diseases:
        severity_color = {
            'None': '🟢',
            'Low': '🟡', 
            'Moderate': '🟠',
            'High': '🔴',
            'Very High': '🔴'
        }.get(disease.get('severity', 'Unknown'), '⚪')
        
        with st.expander(f"{severity_color} {disease.get('disease_name', 'Unknown Disease')}"):
            st.write(f"**Severity:** {disease.get('severity', 'Unknown')}")
            st.write(f"**Description:** {disease.get('description', 'No description available')}")
            
            if disease.get('symptoms'):
                st.write("**Key Symptoms:**")
                for symptom in disease['symptoms'][:3]:
                    st.write(f"• {symptom}")

if __name__ == "__main__":
    main()
