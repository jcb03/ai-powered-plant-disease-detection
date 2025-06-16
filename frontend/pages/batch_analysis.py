"""
Batch analysis page for processing multiple images.
"""

import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from utils.api_client import get_api_client
from utils.ui_components import render_header

def render_batch_analysis_page():
    """Render the batch analysis page."""
    st.set_page_config(
        page_title="📊 Batch Analysis - Plant Disease Detection",
        page_icon="📊",
        layout="wide"
    )
    
    render_header()
    
    st.header("📊 Batch Analysis")
    st.write("Upload multiple plant images for simultaneous disease detection and analysis.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload up to 20 images for batch processing"
    )
    
    if not uploaded_files:
        st.info("👆 Please upload multiple images to start batch analysis")
        
        # Show example of what batch analysis provides
        st.subheader("📋 What You'll Get:")
        st.markdown("""
        - **Individual Results**: Disease detection for each image
        - **Summary Statistics**: Overview of all detections
        - **Comparative Analysis**: Side-by-side comparison
        - **Export Options**: Download results as CSV
        - **Visual Charts**: Graphical representation of findings
        """)
        return
    
    # Display uploaded files preview
    st.subheader(f"📁 Uploaded Files ({len(uploaded_files)})")
    
    # Show preview grid
    cols = st.columns(min(len(uploaded_files), 5))
    for i, file in enumerate(uploaded_files[:5]):
        with cols[i]:
            image = Image.open(file)
            st.image(image, caption=file.name, use_column_width=True)
    
    if len(uploaded_files) > 5:
        st.info(f"... and {len(uploaded_files) - 5} more files")
    
    # Processing options
    with st.expander("⚙️ Processing Options"):
        enhance_images = st.checkbox("Enhance image quality", value=True)
        confidence_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.7, 0.1)
        include_healthy = st.checkbox("Include healthy predictions in summary", value=True)
    
    # Process button
    if st.button("🚀 Process All Images", type="primary"):
        if len(uploaded_files) > 20:
            st.error("❌ Maximum 20 files allowed for batch processing")
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process images
        api_client = get_api_client()
        results = []
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Reset file pointer
            file.seek(0)
            
            # Make prediction
            result = api_client.predict_disease(file, enhance_image=enhance_images)
            result['filename'] = file.name
            result['file_index'] = i
            results.append(result)
        
        status_text.text("✅ Batch processing completed!")
        
        # Display results
        display_batch_results(results, confidence_threshold, include_healthy)

def display_batch_results(results, confidence_threshold, include_healthy):
    """Display comprehensive batch analysis results."""
    
    # Prepare data for analysis
    successful_results = [r for r in results if r.get('success', True) and 'error' not in r]
    failed_results = [r for r in results if not r.get('success', True) or 'error' in r]
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(results))
    
    with col2:
        st.metric("Successful", len(successful_results))
    
    with col3:
        st.metric("Failed", len(failed_results))
    
    with col4:
        if successful_results:
            avg_confidence = sum(r.get('confidence', 0) for r in successful_results) / len(successful_results)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    if not successful_results:
        st.error("No successful predictions to analyze.")
        return
    
    # Create DataFrame for analysis
    df_data = []
    for result in successful_results:
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        disease_info = result.get('disease_info', {})
        
        df_data.append({
            'filename': result['filename'],
            'prediction': prediction,
            'disease_name': disease_info.get('disease_name', prediction),
            'crop': disease_info.get('crop', 'Unknown'),
            'confidence': confidence,
            'is_healthy': 'healthy' in prediction.lower(),
            'is_confident': confidence >= confidence_threshold,
            'severity': disease_info.get('severity', 'Unknown'),
            'processing_time': result.get('processing_time', 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Filter based on options
    if not include_healthy:
        df = df[~df['is_healthy']]
    
    # Disease distribution
    st.subheader("🦠 Disease Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Disease counts
        disease_counts = df['disease_name'].value_counts().head(10)
        if not disease_counts.empty:
            fig_diseases = px.bar(
                x=disease_counts.values,
                y=disease_counts.index,
                orientation='h',
                title='Top 10 Detected Diseases',
                labels={'x': 'Count', 'y': 'Disease'}
            )
            st.plotly_chart(fig_diseases, use_container_width=True)
    
    with col2:
        # Crop distribution
        crop_counts = df['crop'].value_counts()
        if not crop_counts.empty:
            fig_crops = px.pie(
                values=crop_counts.values,
                names=crop_counts.index,
                title='Crop Distribution'
            )
            st.plotly_chart(fig_crops, use_container_width=True)
    
    # Confidence analysis
    st.subheader("📈 Confidence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig_conf_hist = px.histogram(
            df, 
            x='confidence',
            title='Confidence Score Distribution',
            labels={'confidence': 'Confidence Score', 'count': 'Number of Images'}
        )
        fig_conf_hist.add_vline(x=confidence_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig_conf_hist, use_container_width=True)
    
    with col2:
        # Confidence by disease
        if len(df) > 0:
            avg_conf_by_disease = df.groupby('disease_name')['confidence'].mean().sort_values(ascending=True)
            fig_conf_disease = px.bar(
                x=avg_conf_by_disease.values,
                y=avg_conf_by_disease.index,
                orientation='h',
                title='Average Confidence by Disease',
                labels={'x': 'Average Confidence', 'y': 'Disease'}
            )
            st.plotly_chart(fig_conf_disease, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop_filter = st.selectbox("Filter by Crop:", ['All'] + list(df['crop'].unique()))
    
    with col2:
        confidence_filter = st.selectbox("Confidence Level:", ['All', 'High Confidence', 'Low Confidence'])
    
    with col3:
        health_filter = st.selectbox("Health Status:", ['All', 'Healthy', 'Diseased'])
    
    # Apply filters
    filtered_df = df.copy()
    
    if crop_filter != 'All':
        filtered_df = filtered_df[filtered_df['crop'] == crop_filter]
    
    if confidence_filter == 'High Confidence':
        filtered_df = filtered_df[filtered_df['is_confident']]
    elif confidence_filter == 'Low Confidence':
        filtered_df = filtered_df[~filtered_df['is_confident']]
    
    if health_filter == 'Healthy':
        filtered_df = filtered_df[filtered_df['is_healthy']]
    elif health_filter == 'Diseased':
        filtered_df = filtered_df[~filtered_df['is_healthy']]
    
    # Display filtered results
    if len(filtered_df) > 0:
        # Format the dataframe for display
        display_df = filtered_df[['filename', 'disease_name', 'crop', 'confidence', 'severity']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df.columns = ['Filename', 'Disease', 'Crop', 'Confidence', 'Severity']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="batch_analysis_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No results match the selected filters.")
    
    # Individual image results
    st.subheader("🖼️ Individual Results")
    
    # Create expandable sections for each result
    for i, result in enumerate(successful_results):
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        disease_info = result.get('disease_info', {})
        disease_name = disease_info.get('disease_name', prediction)
        
        # Status indicator
        if 'healthy' in prediction.lower():
            status_icon = "✅"
            status_color = "green"
        elif confidence >= confidence_threshold:
            status_icon = "⚠️"
            status_color = "orange"
        else:
            status_icon = "❓"
            status_color = "red"
        
        with st.expander(f"{status_icon} {result['filename']} - {disease_name} ({confidence:.1%})"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Disease:** {disease_name}")
                st.write(f"**Confidence:** {confidence:.1%}")
                st.write(f"**Crop:** {disease_info.get('crop', 'Unknown')}")
                st.write(f"**Severity:** {disease_info.get('severity', 'Unknown')}")
            
            with col2:
                if disease_info.get('description'):
                    st.write(f"**Description:** {disease_info['description']}")
                
                if disease_info.get('treatment'):
                    st.write("**Key Treatments:**")
                    for treatment in disease_info['treatment'][:3]:
                        st.write(f"• {treatment}")
    
    # Failed results
    if failed_results:
        st.subheader("❌ Failed Analyses")
        for result in failed_results:
            st.error(f"**{result['filename']}**: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    render_batch_analysis_page()
