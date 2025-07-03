"""
Data Wrangling & Analytics Platform
Profit Projects Online Virtual Assistance

A comprehensive data processing and visualization application built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import io
import base64
from typing import Dict, List, Any, Optional

# Import custom modules
from config import *
from utils.data_processor import DataProcessor
from utils.database import DatabaseManager
from utils.email_service import EmailService
from utils.visualizations import VisualizationEngine

# Configure Streamlit page
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for dark theme and animations
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Success message styling */
    .success-message {
        background-color: #27ae60;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'email_service' not in st.session_state:
        st.session_state.email_service = EmailService()
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = VisualizationEngine()
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""

# Loading animation
def show_loading_animation(message: str, duration: int = 3):
    """Show animated loading message"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(duration * 10):
        progress_bar.progress((i + 1) / (duration * 10))
        status_text.text(f"{message} {'.' * ((i % 3) + 1)}")
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()

# Header section
def render_header():
    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_ICON} {APP_TITLE}</h1>
        <h3>{COMPANY_NAME}</h3>
        <p>Transform your data into actionable insights</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
def render_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/FF6B6B/FFFFFF?text=PROFIT+PROJECTS", width=200)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Home", "📤 Upload Data", "🔧 Process Data", "📊 Dashboard", "💾 Database", "📧 Share Results"],
            icons=["house", "upload", "gear", "bar-chart", "database", "envelope"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#1a1a1a"},
                "icon": {"color": "#4ECDC4", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#2c3e50"},
                "nav-link-selected": {"background-color": "#FF6B6B"},
            }
        )
        
        # User email input
        st.markdown("---")
        st.markdown("### 👤 User Information")
        user_email = st.text_input("📧 Your Email Address", value=st.session_state.user_email, placeholder="your.email@example.com")
        if user_email != st.session_state.user_email:
            st.session_state.user_email = user_email
        
        # Company info
        st.markdown("---")
        st.markdown("### 🏢 Company Info")
        st.markdown(f"""
        **{COMPANY_NAME}**  
        📍 {COMPANY_LOCATION}  
        📧 {COMPANY_EMAIL}  
        🏢 Enterprise: {ENTERPRISE_NUMBER}
        """)
        
        return selected

# Home page
def render_home():
    st.markdown("## 🎯 Welcome to Our Data Processing Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤 Upload</h3>
            <p>Support for multiple file formats including CSV, Excel, JSON, PDF, and DOCX</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 Process</h3>
            <p>Advanced data cleaning, transformation, and quality improvement tools</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Visualize</h3>
            <p>Interactive dashboards and comprehensive data analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics
    stats = st.session_state.db_manager.get_statistics()
    
    st.markdown("## 📈 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Datasets", stats.get('total_datasets', 0))
    with col2:
        st.metric("Data Processed", f"{stats.get('total_size', 0) / 1024 / 1024:.1f} MB")
    with col3:
        st.metric("Recent Uploads", stats.get('recent_uploads', 0))
    with col4:
        st.metric("File Types", len(stats.get('datasets_by_type', {})))
    
    # Features overview
    st.markdown("## ✨ Key Features")
    
    features = [
        "🔄 **Automated Data Cleaning** - Remove duplicates, handle missing values, standardize formats",
        "📊 **Interactive Visualizations** - Dynamic charts, graphs, and statistical analysis",
        "🗄️ **Searchable Database** - Organize and retrieve your processed datasets",
        "📧 **Email Integration** - Share results directly via email",
        "⚡ **Real-time Processing** - Fast data transformation with progress tracking",
        "🎨 **Modern Interface** - Intuitive design with smooth animations"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    # Getting started
    st.markdown("## 🚀 Getting Started")
    st.markdown("""
    1. **Upload your data** using the Upload Data section
    2. **Process and clean** your data with our advanced tools
    3. **Explore insights** through interactive dashboards
    4. **Save to database** for future access
    5. **Share results** via email or download
    """)

# Upload data page
def render_upload():
    st.markdown("## 📤 Upload Your Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=ALLOWED_FILE_TYPES,
        help=f"Supported formats: {', '.join(ALLOWED_FILE_TYPES)}"
    )
    
    if uploaded_file is not None:
        # File info
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }
        
        st.markdown("### 📋 File Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", file_details["filename"])
        with col2:
            st.metric("File Type", file_details["filetype"])
        with col3:
            st.metric("File Size", f"{file_details['filesize'] / 1024:.1f} KB")
        
        # Load data
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading your data..."):
                show_loading_animation("Processing file", 2)
                
                df = st.session_state.data_processor.load_file(uploaded_file)
                
                if df is not None:
                    st.session_state.current_data = df
                    st.session_state.file_info = file_details
                    
                    st.success("✅ Data loaded successfully!")
                    
                    # Preview data
                    st.markdown("### 👀 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Basic info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    with col4:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Process data page
def render_process():
    st.markdown("## 🔧 Process Your Data")
    
    if st.session_state.current_data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.current_data
    
    # Processing options
    st.markdown("### ⚙️ Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧹 Data Cleaning")
        remove_duplicates = st.checkbox("Remove duplicate rows")
        handle_missing = st.checkbox("Handle missing values")
        if handle_missing:
            missing_strategy = st.selectbox(
                "Missing value strategy",
                ["drop", "fill_mean", "fill_mode"],
                format_func=lambda x: {
                    "drop": "Drop rows with missing values",
                    "fill_mean": "Fill with mean (numeric columns)",
                    "fill_mode": "Fill with most frequent value"
                }[x]
            )
        else:
            missing_strategy = "drop"
    
    with col2:
        st.markdown("#### 🔧 Data Transformation")
        standardize_text = st.checkbox("Standardize text (lowercase, trim)")
        remove_outliers = st.checkbox("Remove outliers (IQR method)")
    
    # Process button
    if st.button("🚀 Process Data", type="primary"):
        processing_options = {
            'remove_duplicates': remove_duplicates,
            'handle_missing': handle_missing,
            'missing_strategy': missing_strategy,
            'standardize_text': standardize_text,
            'remove_outliers': remove_outliers
        }
        
        with st.spinner("Processing your data..."):
            show_loading_animation("Applying transformations", 3)
            
            processed_df = st.session_state.data_processor.clean_data(df, processing_options)
            st.session_state.current_data = processed_df
            st.session_state.processing_complete = True
            
            st.success("✅ Data processing complete!")
            
            # Show processing log
            st.markdown("### 📝 Processing Log")
            processing_log = st.session_state.data_processor.get_processing_log()
            for log_entry in processing_log[-5:]:  # Show last 5 entries
                st.text(log_entry)
            
            # Before/After comparison
            st.markdown("### 📊 Before vs After")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Data**")
                st.metric("Rows", len(df))
                st.metric("Missing Values", df.isnull().sum().sum())
                st.metric("Duplicates", df.duplicated().sum())
            
            with col2:
                st.markdown("**Processed Data**")
                st.metric("Rows", len(processed_df))
                st.metric("Missing Values", processed_df.isnull().sum().sum())
                st.metric("Duplicates", processed_df.duplicated().sum())
            
            # Preview processed data
            st.markdown("### 👀 Processed Data Preview")
            st.dataframe(processed_df.head(10), use_container_width=True)

# Dashboard page
def render_dashboard():
    st.markdown("## 📊 Interactive Dashboard")

    if st.session_state.current_data is None:
        st.warning("⚠️ Please upload and process data first!")
        return

    df = st.session_state.current_data

    # Generate dashboard
    with st.spinner("Creating dashboard..."):
        dashboard_components = st.session_state.viz_engine.create_dashboard(df)

    if not dashboard_components:
        st.error("Failed to create dashboard")
        return

    # Basic statistics
    if 'basic_stats' in dashboard_components:
        st.markdown("### 📈 Basic Statistics")
        stats = dashboard_components['basic_stats']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{stats['total_rows']:,}")
        with col2:
            st.metric("Total Columns", stats['total_columns'])
        with col3:
            st.metric("Missing Values", f"{stats['missing_values']:,}")
        with col4:
            st.metric("Memory Usage", f"{stats['memory_usage'] / 1024 / 1024:.1f} MB")

    # Data quality overview
    if 'data_quality' in dashboard_components:
        st.markdown("### 🔍 Data Quality Overview")
        st.plotly_chart(dashboard_components['data_quality'], use_container_width=True)

    # Numeric analysis
    if 'numeric_analysis' in dashboard_components:
        st.markdown("### 📊 Numeric Data Analysis")
        numeric_figs = dashboard_components['numeric_analysis']

        if 'distributions' in numeric_figs:
            st.plotly_chart(numeric_figs['distributions'], use_container_width=True)

        if 'box_plots' in numeric_figs:
            st.plotly_chart(numeric_figs['box_plots'], use_container_width=True)

    # Categorical analysis
    if 'categorical_analysis' in dashboard_components:
        st.markdown("### 📋 Categorical Data Analysis")
        cat_figs = dashboard_components['categorical_analysis']

        # Display pie chart if available
        if 'pie_chart' in cat_figs:
            st.plotly_chart(cat_figs['pie_chart'], use_container_width=True)

        # Display bar charts
        for key, fig in cat_figs.items():
            if key != 'pie_chart':
                st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    if 'correlation' in dashboard_components:
        st.markdown("### 🔗 Correlation Analysis")
        st.plotly_chart(dashboard_components['correlation'], use_container_width=True)

    # Time series analysis
    if 'time_series' in dashboard_components:
        st.markdown("### ⏰ Time Series Analysis")
        ts_figs = dashboard_components['time_series']
        for key, fig in ts_figs.items():
            st.plotly_chart(fig, use_container_width=True)

    # Custom plotting section
    st.markdown("### 🎨 Custom Plots")

    col1, col2, col3 = st.columns(3)

    with col1:
        plot_type = st.selectbox("Plot Type", ["scatter", "line", "bar", "histogram"])

    with col2:
        x_column = st.selectbox("X-axis Column", df.columns)

    with col3:
        if plot_type in ["scatter", "line"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            y_column = st.selectbox("Y-axis Column", numeric_cols) if numeric_cols else None
        else:
            y_column = None

    if st.button("Generate Custom Plot"):
        custom_fig = st.session_state.viz_engine.create_custom_plot(
            df, plot_type, x_column, y_column
        )
        st.plotly_chart(custom_fig, use_container_width=True)

# Database page
def render_database():
    st.markdown("## 💾 Database Management")

    # Save current dataset
    if st.session_state.current_data is not None:
        st.markdown("### 💾 Save Current Dataset")

        col1, col2 = st.columns(2)
        with col1:
            dataset_name = st.text_input("Dataset Name", value="My Dataset")
        with col2:
            dataset_description = st.text_area("Description", value="")

        tags_input = st.text_input("Tags (comma-separated)", value="")
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        if st.button("💾 Save to Database"):
            metadata = {
                'name': dataset_name,
                'description': dataset_description,
                'file_size': st.session_state.current_data.memory_usage(deep=True).sum(),
                'file_type': st.session_state.file_info.get('filetype', 'unknown') if hasattr(st.session_state, 'file_info') else 'unknown',
                'processing_log': st.session_state.data_processor.get_processing_log(),
                'user_email': st.session_state.user_email,
                'tags': tags
            }

            dataset_id = st.session_state.db_manager.save_dataset(st.session_state.current_data, metadata)

            if dataset_id:
                st.success(f"✅ Dataset saved with ID: {dataset_id}")
            else:
                st.error("❌ Failed to save dataset")

    # Search and browse datasets
    st.markdown("### 🔍 Browse Saved Datasets")

    col1, col2 = st.columns(2)
    with col1:
        search_query = st.text_input("🔍 Search datasets", placeholder="Enter keywords...")
    with col2:
        filter_by_user = st.checkbox("Show only my datasets")

    # Get datasets
    user_email = st.session_state.user_email if filter_by_user else ""
    datasets = st.session_state.db_manager.search_datasets(search_query, user_email)

    if datasets:
        st.markdown(f"Found {len(datasets)} dataset(s)")

        for dataset in datasets:
            with st.expander(f"📊 {dataset['name']} (ID: {dataset['id']})"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Rows:** {dataset['row_count']:,}")
                    st.write(f"**Columns:** {dataset['column_count']}")

                with col2:
                    st.write(f"**Type:** {dataset['file_type']}")
                    st.write(f"**Size:** {dataset['file_size'] / 1024:.1f} KB")

                with col3:
                    st.write(f"**Uploaded:** {dataset['upload_date']}")
                    st.write(f"**User:** {dataset['user_email']}")

                if dataset['description']:
                    st.write(f"**Description:** {dataset['description']}")

                if dataset['tags']:
                    st.write(f"**Tags:** {', '.join(dataset['tags'])}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📥 Load", key=f"load_{dataset['id']}"):
                        loaded_df = st.session_state.db_manager.load_dataset(dataset['id'])
                        if loaded_df is not None:
                            st.session_state.current_data = loaded_df
                            st.success("✅ Dataset loaded successfully!")
                            st.rerun()

                with col2:
                    if st.button(f"📋 View Details", key=f"details_{dataset['id']}"):
                        metadata = st.session_state.db_manager.get_dataset_metadata(dataset['id'])
                        if metadata:
                            st.json(metadata)

                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{dataset['id']}"):
                        if st.session_state.db_manager.delete_dataset(dataset['id']):
                            st.success("✅ Dataset deleted successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete dataset")
    else:
        st.info("No datasets found. Upload and save some data first!")

# Share results page
def render_share_results():
    st.markdown("## 📧 Share Your Results")

    if st.session_state.current_data is None:
        st.warning("⚠️ Please upload and process data first!")
        return

    df = st.session_state.current_data

    # Email sharing section
    st.markdown("### 📧 Email Sharing")

    col1, col2 = st.columns(2)

    with col1:
        recipient_email = st.text_input("📧 Recipient Email", placeholder="recipient@example.com")
        export_format = st.selectbox("📄 Export Format", ["csv", "excel", "json"])

    with col2:
        include_summary = st.checkbox("Include Data Summary", value=True)
        include_visualizations = st.checkbox("Include Visualizations", value=False)

    if st.button("📧 Send Email", type="primary"):
        if not recipient_email:
            st.error("Please enter a recipient email address")
        elif not st.session_state.email_service.validate_email(recipient_email):
            st.error("Please enter a valid email address")
        else:
            with st.spinner("Sending email..."):
                show_loading_animation("Preparing email", 2)

                # Prepare metadata
                metadata = {
                    'filename': getattr(st.session_state, 'file_info', {}).get('filename', 'processed_data'),
                    'original_rows': len(df),  # This should be original data length
                    'processed_rows': len(df),
                    'columns': len(df.columns),
                    'export_format': export_format,
                    'operations': st.session_state.data_processor.get_processing_log()
                }

                # Send email
                success = st.session_state.email_service.send_processed_data(
                    recipient_email, df, metadata, export_format
                )

                if success:
                    st.success(f"✅ Email sent successfully to {recipient_email}!")
                else:
                    st.error("❌ Failed to send email")

    # Download section
    st.markdown("### 💾 Download Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download CSV"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV File",
                data=csv_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📥 Download Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Processed_Data')
            excel_data = output.getvalue()

            st.download_button(
                label="💾 Download Excel File",
                data=excel_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col3:
        if st.button("📥 Download JSON"):
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="💾 Download JSON File",
                data=json_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # Data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    if df.select_dtypes(include=[np.number]).columns.any():
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("No numeric columns available for statistical summary")

def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Route to appropriate page
    if selected_page == "🏠 Home":
        render_home()
    elif selected_page == "📤 Upload Data":
        render_upload()
    elif selected_page == "🔧 Process Data":
        render_process()
    elif selected_page == "📊 Dashboard":
        render_dashboard()
    elif selected_page == "💾 Database":
        render_database()
    elif selected_page == "📧 Share Results":
        render_share_results()

if __name__ == "__main__":
    main()
