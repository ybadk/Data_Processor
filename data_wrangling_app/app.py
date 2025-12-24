"""
Data Wrangling & Analytics Platform
Profit Projects Online Virtual Assistance

A comprehensive data processing, machine learning, and visualization application built with Streamlit
"""

# Machine Learning and Deep Learning imports
from utils.visualizations import VisualizationEngine
from utils.email_service import EmailService
from utils.database import DatabaseManager
from utils.data_processor import DataProcessor
from utils.ml_integration import MLIntegration
from config import *
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.covariance import EllipticEnvelope
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

# App Modules
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import json
import io
import base64
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules

# Configure Streamlit page
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for dark theme and animations


def load_custom_css():
    st.markdown(
        """
    <style>
    /* Main theme */
    .stApp {
        background-color: #black;
        color: #white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #black;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #black, #blue);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 3s ease-in;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #black, #blue);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0000CD;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.10s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }

    /* slideIn animation kept for notifications */
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    /* Animated Button Styles from Galaxy */
    .galaxy-button {
        position: relative;
        padding: 12px 35px;
        background: linear-gradient(135deg, #0000CD, #000080);
        font-size: 17px;
        font-weight: 600;
        color: #fff;
        border: 2px solid #0000CD;
        border-radius: 50px;
        cursor: pointer;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 205, 0.3);
    }

    .galaxy-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 205, 0.5);
        border-color: #4ECDC4;
    }

    .galaxy-button:active {
        transform: translateY(-1px);
    }

    .galaxy-button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(78, 205, 196, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }

    .galaxy-button:hover::before {
        width: 300px;
        height: 300px;
    }

    /* Notification Animation */
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }

    .notification-success {
        animation: slideInRight 0.5s ease-out, pulse 2s infinite;
    }

    /* Loading Spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes blockLoading {
        0%, 100% {
            opacity: 0.3;
        }
        50% {
            opacity: 1;
        }
    }

    .galaxy-loader {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid rgba(0, 0, 205, 0.1);
        border-top: 4px solid #0000CD;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    /* Enhanced Card Hover Effects */
    .enhanced-card {
        position: relative;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(0, 0, 205, 0.3);
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
        overflow: hidden;
    }

    .enhanced-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(78, 205, 196, 0.1), transparent);
        transition: left 0.5s;
    }

    .enhanced-card:hover::before {
        left: 100%;
    }

    .enhanced-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 205, 0.4);
        border-color: #4ECDC4;
    }

    /* Icon Pulse Animation */
    @keyframes iconPulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.2);
        }
    }

    .icon-animated {
        display: inline-block;
        animation: iconPulse 2s ease-in-out infinite;
    }

    /* Tooltip Styles */
    .galaxy-tooltip {
        position: relative;
        display: inline-block;
    }

    .galaxy-tooltip .tooltiptext {
        visibility: hidden;
        background-color: #0000CD;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 8px 12px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s, transform 0.3s;
        white-space: nowrap;
        font-size: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    .galaxy-tooltip .tooltiptext::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #0000CD transparent transparent transparent;
    }

    .galaxy-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(-5px);
    }

    /* Background Pattern */
    .pattern-bg {
        background-image:
            radial-gradient(circle at 20% 50%, rgba(0, 0, 205, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(78, 205, 196, 0.05) 0%, transparent 50%);
        background-size: 100% 100%;
    }

    /* Notification Styles */
    .notification-success {
        background: linear-gradient(135deg, #1a472a, #2d5f3f);
        border-left: 4px solid #4CAF50;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        color: #fff;
        font-weight: 500;
    }

    .notification-error {
        background: linear-gradient(135deg, #4a1a1a, #5f2d2d);
        border-left: 4px solid #f44336;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        color: #fff;
        font-weight: 500;
    }

    .notification-warning {
        background: linear-gradient(135deg, #4a3a1a, #5f4d2d);
        border-left: 4px solid #ff9800;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
        color: #fff;
        font-weight: 500;
    }

    .notification-info {
        background: linear-gradient(135deg, #1a2a4a, #2d3f5f);
        border-left: 4px solid #2196F3;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        color: #fff;
        font-weight: 500;
    }

    /* Loading animation */
    .loading-spinner {
        border: 4px solid #000080;
        border-top: 4px solid #0000CD;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 5s linear infinite;
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
        animation: fadeIn 0.10s ease-in;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #000080, #0000CD);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #000080, #000080);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Initialize session state


def init_session_state():
    if "data_processor" not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if "email_service" not in st.session_state:
        st.session_state.email_service = EmailService()
    if "viz_engine" not in st.session_state:
        st.session_state.viz_engine = VisualizationEngine()
    if "ml_integration" not in st.session_state:
        st.session_state.ml_integration = MLIntegration()
    if "current_data" not in st.session_state:
        st.session_state.current_data = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    # ML Model states
    if "ml_model" not in st.session_state:
        st.session_state.ml_model = None
    if "ml_predictions" not in st.session_state:
        st.session_state.ml_predictions = None
    if "feature_engineered_data" not in st.session_state:
        st.session_state.feature_engineered_data = None
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = {}
    if "auto_ml_metrics" not in st.session_state:
        st.session_state.auto_ml_metrics = None
    if "show_metrics_overlay" not in st.session_state:
        st.session_state.show_metrics_overlay = False
    if "xgboost_predictions_df" not in st.session_state:
        st.session_state.xgboost_predictions_df = None


# Loading animation


def show_loading_animation(message: str, duration: int = 5):
    """Show animated loading message"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(duration * 10):
        progress_bar.progress((i + 1) / (duration * 10))
        status_text.text(f"{message} {'.' * ((i % 3) + 1)}")
        time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()


# Floating Metrics Overlay
def render_metrics_overlay():
    """Render floating metrics overlay with auto-run ML results using Streamlit components"""
    if st.session_state.auto_ml_metrics:
        # Create a container in the sidebar for the floating metrics
        with st.sidebar:
            st.markdown("---")

            # Header with toggle button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 🤖 Auto-ML Metrics")
            with col2:
                if st.button("✕" if st.session_state.show_metrics_overlay else "📊",
                             key="toggle_metrics",
                             help="Close" if st.session_state.show_metrics_overlay else "Show Metrics"):
                    st.session_state.show_metrics_overlay = not st.session_state.show_metrics_overlay
                    st.rerun()

            # Show metrics if overlay is open
            if st.session_state.show_metrics_overlay:
                metrics = st.session_state.auto_ml_metrics

                # Display regression metrics
                if metrics.get('regression_metrics'):
                    st.markdown("#### 📈 Regression Models")

                    for model_name, model_metrics in metrics['regression_metrics'].items():
                        if 'error' not in model_metrics:
                            with st.container():
                                st.markdown(f"**{model_name}**")
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        label="R² Score",
                                        value=f"{model_metrics.get('R2', 0):.4f}"
                                    )

                                with col2:
                                    st.metric(
                                        label="RMSE",
                                        value=f"{model_metrics.get('RMSE', 0):.4f}"
                                    )

                                with col3:
                                    st.metric(
                                        label="MAE",
                                        value=f"{model_metrics.get('MAE', 0):.4f}"
                                    )

                                st.markdown("---")

                # Display best model
                if metrics.get('best_model'):
                    st.success(f"🏆 **Best Model:** {metrics['best_model']}")
                    st.metric(
                        label="Best Score (R²)",
                        value=f"{metrics['best_score']:.4f}",
                        delta=None
                    )

            st.markdown("---")


# Header section


def render_header():
    st.divider()
    st.markdown(
        f"""
    <div class="main-header">
        <h1>{APP_ICON} {APP_TITLE}</h1>
        <h3>{COMPANY_NAME}</h3>
        <p>Transform your data into actionable insights</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
                <div class="tooltip-container">
  <div class="button-content">
    <span class="text">Share</span>
    <svg
      class="share-icon"
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      width="24"
      height="24"
    >
      <path
        d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92c0-1.61-1.31-2.92-2.92-2.92zM18 4c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zM6 13c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zm12 7.02c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1z"
      ></path>
    </svg>
  </div>
  <div class="tooltip-content">
    <div class="social-icons">
      <a href="https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20Data%20Wrangling%20Platform%20by%20Profit%20Projects%20Online!&url=https://profitprojectsonline.com" target="_blank" class="social-icon twitter">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          width="24"
          height="24"
        >
          <path
            d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"
          ></path>
        </svg>
      </a>
      <a href="https://www.facebook.com/sharer/sharer.php?u=https://profitprojectsonline.com" target="_blank" class="social-icon facebook">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          width="24"
          height="24"
        >
          <path
            d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"
          ></path>
        </svg>
      </a>
      <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://profitprojectsonline.com" target="_blank" class="social-icon linkedin">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          width="24"
          height="24"
        >
          <path
            d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"
          ></path>
        </svg>
      </a>
    </div>
  </div>
</div>

<style>
/* From Uiverse.io by Mohammad-Rahme-576  - Tags: tooltip */
/* Container Styles */
.tooltip-container {
  position: relative;
  display: inline-block;
  font-family: "Arial", sans-serif;
  overflow: visible;
}

/* Button Styles */
.button-content {
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #6e8efb, #a777e3);
  color: white;
  padding: 14px 28px;
  border-radius: 50px;
  cursor: pointer;
  transition:
    background 0.4s cubic-bezier(0.25, 0.8, 0.25, 1),
    transform 0.3s ease,
    box-shadow 0.4s ease;
  box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
  position: relative;
  z-index: 10;
  overflow: hidden;
}

.button-content::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: linear-gradient(
    135deg,
    rgba(110, 142, 251, 0.4),
    rgba(167, 119, 227, 0.4)
  );
  filter: blur(15px);
  opacity: 0;
  transition: opacity 0.5s ease;
  z-index: -1;
}

.button-content::after {
  content: "";
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(
    circle,
    rgba(255, 255, 255, 0.3) 0%,
    rgba(255, 255, 255, 0) 70%
  );
  transform: scale(0);
  transition: transform 0.6s ease-out;
  z-index: -1;
}

.button-content:hover::before {
  opacity: 1;
}

.button-content:hover::after {
  transform: scale(1);
}

.button-content:hover {
  background: linear-gradient(135deg, #a777e3, #6e8efb);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
  transform: translateY(-4px) scale(1.03);
}

.button-content:active {
  transform: translateY(-2px) scale(0.98);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.15);
}

.text {
  font-size: 18px;
  font-weight: 600;
  margin-right: 12px;
  white-space: nowrap;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  transition: letter-spacing 0.3s ease;
}

.button-content:hover .text {
  letter-spacing: 1px;
}

.share-icon {
  fill: white;
  transition:
    transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55),
    fill 0.3s ease;
  filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
}

.button-content:hover .share-icon {
  transform: rotate(180deg) scale(1.1);
  fill: #ffffff;
}

/* Tooltip Styles */
.tooltip-content {
  position: absolute;
  top: 102%;
  left: 50%;
  transform: translateX(-50%) scale(0.8);
  background: white;
  border-radius: 15px;
  padding: 22px;
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
  opacity: 0;
  visibility: hidden;
  transition:
    opacity 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55),
    transform 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55),
    visibility 0.5s ease;
  z-index: 100;
  pointer-events: none;
  backdrop-filter: blur(10px);
  background: rgba(255, 255, 255, 0.9);
}

.tooltip-container:hover .tooltip-content {
  opacity: 1;
  visibility: visible;
  transform: translateX(-50%) scale(1);
  pointer-events: auto;
}

/* Social Icons Styles */
.social-icons {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.social-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #f0f0f0;
  transition:
    transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55),
    background 0.3s ease,
    box-shadow 0.4s ease;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow: hidden;
}

.social-icon::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(
    circle at center,
    rgba(255, 255, 255, 0.8) 0%,
    rgba(255, 255, 255, 0) 70%
  );
  opacity: 0;
  transition: opacity 0.3s ease;
}

.social-icon:hover::before {
  opacity: 1;
}

.social-icon svg {
  width: 24px;
  height: 24px;
  fill: #333;
  transition:
    transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55),
    fill 0.3s ease;
  z-index: 1;
}

.social-icon:hover {
  transform: translateY(-5px) scale(1.1);
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
}

.social-icon:active {
  transform: translateY(-2px) scale(1.05);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
}

.social-icon:hover svg {
  transform: scale(1.2);
  fill: white;
}

.social-icon.twitter:hover {
  background: linear-gradient(135deg, #1da1f2, #1a91da);
}

.social-icon.facebook:hover {
  background: linear-gradient(135deg, #1877f2, #165ed0);
}

.social-icon.linkedin:hover {
  background: linear-gradient(135deg, #0077b5, #005e94);
}

/* Animation for Pulse Effect */
@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(110, 142, 251, 0.4);
  }
  70% {
    box-shadow: 0 0 0 20px rgba(110, 142, 251, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(110, 142, 251, 0);
  }
}

.button-content {
  animation: pulse 3s infinite;
}

/* Hover Ripple Effect */
@keyframes ripple {
  0% {
    transform: scale(0);
    opacity: 1;
  }
  100% {
    transform: scale(4);
    opacity: 0;
  }
}

.button-content::before {
  content: "";
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.3);
  border-radius: inherit;
  transform: scale(0);
  opacity: 0;
}

.button-content:active::before {
  animation: ripple 0.6s linear;
}

/* Tooltip Arrow */
.tooltip-content::before {
  content: "";
  position: absolute;
  top: -10px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 0 10px 10px 10px;
  border-style: solid;
  border-color: transparent transparent rgba(255, 255, 255, 0.9) transparent;
  filter: drop-shadow(0 -3px 3px rgba(0, 0, 0, 0.1));
}

/* Accessibility */
.button-content:focus {
  outline: none;
  box-shadow:
    0 0 0 3px rgba(110, 142, 251, 0.5),
    0 8px 15px rgba(0, 0, 0, 0.1);
}

.button-content:focus:not(:focus-visible) {
  box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
}

/* Responsive Design */
@media (max-width: 768px) {
  .button-content {
    padding: 12px 24px;
    border-radius: 40px;
  }

  .text {
    font-size: 16px;
  }

  .tooltip-content {
    width: 240px;
    padding: 18px;
  }

  .social-icon {
    width: 44px;
    height: 44px;
  }

  .social-icon svg {
    width: 20px;
    height: 20px;
  }
}

@media (max-width: 480px) {
  .button-content {
    padding: 10px 20px;
  }

  .text {
    font-size: 14px;
  }

  .tooltip-content {
    width: 200px;
    padding: 15px;
  }

  .social-icon {
    width: 40px;
    height: 40px;
  }

  .social-icon svg {
    width: 18px;
    height: 18px;
  }
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
  .tooltip-content {
    background: rgba(30, 30, 30, 0.9);
    color: white;
  }

  .tooltip-content::before {
    border-color: transparent transparent rgba(30, 30, 30, 0.9) transparent;
  }

  .social-icon {
    background: #2a2a2a;
  }

  .social-icon svg {
    fill: #e0e0e0;
  }
}

/* Print Styles */
@media print {
  .tooltip-container {
    display: none;
  }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
  .button-content,
  .share-icon,
  .social-icon,
  .tooltip-content {
    transition: none;
  }

  .button-content {
    animation: none;
  }
}

/* Custom Scrollbar for Tooltip Content */
.tooltip-content::-webkit-scrollbar {
  width: 6px;
}

.tooltip-content::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.tooltip-content::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}

.tooltip-content::-webkit-scrollbar-thumb:hover {
  background: #555;
}

</style>
    
                """,
        unsafe_allow_html=True,
    )


# Sidebar navigation


def render_sidebar():
    with st.sidebar:
        st.image("image5.jpg", width=300)

        selected = option_menu(
            menu_title="Navigation",
            options=[
                "Home",
                "Upload Data",
                "Process Data",
                "Feature Engineering",
                "Machine Learning",
                "Dashboard",
                "Database",
                "Share Results",
            ],
            icons=["house", "upload", "gear", "sliders",
                   "robot", "bar-chart", "database", "envelope"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#1a1a1a"},
                "icon": {"color": "#4ECDC4", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#191970",
                },
                "nav-link-selected": {"background-color": "#000080"},
            },
        )

        # User email input
        st.markdown("---")
        st.markdown("### 👤 User Information")
        st.write(
            "Enter your email address below, so you can send updated datasets and visualizations to your personal or business email for offline review."
        )
        user_email = st.text_input(
            "📧 Your Email Address",
            value=st.session_state.user_email,
            placeholder="your.email@example.com",
        )
        st.button("Submit")
        if user_email != st.session_state.user_email:
            st.session_state.user_email = user_email

        # Company info
        st.markdown("---")
        st.markdown("### 🏢 Company Info")
        st.markdown(
            f"""
        **{COMPANY_NAME}**  
        📍 {COMPANY_LOCATION}  
        📧 {COMPANY_EMAIL}  
        🏢 Enterprise: {ENTERPRISE_NUMBER}
        """
        )
        st.markdown(
            """
                    <div class="card">
  <center>
  <div class="profileimage">
    <svg class="pfp" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 122.88 122.88"><g><path d="M61.44,0c8.32,0,16.25,1.66,23.5,4.66l0.11,0.05c7.47,3.11,14.2,7.66,19.83,13.3l0,0c5.66,5.65,10.22,12.42,13.34,19.95 c3.01,7.24,4.66,15.18,4.66,23.49c0,8.32-1.66,16.25-4.66,23.5l-0.05,0.11c-3.12,7.47-7.66,14.2-13.3,19.83l0,0 c-5.65,5.66-12.42,10.22-19.95,13.34c-7.24,3.01-15.18,4.66-23.49,4.66c-8.31,0-16.25-1.66-23.5-4.66l-0.11-0.05 c-7.47-3.11-14.2-7.66-19.83-13.29L18,104.87C12.34,99.21,7.78,92.45,4.66,84.94C1.66,77.69,0,69.76,0,61.44s1.66-16.25,4.66-23.5 l0.05-0.11c3.11-7.47,7.66-14.2,13.29-19.83L18.01,18c5.66-5.66,12.42-10.22,19.94-13.34C45.19,1.66,53.12,0,61.44,0L61.44,0z M16.99,94.47l0.24-0.14c5.9-3.29,21.26-4.38,27.64-8.83c0.47-0.7,0.97-1.72,1.46-2.83c0.73-1.67,1.4-3.5,1.82-4.74 c-1.78-2.1-3.31-4.47-4.77-6.8l-4.83-7.69c-1.76-2.64-2.68-5.04-2.74-7.02c-0.03-0.93,0.13-1.77,0.48-2.52 c0.36-0.78,0.91-1.43,1.66-1.93c0.35-0.24,0.74-0.44,1.17-0.59c-0.32-4.17-0.43-9.42-0.23-13.82c0.1-1.04,0.31-2.09,0.59-3.13 c1.24-4.41,4.33-7.96,8.16-10.4c2.11-1.35,4.43-2.36,6.84-3.04c1.54-0.44-1.31-5.34,0.28-5.51c7.67-0.79,20.08,6.22,25.44,12.01 c2.68,2.9,4.37,6.75,4.73,11.84l-0.3,12.54l0,0c1.34,0.41,2.2,1.26,2.54,2.63c0.39,1.53-0.03,3.67-1.33,6.6l0,0 c-0.02,0.05-0.05,0.11-0.08,0.16l-5.51,9.07c-2.02,3.33-4.08,6.68-6.75,9.31C73.75,80,74,80.35,74.24,80.7 c1.09,1.6,2.19,3.2,3.6,4.63c0.05,0.05,0.09,0.1,0.12,0.15c6.34,4.48,21.77,5.57,27.69,8.87l0.24,0.14 c6.87-9.22,10.93-20.65,10.93-33.03c0-15.29-6.2-29.14-16.22-39.15c-10-10.03-23.85-16.23-39.14-16.23 c-15.29,0-29.14,6.2-39.15,16.22C12.27,32.3,6.07,46.15,6.07,61.44C6.07,73.82,10.13,85.25,16.99,94.47L16.99,94.47L16.99,94.47z"></path></g></svg>
  </div>
  <div class="Name">
    <p>Thapelo Kgothatso Thooe</p>
    <p>Python Web Developer</p>
    <p style="font-size: 12px; margin-top: 10px;">💳 <a href="https://paypal.me/dunduonline" target="_blank" style="color: #00457C; text-decoration: none;">paypal@dunduonline</a></p>
  </div>
  <div class="socialbar">
    <a id="github" href="https://github.com/profitprojectsonline" target="_blank"><svg viewBox="0 0 16 16" class="bi bi-github" fill="currentColor" height="16" width="16" xmlns="http://www.w3.org/2000/svg">
  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
</svg></a>
    &nbsp;
    &nbsp;
    &nbsp;
    <a id="instagram" href="https://instagram.com/profitprojectsonline" target="_blank"><svg viewBox="0 0 16 16" class="bi bi-instagram" fill="currentColor" height="16" width="16" xmlns="http://www.w3.org/2000/svg">
  <path d="M8 0C5.829 0 5.556.01 4.703.048 3.85.088 3.269.222 2.76.42a3.917 3.917 0 0 0-1.417.923A3.927 3.927 0 0 0 .42 2.76C.222 3.268.087 3.85.048 4.7.01 5.555 0 5.827 0 8.001c0 2.172.01 2.444.048 3.297.04.852.174 1.433.372 1.942.205.526.478.972.923 1.417.444.445.89.719 1.416.923.51.198 1.09.333 1.942.372C5.555 15.99 5.827 16 8 16s2.444-.01 3.298-.048c.851-.04 1.434-.174 1.943-.372a3.916 3.916 0 0 0 1.416-.923c.445-.445.718-.891.923-1.417.197-.509.332-1.09.372-1.942C15.99 10.445 16 10.173 16 8s-.01-2.445-.048-3.299c-.04-.851-.175-1.433-.372-1.941a3.926 3.926 0 0 0-.923-1.417A3.911 3.911 0 0 0 13.24.42c-.51-.198-1.092-.333-1.943-.372C10.443.01 10.172 0 7.998 0h.003zm-.717 1.442h.718c2.136 0 2.389.007 3.232.046.78.035 1.204.166 1.486.275.373.145.64.319.92.599.28.28.453.546.598.92.11.281.24.705.275 1.485.039.843.047 1.096.047 3.231s-.008 2.389-.047 3.232c-.035.78-.166 1.203-.275 1.485a2.47 2.47 0 0 1-.599.919c-.28.28-.546.453-.92.598-.28.11-.704.24-1.485.276-.843.038-1.096.047-3.232.047s-2.39-.009-3.233-.047c-.78-.036-1.203-.166-1.485-.276a2.478 2.478 0 0 1-.92-.598 2.48 2.48 0 0 1-.6-.92c-.109-.281-.24-.705-.275-1.485-.038-.843-.046-1.096-.046-3.233 0-2.136.008-2.388.046-3.231.036-.78.166-1.204.276-1.486.145-.373.319-.64.599-.92.28-.28.546-.453.92-.598.282-.11.705-.24 1.485-.276.738-.034 1.024-.044 2.515-.045v.002zm4.988 1.328a.96.96 0 1 0 0 1.92.96.96 0 0 0 0-1.92zm-4.27 1.122a4.109 4.109 0 1 0 0 8.217 4.109 4.109 0 0 0 0-8.217zm0 1.441a2.667 2.667 0 1 1 0 5.334 2.667 2.667 0 0 1 0-5.334z"></path>
</svg></a>
    &nbsp;
    &nbsp;
    &nbsp;
    <a id="facebook" href="https://facebook.com/profitprojectsonline" target="_blank"><svg viewBox="0 0 16 16" class="bi bi-facebook" fill="currentColor" height="16" width="16" xmlns="http://www.w3.org/2000/svg">
  <path d="M16 8.049c0-4.446-3.582-8.05-8-8.05C3.58 0-.002 3.603-.002 8.05c0 4.017 2.926 7.347 6.75 7.951v-5.625h-2.03V8.05H6.75V6.275c0-2.017 1.195-3.131 3.022-3.131.876 0 1.791.157 1.791.157v1.98h-1.009c-.993 0-1.303.621-1.303 1.258v1.51h2.218l-.354 2.326H9.25V16c3.824-.604 6.75-3.934 6.75-7.951z"></path>
</svg></a>
    &nbsp;
    &nbsp;
    &nbsp;
    <a id="twitter" href="https://twitter.com/profitprojects" target="_blank"><svg viewBox="0 0 16 16" class="bi bi-twitter" fill="currentColor" height="16" width="16" xmlns="http://www.w3.org/2000/svg">
  <path d="M5.026 15c6.038 0 9.341-5.003 9.341-9.334 0-.14 0-.282-.006-.422A6.685 6.685 0 0 0 16 3.542a6.658 6.658 0 0 1-1.889.518 3.301 3.301 0 0 0 1.447-1.817 6.533 6.533 0 0 1-2.087.793A3.286 3.286 0 0 0 7.875 6.03a9.325 9.325 0 0 1-6.767-3.429 3.289 3.289 0 0 0 1.018 4.382A3.323 3.323 0 0 1 .64 6.575v.045a3.288 3.288 0 0 0 2.632 3.218 3.203 3.203 0 0 1-.865.115 3.23 3.23 0 0 1-.614-.057 3.283 3.283 0 0 0 3.067 2.277A6.588 6.588 0 0 1 .78 13.58a6.32 6.32 0 0 1-.78-.045A9.344 9.344 0 0 0 5.026 15z"></path>
</svg></a>
    </div></center>
  </div>

<style>
/* From Uiverse.io by aadium - Tags: neumorphism, profile, card */
.card {
  width: 230px;
  height: 280px;
  border-radius: 2em;
  padding: 10px;
  background-color: #191919;
  box-shadow: 5px 5px 30px rgb(4, 4, 4),
                   -5px -5px 30px rgb(57, 57, 57);
}

.profileimage {
  background-color: transparent;
  border: none;
  margin-top: 20px;
  border-radius: 5em;
  width: 100px;
  height: 100px;
}

.pfp {
  border-radius: 35em;
  fill: white;
}

.Name {
  color: white;
  font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif;
  padding: 15px;
  font-size: 20px;
  margin-top: 10px;
}

.socialbar {
  background-color: #191919;
  border-radius: 3em;
  width: 90%;
  padding: 14px;
  margin-top: 15px;
  color: white;
  box-shadow: 3px 3px 15px rgb(0, 0, 0),
                   -3px -3px 15px rgb(58, 58, 58);
}

.card a {
  transition: 0.4s;
  color: white
}

#github:hover {
  color: #c9510c;
}

#instagram:hover {
  color: #d62976;
}

#facebook:hover {
  color: #3b5998;
}

#twitter:hover {
  color: #00acee;
}
</style>

                    """,
            unsafe_allow_html=True,
        )

        return selected


# Home page


def render_home():

    st.markdown("""
        <div style="text-align: center; padding: 20px;" class="pattern-bg">
            <h1 style="font-size: 3em; background: linear-gradient(135deg, #0000CD, #4ECDC4);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; margin-bottom: 10px;">
                <span class="icon-animated">📊</span> DWAP
            </h1>
            <p style="font-size: 1.2em; color: #4ECDC4; margin-top: -10px;">Data Wrangling & Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="enhanced-card">
            <h3><span class="icon-animated">📤</span> Upload</h3>
            <p>Support for multiple file formats including CSV, Excel, JSON, PDF, and DOCX</p>
        </div>
        <div class="item-hints">
  <div class="hint" data-position="4">
    <span class="hint-radius"></span>
    <span class="hint-dot">Tip</span>
    <div class="hint-content do--split-children">
      <p>Request for machine learning prediction models to be used on historical datasets. send a direct email for assistance</p>
    </div>
  </div>
</div>

<style>
/* From Uiverse.io by vnuny  - Tags: simple, tooltip, animation, black, animated */
.item-hints {
  --purple: #720c8f;
  cursor: pointer;
  display: flex;
  justify-content: flex-start;
  padding-right: 170px;
}
.item-hints .hint {
  margin: 150px auto;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
}
.item-hints .hint-dot {
  z-index: 3;
  border: 1px solid #ffe4e4;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  -webkit-transform: translate(-0%, -0%) scale(0.95);
  transform: translate(-0%, -0%) scale(0.95);
  margin: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}
.item-hints .hint-radius {
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  position: absolute;
  top: 50%;
  left: 50%;
  margin: -125px 0 0 -125px;
  opacity: 0;
  visibility: hidden;
  -webkit-transform: scale(0);
  transform: scale(0);
}
.item-hints .hint[data-position="1"] .hint-content {
  top: 85px;
  left: 50%;
  margin-left: 56px;
}
.item-hints .hint-content {
  width: 300px;
  position: absolute;
  z-index: 5;
  padding: 35px 0;
  opacity: 0;
  transition: opacity 0.7s ease, visibility 0.7s ease;
  color: #fff;
  visibility: hidden;
  pointer-events: none;
}
.item-hints .hint:hover .hint-content {
  position: absolute;
  z-index: 5;
  padding: 35px 0;
  opacity: 1;
  -webkit-transition: opacity 0.7s ease, visibility 0.7s ease;
  transition: opacity 0.7s ease, visibility 0.7s ease;
  color: #fff;
  visibility: visible;
  pointer-events: none;
}
.item-hints .hint-content::before {
  width: 0px;
  bottom: 29px;
  left: 0;
  content: "";
  background-color: #fff;
  height: 1px;
  position: absolute;
  transition: width 0.4s;
}
.item-hints .hint:hover .hint-content::before {
  width: 180px;
  transition: width 0.4s;
}
.item-hints .hint-content::after {
  -webkit-transform-origin: 0 50%;
  transform-origin: 0 50%;
  -webkit-transform: rotate(-225deg);
  transform: rotate(-225deg);
  bottom: 29px;
  left: 0;
  width: 80px;
  content: "";
  background-color: #fff;
  height: 1px;
  position: absolute;
  opacity: 1;
  -webkit-transition: opacity 0.5s ease;
  transition: opacity 0.5s ease;
  -webkit-transition-delay: 0s;
  transition-delay: 0s;
}
.item-hints .hint:hover .hint-content::after {
  opacity: 1;
  visibility: visible;
}
.item-hints .hint[data-position="4"] .hint-content {
  bottom: 85px;
  left: 50%;
  margin-left: 56px;
}

</style>
    
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="enhanced-card">
            <h3><span class="icon-animated">⚙️</span> Process</h3>
            <p>Advanced data cleaning, transformation, and quality improvement tools</p>
        </div>
        <div class="item-hints">
  <div class="hint" data-position="4">
    <span class="hint-radius"></span>
    <span class="hint-dot">Tip</span>
    <div class="hint-content do--split-children">
      <p>Request for detailed data exploratory analysis using correlated features from historical or present datasets.</p>
    </div>
  </div>
</div>

<style>
/* From Uiverse.io by vnuny  - Tags: simple, tooltip, animation, black, animated */
.item-hints {
  --purple: #720c8f;
  cursor: pointer;
  display: flex;
  justify-content: flex-start;
  padding-right: 170px;
}
.item-hints .hint {
  margin: 150px auto;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
}
.item-hints .hint-dot {
  z-index: 3;
  border: 1px solid #ffe4e4;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  -webkit-transform: translate(-0%, -0%) scale(0.95);
  transform: translate(-0%, -0%) scale(0.95);
  margin: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}
.item-hints .hint-radius {
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  position: absolute;
  top: 50%;
  left: 50%;
  margin: -125px 0 0 -125px;
  opacity: 0;
  visibility: hidden;
  -webkit-transform: scale(0);
  transform: scale(0);
}
.item-hints .hint[data-position="1"] .hint-content {
  top: 85px;
  left: 50%;
  margin-left: 56px;
}
.item-hints .hint-content {
  width: 300px;
  position: absolute;
  z-index: 5;
  padding: 35px 0;
  opacity: 0;
  transition: opacity 0.7s ease, visibility 0.7s ease;
  color: #fff;
  visibility: hidden;
  pointer-events: none;
}
.item-hints .hint:hover .hint-content {
  position: absolute;
  z-index: 5;
  padding: 35px 0;
  opacity: 1;
  -webkit-transition: opacity 0.7s ease, visibility 0.7s ease;
  transition: opacity 0.7s ease, visibility 0.7s ease;
  color: #fff;
  visibility: visible;
  pointer-events: none;
}
.item-hints .hint-content::before {
  width: 0px;
  bottom: 29px;
  left: 0;
  content: "";
  background-color: #fff;
  height: 1px;
  position: absolute;
  transition: width 0.4s;
}
.item-hints .hint:hover .hint-content::before {
  width: 180px;
  transition: width 0.4s;
}
.item-hints .hint-content::after {
  -webkit-transform-origin: 0 50%;
  transform-origin: 0 50%;
  -webkit-transform: rotate(-225deg);
  transform: rotate(-225deg);
  bottom: 29px;
  left: 0;
  width: 80px;
  content: "";
  background-color: #fff;
  height: 1px;
  position: absolute;
  opacity: 1;
  -webkit-transition: opacity 0.5s ease;
  transition: opacity 0.5s ease;
  -webkit-transition-delay: 0s;
  transition-delay: 0s;
}
.item-hints .hint:hover .hint-content::after {
  opacity: 1;
  visibility: visible;
}
.item-hints .hint[data-position="4"] .hint-content {
  bottom: 85px;
  left: 50%;
  margin-left: 56px;
}

</style>
    
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="enhanced-card">
            <h3><span class="icon-animated">📈</span> Visualize</h3>
            <p>Interactive dashboards and comprehensive data analysis</p>
        </div>
        <div class="item-hints">
  <div class="hint" data-position="4">
    <span class="hint-radius"></span>
    <span class="hint-dot">Tip</span>
    <div class="hint-content do--split-children">
      <p>Request for detailed visualizations with feature engineering and machine learning algorithms implemented on datasets.</p>
    </div>
  </div>
</div>

<style>
/* From Uiverse.io by vnuny  - Tags: simple, tooltip, animation, black, animated */
.item-hints {
  --purple: #720c8f;
  cursor: pointer;
  display: flex;
  justify-content: flex-start;
  padding-right: 170px;
}
.item-hints .hint {
  margin: 150px auto;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
}
.item-hints .hint-dot {
  z-index: 3;
  border: 1px solid #ffe4e4;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  -webkit-transform: translate(-0%, -0%) scale(0.95);
  transform: translate(-0%, -0%) scale(0.95);
  margin: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}
.item-hints .hint-radius {
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  position: absolute;
  top: 50%;
  left: 50%;
  margin: -125px 0 0 -125px;
  opacity: 0;
  visibility: hidden;
  -webkit-transform: scale(0);
  transform: scale(0);
}
.item-hints .hint[data-position="1"] .hint-content {
  top: 85px;
  left: 50%;
  margin-left: 56px;
}
.item-hints .hint-content {
  width: 300px;
  position: absolute;
  z-index: 5;
  padding: 35px 0;
  opacity: 0;
  transition: opacity 0.7s ease, visibility 0.7s ease;
  color: #fff;
  visibility: hidden;
  pointer-events: none;
}
.item-hints .hint:hover .hint-content {
  position: absolute;
  z-index: 5;
  padding: 35px 0;
  opacity: 1;
  -webkit-transition: opacity 0.7s ease, visibility 0.7s ease;
  transition: opacity 0.7s ease, visibility 0.7s ease;
  color: #fff;
  visibility: visible;
  pointer-events: none;
}
.item-hints .hint-content::before {
  width: 0px;
  bottom: 29px;
  left: 0;
  content: "";
  background-color: #fff;
  height: 1px;
  position: absolute;
  transition: width 0.4s;
}
.item-hints .hint:hover .hint-content::before {
  width: 180px;
  transition: width 0.4s;
}
.item-hints .hint-content::after {
  -webkit-transform-origin: 0 50%;
  transform-origin: 0 50%;
  -webkit-transform: rotate(-225deg);
  transform: rotate(-225deg);
  bottom: 29px;
  left: 0;
  width: 80px;
  content: "";
  background-color: #fff;
  height: 1px;
  position: absolute;
  opacity: 1;
  -webkit-transition: opacity 0.5s ease;
  transition: opacity 0.5s ease;
  -webkit-transition-delay: 0s;
  transition-delay: 0s;
}
.item-hints .hint:hover .hint-content::after {
  opacity: 1;
  visibility: visible;
}
.item-hints .hint[data-position="4"] .hint-content {
  bottom: 85px;
  left: 50%;
  margin-left: 56px;
}

</style>
    
        """,
            unsafe_allow_html=True,
        )
    st.divider()

    # Statistics
    stats = st.session_state.db_manager.get_statistics()

    st.markdown("##  Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Datasets", stats.get("total_datasets", 0))
    with col2:
        st.metric(
            "Data Processed", f"{stats.get('total_size', 0) / 1024 / 1024:.1f} MB"
        )
    with col3:
        st.metric("Recent Uploads", stats.get("recent_uploads", 0))
    with col4:
        st.metric("File Types", len(stats.get("datasets_by_type", {})))
    st.divider()

    # Features overview
    st.markdown("""
        <h2 style="text-align: center; color: #4ECDC4;">
            <span class="icon-animated">✨</span> Key Features
        </h2>
    """, unsafe_allow_html=True)

    features = [
        ("<span class='icon-animated'>🔄</span>", "**Automated Data Cleaning**",
         "Remove duplicates, handle missing values, standardize formats"),
        ("<span class='icon-animated'>📊</span>", "**Interactive Visualizations**",
         "Dynamic charts, graphs, and statistical analysis"),
        ("<span class='icon-animated'>🗄️</span>", "**Searchable Database**",
         "Organize and retrieve your processed datasets"),
        ("<span class='icon-animated'>📧</span>",
         "**Email Integration**", "Share results directly via email"),
        ("<span class='icon-animated'>⚡</span>", "**Real-time Processing**",
         "Fast data transformation with progress tracking"),
        ("<span class='icon-animated'>🎨</span>", "**Modern Interface**",
         "Intuitive design with smooth animations"),
        ("<span class='icon-animated'>🤖</span>", "**Machine Learning**",
         "Auto-run ML models with XGBoost and linear regression"),
        ("<span class='icon-animated'>🚀</span>", "**Feature Engineering**",
         "Advanced feature scaling, encoding, and selection"),
    ]

    for icon, title, desc in features:
        st.markdown(f"""
            <div class="enhanced-card" style="margin: 10px 0; padding: 15px;">
                <h4>{icon} {title}</h4>
                <p style="color: #aaa; margin: 5px 0 0 0;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    # Getting started
    st.divider()
    col11, col22 = st.columns(2)
    with col11:
        st.markdown("##  Getting Started")
        st.markdown(
            """
        1. **Upload your data** using the Upload Data section
        2. **Process and clean** your data with our advanced tools
        3. **Explore insights** through interactive dashboards
        4. **Save to database** for future access
        5. **Share results** via email or download
        """
        )

    with col22:
        st.markdown(
            """
                <div class="wrapper">
  <input id="tab-1" name="slider" type="radio" />
  <input checked="" id="tab-2" name="slider" type="radio" />
  <input id="tab-3" name="slider" type="radio" />
  <header>
    <label class="tab-1" for="tab-1">Starter</label>
    <label class="tab-2" for="tab-2">Professional</label>
    <label class="tab-3" for="tab-3">Enterprise</label>
    <div class="slider"></div>
  </header>
  <div class="card-area">
    <div class="cards">
      <div class="row row-1">
        <div class="price-details">
          <span class="price">299</span>
          <p>Data Wrangling Package</p>
        </div>
        <ul class="features">
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            <span>Data Cleaning &amp; Preprocessing</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Missing Value Treatment</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Basic Data Visualization (5 charts)</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>CSV/Excel Export Support</span>
          </li>
        </ul>
      </div>
      <div class="row">
        <div class="price-details">
          <span class="price">799</span>
          <p>Data Science Package</p>
        </div>
        <ul class="features">
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>All Starter Features Included</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Feature Engineering &amp; Selection</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Machine Learning Model Training</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Interactive Dashboard Reports</span>
          </li>
        </ul>
      </div>
      <div class="row">
        <div class="price-details">
          <span class="price">2499</span>
          <p>Enterprise AI Package</p>
        </div>
        <ul class="features">
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>All Professional Features</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Deep Learning &amp; Neural Networks</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Predictive Analytics &amp; Forecasting</span>
          </li>
          <li>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white text-2xs bg-green-500 rounded-full mr-2 p-1">
              <polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Custom API &amp; Model Deployment</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
  <button>Contact Us</button>
</div>

<style>
/* From Uiverse.io by Manish-Tamang  - Tags: card, gradients, svg, html, css */
.wrapper {
  width: 400px;
  background: #000000;
  border-radius: 16px;
  padding: 30px;
  box-shadow: 10px 10px 15px rgba(0, 0, 0, 0.05);
}
.wrapper header {
  height: 55px;
  display: flex;
  align-items: center;
  border: 1px solid #ccc;
  border-radius: 30px;
  position: relative;
}
header label {
  height: 100%;
  z-index: 2;
  width: 30%;
  display: flex;
  cursor: pointer;
  font-size: 18px;
  position: relative;
  align-items: center;
  justify-content: center;
  transition: color 0.3s ease;
}
#tab-1:checked ~ header .tab-1,
#tab-2:checked ~ header .tab-2,
#tab-3:checked ~ header .tab-3 {
  color: #fff;
}
header label:nth-child(2) {
  width: 40%;
}
header .slider {
  position: absolute;
  height: 85%;
  border-radius: inherit;
  background: linear-gradient(145deg, #d5a3ff 0%, #77a5f8 100%);
  transition: all 0.3s ease;
}
#tab-1:checked ~ header .slider {
  left: 0%;
  width: 90px;
  transform: translateX(5%);
}
#tab-2:checked ~ header .slider {
  left: 50%;
  width: 120px;
  transform: translateX(-50%);
}
#tab-3:checked ~ header .slider {
  left: 100%;
  width: 95px;
  transform: translateX(-105%);
}
.wrapper input[type="radio"] {
  display: none;
}
.card-area {
  overflow: hidden;
}
.card-area .cards {
  display: flex;
  width: 300%;
}
.cards .row {
  width: 33.4%;
}
.cards .row-1 {
  transition: all 0.3s ease;
}
#tab-1:checked ~ .card-area .cards .row-1 {
  margin-left: 0%;
}
#tab-2:checked ~ .card-area .cards .row-1 {
  margin-left: -33.4%;
}
#tab-3:checked ~ .card-area .cards .row-1 {
  margin-left: -66.8%;
}
.row .price-details {
  margin: 20px 0;
  text-align: center;
  padding-bottom: 25px;
  border-bottom: 1px solid #e6e6e6;
}
.price-details .price {
  font-size: 65px;
  font-weight: 600;
  position: relative;
  font-family: "Noto Sans", sans-serif;
}
.price-details .price::before,
.price-details .price::after {
  position: absolute;
  font-weight: 400;
  font-family: "Poppins", sans-serif;
}
.price-details .price::before {
  content: "R";
  left: -27px;
  top: 17px;
  font-size: 28px;
}
.price-details .price::after {
  content: "/mon";
  right: -33px;
  bottom: 17px;
  font-size: 13px;
}
.price-details p {
  font-size: 18px;
  margin-top: 5px;
}
.row .features li {
  display: flex;
  font-size: 15px;
  list-style: none;
  margin-bottom: 10px;
  align-items: center;
}
.features li i {
  background: linear-gradient(#d5a3ff 0%, #77a5f8 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.features li span {
  margin-left: 10px;
}
.wrapper button {
  width: 100%;
  border-radius: 25px;
  border: none;
  outline: none;
  height: 50px;
  font-size: 18px;
  color: #fff;
  cursor: pointer;
  margin-top: 20px;
  background: linear-gradient(145deg, #d5a3ff 0%, #77a5f8 100%);
  transition: transform 0.3s ease;
}
.wrapper button:hover {
  transform: scale(0.98);
}

</style>
    
                """,
            unsafe_allow_html=True,
        )

    st.divider()


# Upload data page


def render_upload():
    st.markdown("## 📤 Upload Your Data")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=ALLOWED_FILE_TYPES,
        help=f"Supported formats: {', '.join(ALLOWED_FILE_TYPES)}",
    )

    if uploaded_file is not None:
        # File info
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size,
        }

        st.divider()
        st.markdown("###  File Information")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", file_details["filename"])
        with col2:
            st.metric("File Type", file_details["filetype"])
        with col3:
            st.metric("File Size", f"{file_details['filesize'] / 1024:.1f} KB")

        # Load data
        if st.button(" Load Data", type="primary"):
            with st.spinner("Loading your data..."):
                show_loading_animation("Processing file", 2)

                df = st.session_state.data_processor.load_file(uploaded_file)

                if df is not None:
                    st.session_state.current_data = df
                    st.session_state.file_info = file_details

                    st.markdown("""
                        <div class="notification-success" style="animation: slideInRight 0.5s ease-out;">
                            <span class="icon-animated">✅</span> Data loaded successfully!
                        </div>
                    """, unsafe_allow_html=True)

                    # Preview data
                    st.divider()
                    st.markdown("###  Data Preview")
                    st.divider()
                    st.dataframe(df.head(10), use_container_width=True)
                    st.divider()

                    # Basic info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    with col4:
                        st.metric(
                            "Memory Usage",
                            f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
                        )


# Process data page


def render_process():
    st.markdown("##  Process Your Data")
    st.divider()

    if st.session_state.current_data is None:
        st.warning("⚠️ Please upload data first!")
        return

    df = st.session_state.current_data

    # Processing options
    st.markdown("###  Processing Options")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("####  Data Cleaning")
        remove_duplicates = st.checkbox("Remove duplicate rows")
        handle_missing = st.checkbox("Handle missing values")
        if handle_missing:
            missing_strategy = st.selectbox(
                "Missing value strategy",
                ["drop", "fill_mean", "fill_mode"],
                format_func=lambda x: {
                    "drop": "Drop rows with missing values",
                    "fill_mean": "Fill with mean (numeric columns)",
                    "fill_mode": "Fill with most frequent value",
                }[x],
            )
        else:
            missing_strategy = "drop"

    with col2:
        st.markdown("####  Data Transformation")
        standardize_text = st.checkbox("Standardize text (lowercase, trim)")
        remove_outliers = st.checkbox("Remove outliers (IQR method)")

    # Process button
    if st.button(" Process Data", type="primary"):
        processing_options = {
            "remove_duplicates": remove_duplicates,
            "handle_missing": handle_missing,
            "missing_strategy": missing_strategy,
            "standardize_text": standardize_text,
            "remove_outliers": remove_outliers,
        }

        with st.spinner("Processing your data..."):
            show_loading_animation("Applying transformations", 5)

            processed_df = st.session_state.data_processor.clean_data(
                df, processing_options
            )
            st.session_state.current_data = processed_df
            st.session_state.processing_complete = True

            st.markdown("""
                <div class="notification-success" style="animation: slideInRight 0.5s ease-out;">
                    <span class="icon-animated">✅</span> Data processing complete!
                </div>
            """, unsafe_allow_html=True)

            # Auto-run ML models
            st.markdown("""
                <div class="notification-info" style="animation: slideInRight 0.6s ease-out;">
                    <span class="icon-animated">🤖</span> Running automatic ML model evaluation...
                </div>
            """, unsafe_allow_html=True)
            try:
                auto_ml_results = st.session_state.ml_integration.auto_run_models(
                    processed_df)
                st.session_state.auto_ml_metrics = auto_ml_results
                st.session_state.show_metrics_overlay = True

                if auto_ml_results.get('best_model'):
                    st.success(
                        f"✅ Auto-ML complete! Best model: {auto_ml_results['best_model']} (R² Score: {auto_ml_results['best_score']:.4f})")
            except Exception as e:
                st.warning(f"Auto-ML evaluation skipped: {str(e)}")

            # Show processing log
            st.divider()
            st.markdown("###  Processing Log")
            processing_log = st.session_state.data_processor.get_processing_log()
            for log_entry in processing_log[-5:]:  # Show last 5 entries
                st.text(log_entry)

            # Before/After comparison
            st.divider()
            st.markdown("###  Before vs After")
            st.divider()
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
            st.divider()
            st.markdown("###  Processed Data Preview")
            st.divider()
            st.dataframe(processed_df.head(10), use_container_width=True)
            st.divider()


# Dashboard page


def render_dashboard():

    st.markdown("##  Dashboard")
    st.divider()

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
    if "basic_stats" in dashboard_components:
        st.markdown("###  Basic Statistics")
        st.divider()
        stats = dashboard_components["basic_stats"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{stats['total_rows']:,}")
        with col2:
            st.metric("Total Columns", stats["total_columns"])
        with col3:
            st.metric("Missing Values", f"{stats['missing_values']:,}")
        with col4:
            st.metric("Memory Usage",
                      f"{stats['memory_usage'] / 1024 / 1024:.1f} MB")

    # Data quality overview
    if "data_quality" in dashboard_components:
        st.divider()
        st.markdown("###  Data Quality Overview")
        st.divider()
        st.plotly_chart(
            dashboard_components["data_quality"], use_container_width=True)

    # Numeric analysis
    if "numeric_analysis" in dashboard_components:
        st.divider()
        st.markdown("###  Numeric Data Analysis")
        st.divider()
        numeric_figs = dashboard_components["numeric_analysis"]

        if "distributions" in numeric_figs:
            st.plotly_chart(
                numeric_figs["distributions"], use_container_width=True)

        if "box_plots" in numeric_figs:
            st.plotly_chart(numeric_figs["box_plots"],
                            use_container_width=True)

    # Categorical analysis
    if "categorical_analysis" in dashboard_components:
        st.markdown("###  Categorical Data Analysis")
        cat_figs = dashboard_components["categorical_analysis"]

        # Display pie chart if available
        if "pie_chart" in cat_figs:
            st.plotly_chart(cat_figs["pie_chart"], use_container_width=True)

        # Display bar charts
        for key, fig in cat_figs.items():
            if key != "pie_chart":
                st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    if "correlation" in dashboard_components:
        st.divider()
        st.markdown("###  Correlation Analysis")
        st.divider()
        st.plotly_chart(
            dashboard_components["correlation"], use_container_width=True)

    # Time series analysis
    if "time_series" in dashboard_components:
        st.divider()
        st.markdown("### Time Series Analysis")
        st.divider()
        ts_figs = dashboard_components["time_series"]
        for key, fig in ts_figs.items():
            st.plotly_chart(fig, use_container_width=True)

    # Custom plotting section
    st.divider()
    st.markdown("###  Custom Plots")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        plot_type = st.selectbox(
            "Plot Type", ["scatter", "line", "bar", "histogram"])

    with col2:
        x_column = st.selectbox("X-axis Column", df.columns)

    with col3:
        if plot_type in ["scatter", "line"]:
            numeric_cols = df.select_dtypes(
                include=[np.number]).columns.tolist()
            y_column = (
                st.selectbox("Y-axis Column",
                             numeric_cols) if numeric_cols else None
            )
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
    st.divider()

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
                "name": dataset_name,
                "description": dataset_description,
                "file_size": st.session_state.current_data.memory_usage(
                    deep=True
                ).sum(),
                "file_type": (
                    st.session_state.file_info.get("filetype", "unknown")
                    if hasattr(st.session_state, "file_info")
                    else "unknown"
                ),
                "processing_log": st.session_state.data_processor.get_processing_log(),
                "user_email": st.session_state.user_email,
                "tags": tags,
            }

            dataset_id = st.session_state.db_manager.save_dataset(
                st.session_state.current_data, metadata
            )

            if dataset_id:
                st.markdown(f"""
                    <div class="notification-success" style="animation: slideInRight 0.5s ease-out;">
                        <span class="icon-animated">✅</span> Dataset saved with ID: {dataset_id}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="notification-error" style="animation: slideInRight 0.5s ease-out;">
                        <span class="icon-animated">❌</span> Failed to save dataset
                    </div>
                """, unsafe_allow_html=True)

    # Search and browse datasets

    st.markdown("### 🔍 Browse Saved Datasets")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        search_query = st.text_input(
            "🔍 Search datasets", placeholder="Enter keywords..."
        )
    with col2:
        filter_by_user = st.checkbox("Show only my datasets")

    # Get datasets
    user_email = st.session_state.user_email if filter_by_user else ""
    datasets = st.session_state.db_manager.search_datasets(
        search_query, user_email)

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
                # st.write(f"**Size:** {dataset['file_size'] / 1024:.1f} KB")

                with col3:
                    st.write(f"**Uploaded:** {dataset['upload_date']}")
                    st.write(f"**User:** {dataset['user_email']}")

                if dataset["description"]:
                    st.write(f"**Description:** {dataset['description']}")

                if dataset["tags"]:
                    st.write(f"**Tags:** {', '.join(dataset['tags'])}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📥 Load", key=f"load_{dataset['id']}"):
                        loaded_df = st.session_state.db_manager.load_dataset(
                            dataset["id"]
                        )
                        if loaded_df is not None:
                            st.session_state.current_data = loaded_df
                            st.success("✅ Dataset loaded successfully!")
                            st.rerun()

                with col2:
                    if st.button(f"📋 View Details", key=f"details_{dataset['id']}"):
                        metadata = st.session_state.db_manager.get_dataset_metadata(
                            dataset["id"]
                        )
                        if metadata:
                            st.json(metadata)

                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{dataset['id']}"):
                        if st.session_state.db_manager.delete_dataset(dataset["id"]):
                            st.success("✅ Dataset deleted successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete dataset")
    else:
        st.info("No datasets found. Upload and save some data first!")


# Share results page


def render_share_results():
    st.divider()
    st.markdown("## 📧 Share Your Results")
    st.divider()

    if st.session_state.current_data is None:
        st.warning("⚠️ Please upload and process data first!")
        return

    df = st.session_state.current_data

    # Email sharing section
    st.divider()
    st.markdown("### 📧 Email Sharing")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        recipient_email = st.text_input(
            "📧 Recipient Email", placeholder="recipient@example.com"
        )
        export_format = st.selectbox(
            "📄 Export Format", ["csv", "excel", "json"])

    with col2:
        include_summary = st.checkbox("Include Data Summary", value=True)
        include_visualizations = st.checkbox(
            "Include Visualizations", value=False)

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
                    "filename": getattr(st.session_state, "file_info", {}).get(
                        "filename", "processed_data"
                    ),
                    # This should be original data length
                    "original_rows": len(df),
                    "processed_rows": len(df),
                    "columns": len(df.columns),
                    "export_format": export_format,
                    "operations": st.session_state.data_processor.get_processing_log(),
                }

                # Send email
                success = st.session_state.email_service.send_processed_data(
                    recipient_email, df, metadata, export_format
                )

                if success:
                    st.markdown(f"""
                        <div class="notification-success" style="animation: slideInRight 0.5s ease-out;">
                            <span class="icon-animated">✅</span> Email sent successfully to {recipient_email}!
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="notification-error" style="animation: slideInRight 0.5s ease-out;">
                            <span class="icon-animated">❌</span> Failed to send email
                        </div>
                    """, unsafe_allow_html=True)

    # Download section
    st.divider()
    st.markdown("### 💾 Download Data")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download CSV"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV File",
                data=csv_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("📥 Download Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Processed_Data")
            excel_data = output.getvalue()

            st.download_button(
                label="💾 Download Excel File",
                data=excel_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col3:
        if st.button("📥 Download JSON"):
            json_data = df.to_json(orient="records", indent=2)
            st.download_button(
                label="💾 Download JSON File",
                data=json_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    # Data preview
    st.divider()
    st.markdown("### 👀 Data Preview")
    st.divider()
    st.dataframe(df.head(20), use_container_width=True)

    # Summary statistics
    st.divider()
    st.markdown("### 📊 Summary Statistics")
    st.divider()
    if df.select_dtypes(include=[np.number]).columns.any():
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("No numeric columns available for statistical summary")


# Feature Engineering page
def render_feature_engineering():
    st.markdown("## 🔬 Feature Engineering")
    st.divider()

    if st.session_state.current_data is None:
        st.warning("⚠️ Please upload data first!")
        return

    df = st.session_state.current_data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Feature Engineering Options
    st.markdown("### 🛠️ Feature Engineering Tools")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Scaling & Normalization",
        "🔢 Polynomial Features",
        "🏷️ Encoding",
        "🎯 Feature Selection",
        "🔍 Outlier Detection"
    ])

    with tab1:
        st.markdown("#### Scaling & Normalization")
        if not numeric_cols:
            st.warning("No numeric columns available for scaling")
        else:
            scaling_method = st.selectbox(
                "Select Scaling Method",
                ["Standard Scaler (Z-score)",
                 "Min-Max Scaler", "Robust Scaler"]
            )
            cols_to_scale = st.multiselect("Select columns to scale", numeric_cols, default=numeric_cols[:3] if len(
                numeric_cols) >= 3 else numeric_cols)

            if st.button("Apply Scaling", key="apply_scaling"):
                if cols_to_scale:
                    if scaling_method == "Standard Scaler (Z-score)":
                        scaler = StandardScaler()
                    elif scaling_method == "Min-Max Scaler":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()

                    scaled_data = scaler.fit_transform(df[cols_to_scale])
                    for i, col in enumerate(cols_to_scale):
                        df[f"{col}_scaled"] = scaled_data[:, i]

                    st.session_state.feature_engineered_data = df
                    st.success(
                        f"✅ Applied {scaling_method} to {len(cols_to_scale)} columns!")

                    # Show before/after comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Data**")
                        st.dataframe(df[cols_to_scale].head())
                    with col2:
                        st.markdown("**Scaled Data**")
                        st.dataframe(
                            df[[f"{c}_scaled" for c in cols_to_scale]].head())

    with tab2:
        st.markdown("#### Polynomial Features")
        if not numeric_cols:
            st.warning("No numeric columns available for polynomial features")
        else:
            poly_cols = st.multiselect("Select columns for polynomial features", numeric_cols,
                                       default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
            degree = st.slider("Polynomial Degree", 2, 5, 2)
            include_interaction = st.checkbox(
                "Include interaction features only", value=False)

            if st.button("Generate Polynomial Features", key="gen_poly"):
                if poly_cols:
                    poly = PolynomialFeatures(
                        degree=degree, include_bias=False, interaction_only=include_interaction)
                    poly_data = poly.fit_transform(df[poly_cols])
                    feature_names = poly.get_feature_names_out(poly_cols)

                    poly_df = pd.DataFrame(poly_data, columns=feature_names)
                    st.success(
                        f"✅ Generated {len(feature_names)} polynomial features!")
                    st.dataframe(poly_df.head())

                    # Merge with original data
                    for col in poly_df.columns:
                        if col not in df.columns:
                            df[col] = poly_df[col].values
                    st.session_state.feature_engineered_data = df

    with tab3:
        st.markdown("#### Categorical Encoding")
        if not categorical_cols:
            st.warning("No categorical columns available for encoding")
        else:
            encoding_method = st.selectbox(
                "Select Encoding Method",
                ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"]
            )
            cols_to_encode = st.multiselect(
                "Select columns to encode", categorical_cols)

            if st.button("Apply Encoding", key="apply_encoding"):
                if cols_to_encode:
                    if encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        for col in cols_to_encode:
                            df[f"{col}_encoded"] = le.fit_transform(
                                df[col].astype(str))
                    elif encoding_method == "One-Hot Encoding":
                        encoded_df = pd.get_dummies(
                            df[cols_to_encode], prefix=cols_to_encode)
                        df = pd.concat([df, encoded_df], axis=1)

                    st.session_state.feature_engineered_data = df
                    st.success(
                        f"✅ Applied {encoding_method} to {len(cols_to_encode)} columns!")
                    st.dataframe(df.head())

    with tab4:
        st.markdown("#### Feature Selection")
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for feature selection")
        else:
            target_col = st.selectbox("Select Target Column", numeric_cols)
            feature_cols = [c for c in numeric_cols if c != target_col]
            k_features = st.slider("Number of features to select", 1, len(
                feature_cols), min(5, len(feature_cols)))

            if st.button("Analyze Feature Importance", key="feat_importance"):
                X = df[feature_cols].fillna(0)
                y = df[target_col].fillna(0)

                selector = SelectKBest(score_func=f_classif, k=k_features)
                selector.fit(X, y)

                scores = pd.DataFrame({
                    'Feature': feature_cols,
                    'Score': selector.scores_
                }).sort_values('Score', ascending=False)

                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(x=scores['Feature'],
                           y=scores['Score'], marker_color='#0000CD')
                ])
                fig.update_layout(
                    title="Feature Importance Scores",
                    xaxis_title="Features",
                    yaxis_title="Score",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Top Features:**")
                st.dataframe(scores.head(k_features))

    with tab5:
        st.markdown("#### Outlier Detection")
        if not numeric_cols:
            st.warning("No numeric columns available for outlier detection")
        else:
            outlier_col = st.selectbox(
                "Select column for outlier detection", numeric_cols)
            method = st.selectbox("Detection Method", [
                                  "IQR Method", "Elliptic Envelope", "Z-Score"])

            if st.button("Detect Outliers", key="detect_outliers"):
                col_data = df[outlier_col].dropna()

                if method == "IQR Method":
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = (col_data < lower_bound) | (
                        col_data > upper_bound)
                elif method == "Elliptic Envelope":
                    detector = EllipticEnvelope(
                        contamination=0.1, random_state=42)
                    predictions = detector.fit_predict(
                        col_data.values.reshape(-1, 1))
                    outliers = predictions == -1
                else:  # Z-Score
                    z_scores = (col_data - col_data.mean()) / col_data.std()
                    outliers = np.abs(z_scores) > 3

                n_outliers = outliers.sum()
                st.info(
                    f"Found {n_outliers} outliers ({n_outliers/len(col_data)*100:.1f}%)")

                # Visualization
                fig = go.Figure()
                fig.add_trace(
                    go.Box(y=col_data, name=outlier_col, marker_color='#0000CD'))
                fig.update_layout(
                    title=f"Box Plot - {outlier_col}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    # Save engineered features
    st.divider()
    if st.session_state.feature_engineered_data is not None:
        st.markdown("### 💾 Save Engineered Features")
        if st.button("Apply to Current Dataset"):
            st.session_state.current_data = st.session_state.feature_engineered_data
            st.success("✅ Feature engineered data saved to current dataset!")


# Machine Learning page
def render_machine_learning():
    st.markdown("## 🤖 Machine Learning Models")
    st.divider()

    if st.session_state.current_data is None:
        st.warning("⚠️ Please upload data first!")
        return

    df = st.session_state.current_data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for ML models")
        return

    # ML Model tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Regression",
        "🎯 Classification",
        "🔮 Clustering",
        "🧠 Neural Networks",
        "🚀 XGBoost",
        "📊 Linear Models"
    ])

    with tab1:
        st.markdown("### 📈 Regression Models")
        target_col = st.selectbox(
            "Select Target Variable", numeric_cols, key="reg_target")
        feature_cols = st.multiselect("Select Feature Variables", [
                                      c for c in numeric_cols if c != target_col], key="reg_features")

        model_type = st.selectbox("Select Model", [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "K-Neighbors Regressor",
            "Decision Tree Regressor",
            "MLP Regressor"
        ])

        test_size = st.slider("Test Size (%)", 10, 40,
                              20, key="reg_test") / 100

        if st.button("Train Regression Model", key="train_reg"):
            if feature_cols:
                X = df[feature_cols].fillna(0)
                y = df[target_col].fillna(0)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42)

                # Select and train model
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Ridge Regression":
                    model = Ridge(alpha=1.0)
                elif model_type == "Lasso Regression":
                    model = Lasso(alpha=1.0)
                elif model_type == "Random Forest Regressor":
                    model = RandomForestRegressor(
                        n_estimators=100, random_state=42)
                elif model_type == "Gradient Boosting Regressor":
                    model = GradientBoostingRegressor(
                        n_estimators=100, random_state=42)
                elif model_type == "K-Neighbors Regressor":
                    model = KNeighborsRegressor(n_neighbors=5)
                elif model_type == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(random_state=42)
                else:
                    model = MLPRegressor(hidden_layer_sizes=(
                        100, 50), max_iter=500, random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.session_state.trained_models['regression'] = model
                st.session_state.model_metrics['regression'] = {
                    'MSE': mse, 'R2': r2}

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("R² Score", f"{r2:.4f}")

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                              name='Predictions', marker=dict(color='#0000CD')))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max(
                )], mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))
                fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual",
                                  yaxis_title="Predicted", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values(
                        'Importance', ascending=False)
                    fig_imp = go.Figure(data=[go.Bar(
                        x=importance_df['Feature'], y=importance_df['Importance'], marker_color='#0000CD')])
                    fig_imp.update_layout(
                        title="Feature Importance", template="plotly_dark")
                    st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        st.markdown("### 🎯 Classification Models")
        st.info("For classification, ensure your target variable has discrete classes")

        target_col_cls = st.selectbox(
            "Select Target Variable", df.columns.tolist(), key="cls_target")
        feature_cols_cls = st.multiselect(
            "Select Feature Variables", numeric_cols, key="cls_features")

        cls_model_type = st.selectbox("Select Model", [
            "Logistic Regression",
            "Random Forest Classifier",
            "Gradient Boosting Classifier",
            "K-Neighbors Classifier",
            "Decision Tree Classifier",
            "Naive Bayes",
            "SVM Classifier",
            "MLP Classifier"
        ])

        if st.button("Train Classification Model", key="train_cls"):
            if feature_cols_cls:
                X = df[feature_cols_cls].fillna(0)
                y = LabelEncoder().fit_transform(
                    df[target_col_cls].fillna('Unknown'))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                if cls_model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif cls_model_type == "Random Forest Classifier":
                    model = RandomForestClassifier(
                        n_estimators=100, random_state=42)
                elif cls_model_type == "Gradient Boosting Classifier":
                    model = GradientBoostingClassifier(
                        n_estimators=100, random_state=42)
                elif cls_model_type == "K-Neighbors Classifier":
                    model = KNeighborsClassifier(n_neighbors=5)
                elif cls_model_type == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(random_state=42)
                elif cls_model_type == "Naive Bayes":
                    model = GaussianNB()
                elif cls_model_type == "SVM Classifier":
                    model = SVC(kernel='rbf', random_state=42)
                else:
                    model = MLPClassifier(hidden_layer_sizes=(
                        100, 50), max_iter=500, random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.session_state.trained_models['classification'] = model
                st.metric("Accuracy", f"{accuracy:.2%}")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm, colorscale='Blues', text=cm, texttemplate='%{text}'))
                fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted",
                                     yaxis_title="Actual", template="plotly_dark")
                st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.markdown("### 🔮 Clustering Models")
        cluster_cols = st.multiselect(
            "Select Features for Clustering", numeric_cols, key="cluster_features")
        cluster_method = st.selectbox(
            "Select Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])

        if cluster_method == "K-Means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        if st.button("Perform Clustering", key="perform_cluster"):
            if len(cluster_cols) >= 2:
                X = df[cluster_cols].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if cluster_method == "K-Means":
                    model = KMeans(n_clusters=n_clusters,
                                   random_state=42, n_init=10)
                elif cluster_method == "DBSCAN":
                    model = DBSCAN(eps=0.5, min_samples=5)
                else:
                    model = AgglomerativeClustering(n_clusters=3)

                clusters = model.fit_predict(X_scaled)
                df['Cluster'] = clusters

                if len(set(clusters)) > 1:
                    silhouette = silhouette_score(X_scaled, clusters)
                    st.metric("Silhouette Score", f"{silhouette:.4f}")

                # PCA for visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                fig = go.Figure()
                for cluster in set(clusters):
                    mask = clusters == cluster
                    fig.add_trace(go.Scatter(
                        x=X_pca[mask, 0], y=X_pca[mask, 1], mode='markers', name=f'Cluster {cluster}'))
                fig.update_layout(title="Cluster Visualization (PCA)",
                                  xaxis_title="PC1", yaxis_title="PC2", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Cluster Distribution:**")
                cluster_counts = pd.Series(
                    clusters).value_counts().sort_index()
                st.bar_chart(cluster_counts)

    with tab4:
        st.markdown("### 🧠 Neural Network")
        st.info("Configure a Multi-Layer Perceptron neural network")

        nn_target = st.selectbox(
            "Select Target", numeric_cols, key="nn_target")
        nn_features = st.multiselect("Select Features", [
                                     c for c in numeric_cols if c != nn_target], key="nn_features")

        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.text_input(
                "Hidden Layer Sizes (comma-separated)", "100,50")
            max_iterations = st.slider("Max Iterations", 100, 1000, 500)
        with col2:
            learning_rate = st.select_slider("Learning Rate", options=[
                                             0.0001, 0.001, 0.01, 0.1], value=0.001)
            activation = st.selectbox("Activation Function", [
                                      "relu", "tanh", "logistic"])

        task_type = st.radio(
            "Task Type", ["Regression", "Classification"], horizontal=True)

        if st.button("Train Neural Network", key="train_nn"):
            if nn_features:
                X = df[nn_features].fillna(0)
                y = df[nn_target].fillna(0)

                hidden = tuple(int(x.strip())
                               for x in hidden_layers.split(','))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if task_type == "Regression":
                    model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iterations,
                                         learning_rate_init=learning_rate, activation=activation, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.metric("MSE", f"{mse:.4f}")
                    st.metric("R² Score", f"{r2:.4f}")
                else:
                    y_encoded = LabelEncoder().fit_transform(y)
                    y_train_enc = LabelEncoder().fit_transform(y_train)
                    y_test_enc = LabelEncoder().fit_transform(y_test)
                    model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iterations,
                                          learning_rate_init=learning_rate, activation=activation, random_state=42)
                    model.fit(X_train_scaled, y_train_enc)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test_enc, y_pred)
                    st.metric("Accuracy", f"{acc:.2%}")

                st.session_state.trained_models['neural_network'] = model

                # Loss curve
                if hasattr(model, 'loss_curve_'):
                    fig_loss = go.Figure(data=go.Scatter(
                        y=model.loss_curve_, mode='lines', line=dict(color='#0000CD')))
                    fig_loss.update_layout(
                        title="Training Loss Curve", xaxis_title="Iteration", yaxis_title="Loss", template="plotly_dark")
                    st.plotly_chart(fig_loss, use_container_width=True)

    # XGBoost Tab
    with tab5:
        st.markdown("### 🚀 XGBoost - Advanced Gradient Boosting")
        st.info(
            "XGBoost is a powerful gradient boosting algorithm that excels at structured/tabular data.")

        xgb_target = st.selectbox(
            "Select Target Variable", numeric_cols, key="xgb_target")
        xgb_features = st.multiselect("Select Feature Variables",
                                      [c for c in numeric_cols if c != xgb_target],
                                      key="xgb_features")

        col1, col2 = st.columns(2)
        with col1:
            xgb_task = st.radio(
                "Task Type", ["Regression", "Classification"], key="xgb_task")
            max_depth = st.slider("Max Depth", 2, 10, 3, key="xgb_depth")
        with col2:
            n_estimators = st.slider(
                "Number of Estimators", 50, 500, 100, key="xgb_est")
            learning_rate = st.select_slider("Learning Rate",
                                             options=[0.01, 0.05,
                                                      0.1, 0.2, 0.3],
                                             value=0.1, key="xgb_lr")

        col_train, col_pred = st.columns(2)

        with col_train:
            if st.button("🎯 Train XGBoost Model", key="train_xgb"):
                if xgb_features:
                    with st.spinner("Training XGBoost model..."):
                        params = {
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'n_estimators': n_estimators,
                            'random_state': 42
                        }

                        result = st.session_state.ml_integration.train_xgboost_model(
                            df, xgb_target, xgb_features,
                            task_type=xgb_task.lower(),
                            params=params
                        )

                        if 'error' not in result:
                            st.markdown("""
                                <div class="notification-success" style="animation: slideInRight 0.5s ease-out;">
                                    <span class="icon-animated">✅</span> XGBoost model trained successfully!
                                </div>
                            """, unsafe_allow_html=True)

                            # Display metrics
                            st.markdown("#### 📊 Model Metrics")
                            metrics = result['metrics']

                            if xgb_task == "Regression":
                                col_m1, col_m2, col_m3 = st.columns(3)
                                with col_m1:
                                    st.metric(
                                        "R² Score", f"{metrics['R2']:.4f}")
                                with col_m2:
                                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                                with col_m3:
                                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                            else:
                                col_m1, col_m2, col_m3 = st.columns(3)
                                with col_m1:
                                    st.metric(
                                        "Accuracy", f"{metrics['Accuracy']:.4f}")
                                with col_m2:
                                    st.metric(
                                        "Precision", f"{metrics['Precision']:.4f}")
                                with col_m3:
                                    st.metric(
                                        "Recall", f"{metrics['Recall']:.4f}")

                            st.info(
                                f"💾 Model saved to: {result['model_path']}")
                        else:
                            st.error(f"Error: {result['error']}")

        with col_pred:
            if st.button("🔮 Make Predictions", key="xgb_predict"):
                if xgb_features:
                    with st.spinner("Making predictions..."):
                        predictions_df = st.session_state.ml_integration.make_predictions(
                            df, xgb_features, model_name='xgboost'
                        )

                        if predictions_df is not None:
                            st.session_state.xgboost_predictions_df = predictions_df
                            st.markdown("""
                                <div class="notification-success" style="animation: slideInRight 0.5s ease-out;">
                                    <span class="icon-animated">✅</span> Predictions generated!
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown("#### 📊 Predictions vs Original Data")
                            st.dataframe(predictions_df.head(
                                20), use_container_width=True)

                            # Download predictions
                            csv = predictions_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv,
                                file_name="xgboost_predictions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(
                                "No trained model found. Please train a model first.")

    # Linear Models Tab
    with tab6:
        st.markdown("### 📊 Statistical Linear Models")
        st.info("Advanced linear regression models using statsmodels library.")

        lin_target = st.selectbox(
            "Select Target Variable", numeric_cols, key="lin_target")
        lin_features = st.multiselect("Select Feature Variables",
                                      [c for c in numeric_cols if c != lin_target],
                                      key="lin_features")

        if st.button("📈 Run Linear Models Analysis", key="run_linear"):
            if lin_features:
                with st.spinner("Running linear models analysis..."):
                    results = st.session_state.ml_integration.run_linear_models(
                        df, lin_target, lin_features
                    )

                    if 'error' not in results:
                        # OLS Results
                        if 'OLS' in results and 'error' not in results['OLS']:
                            st.markdown(
                                "#### 📊 Ordinary Least Squares (OLS) Results")
                            ols = results['OLS']

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "R-Squared", f"{ols['r_squared']:.4f}")
                            with col2:
                                st.metric("Adj. R-Squared",
                                          f"{ols['adj_r_squared']:.4f}")
                            with col3:
                                st.metric("F-Statistic",
                                          f"{ols['f_statistic']:.4f}")

                            st.markdown("**Model Parameters:**")
                            params_df = pd.DataFrame(list(ols['params'].items()),
                                                     columns=['Variable', 'Coefficient'])
                            st.dataframe(params_df, use_container_width=True)

                            with st.expander("📄 View Full Summary"):
                                st.text(ols['summary'])

                        # WLS Results
                        if 'WLS' in results and 'error' not in results['WLS']:
                            st.markdown(
                                "#### 📊 Weighted Least Squares (WLS) Results")
                            wls = results['WLS']

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "R-Squared", f"{wls['r_squared']:.4f}")
                            with col2:
                                st.metric("Adj. R-Squared",
                                          f"{wls['adj_r_squared']:.4f}")
                    else:
                        st.error(f"Error: {results['error']}")


def main():
    # Load custom CSS
    load_custom_css()

    # Initialize session state
    init_session_state()

    # Render header
    render_header()

    # Render floating metrics overlay
    render_metrics_overlay()

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to appropriate page
    if selected_page == "Home":
        render_home()
    elif selected_page == "Upload Data":
        render_upload()
    elif selected_page == "Process Data":
        render_process()
    elif selected_page == "Feature Engineering":
        render_feature_engineering()
    elif selected_page == "Machine Learning":
        render_machine_learning()
    elif selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "Database":
        render_database()
    elif selected_page == "Share Results":
        render_share_results()


if __name__ == "__main__":
    main()
