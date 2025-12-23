"""
Configuration file for the Data Wrangling Application
Profit Projects Online Virtual Assistance
"""

import os
from pathlib import Path

# Company Information
COMPANY_NAME = "Profit Projects Online Virtual Assistance"
COMPANY_EMAIL = "kgothatsothooe@gmail.com"
ENTERPRISE_NUMBER = "K2025200646"
COMPANY_LOCATION = "Pretoria, Gauteng Province, South Africa"

# Application Settings
APP_TITLE = "DWAP"
APP_ICON = "📊"
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# File Upload Settings
MAX_FILE_SIZE = 200  # MB
ALLOWED_FILE_TYPES = [
    'csv', 'xlsx', 'xls', 'json', 'txt', 'pdf', 'docx', 'doc'
]

# Database Settings
DATABASE_URL = "sqlite:///data_wrangling.db"
DATABASE_PATH = Path("data_wrangling.db")

# Email Settings (for production, use environment variables)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = os.getenv("EMAIL_USER", COMPANY_EMAIL)
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

# UI Theme Settings
THEME = {
    "background_color": "#000000",
    "text_color": "#FFFFFF",
    "primary_color": "#FF6B6B",
    "secondary_color": "#4ECDC4",
    "accent_color": "#45B7D1",
    "success_color": "#96CEB4",
    "warning_color": "#FFEAA7",
    "error_color": "#DDA0DD"
}

# Animation Settings
ANIMATION_SPEED = 1000  # milliseconds
LOADING_MESSAGES = [
    "Processing your data...",
    "Analyzing patterns...",
    "Cleaning and organizing...",
    "Generating insights...",
    "Almost ready...",
    "Finalizing results..."
]

# Data Processing Settings
CHUNK_SIZE = 10000  # for large file processing
MAX_ROWS_DISPLAY = 1000
DEFAULT_ENCODING = 'utf-8'

# Visualization Settings
PLOT_HEIGHT = 400
PLOT_WIDTH = 800
COLOR_PALETTE = ['0000CD', '#000088',
                 '#00FFFF', '#000080', '#ADD8E6', '#87CEFA']

# Security Settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
SESSION_TIMEOUT = 3600  # seconds

# API Settings
API_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
