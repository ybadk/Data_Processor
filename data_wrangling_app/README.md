# 📊 Data Wrangling & Analytics Platform

**Profit Projects Online Virtual Assistance**

A comprehensive data processing and visualization application built with Streamlit for companies needing advanced data wrangling, processing, and visualization solutions.

## 🌟 Features

### 📤 Multi-Format Data Upload
- **CSV Files** - Automatic encoding detection
- **Excel Files** - Support for multiple sheets
- **JSON Files** - Nested data normalization
- **PDF Files** - Text extraction and processing
- **DOCX Files** - Document content analysis
- **Text Files** - Delimiter detection and parsing

### 🔧 Advanced Data Processing
- **Data Cleaning**
  - Remove duplicate rows
  - Handle missing values (drop, fill with mean/mode)
  - Standardize text formatting
  - Remove outliers using IQR method

- **Data Transformation**
  - Automatic data type detection
  - Column standardization
  - Text preprocessing
  - Statistical normalization

### 📊 Interactive Dashboards
- **Real-time Visualizations**
  - Distribution plots and histograms
  - Box plots for outlier detection
  - Correlation heatmaps
  - Time series analysis
  - Custom plot generation

- **Statistical Analysis**
  - Descriptive statistics
  - Data quality metrics
  - Missing value analysis
  - Categorical data insights

### 💾 Database Management
- **SQLite Integration**
  - Searchable dataset storage
  - Metadata management
  - Processing history tracking
  - User session management

- **Data Organization**
  - Tag-based categorization
  - Full-text search
  - User-specific filtering
  - Batch operations

### 📧 Sharing & Export
- **Email Integration**
  - Automated result sharing
  - Professional HTML reports
  - Multiple export formats
  - Processing summaries

- **Download Options**
  - CSV export
  - Excel workbooks
  - JSON format
  - Custom formatting

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd data_wrangling_app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the application**
Open your browser and navigate to `http://localhost:8501`

## 📋 Usage Guide

### 1. Upload Your Data
- Navigate to the "Upload Data" section
- Select your file (CSV, Excel, JSON, PDF, or DOCX)
- Review the file information and preview
- Click "Load Data" to process the file

### 2. Process Your Data
- Go to the "Process Data" section
- Select cleaning and transformation options:
  - Remove duplicates
  - Handle missing values
  - Standardize text
  - Remove outliers
- Click "Process Data" to apply transformations
- Review the processing log and before/after comparison

### 3. Explore with Dashboard
- Visit the "Dashboard" section
- View automatic visualizations:
  - Basic statistics
  - Data quality overview
  - Numeric and categorical analysis
  - Correlation matrices
- Create custom plots with the interactive tools

### 4. Save to Database
- Use the "Database" section to save processed datasets
- Add names, descriptions, and tags
- Search and filter saved datasets
- Load previous datasets for further analysis

### 5. Share Results
- Go to "Share Results" to distribute your findings
- Send via email with professional reports
- Download in multiple formats (CSV, Excel, JSON)
- Include visualizations and summaries

## 🏢 Company Information

**Profit Projects Online Virtual Assistance**
- **Location**: Pretoria, Gauteng Province, South Africa
- **Enterprise Number**: K2025200646
- **Email**: kgothatsothooe@gmail.com
- **Services**: Data processing, inventory management, visualization solutions

## 🛠️ Technical Architecture

### Backend Components
- **Data Processor**: Handles file parsing and data cleaning
- **Database Manager**: SQLite operations and metadata management
- **Email Service**: SMTP integration for result sharing
- **Visualization Engine**: Plotly-based chart generation

### Frontend Features
- **Modern UI**: Dark theme with smooth animations
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Live progress tracking and notifications
- **Interactive Elements**: Dynamic charts and user controls

## 📦 Dependencies

### Core Libraries
- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `sqlalchemy` - Database operations

### Data Processing
- `openpyxl` - Excel file handling
- `PyPDF2` - PDF text extraction
- `python-docx` - Word document processing
- `textblob` - Text analysis
- `scikit-learn` - Machine learning utilities

### UI Components
- `streamlit-option-menu` - Navigation menus
- `streamlit-lottie` - Animations
- `extra-streamlit-components` - Additional widgets

## 🚀 Deployment Instructions

### Local Deployment
1. Clone the repository and install dependencies:
   ```bash
   git clone <repository-url>
   cd data_wrangling_app
   pip install -r requirements.txt
   ```
2. Set environment variables (create a .env file or set in your shell):
   ```bash
   EMAIL_USER=your-email@gmail.com
   EMAIL_PASSWORD=your-app-password
   SECRET_KEY=your-secret-key
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment
1. Push your code to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your repo.
3. Set the following environment variables in the Streamlit Cloud app settings:
   - EMAIL_USER
   - EMAIL_PASSWORD
   - SECRET_KEY
   - SMTP_SERVER
   - SMTP_PORT
4. Deploy with one click and access your app online.

### Notes
- For email features, use an app password if using Gmail and enable "less secure apps" or use OAuth2.
- For production, use strong secrets and secure your credentials.

## 🔧 Configuration

### Environment Variables
```bash
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
SECRET_KEY=your-secret-key
```

### Streamlit Configuration
The application includes a custom `.streamlit/config.toml` file with:
- Dark theme settings
- Performance optimizations
- Security configurations

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Configure environment variables
4. Deploy with one click

### Local Production
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Support

For technical support or business inquiries:
- **Email**: kgothatsothooe@gmail.com
- **Company**: Profit Projects Online Virtual Assistance

## 📄 License

This project is proprietary software developed by Profit Projects Online Virtual Assistance.

## 🔄 Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Enhanced visualizations and email integration
- **v1.2.0** - Database management and search features

## 🐞 Known Issues
- Email sending may fail if SMTP credentials are not set or are incorrect. Use a test account for first-time setup.
- PDF and DOCX parsing may not extract all content perfectly, especially with complex formatting.
- Large files (>50MB) may cause slowdowns or memory errors depending on your system.
- Some advanced Excel features (macros, formulas) are not supported.
- The app is designed for single-user or small team use; multi-user concurrency is not fully tested.

## 🚧 Future Work
- Add user authentication and role-based access control.
- Support for additional file types (e.g., XML, Parquet).
- More advanced data transformation options (custom formulas, pivot tables).
- Enhanced visualization options (geospatial, network graphs).
- Automated scheduled email reports.
- Integration with cloud storage (Google Drive, Dropbox).
- Improved mobile responsiveness and accessibility.

---

**Built with ❤️ by Profit Projects Online Virtual Assistance**
