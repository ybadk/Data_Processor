"""
Test script for the Data Wrangling Application
Run this to verify all components are working correctly
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_processor():
    """Test the data processor functionality"""
    print("Testing Data Processor...")
    
    try:
        from utils.data_processor import DataProcessor
        
        # Create test data
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Alice', None],
            'age': [25, 30, 35, 25, 40],
            'salary': [50000, 60000, 70000, 50000, 80000],
            'department': ['IT', 'HR', 'IT', 'IT', 'Finance']
        })
        
        processor = DataProcessor()
        
        # Test data cleaning
        options = {
            'remove_duplicates': True,
            'handle_missing': True,
            'missing_strategy': 'drop',
            'standardize_text': True,
            'remove_outliers': False
        }
        
        cleaned_data = processor.clean_data(test_data, options)
        print(f"✅ Data Processor: Original {len(test_data)} rows, cleaned {len(cleaned_data)} rows")
        
        # Test summary generation
        summary = processor.get_data_summary(test_data)
        print(f"✅ Data Summary: {len(summary)} metrics generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Processor Error: {str(e)}")
        return False

def test_database():
    """Test the database functionality"""
    print("Testing Database Manager...")
    
    try:
        from utils.database import DatabaseManager
        
        db_manager = DatabaseManager("test_database.db")
        
        # Test database initialization
        stats = db_manager.get_statistics()
        print(f"✅ Database: Initialized with {stats.get('total_datasets', 0)} datasets")
        
        # Clean up test database
        if os.path.exists("test_database.db"):
            os.remove("test_database.db")
        
        return True
        
    except Exception as e:
        print(f"❌ Database Error: {str(e)}")
        return False

def test_email_service():
    """Test the email service functionality"""
    print("Testing Email Service...")
    
    try:
        from utils.email_service import EmailService
        
        email_service = EmailService()
        
        # Test email validation
        valid_email = email_service.validate_email("test@example.com")
        invalid_email = email_service.validate_email("invalid-email")
        
        if valid_email and not invalid_email:
            print("✅ Email Service: Email validation working correctly")
            return True
        else:
            print("❌ Email Service: Email validation failed")
            return False
        
    except Exception as e:
        print(f"❌ Email Service Error: {str(e)}")
        return False

def test_visualizations():
    """Test the visualization engine"""
    print("Testing Visualization Engine...")
    
    try:
        from utils.visualizations import VisualizationEngine
        
        # Create test data
        test_data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        viz_engine = VisualizationEngine()
        
        # Test dashboard creation
        dashboard = viz_engine.create_dashboard(test_data)
        
        if dashboard and 'basic_stats' in dashboard:
            print("✅ Visualization Engine: Dashboard creation successful")
            return True
        else:
            print("❌ Visualization Engine: Dashboard creation failed")
            return False
        
    except Exception as e:
        print(f"❌ Visualization Engine Error: {str(e)}")
        return False

def test_imports():
    """Test all required imports"""
    print("Testing Required Imports...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sqlalchemy',
        'openpyxl',
        'PyPDF2',
        'docx',
        'streamlit_option_menu'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Missing packages: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def main():
    """Run all tests"""
    print("🧪 Running Data Wrangling Application Tests\n")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_processor,
        test_database,
        test_email_service,
        test_visualizations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    main()
