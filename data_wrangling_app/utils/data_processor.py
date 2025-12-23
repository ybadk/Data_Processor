"""
Data Processing Utilities for the Data Wrangling Application
Handles file uploads, data cleaning, and transformation
"""
#Machine Learning modules
from datasets import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#App Modules
import pandas as pd
import numpy as np
import streamlit as st
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import PyPDF2
import docx
from textblob import TextBlob
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for handling data processing operations"""

    def __init__(self):
        self.supported_formats = ['csv', 'xlsx',
                                  'xls', 'json', 'txt', 'pdf', 'docx']
        self.processed_data = None
        self.original_data = None
        self.processing_log = []

    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load and parse uploaded file into a pandas DataFrame"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                return self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._load_excel(uploaded_file)
            elif file_extension == 'json':
                return self._load_json(uploaded_file)
            elif file_extension == 'txt':
                return self._load_text(uploaded_file)
            elif file_extension == 'pdf':
                return self._load_pdf(uploaded_file)
            elif file_extension in ['docx', 'doc']:
                return self._load_docx(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None

        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            st.error(f"Error loading file: {str(e)}")
            return None

    def _load_csv(self, file) -> pd.DataFrame:
        """Load CSV file with encoding detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    self._log_action(
                        f"Successfully loaded CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, try with error handling
            file.seek(0)
            df = pd.read_csv(file, encoding='utf-8', errors='ignore')
            self._log_action("Loaded CSV with error handling")
            return df

        except Exception as e:
            raise Exception(f"Failed to load CSV: {str(e)}")

    def _load_excel(self, file) -> pd.DataFrame:
        """Load Excel file"""
        try:
            # Read all sheets and let user choose
            excel_file = pd.ExcelFile(file)

            if len(excel_file.sheet_names) == 1:
                df = pd.read_excel(file, sheet_name=0)
            else:
                # For multiple sheets, read the first one by default
                # In a full implementation, you'd let the user choose
                df = pd.read_excel(file, sheet_name=0)
                st.info(
                    f"Multiple sheets found. Using sheet: {excel_file.sheet_names[0]}")

            self._log_action(f"Successfully loaded Excel file")
            return df

        except Exception as e:
            raise Exception(f"Failed to load Excel: {str(e)}")

    def _load_json(self, file) -> pd.DataFrame:
        """Load JSON file"""
        try:
            file.seek(0)
            data = json.load(file)

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.json_normalize(data)
            else:
                raise ValueError("JSON format not supported")

            self._log_action("Successfully loaded JSON file")
            return df

        except Exception as e:
            raise Exception(f"Failed to load JSON: {str(e)}")

    def _load_text(self, file) -> pd.DataFrame:
        """Load text file and attempt to parse it"""
        try:
            file.seek(0)
            content = file.read().decode('utf-8')

            # Try to detect delimiter
            lines = content.split('\n')
            if len(lines) > 1:
                # Check for common delimiters
                delimiters = [',', '\t', ';', '|']
                for delimiter in delimiters:
                    if delimiter in lines[0]:
                        # Create a StringIO object and read as CSV
                        string_data = io.StringIO(content)
                        df = pd.read_csv(string_data, delimiter=delimiter)
                        self._log_action(
                            f"Loaded text file with delimiter: {delimiter}")
                        return df

            # If no delimiter found, create a single column DataFrame
            df = pd.DataFrame({'text': lines})
            self._log_action("Loaded text file as single column")
            return df

        except Exception as e:
            raise Exception(f"Failed to load text file: {str(e)}")

    def _load_pdf(self, file) -> pd.DataFrame:
        """Extract text from PDF and create DataFrame"""
        try:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)

            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_content.append({
                    'page': page_num + 1,
                    'text': text.strip()
                })

            df = pd.DataFrame(text_content)
            self._log_action(
                f"Extracted text from {len(pdf_reader.pages)} PDF pages")
            return df

        except Exception as e:
            raise Exception(f"Failed to load PDF: {str(e)}")

    def _load_docx(self, file) -> pd.DataFrame:
        """Extract text from DOCX and create DataFrame"""
        try:
            file.seek(0)
            doc = docx.Document(file)

            paragraphs = []
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    paragraphs.append({
                        'paragraph_number': i + 1,
                        'text': paragraph.text.strip()
                    })

            df = pd.DataFrame(paragraphs)
            self._log_action(
                f"Extracted {len(paragraphs)} paragraphs from DOCX")
            return df

        except Exception as e:
            raise Exception(f"Failed to load DOCX: {str(e)}")

    def clean_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Clean data based on user-selected options"""
        cleaned_df = df.copy()

        try:
            if options.get('remove_duplicates', False):
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed_rows = initial_rows - len(cleaned_df)
                self._log_action(f"Removed {removed_rows} duplicate rows")

            if options.get('handle_missing', False):
                missing_strategy = options.get('missing_strategy', 'drop')
                if missing_strategy == 'drop':
                    cleaned_df = cleaned_df.dropna()
                    self._log_action("Dropped rows with missing values")
                elif missing_strategy == 'fill_mean':
                    numeric_cols = cleaned_df.select_dtypes(
                        include=[np.number]).columns
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                        cleaned_df[numeric_cols].mean())
                    self._log_action("Filled missing numeric values with mean")
                elif missing_strategy == 'fill_mode':
                    for col in cleaned_df.columns:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode(
                        ).iloc[0] if not cleaned_df[col].mode().empty else 'Unknown')
                    self._log_action("Filled missing values with mode")

            if options.get('standardize_text', False):
                text_cols = cleaned_df.select_dtypes(
                    include=['object']).columns
                for col in text_cols:
                    cleaned_df[col] = cleaned_df[col].astype(
                        str).str.strip().str.lower()
                self._log_action("Standardized text columns")

            if options.get('remove_outliers', False):
                numeric_cols = cleaned_df.select_dtypes(
                    include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (
                        cleaned_df[col] <= upper_bound)]
                self._log_action("Removed outliers using IQR method")

            self.processed_data = cleaned_df
            return cleaned_df

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            st.error(f"Error cleaning data: {str(e)}")
            return df

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        try:
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
                'categorical_summary': {}
            }

            # Add categorical summary
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                summary['categorical_summary'][col] = {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head().to_dict()
                }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {}

    def _log_action(self, action: str):
        """Log processing actions"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.processing_log.append(f"[{timestamp}] {action}")
        logger.info(action)

    def get_processing_log(self) -> List[str]:
        """Get the processing log"""
        return self.processing_log

    def export_data(self, df: pd.DataFrame, format: str = 'csv') -> bytes:
        """Export processed data in specified format"""
        try:
            if format == 'csv':
                return df.to_csv(index=False).encode('utf-8')
            elif format == 'excel':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False,
                                sheet_name='Processed_Data')
                return output.getvalue()
            elif format == 'json':
                return df.to_json(orient='records', indent=2).encode('utf-8')
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise Exception(f"Failed to export data: {str(e)}")
