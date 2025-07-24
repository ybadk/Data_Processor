"""
Database utilities for the Data Wrangling Application
Handles data storage, retrieval, and search functionality
"""

import sqlite3
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import logging
from pathlib import Path
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for the data wrangling application"""
    
    def __init__(self, db_path: str = "data_wrangling.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with self.engine.connect() as conn:
                # Create datasets table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        description TEXT,
                        file_hash TEXT UNIQUE,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        row_count INTEGER,
                        column_count INTEGER,
                        file_type TEXT,
                        processing_log TEXT,
                        user_email TEXT,
                        tags TEXT
                    )
                """))
                
                # Create processing_history table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_id INTEGER,
                        operation TEXT,
                        parameters TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_email TEXT,
                        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                    )
                """))
                
                # Create user_sessions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE,
                        user_email TEXT,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        datasets_processed INTEGER DEFAULT 0
                    )
                """))
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except SQLAlchemyError as e:
            logger.error(f"Database initialization error: {str(e)}")
            st.error(f"Database error: {str(e)}")
    
    def save_dataset(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Optional[int]:
        """Save a dataset to the database"""
        try:
            # Generate file hash for uniqueness
            file_hash = self._generate_hash(df)
            
            # Save the actual data
            table_name = f"dataset_{file_hash[:8]}"
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            
            # Save metadata
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO datasets (
                        name, description, file_hash, file_size, row_count, 
                        column_count, file_type, processing_log, user_email, tags
                    ) VALUES (
                        :name, :description, :file_hash, :file_size, :row_count,
                        :column_count, :file_type, :processing_log, :user_email, :tags
                    )
                """), {
                    'name': metadata.get('name', 'Unnamed Dataset'),
                    'description': metadata.get('description', ''),
                    'file_hash': file_hash,
                    'file_size': metadata.get('file_size', 0),
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'file_type': metadata.get('file_type', 'unknown'),
                    'processing_log': json.dumps(metadata.get('processing_log', [])),
                    'user_email': metadata.get('user_email', ''),
                    'tags': json.dumps(metadata.get('tags', []))
                })
                
                dataset_id = result.lastrowid
                conn.commit()
                
                logger.info(f"Dataset saved with ID: {dataset_id}")
                return dataset_id
                
        except SQLAlchemyError as e:
            logger.error(f"Error saving dataset: {str(e)}")
            st.error(f"Database error: {str(e)}")
            return None
    
    def load_dataset(self, dataset_id: int) -> Optional[pd.DataFrame]:
        """Load a dataset from the database"""
        try:
            with self.engine.connect() as conn:
                # Get dataset metadata
                result = conn.execute(text("""
                    SELECT file_hash FROM datasets WHERE id = :id
                """), {'id': dataset_id})
                
                row = result.fetchone()
                if not row:
                    return None
                
                file_hash = row[0]
                table_name = f"dataset_{file_hash[:8]}"
                
                # Load the actual data
                df = pd.read_sql_table(table_name, self.engine)
                return df
                
        except SQLAlchemyError as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None
    
    def search_datasets(self, query: str = "", user_email: str = "") -> List[Dict[str, Any]]:
        """Search datasets based on query and user email"""
        try:
            with self.engine.connect() as conn:
                sql_query = """
                    SELECT id, name, description, upload_date, file_size, 
                           row_count, column_count, file_type, user_email, tags
                    FROM datasets
                    WHERE 1=1
                """
                params = {}
                
                if query:
                    sql_query += " AND (name LIKE :query OR description LIKE :query OR tags LIKE :query)"
                    params['query'] = f"%{query}%"
                
                if user_email:
                    sql_query += " AND user_email = :user_email"
                    params['user_email'] = user_email
                
                sql_query += " ORDER BY upload_date DESC"
                
                result = conn.execute(text(sql_query), params)
                
                datasets = []
                for row in result:
                    datasets.append({
                        'id': row[0],
                        'name': row[1],
                        'description': row[2],
                        'upload_date': row[3],
                        'file_size': row[4],
                        'row_count': row[5],
                        'column_count': row[6],
                        'file_type': row[7],
                        'user_email': row[8],
                        'tags': json.loads(row[9]) if row[9] else []
                    })
                
                return datasets
                
        except SQLAlchemyError as e:
            logger.error(f"Error searching datasets: {str(e)}")
            return []
    
    def get_dataset_metadata(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific dataset"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM datasets WHERE id = :id
                """), {'id': dataset_id})
                
                row = result.fetchone()
                if not row:
                    return None
                
                return {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'file_hash': row[3],
                    'upload_date': row[4],
                    'file_size': row[5],
                    'row_count': row[6],
                    'column_count': row[7],
                    'file_type': row[8],
                    'processing_log': json.loads(row[9]) if row[9] else [],
                    'user_email': row[10],
                    'tags': json.loads(row[11]) if row[11] else []
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting dataset metadata: {str(e)}")
            return None
    
    def log_processing_operation(self, dataset_id: int, operation: str, 
                               parameters: Dict[str, Any], user_email: str = ""):
        """Log a processing operation"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO processing_history (dataset_id, operation, parameters, user_email)
                    VALUES (:dataset_id, :operation, :parameters, :user_email)
                """), {
                    'dataset_id': dataset_id,
                    'operation': operation,
                    'parameters': json.dumps(parameters),
                    'user_email': user_email
                })
                conn.commit()
                
        except SQLAlchemyError as e:
            logger.error(f"Error logging operation: {str(e)}")
    
    def get_processing_history(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Get processing history for a dataset"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT operation, parameters, timestamp, user_email
                    FROM processing_history
                    WHERE dataset_id = :dataset_id
                    ORDER BY timestamp DESC
                """), {'dataset_id': dataset_id})
                
                history = []
                for row in result:
                    history.append({
                        'operation': row[0],
                        'parameters': json.loads(row[1]) if row[1] else {},
                        'timestamp': row[2],
                        'user_email': row[3]
                    })
                
                return history
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting processing history: {str(e)}")
            return []
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset and its associated data"""
        try:
            with self.engine.connect() as conn:
                # Get file hash first
                result = conn.execute(text("""
                    SELECT file_hash FROM datasets WHERE id = :id
                """), {'id': dataset_id})
                
                row = result.fetchone()
                if not row:
                    return False
                
                file_hash = row[0]
                table_name = f"dataset_{file_hash[:8]}"
                
                # Delete the data table
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                
                # Delete from datasets table
                conn.execute(text("""
                    DELETE FROM datasets WHERE id = :id
                """), {'id': dataset_id})
                
                # Delete processing history
                conn.execute(text("""
                    DELETE FROM processing_history WHERE dataset_id = :id
                """), {'id': dataset_id})
                
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error deleting dataset: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.engine.connect() as conn:
                # Total datasets
                result = conn.execute(text("SELECT COUNT(*) FROM datasets"))
                total_datasets = result.fetchone()[0]
                
                # Total data size
                result = conn.execute(text("SELECT SUM(file_size) FROM datasets"))
                total_size = result.fetchone()[0] or 0
                
                # Datasets by type
                result = conn.execute(text("""
                    SELECT file_type, COUNT(*) FROM datasets GROUP BY file_type
                """))
                datasets_by_type = dict(result.fetchall())
                
                # Recent activity
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM datasets 
                    WHERE upload_date >= datetime('now', '-7 days')
                """))
                recent_uploads = result.fetchone()[0]
                
                return {
                    'total_datasets': total_datasets,
                    'total_size': total_size,
                    'datasets_by_type': datasets_by_type,
                    'recent_uploads': recent_uploads
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def _generate_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash for the dataset"""
        # Create a hash based on the dataframe content
        content = df.to_string()
        return hashlib.md5(content.encode()).hexdigest()
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
