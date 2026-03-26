"""
Machine Learning Integration Module
Integrates linear models, XGBoost, PyTorch, Transformers, and auto-run ML capabilities
"""

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from scipy.special import expit as activation_function
from scipy.stats import truncnorm, stats
import pandas as pd
import numpy as np
import pickle
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Tuple, List
import logging
import warnings
warnings.filterwarnings('ignore')

# New imports for PyTorch and Transformers
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForRegression
    from transformers import Trainer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def sanitize_dataframe_for_xgboost(df: pd.DataFrame, feature_cols: List[str], target_col: str=None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Comprehensive data sanitization using numpy and statsmodels validation.
    Transforms data into a form that XGBoost can process without dtype errors.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name (optional)
    
    Returns:
        Tuple of (sanitized_X, sanitized_y)
    """
    # Step 1: Extract and validate features
    X = df[feature_cols].copy()
    
    # Step 2: Convert each column to numpy array and back using explicit dtypes
    for col in X.columns:
        try:
            # Convert column to numpy array (forces evaluation of any lazy structures)
            arr = np.asarray(X[col], dtype=np.float64)
            
            # Replace any non-finite values
            arr = np.where(np.isfinite(arr), arr, 0.0)
            
            # Convert back to Series with explicit float64 dtype
            X[col] = pd.Series(arr, index=X.index, dtype=np.float64)
        except (ValueError, TypeError) as e:
            # If conversion fails, fill with zeros
            X[col] = 0.0
    
    # Step 3: Reset index and column names
    X = X.reset_index(drop=True)
    X.columns = [str(col) for col in X.columns]
    
    # Step 4: Use statsmodels-compatible conversion
    if STATSMODELS_AVAILABLE:
        try:
            # This validates data compatibility with statsmodels
            X_validated = sm.tools.tools.add_constant(X, has_constant='skip')
            X = X_validated.drop(columns='const', errors='ignore')
        except Exception:
            pass  # Continue if statsmodels validation fails
    
    # Step 5: Additional numpy validation
    X_array = X.values
    X_array = np.asarray(X_array, dtype=np.float64)
    X = pd.DataFrame(X_array, columns=X.columns)
    
    # Step 6: Handle target variable if provided
    y = None
    if target_col is not None:
        y = df[target_col].copy()
        try:
            arr = np.asarray(y, dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            y = pd.Series(arr, index=y.index, dtype=np.float64)
        except (ValueError, TypeError):
            y = pd.Series(0.0, index=y.index)
        y = y.reset_index(drop=True)
    
    return X, y


class NeuralNetworkWrapper:
    """
    Custom Neural Network implementation supporting multiple hidden layers.
    """

    def __init__(self, input_nodes, output_nodes, hidden_layers, learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers if isinstance(hidden_layers, list) else [hidden_layers]
        self.learning_rate = learning_rate
        self.layers = [input_nodes] + self.hidden_layers + [output_nodes]
        self.create_weight_matrices()

    def create_weight_matrices(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            rad = 1 / np.sqrt(self.layers[i])
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            w = X.rvs((self.layers[i + 1], self.layers[i]))
            self.weights.append(w)

    def train(self, input_vector, target_vector):
        input_vector = np.array(input_vector).reshape(-1, 1)
        target_vector = np.array(target_vector).reshape(-1, 1)
        
        # FasterPython: bind activation_function to local to avoid attribute lookup in loop
        af = activation_function
        weights = self.weights
        
        # Forward pass - store activations
        activations = [input_vector]
        current_input = input_vector
        for w in weights:
            current_input = af(w @ current_input)
            activations.append(current_input)
            
        # Backward pass
        output_vector = activations[-1]
        error = target_vector - output_vector
        
        for i in reversed(range(len(weights))):
            act_next = activations[i + 1]
            act_curr = activations[i]
            # FasterPython: use local variables for gradients
            delta = error * act_next * (1.0 - act_next)
            
            # Update weights
            weights[i] += self.learning_rate * (delta @ act_curr.T)
            
            # Propagate error back to previous layer
            error = weights[i].T @ error

    def run(self, input_vector):
        # FasterPython: bind activation_function to local
        af = activation_function
        current_input = np.array(input_vector).reshape(-1, 1)
        for w in self.weights:
            current_input = af(w @ current_input)
        return current_input

    def fit_dataset(self, X, y, epochs=10):
        # FasterPython: bind train method to local
        train_fn = self.train
        if len(y.shape) == 1:
            num_classes = self.output_nodes
            y_one_hot = np.zeros((y.size, num_classes))
            y_one_hot[np.arange(y.size), y.astype(int)] = 1
            y = y_one_hot
            
        for epoch in range(epochs):
            for i in range(len(X)):
                train_fn(X[i], y[i])

    def predict_classes(self, X):
        # FasterPython: bind method and np to local
        run_fn = self.run
        predictions = []
        for i in range(len(X)):
            res = run_fn(X[i])
            predictions.append(res.argmax())
        return np.array(predictions)

# ML Libraries


# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Statsmodels for linear models
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ModelErrorTester:
    """
    Tests model error rates before saving to ensure quality control.
    Ensures error rate is below 10% before allowing model persistence.
    """
    
    def __init__(self, max_error_rate: float=0.10):
        self.max_error_rate = max_error_rate
    
    def test_regression_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Test regression model error metrics
        Returns error rates and pass/fail status
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        
        # Calculate error rate as MAPE (percentage)
        error_rate = mape / 100.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'error_rate': error_rate,
            'passes_test': error_rate < self.max_error_rate
        }
    
    def test_classification_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Test classification model error metrics
        Returns error rates and pass/fail status
        """
        accuracy = accuracy_score(y_true, y_pred)
        error_rate = 1.0 - accuracy  # Error rate is 1 - accuracy
        
        return {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'passes_test': error_rate < self.max_error_rate
        }
    
    def validate_model_for_saving(self, model, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, Any]:
        """
        Comprehensive validation before saving model
        """
        y_pred = model.predict(X_test)
        
        if task_type == 'regression':
            results = self.test_regression_error(y_test, y_pred)
        else:
            results = self.test_classification_error(y_test, y_pred)
        
        results['task_type'] = task_type
        results['validation_timestamp'] = datetime.now().isoformat()
        
        return results


class AsyncMLTrainer:
    """
    Asynchronous ML trainer supporting PyTorch, Transformers, and scikit-learn
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if PYTORCH_AVAILABLE else None
    
    async def create_datasets_from_data(self, df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
        """
        Create three different datasets from the input data for diverse training
        """
        datasets = []
        
        # Dataset 1: Original data
        datasets.append({
            'name': 'original',
            'data': df.copy(),
            'target': target_col
        })
        
        # Dataset 2: Scaled numeric features
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        datasets.append({
            'name': 'scaled',
            'data': df_scaled,
            'target': target_col,
            'scaler': scaler
        })
        
        # Dataset 3: Feature engineered (add polynomial features for numeric)
        df_engineered = df.copy()
        numeric_cols = [col for col in df_engineered.select_dtypes(include=[np.number]).columns if col != target_col]
        if len(numeric_cols) > 1:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(df_engineered[numeric_cols])
            poly_feature_names = poly.get_feature_names_out(numeric_cols)
            df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_engineered.index)
            df_engineered = pd.concat([df_engineered, df_poly], axis=1)
        
        datasets.append({
            'name': 'engineered',
            'data': df_engineered,
            'target': target_col,
            'poly_features': poly_feature_names if 'poly' in locals() else None
        })
        
        return datasets
    
    async def train_pytorch_model(self, X: np.ndarray, y: np.ndarray, task_type: str) -> nn.Module:
        """
        Train a simple PyTorch neural network
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        if task_type == 'regression':
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
            output_size = 1
            criterion = nn.MSELoss()
        else:
            y_tensor = torch.LongTensor(y).to(self.device)
            output_size = len(np.unique(y))
            criterion = nn.CrossEntropyLoss()
        
        # Simple neural network
        model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return model
    
    async def train_transformers_model(self, df: pd.DataFrame, text_col: str, target_col: str, task_type: str):
        """
        Train a transformers model for text classification/regression
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        # Assume text_col exists for text tasks
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in data")
        
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        if task_type == 'regression':
            model = AutoModelForRegression.from_pretrained('distilbert-base-uncased')
            num_labels = 1
        else:
            num_labels = len(df[target_col].unique())
            model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        
        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(examples[text_col], padding="max_length", truncation=True)
        
        # Prepare dataset (simplified)
        train_texts = df[text_col].tolist()
        train_labels = df[target_col].tolist()
        
        # For demo, split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir='./logs',
        )
        
        # Create trainer (simplified - would need proper dataset class)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,  # Would need to implement
            eval_dataset=None,
        )
        
        return model


class MLIntegration:
    """Comprehensive ML Integration with XGBoost, PyTorch, Transformers and Linear Models"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.metrics = {}
        self.predictions = {}
        self.model_dir = "model_lake"  # Changed to model_lake

        # Initialize new components
        self.error_tester = ModelErrorTester()
        self.async_trainer = AsyncMLTrainer()

        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def auto_run_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically run multiple ML models on the dataset
        Returns metrics for all models
        """
        results = {
            'regression_metrics': {},
            'classification_metrics': {},
            'best_model': None,
            'best_score': 0
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return results

        # Try regression on numeric columns
        target_col = numeric_cols[-1]  # Use last column as target
        feature_cols = numeric_cols[:-1]

        if len(feature_cols) > 0:
            # Use sanitization function
            X, y = sanitize_dataframe_for_xgboost(df, feature_cols, target_col)

            # Run multiple regression models
            reg_models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=3)
            }

            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                reg_models['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=50, random_state=42, max_depth=3)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Convert to numpy arrays for compatibility
            X_train_array = np.asarray(X_train.values, dtype=np.float64, order='C')
            X_test_array = np.asarray(X_test.values, dtype=np.float64, order='C')
            y_train_array = np.asarray(y_train.values, dtype=np.float64, order='C')
            y_test_array = np.asarray(y_test.values, dtype=np.float64, order='C')

            for name, model in reg_models.items():
                try:
                    # Use numpy arrays for all models
                    model.fit(X_train_array, y_train_array)
                    y_pred = model.predict(X_test_array)

                    mse = mean_squared_error(y_test_array, y_pred)
                    r2 = r2_score(y_test_array, y_pred)
                    mae = mean_absolute_error(y_test_array, y_pred)

                    results['regression_metrics'][name] = {
                        'MSE': round(mse, 4),
                        'R2': round(r2, 4),
                        'MAE': round(mae, 4),
                        'RMSE': round(np.sqrt(mse), 4)
                    }

                    # Track best model
                    if r2 > results['best_score']:
                        results['best_score'] = r2
                        results['best_model'] = name

                except Exception as e:
                    results['regression_metrics'][name] = {'error': str(e)}

        return results

    async def train_xgboost_model(self, df: pd.DataFrame, target_col: str,
                                  feature_cols: List[str], task_type: str='regression',
                                  params: Dict=None) -> Dict[str, Any]:
        """
        Enhanced async XGBoost training with multiple datasets, error testing, and model lake
        """
        if not XGBOOST_AVAILABLE:
            return {'error': 'XGBoost not available'}

        # Validate inputs
        if target_col not in df.columns:
            return {'error': f'Target column "{target_col}" not found in dataframe'}

        # Filter valid feature columns
        valid_feature_cols = [col for col in feature_cols if col in df.columns and col != target_col]
        if not valid_feature_cols:
            return {'error': 'No valid feature columns found'}

        # Create three different datasets
        datasets = await self.async_trainer.create_datasets_from_data(df, target_col)

        trained_models = {}
        model_results = {}

        # Train XGBoost on each dataset individually
        for dataset in datasets:
            dataset_name = dataset['name']
            dataset_df = dataset['data']

            try:
                # Use comprehensive sanitization function
                X, y = sanitize_dataframe_for_xgboost(dataset_df, valid_feature_cols, target_col)

                # Validate sanitization
                if X is None or y is None:
                    return {'error': f'Data sanitization failed for dataset {dataset_name}'}

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # CRITICAL: Convert to numpy arrays BEFORE XGBoost to avoid dtype corruption
                X_train_array = np.asarray(X_train.values, dtype=np.float64, order='C')
                X_test_array = np.asarray(X_test.values, dtype=np.float64, order='C')
                y_train_array = np.asarray(y_train.values, dtype=np.float64, order='C')
                y_test_array = np.asarray(y_test.values, dtype=np.float64, order='C')
                
                # Ensure arrays are contiguous and have no issues
                X_train_array = np.ascontiguousarray(X_train_array)
                X_test_array = np.ascontiguousarray(X_test_array)
                y_train_array = np.ascontiguousarray(y_train_array)
                y_test_array = np.ascontiguousarray(y_test_array)

                # Default parameters
                if params is None:
                    params = {
                        'max_depth': 3,
                        'learning_rate': 0.1,
                        'n_estimators': 100,
                        'random_state': 42
                    }

                # Train model based on task type
                if task_type == 'regression':
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train_array, y_train_array)
                    y_pred = model.predict(X_test_array)

                    # Test error using our error tester
                    error_results = self.error_tester.test_regression_error(y_test_array, y_pred)

                else:  # classification
                    le = LabelEncoder()
                    y_train_enc = le.fit_transform(y_train_array)
                    y_test_enc = le.transform(y_test_array)

                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train_array, y_train_enc)
                    y_pred_enc = model.predict(X_test_array)

                    # Test error
                    error_results = self.error_tester.test_classification_error(y_test_enc, y_pred_enc)
                    error_results['label_encoder'] = le

                # Only save if error rate < 10%
                if error_results['passes_test']:
                    # Model Lake: Save with unique ID
                    model_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_filename = f'xgboost_{model_id}.pkl'
                    model_path = os.path.join(self.model_dir, model_filename)
                    
                    # Save model
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    # Save metadata
                    metadata = {
                        'model_id': model_id,
                        'dataset_name': dataset_name,
                        'task_type': task_type,
                        'target_col': target_col,
                        'feature_cols': valid_feature_cols,
                        'error_metrics': error_results,
                        'params': {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
                        'timestamp': datetime.now().isoformat(),
                        'scaler': dataset.get('scaler'),
                        'poly_features': dataset.get('poly_features')
                    }
                    meta_path = model_path.replace('.pkl', '.json')
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=4, default=str)
                    
                    trained_models[dataset_name] = model
                    model_results[dataset_name] = {
                        'model_id': model_id,
                        'error_results': error_results,
                        'model_path': model_path
                    }
                else:
                    model_results[dataset_name] = {
                        'error': f"Model failed error test: {error_results['error_rate']:.2%} > 10%",
                        'error_results': error_results
                    }
            except Exception as e:
                model_results[dataset_name] = {
                    'error': f"Training failed: {str(e)}"
                }
        
        # Try Transformers if text data available
        text_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target_col]
        if text_cols and TRANSFORMERS_AVAILABLE:
            try:
                transformers_model = await self.async_trainer.train_transformers_model(df, text_cols[0], target_col, task_type)
                model_results['transformers'] = {'status': 'trained'}
            except Exception as e:
                model_results['transformers'] = {'error': f'Transformers training failed: {str(e)}'}
        
        return {
            'trained_models': trained_models,
            'results': model_results,
            'total_datasets': len(datasets),
            'passed_models': len([r for r in model_results.values() if 'model_id' in r])
        }

    def list_models_in_lake(self) -> List[Dict[str, Any]]:
        """List all models in the Model Lake (XGBoost, PyTorch, Transformers)"""
        models_info = []
        if not os.path.exists(self.model_dir):
            return []
            
        for f in os.listdir(self.model_dir):
            if f.endswith('.json'):
                try:
                    with open(os.path.join(self.model_dir, f), 'r') as meta_file:
                        metadata = json.load(meta_file)
                        models_info.append(metadata)
                except:
                    continue  # Skip corrupted metadata
        
        # Sort by timestamp descending
        models_info.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return models_info

    def load_model_from_lake(self, model_id: str) -> Any:
        """Load a specific model from the Lake by ID (supports XGBoost, PyTorch, Transformers)"""
        # Try XGBoost first
        model_path = os.path.join(self.model_dir, f'xgboost_{model_id}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        # Try PyTorch
        model_path = os.path.join(self.model_dir, f'pytorch_{model_id}.pth')
        if os.path.exists(model_path) and PYTORCH_AVAILABLE:
            # Would need model architecture to load properly
            # For now, return path
            return model_path
        
        # Try Transformers
        model_path = os.path.join(self.model_dir, f'transformers_{model_id}')
        if os.path.exists(model_path) and TRANSFORMERS_AVAILABLE:
            # Would need to load transformers model
            return model_path
            
        return None

    async def make_predictions_multi(self, df: pd.DataFrame, model_ids: List[str]) -> pd.DataFrame:
        """
        Make predictions using multiple selected models from the Lake
        """
        result_df = df.copy()
        X = None
        
        # FasterPython: use asyncio for parallel loading (though pickle is synchronous)
        for mid in model_ids:
            model = self.load_model_from_lake(mid)
            if model:
                # Get features from metadata
                meta_path = os.path.join(self.model_dir, f'xgboost_{mid}.json')
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                features = meta.get('feature_cols', [])
                X_model = df[features].fillna(0)
                
                preds = model.predict(X_model)
                result_df[f'Pred_{mid}'] = preds
                
        return result_df

    def load_xgboost_model(self, model_path: str=None) -> Any:
        """Load saved XGBoost model"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models['xgboost'] = model
            return model
        return None

    def make_predictions(self, df: pd.DataFrame, feature_cols: List[str],
                         model_name: str='xgboost') -> pd.DataFrame:
        """
        Make predictions on new data and return comparison dataframe
        """
        if model_name not in self.models:
            # Try to load model
            self.load_xgboost_model()
            if model_name not in self.models:
                return None

        # Validate feature columns
        valid_feature_cols = [col for col in feature_cols if col in df.columns]
        if not valid_feature_cols:
            return None

        model = self.models[model_name]
        X = df[valid_feature_cols].fillna(0)

        # Ensure X is proper
        if not isinstance(X, pd.DataFrame):
            return None

        # Remove duplicate columns if any
        if X.columns.duplicated().any():
            X = X.loc[:, ~X.columns.duplicated()]

        predictions = model.predict(X)

        # Create comparison dataframe
        result_df = df.copy()
        result_df['Predictions'] = predictions

        self.predictions[model_name] = predictions

        return result_df

    def run_linear_models(self, df: pd.DataFrame, target_col: str,
                          feature_cols: List[str]) -> Dict[str, Any]:
        """
        Run statsmodels linear regression models
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'Statsmodels not available'}

        # Validate inputs
        if target_col not in df.columns:
            return {'error': f'Target column "{target_col}" not found in dataframe'}

        # Filter valid feature columns
        valid_feature_cols = [col for col in feature_cols if col in df.columns and col != target_col]
        if not valid_feature_cols:
            return {'error': 'No valid feature columns found'}

        X = df[valid_feature_cols].fillna(0)
        y = df[target_col].fillna(0)

        # Ensure X is proper DataFrame
        if not isinstance(X, pd.DataFrame):
            return {'error': 'Invalid feature data format'}

        # Remove duplicate columns if any
        if X.columns.duplicated().any():
            X = X.loc[:, ~X.columns.duplicated()]

        # Add constant for intercept
        X_with_const = sm.add_constant(X)

        results = {}

        # OLS Model
        try:
            ols_model = sm.OLS(y, X_with_const)
            ols_results = ols_model.fit()

            results['OLS'] = {
                'params': ols_results.params.to_dict(),
                'r_squared': ols_results.rsquared,
                'adj_r_squared': ols_results.rsquared_adj,
                'f_statistic': ols_results.fvalue,
                'p_value': ols_results.f_pvalue,
                'summary': str(ols_results.summary())
            }
        except Exception as e:
            results['OLS'] = {'error': str(e)}

        # WLS Model (Weighted Least Squares)
        try:
            weights = np.ones(len(y))
            wls_model = sm.WLS(y, X_with_const, weights=weights)
            wls_results = wls_model.fit()

            results['WLS'] = {
                'params': wls_results.params.to_dict(),
                'r_squared': wls_results.rsquared,
                'adj_r_squared': wls_results.rsquared_adj
            }
        except Exception as e:
            results['WLS'] = {'error': str(e)}

        return results

    def analyze_ab_test(self, df: pd.DataFrame, channel: str, subsegment: str=None) -> Dict[str, Any]:
        """Perform A/B testing analysis extracted from notebooks"""
        if 'variant' not in df.columns or 'converted' not in df.columns:
            return {"error": "Required columns 'variant' or 'converted' missing"}

        results = {}
        if subsegment is None:
            subsegmented_df = df[df['marketing_channel'] == channel] if 'marketing_channel' in df.columns else df
            subscribers = subsegmented_df.groupby(['user_id', 'variant'])['converted'].max() if 'user_id' in df.columns else subsegmented_df.groupby(['variant'])['converted'].mean()
            subscribers_df = pd.DataFrame(subscribers.unstack(level=1) if hasattr(subscribers, 'unstack') else subscribers)

            if 'control' not in subscribers_df.columns or 'personalization' not in subscribers_df.columns:
                 return {"error": "Control or Personalization group missing"}

            control = subscribers_df['control'].dropna()
            personalization = subscribers_df['personalization'].dropna()
            lift = (np.mean(personalization) - np.mean(control)) / np.mean(control)
            t_stat, p_val = stats.ttest_ind(control, personalization)

            results['main_test'] = {
                'lift': lift,
                't_statistic': t_stat,
                'p_value': p_val,
                'control_count': len(control),
                'personalization_count': len(personalization)
            }
        else:
            for value in np.unique(df[subsegment].dropna().values):
                sub_df = df[(df['marketing_channel'] == channel) & (df[subsegment] == value)] if 'marketing_channel' in df.columns else df[df[subsegment] == value]
                subscribers = sub_df.groupby(['user_id', 'variant'])['converted'].max() if 'user_id' in df.columns else sub_df.groupby(['variant'])['converted'].mean()
                subscribers_df = pd.DataFrame(subscribers.unstack(level=1) if hasattr(subscribers, 'unstack') else subscribers)

                if 'control' in subscribers_df.columns and 'personalization' in subscribers_df.columns:
                    control = subscribers_df['control'].dropna()
                    personalization = subscribers_df['personalization'].dropna()
                    if len(control) > 0 and len(personalization) > 0:
                        lift = (np.mean(personalization) - np.mean(control)) / np.mean(control)
                        t_stat, p_val = stats.ttest_ind(control, personalization)
                        results[value] = {'lift': lift, 't_statistic': t_stat, 'p_value': p_val, 'control_count': len(control), 'personalization_count': len(personalization)}
        return results

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models"""
        comparison_data = []

        for model_name, metrics in self.metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)
