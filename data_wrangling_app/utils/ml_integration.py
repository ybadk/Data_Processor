"""
Machine Learning Integration Module
Integrates linear models, XGBoost, and auto-run ML capabilities
"""

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

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


class MLIntegration:
    """Comprehensive ML Integration with XGBoost and Linear Models"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.metrics = {}
        self.predictions = {}
        self.model_dir = "xg-boost"

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
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)

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

            for name, model in reg_models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

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

    def train_xgboost_model(self, df: pd.DataFrame, target_col: str,
                            feature_cols: List[str], task_type: str = 'regression',
                            params: Dict = None) -> Dict[str, Any]:
        """
        Train XGBoost model with feature engineering integration
        """
        if not XGBOOST_AVAILABLE:
            return {'error': 'XGBoost not available'}

        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                'MSE': mean_squared_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        else:  # classification
            # Encode labels
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train_enc)
            y_pred_enc = model.predict(X_test)

            metrics = {
                'Accuracy': accuracy_score(y_test_enc, y_pred_enc),
                'Precision': precision_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0),
                'Recall': recall_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0),
                'F1': f1_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0)
            }
            self.encoders['target'] = le

        # Save model
        model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Also save in XGBoost native format
        model.save_model(os.path.join(self.model_dir, 'xgboost_model.bin'))

        self.models['xgboost'] = model
        self.metrics['xgboost'] = metrics

        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'model_path': model_path
        }

    def load_xgboost_model(self, model_path: str = None) -> Any:
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
                         model_name: str = 'xgboost') -> pd.DataFrame:
        """
        Make predictions on new data and return comparison dataframe
        """
        if model_name not in self.models:
            # Try to load model
            self.load_xgboost_model()
            if model_name not in self.models:
                return None

        model = self.models[model_name]
        X = df[feature_cols].fillna(0)

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

        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)

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

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models"""
        comparison_data = []

        for model_name, metrics in self.metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)
