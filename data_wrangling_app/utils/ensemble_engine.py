import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from .custom_ml_wrapper import CustomModelWrapper

class EnsembleEngine:
    """
    Manages ensembles of custom machine learning models.
    Supports Voting (Classification) and Averaging (Regression).
    """
    def __init__(self, model_paths: List[str], model_type: str = 'classification'):
        self.model_paths = model_paths
        self.model_type = model_type
        self.wrappers = [CustomModelWrapper(path, model_type) for path in model_paths]
        self.is_trained = False

    def train_ensemble(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train all base models in the ensemble"""
        success_count = 0
        for wrapper in self.wrappers:
            if wrapper.fit(X, y, **kwargs):
                success_count += 1
        
        if success_count > 0:
            self.is_trained = True
            return True
        return False

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Combine predictions from all base models"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first.")

        all_preds = []
        for wrapper in self.wrappers:
            pred = wrapper.predict(X)
            if pred is not None:
                all_preds.append(pred)

        if not all_preds:
            return None

        all_preds = np.array(all_preds)

        if self.model_type == 'classification':
            # Majority voting
            from scipy import stats
            mode_result = stats.mode(all_preds, axis=0)
            return mode_result.mode.flatten()
        else:
            # Simple averaging
            return np.mean(all_preds, axis=0)

    def get_individual_performances(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate performance for each base model for comparison"""
        from sklearn.metrics import accuracy_score, mean_squared_error
        performances = {}
        
        for wrapper in self.wrappers:
            pred = wrapper.predict(X)
            if pred is not None:
                if self.model_type == 'classification':
                    # Ensure same shape for accuracy
                    y_true = y.flatten()
                    if len(pred.shape) > 1 and pred.shape[1] > 1:
                        p = np.argmax(pred, axis=1)
                    else:
                        p = pred.flatten()
                    performances[wrapper.module_name] = accuracy_score(y_true, p)
                else:
                    performances[wrapper.module_name] = mean_squared_error(y, pred)
        
        return performances
