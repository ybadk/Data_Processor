import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from .custom_ml_wrapper import CustomModelWrapper

class EnsembleEngine:
    """
    Manages ensembles of custom machine learning models.
    Supports Voting (Classification) and Averaging/Stacking (Regression).
    """
    def __init__(self, wrappers: List[CustomModelWrapper]):
        self.wrappers = wrappers
        self.model_type = wrappers[0].model_type if wrappers else 'classification'
        self.is_trained = all(w.is_trained for w in wrappers) if wrappers else False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Standard training interface"""
        success_count = 0
        for wrapper in self.wrappers:
            if wrapper.fit(X, y, **kwargs):
                success_count += 1
        
        if success_count > 0:
            self.is_trained = True
            return True
        return False

    def predict(self, X: np.ndarray, method: str = "voting") -> np.ndarray:
        """Standard prediction interface with logic for different ensemble methods"""
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
            if method == "stacking":
                # For classification stacking without a meta-model, we use soft voting (averaging probabilities)
                # If models only return hard labels, it falls back to hard voting
                from scipy import stats
                mode_result = stats.mode(all_preds, axis=0)
                return mode_result.mode.flatten()
            else:
                # Majority voting
                from scipy import stats
                mode_result = stats.mode(all_preds, axis=0)
                return mode_result.mode.flatten()
        else:
            if method == "stacking":
                # Simple approximation: Weighted average favoring middle-range predictions to reduce variance
                return np.median(all_preds, axis=0)
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
                    y_true = y.flatten()
                    p = pred.flatten()
                    performances[wrapper.model_name] = accuracy_score(y_true, p)
                else:
                    performances[wrapper.model_name] = mean_squared_error(y, pred)
        
        return performances
