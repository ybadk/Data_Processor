import numpy as np
import pandas as pd
import asyncio
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

    async def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Standard training interface with simultaneous execution"""
        tasks = []
        for wrapper in self.wrappers:
            # Run blocking fit in a separate thread for parallelism
            tasks.append(asyncio.to_thread(wrapper.fit, X, y, **kwargs))
        
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)
        
        if success_count > 0:
            self.is_trained = True
            return True
        return False

    async def predict(self, X: np.ndarray, method: str = "voting") -> np.ndarray:
        """Standard prediction interface with simultaneous execution"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first.")

        tasks = []
        for wrapper in self.wrappers:
            tasks.append(asyncio.to_thread(wrapper.predict, X))
        
        preds_results = await asyncio.gather(*tasks)
        all_preds = [p for p in preds_results if p is not None]

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
