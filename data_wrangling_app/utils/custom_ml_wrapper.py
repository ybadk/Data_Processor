import numpy as np
import pandas as pd
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from utils.integrated_models import DecisionTree, LogisticRegression, KMeans, BPNN, DenseLayer, sigmoid

class CustomModelWrapper:
    """
    Wraps integrated Python models to provide a standardized fit/predict interface.
    """
    def __init__(self, model_name: str, category: str = 'machine_learning', model_type: str = 'classification'):
        self.model_name = model_name
        self.category = category
        self.model_type = model_type
        self.model_instance = None
        self.is_trained = False
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate integrated model class."""
        if self.model_name == 'DecisionTree':
            self.model_instance = DecisionTree(depth=5)
        elif self.model_name == 'LogisticRegression':
            self.model_instance = LogisticRegression()
        elif self.model_name == 'KMeans':
            self.model_instance = KMeans(k=3)
        elif self.model_name == 'BPNN':
            self.model_instance = BPNN()
            # Default layers for BPNN integration
            self.model_instance.add_layer(DenseLayer(10)) # Placeholder input/hidden
            self.model_instance.add_layer(DenseLayer(1))  # Placeholder output
            self.model_instance.build()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Standardized training interface"""
        try:
            if isinstance(self.model_instance, (DecisionTree, LogisticRegression, KMeans)):
                if isinstance(self.model_instance, LogisticRegression):
                    self.model_instance.fit(X, y)
                elif isinstance(self.model_instance, KMeans):
                    self.model_instance.fit(X)
                else:
                    self.model_instance.train(X, y)
            elif isinstance(self.model_instance, BPNN):
                self.model_instance.train(X, y, train_round=kwargs.get('epochs', 10), accuracy=kwargs.get('accuracy', 0.01))
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error fitting model {self.model_name}: {str(e)}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Standardized prediction interface"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        try:
            return self.model_instance.predict(X)
        except Exception as e:
            print(f"Error predicting with model {self.model_name}: {str(e)}")
            return None

def get_available_custom_models(base_dir: str = ".") -> Dict[str, List[str]]:
    """Return a static registry of integrated models."""
    return {
        'machine_learning': ['DecisionTree', 'LogisticRegression', 'KMeans'],
        'neural_network': ['BPNN']
    }
