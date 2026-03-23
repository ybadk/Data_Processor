import numpy as np
import pandas as pd
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

class CustomModelWrapper:
    """
    Wraps custom Python scripts from machine_learning/ and neural_network/
    to provide a standardized fit/predict interface.
    """
    def __init__(self, script_path: str, model_type: str = 'classification'):
        self.script_path = script_path
        self.model_type = model_type
        self.module_name = os.path.basename(script_path).replace('.py', '').replace('.', '_')
        self.model_instance = None
        self.is_trained = False
        self._load_module()

    def _load_module(self):
        """Dynamically load the module from the script path"""
        spec = importlib.util.spec_from_file_location(self.module_name, self.script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        self.module = module

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Standardized training interface"""
        # Mapping common class names/functions in the custom scripts
        try:
            if 'BPNN' in dir(self.module):
                self.model_instance = self.module.BPNN()
                # Assuming standard DenseLayers for simplicity in automated wrapper
                input_dim = X.shape[1]
                output_dim = y.shape[1] if len(y.shape) > 1 else (len(np.unique(y)) if self.model_type == 'classification' else 1)
                
                # Check if it has DenseLayer
                if 'DenseLayer' in dir(self.module):
                    self.model_instance.add_layer(self.module.DenseLayer(input_dim))
                    self.model_instance.add_layer(self.module.DenseLayer(32))
                    self.model_instance.add_layer(self.module.DenseLayer(output_dim))
                
                self.model_instance.build()
                self.model_instance.train(X, y, train_round=kwargs.get('epochs', 100), accuracy=kwargs.get('accuracy', 0.01))
                
            elif 'DecisionTree' in dir(self.module):
                self.model_instance = self.module.DecisionTree(depth=kwargs.get('depth', 5))
                # DecisionTree in the provided code is 1D, we might need to adapt or use 1st feature
                self.model_instance.train(X[:, 0], y.flatten())
                
            elif 'logistic_reg' in dir(self.module):
                # Function based implementation
                self.theta = self.module.logistic_reg(kwargs.get('alpha', 0.1), X, y, max_iterations=kwargs.get('iterations', 1000))
                self.model_instance = "function_based_logit"
                
            elif 'kmeans' in dir(self.module):
                k = kwargs.get('k', 3)
                initial_centroids = self.module.get_initial_centroids(X, k)
                centroids, cluster_assignment = self.module.kmeans(X, k, initial_centroids)
                self.model_instance = {"centroids": centroids, "function": self.module.assign_clusters}
            
            # Add more mappings as needed based on file inspection
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error fitting model {self.module_name}: {str(e)}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Standardized prediction interface"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        try:
            if hasattr(self.model_instance, 'predict'):
                if isinstance(self.model_instance, self.module.DecisionTree):
                     return np.array([self.model_instance.predict(x) for x in X[:, 0]])
                return self.model_instance.predict(X)
            
            elif self.model_instance == "function_based_logit":
                return self.module.sigmoid_function(np.dot(X, self.theta)) > 0.5
            
            elif isinstance(self.model_instance, dict) and "function" in self.model_instance:
                return self.model_instance["function"](X, self.model_instance["centroids"])
            
            elif hasattr(self.model_instance, 'run'): # For BPNN
                results = []
                for x in X:
                    res = self.model_instance.run(x.reshape(-1, 1))
                    results.append(res.argmax() if self.model_type == 'classification' else res.flatten()[0])
                return np.array(results)
                
            return None
        except Exception as e:
            print(f"Error predicting with model {self.module_name}: {str(e)}")
            return None

def get_available_custom_models(base_dir: str) -> List[Dict[str, str]]:
    """Scan directories for usable custom model scripts"""
    models = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                path = os.path.join(root, file)
                models.append({
                    'name': file.replace('.py', ''),
                    'path': path,
                    'category': os.path.basename(root)
                })
    return models
