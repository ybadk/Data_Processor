"""
Normal-equation and cost-function optimization utilities.

Derived from The_Normal_equation_minimizes-the-cost-function/ and
gradient_Descent_algorithm/ reference implementations. Used to establish
a stable linear baseline and tune XGBoost hyperparameters toward the
10% error-rate acceptance threshold without runtime failures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _add_bias(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.c_[np.ones((len(X), 1)), X]


def fit_normal_equation(
    X: np.ndarray, y: np.ndarray, reg_lambda: float = 1e-6
) -> np.ndarray:
    """
    Closed-form least squares: theta = (X'X + lambda*I)^(-1) X'y.
    Ridge regularization avoids singular-matrix runtime errors.
    """
    X_b = _add_bias(X)
    y_vec = np.asarray(y, dtype=np.float64).ravel()
    n_features = X_b.shape[1]
    reg = reg_lambda * np.eye(n_features)
    reg[0, 0] = 0.0

    xtx = X_b.T @ X_b + reg
    xty = X_b.T @ y_vec
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ xty


def predict_linear(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return (_add_bias(X) @ theta).ravel()


def minimize_cost_function(
    X: np.ndarray, y: np.ndarray, theta_init: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Scipy cost-function minimization (MSE) from the reference
    main-ollama-llama-falcon3-update implementation.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y_vec = np.asarray(y, dtype=np.float64).ravel()
    X_b = _add_bias(X)
    n_params = X_b.shape[1]

    if theta_init is None:
        theta_init = np.zeros(n_params)

    def cost_function(theta: np.ndarray) -> float:
        predictions = X_b @ theta
        return float(np.sum((predictions - y_vec) ** 2))

    try:
        result = minimize(cost_function, theta_init, method="L-BFGS-B")
        if result.success:
            return result.x
    except Exception as exc:
        logger.warning("Cost-function minimization failed: %s", exc)

    return fit_normal_equation(X, y_vec)


def gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    eta: float = 0.1,
    n_iterations: int = 1000,
) -> np.ndarray:
    """
    Batch gradient descent from gradient_Descent_algorithm/main.py.
    Falls back to the normal equation when inputs are invalid.
    """
    X_b = _add_bias(X)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    m = max(len(y_vec), 1)
    theta = np.random.randn(X_b.shape[1], 1) * 0.01

    try:
        for _ in range(n_iterations):
            gradients = (2.0 / m) * X_b.T @ (X_b @ theta - y_vec)
            theta -= eta * gradients
        return theta.ravel()
    except Exception as exc:
        logger.warning("Gradient descent failed: %s", exc)
        return fit_normal_equation(X, y_vec)


def evaluate_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    error_tester: Any,
    task_type: str,
) -> Dict[str, Any]:
    """
    Fit three linear baselines and return the best one plus its test metrics.
    """
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_test = np.asarray(y_test, dtype=np.float64).ravel()

    if len(y_train) == 0 or len(y_test) == 0:
        return {"method": "none", "error_rate": 1.0, "passes_test": False}

    candidates: List[Tuple[str, np.ndarray]] = []
    for name, fit_fn in (
        ("normal_equation", fit_normal_equation),
        ("cost_minimize", minimize_cost_function),
        ("gradient_descent", gradient_descent_fit),
    ):
        try:
            theta = fit_fn(X_train, y_train)
            candidates.append((name, theta))
        except Exception as exc:
            logger.debug("%s baseline skipped: %s", name, exc)

    best: Dict[str, Any] = {
        "method": "none",
        "error_rate": 1.0,
        "passes_test": False,
    }

    for name, theta in candidates:
        try:
            y_pred = predict_linear(X_test, theta)
            if task_type == "regression":
                metrics = error_tester.test_regression_error(y_test, y_pred)
            else:
                y_pred_cls = np.round(y_pred).astype(int)
                metrics = error_tester.test_classification_error(y_test, y_pred_cls)
            if metrics["error_rate"] < best["error_rate"]:
                best = {"method": name, **metrics}
        except Exception as exc:
            logger.debug("%s evaluation skipped: %s", name, exc)

    return best


def tune_xgboost_params(
    baseline_error: float,
    base_params: Dict[str, Any],
    passes_baseline: bool,
) -> Dict[str, Any]:
    """
    Adjust XGBoost hyperparameters based on the linear baseline error rate.
    Conservative settings when the baseline already passes; more capacity
    when it does not.
    """
    params = dict(base_params)
    params.setdefault("max_depth", 3)
    params.setdefault("learning_rate", 0.1)
    params.setdefault("n_estimators", 100)
    params.setdefault("random_state", 42)

    if passes_baseline:
        params["max_depth"] = min(int(params["max_depth"]), 4)
        params["n_estimators"] = min(int(params["n_estimators"]), 120)
        params["learning_rate"] = max(float(params["learning_rate"]), 0.08)
    elif baseline_error > 0.25:
        params["max_depth"] = min(int(params["max_depth"]) + 2, 10)
        params["n_estimators"] = min(int(params["n_estimators"]) + 150, 500)
        params["learning_rate"] = min(float(params["learning_rate"]), 0.05)
    else:
        params["max_depth"] = min(int(params["max_depth"]) + 1, 8)
        params["n_estimators"] = min(int(params["n_estimators"]) + 75, 350)
        params["learning_rate"] = max(min(float(params["learning_rate"]), 0.08), 0.03)

    return params


def build_param_candidates(
    tuned_params: Dict[str, Any], max_candidates: int = 3
) -> List[Dict[str, Any]]:
    """Generate a small list of param sets to try when the first pass fails."""
    candidates = [tuned_params]
    aggressive = dict(tuned_params)
    aggressive["n_estimators"] = min(int(aggressive.get("n_estimators", 100)) + 100, 500)
    aggressive["learning_rate"] = max(float(aggressive.get("learning_rate", 0.1)) * 0.7, 0.01)
    aggressive["max_depth"] = min(int(aggressive.get("max_depth", 3)) + 1, 10)
    candidates.append(aggressive)

    conservative = dict(tuned_params)
    conservative["n_estimators"] = max(int(conservative.get("n_estimators", 100)) - 30, 50)
    conservative["learning_rate"] = min(float(conservative.get("learning_rate", 0.1)) * 1.2, 0.3)
    conservative["max_depth"] = max(int(conservative.get("max_depth", 3)) - 1, 2)
    candidates.append(conservative)

    return candidates[:max_candidates]
