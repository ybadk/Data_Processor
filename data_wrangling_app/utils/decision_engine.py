"""
Decision-Making Engine
Practical decision-making utilities derived from the decision-making-ML
research collection (probabilistic reasoning, ensembles, model
uncertainty, sequential problems, utility theory).

Provides:
- DataProfile / profile_dataset
- ModelRecommender (probabilistic / rule-based recommendation)
- BootstrapUncertainty (prediction intervals via bagging)
- BayesianModelAverager (softmax-weighted model averaging)
- EpsilonGreedyBandit (multi-armed bandit for adaptive model selection)
- DecisionMatrix (expected-utility decision support)
- build_voting_ensemble / build_bagging_ensemble / build_boosting_ensemble
"""

from __future__ import annotations

import json
import os
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data profiling
# ---------------------------------------------------------------------------
@dataclass
class DataProfile:
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    target: Optional[str]
    task_type: str  # 'regression' | 'binary' | 'multiclass' | 'unknown'
    n_classes: Optional[int]
    class_balance: Optional[float]  # min_class / max_class for classification
    missing_ratio: float
    high_dimensional: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def profile_dataset(df: pd.DataFrame, target: Optional[str]=None) -> DataProfile:
    """Inspect a dataframe and produce a structured profile."""
    n_rows, n_cols = df.shape
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    feature_cols = [c for c in df.columns if c != target] if target else list(df.columns)
    n_features = len(feature_cols)

    missing_ratio = float(df.isna().mean().mean()) if n_cols else 0.0

    task_type = 'unknown'
    n_classes: Optional[int] = None
    class_balance: Optional[float] = None
    if target and target in df.columns:
        y = df[target].dropna()
        if y.dtype.kind in 'biu' and y.nunique() <= max(20, int(0.05 * len(y))):
            n_classes = int(y.nunique())
            task_type = 'binary' if n_classes == 2 else 'multiclass'
            counts = y.value_counts()
            class_balance = float(counts.min() / counts.max()) if counts.max() else 0.0
        elif y.dtype.kind in 'fcb' or y.nunique() > 20:
            task_type = 'regression'
        else:
            n_classes = int(y.nunique())
            task_type = 'binary' if n_classes == 2 else 'multiclass'
            counts = y.value_counts()
            class_balance = float(counts.min() / counts.max()) if counts.max() else 0.0

    return DataProfile(
        n_rows=n_rows,
        n_features=n_features,
        n_numeric=len([c for c in numeric if c != target]),
        n_categorical=len([c for c in categorical if c != target]),
        target=target,
        task_type=task_type,
        n_classes=n_classes,
        class_balance=class_balance,
        missing_ratio=missing_ratio,
        high_dimensional=n_features > 20,
    )


# ---------------------------------------------------------------------------
# Model recommendation (probabilistic / rule-based)
# ---------------------------------------------------------------------------
@dataclass
class ModelRecommendation:
    name: str
    score: float
    rationale: str


class ModelRecommender:
    """
    Recommend candidate model families given a DataProfile, using a
    transparent rule-based scoring system inspired by the probabilistic
    decision-making notes (utility/score per candidate action).
    """

    def recommend(self, profile: DataProfile, top_k: int=5) -> List[ModelRecommendation]:
        recs: List[ModelRecommendation] = []
        task = profile.task_type

        if task == 'regression':
            recs += self._regression_candidates(profile)
        elif task in ('binary', 'multiclass'):
            recs += self._classification_candidates(profile)
        else:
            recs.append(ModelRecommendation(
                'KMeans (clustering)', 0.5,
                'No clear target detected. Consider unsupervised methods.'))

        recs.sort(key=lambda r: r.score, reverse=True)
        return recs[:top_k]

    def _classification_candidates(self, p: DataProfile) -> List[ModelRecommendation]:
        out: List[ModelRecommendation] = []
        small = p.n_rows < 1000
        imbalanced = (p.class_balance or 1.0) < 0.3

        out.append(ModelRecommendation(
            'LogisticRegression', 0.85 if (small and not p.high_dimensional) else 0.7,
            'Strong, interpretable baseline; well-suited to low-to-medium dimensional data.'))
        out.append(ModelRecommendation(
            'RandomForestClassifier', 0.9 if not small else 0.8,
            'Handles mixed types, non-linearities, and missingness with little tuning.'))
        out.append(ModelRecommendation(
            'GradientBoostingClassifier', 0.92 if p.n_rows >= 500 else 0.75,
            'Typically top-tier accuracy on tabular data when given enough rows.'))
        out.append(ModelRecommendation(
            'AdaBoostClassifier', 0.78 if imbalanced else 0.7,
            'Boosting weak learners; useful when classes are imbalanced.'))
        out.append(ModelRecommendation(
            'KNeighborsClassifier', 0.65 if small and not p.high_dimensional else 0.4,
            'Simple instance-based learner; degrades in high dimensions.'))
        out.append(ModelRecommendation(
            'SVC', 0.7 if small else 0.5,
            'Strong with clear margins; training cost grows with n_rows^2.'))
        return out

    def _regression_candidates(self, p: DataProfile) -> List[ModelRecommendation]:
        out: List[ModelRecommendation] = []
        out.append(ModelRecommendation(
            'Ridge', 0.8 if not p.high_dimensional else 0.85,
            'Regularised linear baseline; stable when features are correlated.'))
        out.append(ModelRecommendation(
            'RandomForestRegressor', 0.88,
            'Robust non-linear baseline with built-in feature interactions.'))
        out.append(ModelRecommendation(
            'GradientBoostingRegressor', 0.92 if p.n_rows >= 500 else 0.78,
            'Usually best-in-class on tabular regression tasks.'))
        out.append(ModelRecommendation(
            'KNeighborsRegressor', 0.6 if p.n_rows < 2000 and not p.high_dimensional else 0.4,
            'Locally-weighted predictions; sensitive to scale and dimensionality.'))
        out.append(ModelRecommendation(
            'SVR', 0.65 if p.n_rows < 5000 else 0.45,
            'Kernel-based regression; expensive on large datasets.'))
        return out


# ---------------------------------------------------------------------------
# Bootstrap uncertainty (prediction intervals)
# ---------------------------------------------------------------------------
class BootstrapUncertainty:
    """
    Fit a base estimator on B bootstrap resamples and produce
    point predictions plus empirical prediction intervals.
    Inspired by the bootstrap_aggregating and monte_carlo notes.
    """

    def __init__(self, base_estimator: BaseEstimator, n_bootstrap: int=30,
                 random_state: int=42):
        self.base_estimator = base_estimator
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = int(random_state)
        self.estimators_: List[BaseEstimator] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BootstrapUncertainty':
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.estimators_ = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            est = clone(self.base_estimator)
            est.fit(np.asarray(X)[idx], np.asarray(y)[idx])
            self.estimators_.append(est)
        return self

    def predict(self, X: np.ndarray, alpha: float=0.05) -> Dict[str, np.ndarray]:
        if not self.estimators_:
            raise RuntimeError("BootstrapUncertainty.fit must be called first")
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        lower = np.quantile(preds, alpha / 2, axis=1)
        upper = np.quantile(preds, 1 - alpha / 2, axis=1)
        return {
            'mean': preds.mean(axis=1),
            'std': preds.std(axis=1),
            'lower': lower,
            'upper': upper,
            'samples': preds,
        }


# ---------------------------------------------------------------------------
# Bayesian model averaging (softmax weights from losses)
# ---------------------------------------------------------------------------
class BayesianModelAverager:
    """
    Weight a collection of fitted models by softmax(-loss / T) and
    combine their predictions. Approximates a posterior over models
    using their validation losses as evidence.
    """

    def __init__(self, temperature: float=1.0):
        self.temperature = float(temperature)
        self.weights_: Optional[np.ndarray] = None
        self.model_names_: List[str] = []

    def fit_weights(self, losses: Dict[str, float]) -> Dict[str, float]:
        self.model_names_ = list(losses.keys())
        loss_arr = np.array([losses[n] for n in self.model_names_], dtype=np.float64)
        # Smaller loss -> larger weight
        scaled = -loss_arr / max(self.temperature, 1e-9)
        scaled -= scaled.max()  # numerical stability
        w = np.exp(scaled)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        self.weights_ = w
        return dict(zip(self.model_names_, w.tolist()))

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("fit_weights must be called before predict")
        stack = np.column_stack([predictions[n] for n in self.model_names_])
        return stack @ self.weights_


# ---------------------------------------------------------------------------
# Epsilon-greedy bandit for adaptive model selection
# ---------------------------------------------------------------------------
class EpsilonGreedyBandit:
    """
    Treat each candidate model as an arm of a multi-armed bandit.
    `select_arm` returns the arm to pull (epsilon-greedy exploration)
    and `update` records the observed reward (e.g. accuracy or -RMSE).
    Mirrors the e-greedy_exploration_strategy / multiarmed_bandit notes.
    """

    def __init__(self, arms: List[str], epsilon: float=0.1, seed: int=0):
        self.arms = list(arms)
        self.epsilon = float(epsilon)
        self.counts: Dict[str, int] = {a: 0 for a in self.arms}
        self.values: Dict[str, float] = {a: 0.0 for a in self.arms}
        self.rng = np.random.default_rng(seed)

    def select_arm(self) -> str:
        if self.rng.random() < self.epsilon or all(c == 0 for c in self.counts.values()):
            return str(self.rng.choice(self.arms))
        return max(self.values, key=lambda a: self.values[a])

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.counts:
            self.counts[arm] = 0
            self.values[arm] = 0.0
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental mean update (from incremental_estimate notes)
        self.values[arm] += (reward - self.values[arm]) / n

    def state(self) -> Dict[str, Dict[str, float]]:
        return {a: {'count': float(self.counts[a]), 'value': self.values[a]}
                for a in self.arms}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'arms': self.arms, 'epsilon': self.epsilon,
                       'counts': self.counts, 'values': self.values}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'EpsilonGreedyBandit':
        with open(path) as f:
            d = json.load(f)
        b = cls(d['arms'], epsilon=d.get('epsilon', 0.1))
        b.counts = {a: int(d['counts'].get(a, 0)) for a in b.arms}
        b.values = {a: float(d['values'].get(a, 0.0)) for a in b.arms}
        return b


# ---------------------------------------------------------------------------
# Decision matrix (expected-utility decision support)
# ---------------------------------------------------------------------------
@dataclass
class DecisionOption:
    name: str
    attributes: Dict[str, float]  # e.g. {'accuracy': 0.91, 'speed': 0.4, ...}


class DecisionMatrix:
    """
    Expected-utility decision support inspired by the
    utility_variables_bayesian_decision_network notes.

    Each option has a vector of attribute scores in [0, 1].
    The user supplies weights (importance) for each attribute.
    Utility(option) = sum_i weight_i * attribute_i.
    """

    def __init__(self, weights: Dict[str, float]):
        total = sum(max(0.0, w) for w in weights.values()) or 1.0
        self.weights = {k: max(0.0, v) / total for k, v in weights.items()}

    def score(self, options: List[DecisionOption]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for opt in options:
            utility = 0.0
            row: Dict[str, Any] = {'option': opt.name}
            for attr, w in self.weights.items():
                a = float(opt.attributes.get(attr, 0.0))
                row[attr] = a
                utility += w * a
            row['utility'] = utility
            rows.append(row)
        df = pd.DataFrame(rows).sort_values('utility', ascending=False).reset_index(drop=True)
        df['recommended'] = False
        if len(df):
            df.loc[0, 'recommended'] = True
        return df


# ---------------------------------------------------------------------------
# Ensemble builders (voting / bagging / boosting)
# ---------------------------------------------------------------------------
_CLF_FAMILY = {
    'logistic_regression': lambda: LogisticRegression(max_iter=1000),
    'decision_tree': lambda: DecisionTreeClassifier(),
    'random_forest': lambda: RandomForestClassifier(n_estimators=100),
    'gradient_boosting': lambda: GradientBoostingClassifier(),
    'knn': lambda: KNeighborsClassifier(),
    'svc': lambda: SVC(probability=True),
}

_REG_FAMILY = {
    'ridge': lambda: Ridge(),
    'decision_tree': lambda: DecisionTreeRegressor(),
    'random_forest': lambda: RandomForestRegressor(n_estimators=100),
    'gradient_boosting': lambda: GradientBoostingRegressor(),
    'knn': lambda: KNeighborsRegressor(),
    'svr': lambda: SVR(),
}


def _resolve(names: List[str], task: str) -> List[Tuple[str, BaseEstimator]]:
    family = _CLF_FAMILY if task == 'classification' else _REG_FAMILY
    resolved: List[Tuple[str, BaseEstimator]] = []
    for n in names:
        key = n.lower()
        if key in family:
            resolved.append((key, family[key]()))
    return resolved


def build_voting_ensemble(names: List[str], task: str='classification',
                          voting: str='soft') -> BaseEstimator:
    estimators = _resolve(names, task)
    if not estimators:
        raise ValueError("No valid estimators resolved from the given names")
    if task == 'classification':
        return VotingClassifier(estimators=estimators, voting=voting)
    return VotingRegressor(estimators=estimators)


def build_bagging_ensemble(base_name: str, task: str='classification',
                           n_estimators: int=50) -> BaseEstimator:
    base = _resolve([base_name], task)
    if not base:
        raise ValueError(f"Unknown base estimator: {base_name}")
    estimator = base[0][1]
    if task == 'classification':
        return BaggingClassifier(estimator=estimator, n_estimators=n_estimators, n_jobs=-1)
    return BaggingRegressor(estimator=estimator, n_estimators=n_estimators, n_jobs=-1)


def build_boosting_ensemble(task: str='classification',
                            n_estimators: int=100,
                            learning_rate: float=0.5) -> BaseEstimator:
    if task == 'classification':
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_estimators, learning_rate=learning_rate)
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=n_estimators, learning_rate=learning_rate)

