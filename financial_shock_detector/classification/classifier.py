"""Multi-model classifier for financial shock events."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb
from typing import Optional, Dict, Tuple, Any
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class ShockEventClassifier:
    """Classifier for detecting financial shock events."""

    def __init__(
        self,
        model_type: str = "logistic",
        **kwargs,
    ):
        """
        Initialize shock event classifier.

        Args:
            model_type: Type of classifier ('logistic', 'random_forest', 'neural_network', 'xgboost')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type.lower()
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self._init_model()

    def _init_model(self):
        """Initialize the classification model."""
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=self.kwargs.get("max_iter", 1000),
                random_state=self.kwargs.get("random_state", 42),
                C=self.kwargs.get("C", 1.0),
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", None),
                random_state=self.kwargs.get("random_state", 42),
            )
        elif self.model_type == "neural_network":
            hidden_layer_sizes = self.kwargs.get("hidden_layer_sizes", (100, 50))
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=self.kwargs.get("max_iter", 500),
                random_state=self.kwargs.get("random_state", 42),
                early_stopping=self.kwargs.get("early_stopping", True),
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", 6),
                learning_rate=self.kwargs.get("learning_rate", 0.1),
                random_state=self.kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShockEventClassifier":
        """
        Train the classifier.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for data.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict_proba")

        return self.model.predict_proba(X)

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray, average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.

        Args:
            X: Input features
            y_true: True labels
            average: Averaging method for multi-class metrics

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before evaluation")

        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        return metrics

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation.

        Args:
            X: Input features
            y: Labels
            cv: Number of folds

        Returns:
            Dictionary with cross-validation scores
        """
        scores = {
            "accuracy": cross_val_score(self.model, X, y, cv=cv, scoring="accuracy"),
            "precision": cross_val_score(
                self.model, X, y, cv=cv, scoring="precision_weighted"
            ),
            "recall": cross_val_score(self.model, X, y, cv=cv, scoring="recall_weighted"),
            "f1": cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted"),
        }

        return scores

    def get_classification_report(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> str:
        """
        Generate classification report.

        Args:
            X: Input features
            y_true: True labels

        Returns:
            Classification report string
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before generating report")

        y_pred = self.predict(X)
        return classification_report(y_true, y_pred)

    def plot_confusion_matrix(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            X: Input features
            y_true: True labels
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before plotting")

        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix ({self.model_type})")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (if supported by model).

        Returns:
            Feature importance array or None
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        if self.model_type in ["random_forest", "xgboost"]:
            return self.model.feature_importances_
        elif self.model_type == "logistic":
            # For binary classification, use coefficients
            return np.abs(self.model.coef_[0])
        else:
            return None

    def plot_feature_importance(
        self,
        feature_names: Optional[list] = None,
        top_k: int = 10,
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Plot feature importance.

        Args:
            feature_names: List of feature names
            top_k: Number of top features to show
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure or None if not supported
        """
        importances = self.get_feature_importance()

        if importances is None:
            print(f"Feature importance not supported for {self.model_type}")
            return None

        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=(10, 6))

        names = (
            [feature_names[i] for i in sorted_idx]
            if feature_names
            else [f"Feature {i}" for i in sorted_idx]
        )

        ax.barh(range(top_k), importances[sorted_idx])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_k} Feature Importance ({self.model_type})")
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def train_and_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Train model and validate on test set.

        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            random_state: Random state for split

        Returns:
            Tuple of (train_metrics, test_metrics)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train
        self.fit(X_train, y_train)

        # Evaluate
        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)

        print("Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        return train_metrics, test_metrics

    def save(self, filepath: str) -> None:
        """
        Save classifier to file.

        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before saving")

        joblib.dump(
            {
                "model_type": self.model_type,
                "model": self.model,
                "kwargs": self.kwargs,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "ShockEventClassifier":
        """
        Load classifier from file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded ShockEventClassifier instance
        """
        data = joblib.load(filepath)

        classifier = cls(model_type=data["model_type"], **data["kwargs"])
        classifier.model = data["model"]
        classifier.is_fitted = True

        return classifier
