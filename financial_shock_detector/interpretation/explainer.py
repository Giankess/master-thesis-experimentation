"""Explainable AI for cluster interpretation using SHAP."""

import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Optional, List, Any
import pandas as pd


class ClusterExplainer:
    """Explain clusters and predictions using SHAP and other XAI techniques."""

    def __init__(self, model: Any = None, feature_names: Optional[List[str]] = None):
        """
        Initialize cluster explainer.

        Args:
            model: Trained model to explain
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def create_shap_explainer(
        self, X_background: np.ndarray, explainer_type: str = "tree"
    ) -> None:
        """
        Create SHAP explainer for the model.

        Args:
            X_background: Background data for SHAP explainer
            explainer_type: Type of explainer ('tree', 'linear', 'kernel')
        """
        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model, X_background)
        elif explainer_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, X_background)
        elif explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(self.model.predict, X_background)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Get SHAP values for data.

        Args:
            X: Input data

        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not created. Call create_shap_explainer first.")

        return self.explainer.shap_values(X)

    def plot_shap_summary(
        self,
        X: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot SHAP summary plot.

        Args:
            X: Input data
            shap_values: Pre-computed SHAP values (optional)
            save_path: Optional path to save figure
        """
        if shap_values is None:
            shap_values = self.get_shap_values(X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_shap_waterfall(
        self,
        X: np.ndarray,
        index: int = 0,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot SHAP waterfall plot for a single instance.

        Args:
            X: Input data
            index: Index of instance to explain
            shap_values: Pre-computed SHAP values (optional)
            save_path: Optional path to save figure
        """
        if shap_values is None:
            shap_values = self.get_shap_values(X)

        # Create explanation object
        if self.explainer is not None:
            explanation = shap.Explanation(
                values=shap_values[index],
                base_values=self.explainer.expected_value,
                data=X[index],
                feature_names=self.feature_names,
            )

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()

    def analyze_cluster_characteristics(
        self, X: np.ndarray, labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze characteristics of each cluster.

        Args:
            X: Input data
            labels: Cluster labels

        Returns:
            DataFrame with cluster statistics
        """
        results = []

        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue

            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]

            stats = {
                "cluster_id": cluster_id,
                "size": cluster_data.shape[0],
                "mean": cluster_data.mean(axis=0),
                "std": cluster_data.std(axis=0),
                "min": cluster_data.min(axis=0),
                "max": cluster_data.max(axis=0),
            }
            results.append(stats)

        # Create summary DataFrame
        summary_data = []
        for result in results:
            row = {"cluster_id": result["cluster_id"], "size": result["size"]}

            # Add feature statistics
            if self.feature_names:
                for i, feature_name in enumerate(self.feature_names):
                    row[f"{feature_name}_mean"] = result["mean"][i]
                    row[f"{feature_name}_std"] = result["std"][i]
            else:
                for i in range(len(result["mean"])):
                    row[f"feature_{i}_mean"] = result["mean"][i]
                    row[f"feature_{i}_std"] = result["std"][i]

            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def plot_feature_importance(
        self,
        X: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        top_k: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot feature importance based on SHAP values.

        Args:
            X: Input data
            shap_values: Pre-computed SHAP values (optional)
            top_k: Number of top features to show
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if shap_values is None:
            shap_values = self.get_shap_values(X)

        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Sort by importance
        sorted_idx = np.argsort(mean_shap)[::-1][:top_k]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        feature_names = (
            [self.feature_names[i] for i in sorted_idx]
            if self.feature_names
            else [f"Feature {i}" for i in sorted_idx]
        )

        ax.barh(range(top_k), mean_shap[sorted_idx])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_k} Feature Importance")
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def explain_cluster_differences(
        self, X: np.ndarray, labels: np.ndarray, cluster_a: int, cluster_b: int
    ) -> pd.DataFrame:
        """
        Explain differences between two clusters.

        Args:
            X: Input data
            labels: Cluster labels
            cluster_a: First cluster ID
            cluster_b: Second cluster ID

        Returns:
            DataFrame with differences
        """
        mask_a = labels == cluster_a
        mask_b = labels == cluster_b

        data_a = X[mask_a]
        data_b = X[mask_b]

        differences = []

        for i in range(X.shape[1]):
            feature_name = (
                self.feature_names[i] if self.feature_names else f"Feature {i}"
            )

            diff = {
                "feature": feature_name,
                f"cluster_{cluster_a}_mean": data_a[:, i].mean(),
                f"cluster_{cluster_b}_mean": data_b[:, i].mean(),
                "difference": abs(data_a[:, i].mean() - data_b[:, i].mean()),
                f"cluster_{cluster_a}_std": data_a[:, i].std(),
                f"cluster_{cluster_b}_std": data_b[:, i].std(),
            }
            differences.append(diff)

        df = pd.DataFrame(differences)
        return df.sort_values("difference", ascending=False)

    def interpret_predictions(
        self, X: np.ndarray, predictions: np.ndarray
    ) -> pd.DataFrame:
        """
        Interpret predictions for each sample.

        Args:
            X: Input data
            predictions: Model predictions

        Returns:
            DataFrame with interpretations
        """
        interpretations = []

        for i in range(len(X)):
            interpretation = {
                "sample_id": i,
                "prediction": predictions[i],
            }

            # Add feature values
            if self.feature_names:
                for j, feature_name in enumerate(self.feature_names):
                    interpretation[feature_name] = X[i, j]
            else:
                for j in range(X.shape[1]):
                    interpretation[f"feature_{j}"] = X[i, j]

            interpretations.append(interpretation)

        return pd.DataFrame(interpretations)
