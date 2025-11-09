"""Clustering engine with GMM and DBSCAN."""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
import joblib


class ClusterEngine:
    """Clustering engine supporting GMM and DBSCAN."""

    def __init__(self, method: str = "gmm", **kwargs):
        """
        Initialize clustering engine.

        Args:
            method: Clustering method ('gmm' or 'dbscan')
            **kwargs: Additional parameters for the clustering algorithm
        """
        self.method = method.lower()
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
        self.is_fitted = False

        if self.method == "gmm":
            self._init_gmm()
        elif self.method == "dbscan":
            self._init_dbscan()
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _init_gmm(self):
        """Initialize Gaussian Mixture Model."""
        n_components = self.kwargs.get("n_components", 3)
        covariance_type = self.kwargs.get("covariance_type", "full")
        random_state = self.kwargs.get("random_state", 42)
        max_iter = self.kwargs.get("max_iter", 100)

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter,
        )

    def _init_dbscan(self):
        """Initialize DBSCAN."""
        eps = self.kwargs.get("eps", 0.5)
        min_samples = self.kwargs.get("min_samples", 5)
        metric = self.kwargs.get("metric", "euclidean")

        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def fit(self, X: np.ndarray) -> "ClusterEngine":
        """
        Fit clustering model on data.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Self
        """
        self.model.fit(X)

        if self.method == "gmm":
            self.labels_ = self.model.predict(X)
        elif self.method == "dbscan":
            self.labels_ = self.model.labels_

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for data.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("ClusterEngine must be fitted before predict")

        if self.method == "gmm":
            return self.model.predict(X)
        elif self.method == "dbscan":
            # DBSCAN doesn't have a predict method, use fit_predict
            # For new data, use nearest neighbor approach
            from sklearn.neighbors import NearestNeighbors

            # This is a simple approximation for DBSCAN prediction
            print(
                "Warning: DBSCAN doesn't support prediction on new data directly. "
                "Using fitted labels."
            )
            return self.labels_
        else:
            raise ValueError(f"Prediction not implemented for {self.method}")

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and predict cluster labels.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers (only for GMM).

        Returns:
            Cluster centers or None if not applicable
        """
        if self.method == "gmm" and self.is_fitted:
            return self.model.means_
        return None

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("ClusterEngine must be fitted before evaluation")

        metrics = {}

        # Remove noise points for DBSCAN (label -1)
        if self.method == "dbscan":
            mask = self.labels_ != -1
            if mask.sum() < 2:
                print("Warning: Too few non-noise points for evaluation")
                return {"n_clusters": 0, "n_noise": len(self.labels_)}

            X_eval = X[mask]
            labels_eval = self.labels_[mask]
        else:
            X_eval = X
            labels_eval = self.labels_

        # Number of clusters
        n_clusters = len(np.unique(labels_eval))
        metrics["n_clusters"] = n_clusters

        if self.method == "dbscan":
            metrics["n_noise"] = (self.labels_ == -1).sum()

        # Silhouette score (requires at least 2 clusters)
        if n_clusters > 1:
            try:
                metrics["silhouette_score"] = silhouette_score(X_eval, labels_eval)
                metrics["davies_bouldin_score"] = davies_bouldin_score(X_eval, labels_eval)
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_eval, labels_eval)
            except Exception as e:
                print(f"Error computing metrics: {e}")

        # GMM-specific metrics
        if self.method == "gmm":
            metrics["bic"] = self.model.bic(X)
            metrics["aic"] = self.model.aic(X)

        return metrics

    def find_optimal_n_clusters(
        self, X: np.ndarray, n_range: Tuple[int, int] = (2, 10)
    ) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters for GMM using BIC.

        Args:
            X: Input data (n_samples, n_features)
            n_range: Range of n_components to try (min, max)

        Returns:
            Tuple of (optimal_n, results_dict)
        """
        if self.method != "gmm":
            raise ValueError("Optimal cluster search only supported for GMM")

        results = {"n_components": [], "bic": [], "aic": [], "silhouette": []}

        for n in range(n_range[0], n_range[1] + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.kwargs.get("covariance_type", "full"),
                random_state=self.kwargs.get("random_state", 42),
            )
            gmm.fit(X)
            labels = gmm.predict(X)

            results["n_components"].append(n)
            results["bic"].append(gmm.bic(X))
            results["aic"].append(gmm.aic(X))

            if len(np.unique(labels)) > 1:
                results["silhouette"].append(silhouette_score(X, labels))
            else:
                results["silhouette"].append(0)

        # Optimal n is where BIC is minimum
        optimal_idx = np.argmin(results["bic"])
        optimal_n = results["n_components"][optimal_idx]

        print(f"Optimal number of clusters: {optimal_n}")

        return optimal_n, results

    def plot_clusters_2d(
        self, X: np.ndarray, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot clusters in 2D (assumes X has 2 features or uses first 2).

        Args:
            X: Input data (n_samples, n_features)
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("ClusterEngine must be fitted before plotting")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use first 2 dimensions if X has more features
        X_plot = X[:, :2] if X.shape[1] >= 2 else X

        # Plot points colored by cluster
        scatter = ax.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            c=self.labels_,
            cmap="viridis",
            alpha=0.6,
            edgecolors="k",
            linewidth=0.5,
        )

        # Plot cluster centers for GMM
        if self.method == "gmm":
            centers = self.get_cluster_centers()
            if centers is not None:
                centers_plot = centers[:, :2] if centers.shape[1] >= 2 else centers
                ax.scatter(
                    centers_plot[:, 0],
                    centers_plot[:, 1],
                    c="red",
                    marker="X",
                    s=200,
                    edgecolors="black",
                    linewidth=2,
                    label="Cluster Centers",
                )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"Clusters ({self.method.upper()})")
        plt.colorbar(scatter, ax=ax, label="Cluster Label")

        if self.method == "gmm":
            ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def save(self, filepath: str) -> None:
        """
        Save clustering model to file.

        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("ClusterEngine must be fitted before saving")

        joblib.dump(
            {
                "method": self.method,
                "model": self.model,
                "labels_": self.labels_,
                "kwargs": self.kwargs,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "ClusterEngine":
        """
        Load clustering model from file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded ClusterEngine instance
        """
        data = joblib.load(filepath)

        engine = cls(method=data["method"], **data["kwargs"])
        engine.model = data["model"]
        engine.labels_ = data["labels_"]
        engine.is_fitted = True

        return engine
