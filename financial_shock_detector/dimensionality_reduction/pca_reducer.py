"""PCA-based dimensionality reduction."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import joblib


class PCAReducer:
    """Dimensionality reduction using PCA."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        standardize: bool = True,
    ):
        """
        Initialize PCA reducer.

        Args:
            n_components: Number of components to keep. If None, use variance_threshold
            variance_threshold: Minimum cumulative variance to explain (if n_components is None)
            standardize: Whether to standardize data before PCA
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.pca = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """
        Fit PCA on data.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Self
        """
        # Standardize if needed
        if self.standardize:
            X = self.scaler.fit_transform(X)

        # Fit PCA
        if self.n_components is not None:
            self.pca = PCA(n_components=self.n_components)
        else:
            # Fit with all components first to determine n_components
            temp_pca = PCA()
            temp_pca.fit(X)

            # Find number of components for desired variance
            cumsum_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= self.variance_threshold) + 1

            print(
                f"Selected {n_components} components to explain "
                f"{self.variance_threshold * 100:.1f}% variance"
            )

            self.pca = PCA(n_components=n_components)

        self.pca.fit(X)
        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Transformed data (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted before transform")

        # Standardize if needed
        if self.standardize:
            X = self.scaler.transform(X)

        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Transformed data (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.

        Args:
            X_reduced: Reduced data (n_samples, n_components)

        Returns:
            Reconstructed data (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted before inverse_transform")

        X_reconstructed = self.pca.inverse_transform(X_reduced)

        # Inverse standardization if needed
        if self.standardize:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)

        return X_reconstructed

    def get_explained_variance(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.

        Returns:
            Array of explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted first")

        return self.pca.explained_variance_ratio_

    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance.

        Returns:
            Array of cumulative explained variance
        """
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted first")

        return np.cumsum(self.pca.explained_variance_ratio_)

    def plot_variance_explained(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot explained variance by component.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        ax1.bar(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            self.pca.explained_variance_ratio_,
        )
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("Variance Explained by Each Component")

        # Cumulative variance
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, marker="o")
        ax2.axhline(
            y=self.variance_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold ({self.variance_threshold * 100:.0f}%)",
        )
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title("Cumulative Variance Explained")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def get_n_components(self) -> int:
        """
        Get number of components.

        Returns:
            Number of components
        """
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted first")

        return self.pca.n_components_

    def save(self, filepath: str) -> None:
        """
        Save PCA reducer to file.

        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("PCAReducer must be fitted before saving")

        joblib.dump(
            {
                "pca": self.pca,
                "scaler": self.scaler,
                "n_components": self.n_components,
                "variance_threshold": self.variance_threshold,
                "standardize": self.standardize,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "PCAReducer":
        """
        Load PCA reducer from file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded PCAReducer instance
        """
        data = joblib.load(filepath)

        reducer = cls(
            n_components=data["n_components"],
            variance_threshold=data["variance_threshold"],
            standardize=data["standardize"],
        )
        reducer.pca = data["pca"]
        reducer.scaler = data["scaler"]
        reducer.is_fitted = True

        return reducer
