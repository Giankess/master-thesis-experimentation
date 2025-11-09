"""Main pipeline for financial shock detection."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import joblib

from .data_collection import FinancialDataScraper, FinancialAPIClient
from .nlp_processing import FinBERTProcessor
from .dimensionality_reduction import PCAReducer
from .clustering import ClusterEngine
from .classification import ShockEventClassifier
from .interpretation import ClusterExplainer
from .utils import load_config, save_config, get_default_config


class FinancialShockPipeline:
    """End-to-end pipeline for financial shock event detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline.

        Args:
            config: Configuration dictionary. If None, uses default config.
        """
        self.config = config or get_default_config()
        self.scraper = None
        self.api_client = None
        self.nlp_processor = None
        self.pca_reducer = None
        self.cluster_engine = None
        self.classifier = None
        self.explainer = None

        # Data storage
        self.raw_texts = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None
        self.predictions = None

    def setup_data_collection(self, api_key: Optional[str] = None):
        """
        Set up data collection components.

        Args:
            api_key: Optional API key for financial data APIs
        """
        self.scraper = FinancialDataScraper(
            user_agent=self.config.get("data_collection", {}).get("user_agent")
        )
        self.api_client = FinancialAPIClient(api_key=api_key)

    def setup_nlp_processor(self):
        """Set up NLP processor with FinBERT."""
        nlp_config = self.config.get("nlp_processing", {})
        self.nlp_processor = FinBERTProcessor(
            model_name=nlp_config.get("model_name", "yiyanghkust/finbert-tone"),
            max_length=nlp_config.get("max_length", 512),
        )

    def collect_data_from_urls(self, urls: list, delay: float = 1.0) -> pd.DataFrame:
        """
        Collect data from URLs.

        Args:
            urls: List of URLs to scrape
            delay: Delay between requests

        Returns:
            DataFrame with scraped data
        """
        if self.scraper is None:
            self.setup_data_collection()

        return self.scraper.scrape_to_dataframe(urls, delay=delay)

    def collect_data_from_api(
        self, query: str, from_date: str, to_date: str, limit: int = 100
    ) -> list:
        """
        Collect data from API.

        Args:
            query: Search query
            from_date: Start date
            to_date: End date
            limit: Maximum number of articles

        Returns:
            List of articles
        """
        if self.api_client is None:
            self.setup_data_collection()

        return self.api_client.fetch_news(query, from_date, to_date, limit)

    def process_texts(self, texts: list) -> np.ndarray:
        """
        Process texts and generate embeddings.

        Args:
            texts: List of texts

        Returns:
            Embeddings array
        """
        if self.nlp_processor is None:
            self.setup_nlp_processor()

        nlp_config = self.config.get("nlp_processing", {})
        self.raw_texts = texts
        self.embeddings = self.nlp_processor.get_embeddings(
            texts,
            batch_size=nlp_config.get("batch_size", 8),
            pooling=nlp_config.get("pooling", "cls"),
        )

        return self.embeddings

    def reduce_dimensionality(self, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reduce dimensionality using PCA.

        Args:
            embeddings: Input embeddings. If None, uses self.embeddings

        Returns:
            Reduced embeddings
        """
        if embeddings is None:
            embeddings = self.embeddings

        if embeddings is None:
            raise ValueError("No embeddings available. Call process_texts first.")

        pca_config = self.config.get("dimensionality_reduction", {})
        self.pca_reducer = PCAReducer(
            variance_threshold=pca_config.get("variance_threshold", 0.95),
            standardize=pca_config.get("standardize", True),
        )

        self.reduced_embeddings = self.pca_reducer.fit_transform(embeddings)

        return self.reduced_embeddings

    def cluster_data(self, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cluster data using GMM or DBSCAN.

        Args:
            embeddings: Input embeddings. If None, uses reduced_embeddings or embeddings

        Returns:
            Cluster labels
        """
        if embeddings is None:
            embeddings = self.reduced_embeddings if self.reduced_embeddings is not None else self.embeddings

        if embeddings is None:
            raise ValueError("No embeddings available. Call process_texts first.")

        cluster_config = self.config.get("clustering", {})
        self.cluster_engine = ClusterEngine(
            method=cluster_config.get("method", "gmm"),
            **{k: v for k, v in cluster_config.items() if k != "method"},
        )

        self.cluster_labels = self.cluster_engine.fit_predict(embeddings)

        return self.cluster_labels

    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Train classifier on labeled data.

        Args:
            X: Features
            y: Labels
            model_type: Type of classifier (overrides config)

        Returns:
            Training metrics
        """
        classifier_config = self.config.get("classification", {})

        if model_type is None:
            model_type = classifier_config.get("model_type", "xgboost")

        self.classifier = ShockEventClassifier(
            model_type=model_type,
            **{k: v for k, v in classifier_config.items() if k not in ["model_type", "test_size"]},
        )

        train_metrics, test_metrics = self.classifier.train_and_validate(
            X, y, test_size=classifier_config.get("test_size", 0.2)
        )

        return test_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict shock events for new data.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")

        return self.classifier.predict(X)

    def explain_clusters(
        self, embeddings: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None
    ):
        """
        Explain clusters using XAI.

        Args:
            embeddings: Input embeddings
            labels: Cluster labels
        """
        if embeddings is None:
            embeddings = self.reduced_embeddings if self.reduced_embeddings is not None else self.embeddings

        if labels is None:
            labels = self.cluster_labels

        if embeddings is None or labels is None:
            raise ValueError("No embeddings or labels available.")

        self.explainer = ClusterExplainer()
        return self.explainer.analyze_cluster_characteristics(embeddings, labels)

    def run_full_pipeline(
        self,
        texts: list,
        labels: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from text to prediction.

        Args:
            texts: List of texts to process
            labels: Optional labels for supervised learning
            output_dir: Directory to save outputs

        Returns:
            Dictionary with pipeline results
        """
        print("=" * 60)
        print("FINANCIAL SHOCK DETECTION PIPELINE")
        print("=" * 60)

        # Step 1: Process texts
        print("\n[1/6] Processing texts with FinBERT...")
        embeddings = self.process_texts(texts)
        print(f"Generated embeddings: {embeddings.shape}")

        # Step 2: Reduce dimensionality
        print("\n[2/6] Reducing dimensionality with PCA...")
        reduced = self.reduce_dimensionality(embeddings)
        print(f"Reduced embeddings: {reduced.shape}")
        print(f"Explained variance: {self.pca_reducer.get_cumulative_variance()[-1]:.2%}")

        # Step 3: Cluster data
        print("\n[3/6] Clustering data...")
        cluster_labels = self.cluster_data(reduced)
        print(f"Found {len(np.unique(cluster_labels))} clusters")

        # Step 4: Explain clusters
        print("\n[4/6] Analyzing clusters...")
        cluster_stats = self.explain_clusters(reduced, cluster_labels)
        print("\nCluster statistics:")
        print(cluster_stats)

        # Step 5: Train classifier (if labels provided)
        metrics = None
        if labels is not None:
            print("\n[5/6] Training classifier...")
            metrics = self.train_classifier(reduced, labels)
            print(f"\nTest metrics: {metrics}")
        else:
            print("\n[5/6] Skipping classifier training (no labels provided)")

        # Step 6: Save outputs
        if output_dir:
            print("\n[6/6] Saving outputs...")
            self.save_pipeline(output_dir)
            print(f"Outputs saved to {output_dir}")
        else:
            print("\n[6/6] Skipping output saving")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)

        return {
            "embeddings": embeddings,
            "reduced_embeddings": reduced,
            "cluster_labels": cluster_labels,
            "cluster_stats": cluster_stats,
            "metrics": metrics,
        }

    def save_pipeline(self, output_dir: str):
        """
        Save pipeline components and results.

        Args:
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save models
        if self.pca_reducer:
            self.pca_reducer.save(output_path / "pca_reducer.pkl")

        if self.cluster_engine:
            self.cluster_engine.save(output_path / "cluster_engine.pkl")

        if self.classifier:
            self.classifier.save(output_path / "classifier.pkl")

        # Save config
        save_config(self.config, str(output_path / "config.yaml"))

        # Save data
        if self.embeddings is not None:
            np.save(output_path / "embeddings.npy", self.embeddings)

        if self.reduced_embeddings is not None:
            np.save(output_path / "reduced_embeddings.npy", self.reduced_embeddings)

        if self.cluster_labels is not None:
            np.save(output_path / "cluster_labels.npy", self.cluster_labels)

        # Save plots
        if self.config.get("output", {}).get("save_plots", True):
            if self.pca_reducer:
                self.pca_reducer.plot_variance_explained(
                    save_path=str(output_path / "pca_variance.png")
                )

            if self.cluster_engine and self.reduced_embeddings is not None:
                self.cluster_engine.plot_clusters_2d(
                    self.reduced_embeddings,
                    save_path=str(output_path / "clusters.png"),
                )

    @classmethod
    def load_pipeline(cls, output_dir: str) -> "FinancialShockPipeline":
        """
        Load saved pipeline.

        Args:
            output_dir: Directory with saved pipeline

        Returns:
            Loaded pipeline instance
        """
        output_path = Path(output_dir)

        # Load config
        config = load_config(str(output_path / "config.yaml"))

        pipeline = cls(config=config)

        # Load models
        if (output_path / "pca_reducer.pkl").exists():
            pipeline.pca_reducer = PCAReducer.load(str(output_path / "pca_reducer.pkl"))

        if (output_path / "cluster_engine.pkl").exists():
            pipeline.cluster_engine = ClusterEngine.load(str(output_path / "cluster_engine.pkl"))

        if (output_path / "classifier.pkl").exists():
            pipeline.classifier = ShockEventClassifier.load(str(output_path / "classifier.pkl"))

        # Load data
        if (output_path / "embeddings.npy").exists():
            pipeline.embeddings = np.load(output_path / "embeddings.npy")

        if (output_path / "reduced_embeddings.npy").exists():
            pipeline.reduced_embeddings = np.load(output_path / "reduced_embeddings.npy")

        if (output_path / "cluster_labels.npy").exists():
            pipeline.cluster_labels = np.load(output_path / "cluster_labels.npy")

        return pipeline
