"""Advanced usage example with custom configuration."""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_shock_detector import (
    FinancialShockPipeline,
    FinBERTProcessor,
    PCAReducer,
    ClusterEngine,
    ShockEventClassifier,
)
from financial_shock_detector.utils import create_sample_financial_data


def main():
    """Run advanced example with custom configuration."""
    print("Financial Shock Detector - Advanced Example")
    print("=" * 60)

    # Custom configuration
    config = {
        "nlp_processing": {
            "model_name": "yiyanghkust/finbert-tone",
            "max_length": 512,
            "batch_size": 4,
            "pooling": "cls",
        },
        "dimensionality_reduction": {
            "variance_threshold": 0.90,
            "standardize": True,
        },
        "clustering": {
            "method": "gmm",
            "n_components": 4,
            "covariance_type": "full",
        },
        "classification": {
            "model_type": "random_forest",
            "n_estimators": 50,
            "test_size": 0.25,
        },
    }

    # Create sample data
    print("\n1. Creating sample data...")
    data = create_sample_financial_data(n_samples=80)

    texts = data["text"].tolist()
    labels = data["label"].values

    # Initialize pipeline with custom config
    print("\n2. Initializing pipeline with custom configuration...")
    pipeline = FinancialShockPipeline(config=config)

    # Run individual steps with more control
    print("\n3. Running pipeline steps individually...")

    print("\n  Step 1: NLP Processing...")
    embeddings = pipeline.process_texts(texts)
    print(f"  Generated embeddings: {embeddings.shape}")

    print("\n  Step 2: Dimensionality Reduction...")
    reduced = pipeline.reduce_dimensionality(embeddings)
    print(f"  Reduced to: {reduced.shape}")
    print(f"  Cumulative variance: {pipeline.pca_reducer.get_cumulative_variance()[-1]:.2%}")

    print("\n  Step 3: Clustering...")
    cluster_labels = pipeline.cluster_data(reduced)
    unique_clusters = np.unique(cluster_labels)
    print(f"  Found {len(unique_clusters)} clusters")
    for cluster_id in unique_clusters:
        count = (cluster_labels == cluster_id).sum()
        print(f"    Cluster {cluster_id}: {count} samples")

    print("\n  Step 4: Cluster Interpretation...")
    cluster_stats = pipeline.explain_clusters(reduced, cluster_labels)
    print("\n  Cluster Statistics:")
    print(cluster_stats)

    print("\n  Step 5: Classification...")
    metrics = pipeline.train_classifier(reduced, labels)
    print("\n  Test Metrics:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.4f}")

    # Make predictions
    print("\n4. Making predictions on same data...")
    predictions = pipeline.predict(reduced)
    print(f"  Predicted {predictions.sum()} shock events")
    print(f"  Actual {labels.sum()} shock events")

    # Save pipeline
    output_dir = "output/advanced_example"
    print(f"\n5. Saving pipeline to {output_dir}...")
    pipeline.save_pipeline(output_dir)

    print("\n" + "=" * 60)
    print("Advanced example completed successfully!")


if __name__ == "__main__":
    main()
