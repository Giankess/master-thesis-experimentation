"""Basic usage example of the financial shock detector."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_shock_detector import FinancialShockPipeline
from financial_shock_detector.utils import create_sample_financial_data


def main():
    """Run basic example."""
    print("Financial Shock Detector - Basic Example")
    print("=" * 60)

    # Create sample data
    print("\n1. Creating sample financial data...")
    data = create_sample_financial_data(n_samples=50)
    print(f"Generated {len(data)} samples")
    print("\nSample texts:")
    for i, text in enumerate(data["text"].head(3)):
        print(f"  {i+1}. {text}")

    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = FinancialShockPipeline()

    # Extract texts and labels
    texts = data["text"].tolist()
    labels = data["label"].values

    print(f"\nTexts: {len(texts)}")
    print(f"Labels: {labels.sum()} shock events, {len(labels) - labels.sum()} normal")

    # Run pipeline
    print("\n3. Running pipeline...")
    results = pipeline.run_full_pipeline(
        texts=texts,
        labels=labels,
        output_dir="output/basic_example"
    )

    print("\n4. Results Summary:")
    print(f"  - Embeddings shape: {results['embeddings'].shape}")
    print(f"  - Reduced embeddings shape: {results['reduced_embeddings'].shape}")
    print(f"  - Number of clusters: {len(set(results['cluster_labels']))}")

    if results["metrics"]:
        print("\n5. Classification Metrics:")
        for metric, value in results["metrics"].items():
            print(f"  - {metric}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
