"""Quick start example that doesn't require heavy dependencies."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from financial_shock_detector.utils import (
    create_sample_financial_data,
    preprocess_text,
    get_default_config,
)


def main():
    """Quick start demonstration."""
    print("=" * 70)
    print("FINANCIAL SHOCK DETECTOR - QUICK START")
    print("=" * 70)

    # 1. Create sample data
    print("\n[1] Creating sample financial news data...")
    data = create_sample_financial_data(n_samples=30)
    print(f"    Created {len(data)} samples")
    print(f"    Shock events: {data['label'].sum()}")
    print(f"    Normal events: {(data['label'] == 0).sum()}")

    # 2. Show sample texts
    print("\n[2] Sample financial news headlines:")
    for i, row in data.head(5).iterrows():
        label = "SHOCK" if row["label"] == 1 else "NORMAL"
        print(f"    [{label}] {row['text']}")

    # 3. Text preprocessing
    print("\n[3] Text preprocessing example:")
    sample_text = data.iloc[0]["text"]
    print(f"    Original: {sample_text}")
    cleaned = preprocess_text(sample_text)
    print(f"    Cleaned:  {cleaned}")

    # 4. Show configuration
    print("\n[4] Default pipeline configuration:")
    config = get_default_config()
    print(f"    NLP Model: {config['nlp_processing']['model_name']}")
    print(f"    Embedding Pooling: {config['nlp_processing']['pooling']}")
    print(f"    PCA Variance: {config['dimensionality_reduction']['variance_threshold']}")
    print(f"    Clustering: {config['clustering']['method'].upper()} with {config['clustering']['n_components']} components")
    print(f"    Classification: {config['classification']['model_type'].upper()}")

    # 5. Data statistics
    print("\n[5] Data statistics:")
    print(f"    Total samples: {len(data)}")
    print(f"    Average text length: {data['text'].str.len().mean():.1f} characters")
    print(f"    Label distribution: {dict(data['label'].value_counts())}")

    # 6. Pipeline overview
    print("\n[6] Pipeline workflow:")
    print("    Step 1: Data Collection (API / Web Scraping)")
    print("    Step 2: NLP Processing (FinBERT Embeddings)")
    print("    Step 3: Dimensionality Reduction (PCA)")
    print("    Step 4: Clustering (GMM / DBSCAN)")
    print("    Step 5: Classification (LR / RF / NN / XGBoost)")
    print("    Step 6: Interpretation (SHAP / XAI)")

    print("\n" + "=" * 70)
    print("To run the full pipeline with FinBERT, use:")
    print("  python examples/basic_usage.py")
    print("Or with custom configuration:")
    print("  python examples/advanced_usage.py")
    print("\nTo use the CLI:")
    print("  financial-shock-detector demo --n-samples 100")
    print("=" * 70)


if __name__ == "__main__":
    main()
