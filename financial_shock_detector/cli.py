"""Command-line interface for financial shock detector."""

import argparse
import sys
from pathlib import Path

from .pipeline import FinancialShockPipeline
from .utils import load_data, create_sample_financial_data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Financial Shock Detector - Detect financial shock events using NLP"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run the detection pipeline")
    run_parser.add_argument("--input", type=str, required=True, help="Input data file (CSV)")
    run_parser.add_argument("--text-column", type=str, default="text", help="Name of text column")
    run_parser.add_argument("--label-column", type=str, help="Name of label column (optional)")
    run_parser.add_argument("--output", type=str, default="output", help="Output directory")
    run_parser.add_argument("--config", type=str, help="Configuration file")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample data")
    demo_parser.add_argument("--n-samples", type=int, default=100, help="Number of samples")
    demo_parser.add_argument("--output", type=str, default="demo_output", help="Output directory")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with trained model")
    predict_parser.add_argument("--model-dir", type=str, required=True, help="Model directory")
    predict_parser.add_argument("--input", type=str, required=True, help="Input data file")
    predict_parser.add_argument("--text-column", type=str, default="text", help="Name of text column")
    predict_parser.add_argument("--output", type=str, help="Output file for predictions")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_pipeline(args):
    """Run the detection pipeline."""
    print("Loading data...")
    data = load_data(args.input)

    if args.text_column not in data.columns:
        print(f"Error: Column '{args.text_column}' not found in data")
        sys.exit(1)

    texts = data[args.text_column].tolist()
    labels = None

    if args.label_column and args.label_column in data.columns:
        labels = data[args.label_column].values

    # Create pipeline
    pipeline = FinancialShockPipeline()

    # Run pipeline
    results = pipeline.run_full_pipeline(texts, labels=labels, output_dir=args.output)

    print(f"\nPipeline completed successfully!")
    print(f"Results saved to: {args.output}")


def run_demo(args):
    """Run demo with sample data."""
    print("Generating sample financial data...")
    data = create_sample_financial_data(n_samples=args.n_samples)

    print(f"Generated {len(data)} samples")
    print("\nSample data:")
    print(data.head())

    texts = data["text"].tolist()
    labels = data["label"].values

    # Create pipeline
    pipeline = FinancialShockPipeline()

    # Run pipeline
    results = pipeline.run_full_pipeline(texts, labels=labels, output_dir=args.output)

    print(f"\nDemo completed successfully!")
    print(f"Results saved to: {args.output}")


def run_predict(args):
    """Make predictions with trained model."""
    print("Loading model...")
    pipeline = FinancialShockPipeline.load_pipeline(args.model_dir)

    print("Loading data...")
    data = load_data(args.input)

    if args.text_column not in data.columns:
        print(f"Error: Column '{args.text_column}' not found in data")
        sys.exit(1)

    texts = data[args.text_column].tolist()

    print("Processing texts...")
    embeddings = pipeline.process_texts(texts)

    if pipeline.pca_reducer:
        print("Reducing dimensionality...")
        embeddings = pipeline.pca_reducer.transform(embeddings)

    print("Making predictions...")
    predictions = pipeline.predict(embeddings)

    data["prediction"] = predictions

    if args.output:
        print(f"Saving predictions to {args.output}...")
        data.to_csv(args.output, index=False)
    else:
        print("\nPredictions:")
        print(data[["text", "prediction"]].head(10))

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
