# Financial Shock Detector - Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Examples](#detailed-examples)
4. [Component Usage](#component-usage)
5. [Configuration](#configuration)
6. [CLI Usage](#cli-usage)
7. [Advanced Topics](#advanced-topics)

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed:

```bash
python --version
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package

For development:
```bash
pip install -e .
```

For production:
```bash
pip install .
```

## Quick Start

### 1. Run Demo with Sample Data

The fastest way to see the package in action:

```bash
python examples/quick_start.py
```

This creates sample financial news data and demonstrates the package structure without requiring heavy dependencies.

### 2. Run Full Pipeline Demo

With dependencies installed:

```bash
python examples/basic_usage.py
```

Or use the CLI:

```bash
financial-shock-detector demo --n-samples 100 --output demo_output
```

## Detailed Examples

### Example 1: Basic Pipeline

```python
from financial_shock_detector import FinancialShockPipeline
from financial_shock_detector.utils import create_sample_financial_data

# Create sample data
data = create_sample_financial_data(n_samples=100)
texts = data["text"].tolist()
labels = data["label"].values

# Initialize pipeline
pipeline = FinancialShockPipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline(
    texts=texts,
    labels=labels,
    output_dir="output/my_results"
)

# Access results
print(f"Number of clusters: {len(set(results['cluster_labels']))}")
print(f"Classification accuracy: {results['metrics']['accuracy']:.3f}")
```

### Example 2: Custom Configuration

```python
from financial_shock_detector import FinancialShockPipeline

# Custom configuration
config = {
    "nlp_processing": {
        "model_name": "yiyanghkust/finbert-tone",
        "batch_size": 16,
        "pooling": "mean",  # Try different pooling
    },
    "dimensionality_reduction": {
        "variance_threshold": 0.90,  # Keep less variance
    },
    "clustering": {
        "method": "dbscan",  # Use DBSCAN instead of GMM
        "eps": 0.5,
        "min_samples": 5,
    },
    "classification": {
        "model_type": "random_forest",  # Use Random Forest
        "n_estimators": 200,
    },
}

pipeline = FinancialShockPipeline(config=config)
results = pipeline.run_full_pipeline(texts, labels, output_dir="output/custom")
```

### Example 3: Step-by-Step Execution

```python
from financial_shock_detector import FinancialShockPipeline

pipeline = FinancialShockPipeline()

# Step 1: Process texts
print("Processing texts...")
embeddings = pipeline.process_texts(texts)
print(f"Embeddings shape: {embeddings.shape}")

# Step 2: Reduce dimensionality
print("Reducing dimensionality...")
reduced = pipeline.reduce_dimensionality(embeddings)
print(f"Reduced shape: {reduced.shape}")

# Step 3: Cluster
print("Clustering...")
clusters = pipeline.cluster_data(reduced)
print(f"Found {len(set(clusters))} clusters")

# Step 4: Train classifier
print("Training classifier...")
metrics = pipeline.train_classifier(reduced, labels)
print(f"Test F1 score: {metrics['f1_score']:.3f}")

# Step 5: Make predictions
print("Making predictions...")
predictions = pipeline.predict(reduced)
print(f"Predicted {predictions.sum()} shock events")
```

## Component Usage

### Data Collection

#### Web Scraping

```python
from financial_shock_detector import FinancialDataScraper

scraper = FinancialDataScraper()

# Scrape single URL
urls = ["https://example.com/article1", "https://example.com/article2"]
data = scraper.scrape_to_dataframe(urls, delay=1.0)
```

#### API Integration

```python
from financial_shock_detector import FinancialAPIClient

client = FinancialAPIClient(api_key="your_api_key")

# Fetch news
articles = client.fetch_news(
    query="financial crisis",
    from_date="2024-01-01",
    to_date="2024-12-31",
    limit=100
)

# Fetch market data
market_data = client.fetch_market_data(
    symbol="AAPL",
    from_date="2024-01-01",
    to_date="2024-12-31"
)
```

### NLP Processing

```python
from financial_shock_detector import FinBERTProcessor

# Initialize processor
processor = FinBERTProcessor(
    model_name="yiyanghkust/finbert-tone",
    max_length=512
)

# Generate embeddings
texts = ["Market crashes", "Economy grows"]
embeddings = processor.get_embeddings(texts, batch_size=8, pooling="cls")

# Get embedding dimension
dim = processor.get_embedding_dim()
print(f"Embedding dimension: {dim}")
```

### Dimensionality Reduction

```python
from financial_shock_detector import PCAReducer

# Initialize PCA
pca = PCAReducer(variance_threshold=0.95, standardize=True)

# Fit and transform
reduced = pca.fit_transform(embeddings)

# Get explained variance
variance = pca.get_explained_variance()
print(f"Explained variance per component: {variance}")

# Plot variance
pca.plot_variance_explained(save_path="variance.png")

# Save model
pca.save("models/pca_model.pkl")

# Load model
pca_loaded = PCAReducer.load("models/pca_model.pkl")
```

### Clustering

```python
from financial_shock_detector import ClusterEngine

# Gaussian Mixture Model
gmm = ClusterEngine(method="gmm", n_components=3)
labels = gmm.fit_predict(reduced)

# Evaluate
metrics = gmm.evaluate(reduced)
print(f"Silhouette score: {metrics['silhouette_score']:.3f}")

# Find optimal clusters
optimal_n, results = gmm.find_optimal_n_clusters(reduced, n_range=(2, 10))

# DBSCAN
dbscan = ClusterEngine(method="dbscan", eps=0.5, min_samples=5)
labels = dbscan.fit_predict(reduced)

# Plot clusters
gmm.plot_clusters_2d(reduced, save_path="clusters.png")
```

### Classification

```python
from financial_shock_detector import ShockEventClassifier

# XGBoost (recommended)
xgb = ShockEventClassifier(
    model_type="xgboost",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# Train and validate
train_metrics, test_metrics = xgb.train_and_validate(X, y, test_size=0.2)

# Make predictions
predictions = xgb.predict(X_new)
probabilities = xgb.predict_proba(X_new)

# Get feature importance
importance = xgb.get_feature_importance()
xgb.plot_feature_importance(feature_names=["feat1", "feat2"], top_k=10)

# Cross-validation
cv_scores = xgb.cross_validate(X, y, cv=5)
print(f"CV accuracy: {cv_scores['accuracy'].mean():.3f}")

# Other models
rf = ShockEventClassifier(model_type="random_forest", n_estimators=100)
lr = ShockEventClassifier(model_type="logistic", C=1.0)
nn = ShockEventClassifier(model_type="neural_network", hidden_layer_sizes=(100, 50))
```

### Interpretation

```python
from financial_shock_detector import ClusterExplainer

# Initialize explainer
explainer = ClusterExplainer(model=xgb.model, feature_names=["feat1", "feat2"])

# Create SHAP explainer
explainer.create_shap_explainer(X_background, explainer_type="tree")

# Get SHAP values
shap_values = explainer.get_shap_values(X_test)

# Plot SHAP summary
explainer.plot_shap_summary(X_test, save_path="shap_summary.png")

# Analyze clusters
stats = explainer.analyze_cluster_characteristics(X, cluster_labels)
print(stats)

# Compare clusters
diff = explainer.explain_cluster_differences(X, cluster_labels, cluster_a=0, cluster_b=1)
print(diff.head())
```

## Configuration

### YAML Configuration File

Create `config.yaml`:

```yaml
data_collection:
  user_agent: "Mozilla/5.0"
  delay: 1.0

nlp_processing:
  model_name: "yiyanghkust/finbert-tone"
  max_length: 512
  batch_size: 8
  pooling: "cls"

dimensionality_reduction:
  variance_threshold: 0.95
  standardize: true

clustering:
  method: "gmm"
  n_components: 3

classification:
  model_type: "xgboost"
  test_size: 0.2

output:
  save_models: true
  output_dir: "output"
```

Use configuration:

```python
from financial_shock_detector import FinancialShockPipeline
from financial_shock_detector.utils import load_config

config = load_config("config.yaml")
pipeline = FinancialShockPipeline(config=config)
```

## CLI Usage

### Run Pipeline on Data

```bash
financial-shock-detector run \
    --input data.csv \
    --text-column text \
    --label-column label \
    --output results \
    --config config.yaml
```

### Run Demo

```bash
financial-shock-detector demo \
    --n-samples 200 \
    --output demo_results
```

### Make Predictions

```bash
financial-shock-detector predict \
    --model-dir trained_model \
    --input new_data.csv \
    --text-column text \
    --output predictions.csv
```

## Advanced Topics

### Working with Large Datasets

For datasets too large to fit in memory:

```python
# Process in batches
batch_size = 1000
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = processor.get_embeddings(batch)
    all_embeddings.append(embeddings)

embeddings = np.vstack(all_embeddings)
```

### GPU Acceleration

FinBERT automatically uses GPU if available:

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Force CPU usage
processor = FinBERTProcessor(device="cpu")
```

### Custom FinBERT Models

Use different FinBERT variants:

```python
# Different FinBERT models
models = [
    "yiyanghkust/finbert-tone",          # Sentiment
    "ProsusAI/finbert",                  # General financial
    "yiyanghkust/finbert-pretrain",      # Pre-trained base
]

processor = FinBERTProcessor(model_name=models[0])
```

### Ensemble Predictions

Combine multiple classifiers:

```python
from sklearn.ensemble import VotingClassifier

# Train individual models
xgb = ShockEventClassifier("xgboost")
rf = ShockEventClassifier("random_forest")
lr = ShockEventClassifier("logistic")

xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Ensemble predictions
pred_xgb = xgb.predict(X_test)
pred_rf = rf.predict(X_test)
pred_lr = lr.predict(X_test)

# Majority voting
from scipy.stats import mode
ensemble_pred = mode([pred_xgb, pred_rf, pred_lr], axis=0)[0]
```

### Saving and Loading Pipeline

```python
# Save complete pipeline
pipeline.save_pipeline("saved_model")

# Load pipeline
from financial_shock_detector import FinancialShockPipeline
loaded_pipeline = FinancialShockPipeline.load_pipeline("saved_model")

# Use loaded pipeline
predictions = loaded_pipeline.predict(new_embeddings)
```

### Real-time Processing

Set up a simple real-time processing loop:

```python
import time

# Load pre-trained pipeline
pipeline = FinancialShockPipeline.load_pipeline("trained_model")

def process_new_article(text):
    """Process a single article."""
    # Generate embedding
    embedding = pipeline.nlp_processor.get_embeddings([text])
    
    # Reduce dimensionality
    reduced = pipeline.pca_reducer.transform(embedding)
    
    # Predict
    prediction = pipeline.classifier.predict(reduced)[0]
    probability = pipeline.classifier.predict_proba(reduced)[0]
    
    return {
        "prediction": "SHOCK" if prediction == 1 else "NORMAL",
        "probability": probability[1],
        "timestamp": time.time()
    }

# Process new articles as they arrive
new_article = "Stock market crashes by 20% in worst day since 2008"
result = process_new_article(new_article)
print(result)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Process data in chunks
   - Use CPU instead of GPU

2. **Slow Processing**
   - Use GPU if available
   - Increase batch size
   - Reduce max_length for tokenization

3. **Poor Performance**
   - Increase training data
   - Tune hyperparameters
   - Try different models
   - Check data quality

4. **Import Errors**
   - Ensure all dependencies installed
   - Check Python version (3.8+)
   - Reinstall package

## Next Steps

- Explore the [Architecture Guide](ARCHITECTURE.md)
- Check out [Example Scripts](examples/)
- Read the [API Documentation](docs/)
- Join the community discussions
