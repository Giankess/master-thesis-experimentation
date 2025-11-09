# Financial Shock Detector

A comprehensive Python package for detecting financial shock events using Natural Language Processing (NLP) and machine learning.

## Features

- **Data Collection**: Web scraping with BeautifulSoup and API integration
- **NLP Processing**: Text embeddings using FinBERT (financial domain BERT model)
- **Dimensionality Reduction**: PCA for efficient feature representation
- **Clustering**: Unsupervised learning with Gaussian Mixture Models (GMM) and DBSCAN
- **Classification**: Multiple classifiers including Logistic Regression, Random Forest, Neural Networks, and XGBoost
- **Explainability**: Cluster interpretation using SHAP and other XAI techniques
- **End-to-End Pipeline**: Seamless workflow from data collection to prediction

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/Giankess/master-thesis.git
cd master-thesis
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Using the CLI

Run demo with sample data:

```bash
financial-shock-detector demo --n-samples 100 --output demo_output
```

Run pipeline on your own data:

```bash
financial-shock-detector run --input data.csv --text-column text --label-column label --output results
```

Make predictions with a trained model:

```bash
financial-shock-detector predict --model-dir results --input new_data.csv --output predictions.csv
```

### Using Python API

#### Basic Usage

```python
from financial_shock_detector import FinancialShockPipeline
from financial_shock_detector.utils import create_sample_financial_data

# Create sample data
data = create_sample_financial_data(n_samples=100)
texts = data["text"].tolist()
labels = data["label"].values

# Initialize and run pipeline
pipeline = FinancialShockPipeline()
results = pipeline.run_full_pipeline(
    texts=texts,
    labels=labels,
    output_dir="output"
)

# Access results
print(f"Embeddings shape: {results['embeddings'].shape}")
print(f"Test metrics: {results['metrics']}")
```

#### Advanced Usage

```python
from financial_shock_detector import (
    FinBERTProcessor,
    PCAReducer,
    ClusterEngine,
    ShockEventClassifier,
)

# Custom configuration
config = {
    "nlp_processing": {
        "model_name": "yiyanghkust/finbert-tone",
        "batch_size": 8,
        "pooling": "cls",
    },
    "clustering": {
        "method": "gmm",
        "n_components": 3,
    },
    "classification": {
        "model_type": "xgboost",
    },
}

pipeline = FinancialShockPipeline(config=config)

# Run individual steps
embeddings = pipeline.process_texts(texts)
reduced = pipeline.reduce_dimensionality(embeddings)
clusters = pipeline.cluster_data(reduced)
metrics = pipeline.train_classifier(reduced, labels)
```

## Components

### 1. Data Collection

Collect financial news and data from various sources:

```python
from financial_shock_detector import FinancialDataScraper, FinancialAPIClient

# Web scraping
scraper = FinancialDataScraper()
data = scraper.scrape_to_dataframe(urls, delay=1.0)

# API client
api_client = FinancialAPIClient(api_key="your_key")
articles = api_client.fetch_news("financial crisis", "2020-01-01", "2024-12-31")
```

### 2. NLP Processing with FinBERT

Generate embeddings for financial text:

```python
from financial_shock_detector import FinBERTProcessor

processor = FinBERTProcessor(model_name="yiyanghkust/finbert-tone")
embeddings = processor.get_embeddings(texts, batch_size=8, pooling="cls")
```

### 3. Dimensionality Reduction

Reduce embedding dimensions while preserving variance:

```python
from financial_shock_detector import PCAReducer

pca = PCAReducer(variance_threshold=0.95)
reduced = pca.fit_transform(embeddings)
pca.plot_variance_explained(save_path="variance.png")
```

### 4. Clustering

Discover patterns in financial text:

```python
from financial_shock_detector import ClusterEngine

# Gaussian Mixture Model
gmm = ClusterEngine(method="gmm", n_components=3)
labels = gmm.fit_predict(reduced)

# DBSCAN
dbscan = ClusterEngine(method="dbscan", eps=0.5, min_samples=5)
labels = dbscan.fit_predict(reduced)

# Evaluate clustering
metrics = gmm.evaluate(reduced)
print(metrics)
```

### 5. Classification

Train models to detect shock events:

```python
from financial_shock_detector import ShockEventClassifier

# XGBoost (recommended)
classifier = ShockEventClassifier(model_type="xgboost", n_estimators=100)
train_metrics, test_metrics = classifier.train_and_validate(X, y, test_size=0.2)

# Other models
rf = ShockEventClassifier(model_type="random_forest")
lr = ShockEventClassifier(model_type="logistic")
nn = ShockEventClassifier(model_type="neural_network")
```

### 6. Explainability

Interpret clusters and predictions:

```python
from financial_shock_detector import ClusterExplainer

explainer = ClusterExplainer(model=classifier.model)
explainer.create_shap_explainer(X_background, explainer_type="tree")
explainer.plot_shap_summary(X)

# Analyze cluster characteristics
stats = explainer.analyze_cluster_characteristics(X, cluster_labels)
```

## Configuration

Create a `config.yaml` file:

```yaml
data_collection:
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
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
  covariance_type: "full"

classification:
  model_type: "xgboost"
  test_size: 0.2
  random_state: 42

output:
  save_models: true
  output_dir: "output"
  save_plots: true
```

## Examples

See the `examples/` directory for detailed examples:

- `basic_usage.py`: Simple end-to-end example
- `advanced_usage.py`: Custom configuration and individual component usage

Run examples:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## Project Structure

```
financial_shock_detector/
├── __init__.py
├── pipeline.py                 # Main pipeline orchestrator
├── cli.py                      # Command-line interface
├── data_collection/
│   ├── scraper.py             # Web scraping with BeautifulSoup
│   └── api_client.py          # API integration
├── nlp_processing/
│   └── finbert_processor.py   # FinBERT embeddings
├── dimensionality_reduction/
│   └── pca_reducer.py         # PCA implementation
├── clustering/
│   └── cluster_engine.py      # GMM and DBSCAN
├── classification/
│   └── classifier.py          # Multi-model classifier
├── interpretation/
│   └── explainer.py           # XAI with SHAP
└── utils/
    ├── config.py              # Configuration management
    └── data_utils.py          # Data utilities
```

## Pipeline Architecture

```
┌─────────────────┐
│  Data Sources   │
│  (API, Web)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Collection│
│  (BeautifulSoup)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization & │
│  Embeddings     │
│  (FinBERT)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dimensionality │
│  Reduction (PCA)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Clustering     │
│  (GMM/DBSCAN)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classification │
│  (LR/RF/NN/XGB) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Interpretation │
│  (SHAP/XAI)     │
└─────────────────┘
```

## Requirements

See `requirements.txt` for full dependencies:

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- torch >= 1.10.0
- transformers >= 4.20.0
- beautifulsoup4 >= 4.10.0
- xgboost >= 1.5.0
- shap >= 0.41.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black financial_shock_detector/
flake8 financial_shock_detector/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Citation

If you use this package in your research, please cite:

```bibtex
@software{financial_shock_detector,
  title={Financial Shock Detector: NLP-based Detection of Financial Shock Events},
  author={Master Thesis Project},
  year={2024},
  url={https://github.com/Giankess/master-thesis}
}
```

## Acknowledgments

- FinBERT model from [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Built with scikit-learn, PyTorch, and Transformers

## Contact

For questions or issues, please open an issue on GitHub.