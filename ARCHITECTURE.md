# Financial Shock Detector - Architecture

## Overview

The Financial Shock Detector is a modular Python package designed to detect financial shock events using Natural Language Processing (NLP) and Machine Learning techniques. The architecture follows a pipeline pattern with six main stages.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Financial Shock Detector                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│     CLI      │    │  Python API  │    │   Pipeline   │
└──────────────┘    └──────────────┘    └──────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
        ┌───────────▼───────────┐       │
        │   Data Collection     │       │
        │   - Scraper           │       │
        │   - API Client        │       │
        └───────────┬───────────┘       │
                    │                   │
        ┌───────────▼───────────┐       │
        │   NLP Processing      │       │
        │   - FinBERT           │       │
        │   - Tokenization      │       │
        │   - Embeddings        │       │
        └───────────┬───────────┘       │
                    │                   │
        ┌───────────▼───────────┐       │
        │   Dimensionality      │       │
        │   Reduction           │       │
        │   - PCA               │       │
        └───────────┬───────────┘       │
                    │                   │
        ┌───────────▼───────────┐       │
        │   Clustering          │       │
        │   - GMM               │       │
        │   - DBSCAN            │       │
        └───────────┬───────────┘       │
                    │                   │
        ┌───────────▼───────────┐       │
        │   Classification      │       │
        │   - Logistic Regr.    │       │
        │   - Random Forest     │       │
        │   - Neural Network    │       │
        │   - XGBoost           │       │
        └───────────┬───────────┘       │
                    │                   │
        ┌───────────▼───────────┐       │
        │   Interpretation      │       │
        │   - SHAP              │       │
        │   - XAI               │       │
        └───────────────────────┘       │
                                        │
                    ┌───────────────────▼────┐
                    │   Utils & Config       │
                    └────────────────────────┘
```

## Module Descriptions

### 1. Data Collection Module

**Location**: `financial_shock_detector/data_collection/`

**Components**:
- `scraper.py`: Web scraping using BeautifulSoup
- `api_client.py`: API integration for financial data sources

**Purpose**: Collect financial news articles and data from various sources including web pages and APIs.

**Key Features**:
- BeautifulSoup-based web scraping
- Rate-limited requests to avoid blocking
- Metadata extraction (title, date, author)
- Support for multiple data sources

### 2. NLP Processing Module

**Location**: `financial_shock_detector/nlp_processing/`

**Components**:
- `finbert_processor.py`: FinBERT-based text processing

**Purpose**: Transform raw text into numerical embeddings suitable for machine learning.

**Key Features**:
- FinBERT tokenization (financial domain-specific BERT)
- Multiple pooling strategies (CLS, mean, max)
- Batch processing for efficiency
- GPU acceleration support

**Technical Details**:
- Model: `yiyanghkust/finbert-tone` (default)
- Embedding dimension: 768 (BERT-base)
- Max sequence length: 512 tokens

### 3. Dimensionality Reduction Module

**Location**: `financial_shock_detector/dimensionality_reduction/`

**Components**:
- `pca_reducer.py`: PCA implementation

**Purpose**: Reduce high-dimensional embeddings to a manageable size while preserving variance.

**Key Features**:
- Automatic component selection based on variance threshold
- Standardization option
- Visualization of explained variance
- Save/load functionality

**Technical Details**:
- Default variance threshold: 95%
- Standardization: Z-score normalization
- Incremental PCA support for large datasets

### 4. Clustering Module

**Location**: `financial_shock_detector/clustering/`

**Components**:
- `cluster_engine.py`: GMM and DBSCAN implementations

**Purpose**: Discover patterns and groupings in financial news without labels.

**Key Features**:
- Gaussian Mixture Models (GMM)
- DBSCAN for density-based clustering
- Automatic optimal cluster selection
- Multiple evaluation metrics

**Technical Details**:
- GMM: Supports full, tied, diag, and spherical covariance
- DBSCAN: Configurable epsilon and min_samples
- Evaluation metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz

### 5. Classification Module

**Location**: `financial_shock_detector/classification/`

**Components**:
- `classifier.py`: Multi-model classifier

**Purpose**: Train supervised models to identify financial shock events.

**Supported Models**:
1. **Logistic Regression**: Linear baseline model
2. **Random Forest**: Ensemble decision trees
3. **Neural Network**: MLP with configurable layers
4. **XGBoost**: Gradient boosting (recommended)

**Key Features**:
- Automatic train/test splitting
- Cross-validation support
- Feature importance extraction
- Confusion matrix visualization

### 6. Interpretation Module

**Location**: `financial_shock_detector/interpretation/`

**Components**:
- `explainer.py`: XAI using SHAP

**Purpose**: Explain model predictions and cluster characteristics.

**Key Features**:
- SHAP value computation
- Feature importance ranking
- Cluster characteristic analysis
- Waterfall and summary plots

**Technical Details**:
- Supports TreeExplainer, LinearExplainer, KernelExplainer
- Background data sampling for efficiency
- Integration with classification module

### 7. Pipeline Module

**Location**: `financial_shock_detector/pipeline.py`

**Purpose**: Orchestrate all components in an end-to-end workflow.

**Key Features**:
- Sequential execution of all stages
- Configuration-driven behavior
- Model persistence and loading
- Progress tracking and visualization

**Workflow**:
1. Load and preprocess text data
2. Generate embeddings using FinBERT
3. Reduce dimensionality with PCA
4. Cluster data to find patterns
5. Train classifier on labeled data
6. Interpret results with XAI

### 8. Utilities Module

**Location**: `financial_shock_detector/utils/`

**Components**:
- `config.py`: Configuration management
- `data_utils.py`: Data loading/saving utilities

**Purpose**: Provide common utilities and helper functions.

**Key Features**:
- YAML/JSON configuration support
- Data serialization (CSV, pickle, numpy)
- Text preprocessing
- Sample data generation

## Data Flow

```
Input Text
    │
    ▼
[Tokenization]
    │
    ▼
Token IDs
    │
    ▼
[FinBERT Encoding]
    │
    ▼
Embeddings (768-dim)
    │
    ▼
[PCA Reduction]
    │
    ▼
Reduced Embeddings (n-dim)
    │
    ├──────────────────┐
    │                  │
    ▼                  ▼
[Clustering]    [Classification]
    │                  │
    ▼                  ▼
Cluster Labels    Predictions
    │                  │
    └──────────────────┤
                       ▼
              [Interpretation]
                       │
                       ▼
            Explanations & Insights
```

## Configuration System

The package uses a hierarchical configuration system:

1. **Default Config**: Hard-coded in `utils/config.py`
2. **File Config**: YAML/JSON files in `configs/`
3. **Runtime Config**: Passed directly to components

Configuration precedence: Runtime > File > Default

## Design Patterns

### 1. Pipeline Pattern
The main `FinancialShockPipeline` class orchestrates all components in sequence.

### 2. Strategy Pattern
Multiple algorithms (GMM/DBSCAN, LR/RF/NN/XGB) can be swapped via configuration.

### 3. Factory Pattern
Components are instantiated based on configuration parameters.

### 4. Lazy Loading
Modules are imported only when needed to reduce startup time.

## Extension Points

### Adding New Data Sources
Extend `FinancialDataScraper` or `FinancialAPIClient` with new methods.

### Adding New Clustering Algorithms
Extend `ClusterEngine` with new clustering methods.

### Adding New Classifiers
Add new model types to `ShockEventClassifier._init_model()`.

### Adding New NLP Models
Replace FinBERT with other transformer models in `FinBERTProcessor`.

## Performance Considerations

### Memory
- Batch processing for large datasets
- Incremental PCA for huge datasets
- Clear intermediate results when not needed

### Speed
- GPU acceleration for FinBERT
- Vectorized operations with NumPy
- Parallel processing where possible

### Storage
- Efficient model serialization with joblib
- Compressed numpy arrays for large embeddings
- Incremental saving of results

## Testing Strategy

### Unit Tests
- Individual component functionality
- Edge cases and error handling

### Integration Tests
- Component interactions
- Pipeline execution

### Performance Tests
- Speed benchmarks
- Memory profiling

## Deployment

### As a Package
```bash
pip install -e .
```

### As a CLI Tool
```bash
financial-shock-detector run --input data.csv
```

### As a Library
```python
from financial_shock_detector import FinancialShockPipeline
pipeline = FinancialShockPipeline()
```

## Future Enhancements

1. **Real-time Processing**: Stream processing for live news feeds
2. **Multi-language Support**: Extend beyond English
3. **Advanced Models**: GPT-based embeddings, LLMs
4. **Web Interface**: Dashboard for visualization
5. **API Service**: REST API for predictions
6. **Database Integration**: Store and query historical data
7. **Ensemble Methods**: Combine multiple models
8. **Active Learning**: Interactive labeling interface
