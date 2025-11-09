"""Financial Shock Detector - A package for detecting financial shock events using NLP."""

__version__ = "0.1.0"
__author__ = "Master Thesis Project"

# Lazy imports to avoid dependency issues
def __getattr__(name):
    """Lazy loading of modules."""
    if name == "FinancialDataScraper":
        from .data_collection.scraper import FinancialDataScraper
        return FinancialDataScraper
    elif name == "FinancialAPIClient":
        from .data_collection.api_client import FinancialAPIClient
        return FinancialAPIClient
    elif name == "FinBERTProcessor":
        from .nlp_processing.finbert_processor import FinBERTProcessor
        return FinBERTProcessor
    elif name == "PCAReducer":
        from .dimensionality_reduction.pca_reducer import PCAReducer
        return PCAReducer
    elif name == "ClusterEngine":
        from .clustering.cluster_engine import ClusterEngine
        return ClusterEngine
    elif name == "ShockEventClassifier":
        from .classification.classifier import ShockEventClassifier
        return ShockEventClassifier
    elif name == "ClusterExplainer":
        from .interpretation.explainer import ClusterExplainer
        return ClusterExplainer
    elif name == "FinancialShockPipeline":
        from .pipeline import FinancialShockPipeline
        return FinancialShockPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FinancialDataScraper",
    "FinancialAPIClient",
    "FinBERTProcessor",
    "PCAReducer",
    "ClusterEngine",
    "ShockEventClassifier",
    "ClusterExplainer",
    "FinancialShockPipeline",
]
