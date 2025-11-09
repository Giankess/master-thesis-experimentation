"""Financial Shock Detector - A package for detecting financial shock events using NLP."""

__version__ = "0.1.0"
__author__ = "Master Thesis Project"

from .data_collection.scraper import FinancialDataScraper
from .nlp_processing.finbert_processor import FinBERTProcessor
from .dimensionality_reduction.pca_reducer import PCAReducer
from .clustering.cluster_engine import ClusterEngine
from .classification.classifier import ShockEventClassifier
from .interpretation.explainer import ClusterExplainer
from .pipeline import FinancialShockPipeline

__all__ = [
    "FinancialDataScraper",
    "FinBERTProcessor",
    "PCAReducer",
    "ClusterEngine",
    "ShockEventClassifier",
    "ClusterExplainer",
    "FinancialShockPipeline",
]
