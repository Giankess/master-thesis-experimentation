"""Data utilities for loading, saving, and preprocessing."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Union, Any
import re


def load_data(filepath: str) -> Union[pd.DataFrame, np.ndarray, Any]:
    """
    Load data from file (CSV, pickle, or numpy).

    Args:
        filepath: Path to data file

    Returns:
        Loaded data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix in [".pkl", ".pickle"]:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif filepath.suffix in [".npy"]:
        return np.load(filepath)
    elif filepath.suffix == ".npz":
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_data(data: Any, filepath: str) -> None:
    """
    Save data to file (CSV, pickle, or numpy).

    Args:
        data: Data to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == ".csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    elif filepath.suffix in [".pkl", ".pickle"]:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    elif filepath.suffix == ".npy":
        np.save(filepath, data)
    elif filepath.suffix == ".npz":
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            np.savez(filepath, data=data)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def preprocess_text(text: str, lowercase: bool = True) -> str:
    """
    Preprocess text for NLP.

    Args:
        text: Input text
        lowercase: Whether to convert to lowercase

    Returns:
        Preprocessed text
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    # Lowercase
    if lowercase:
        text = text.lower()

    return text.strip()


def split_train_test(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets.

    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_sample_financial_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample financial news data for testing.

    Args:
        n_samples: Number of samples to create

    Returns:
        DataFrame with sample financial news
    """
    np.random.seed(42)

    # Sample financial news headlines
    templates = [
        "Stock market crashes by {pct}% due to {event}",
        "Federal Reserve announces {action} to combat {issue}",
        "Major bank reports {result} earnings amid {condition}",
        "Economic indicators show {trend} in {sector}",
        "Investors react to {event} with {reaction}",
    ]

    events = ["inflation concerns", "rate hikes", "geopolitical tensions", "market volatility"]
    actions = ["rate increase", "bond purchases", "policy changes", "stimulus measures"]
    results = ["strong", "weak", "mixed", "disappointing"]
    conditions = ["uncertainty", "growth", "recovery", "recession fears"]
    trends = ["improvement", "decline", "stability", "growth"]
    sectors = ["technology", "finance", "energy", "manufacturing"]
    reactions = ["selling", "buying", "caution", "optimism"]

    data = []
    for i in range(n_samples):
        template = np.random.choice(templates)
        text = template.format(
            pct=np.random.randint(1, 10),
            event=np.random.choice(events),
            action=np.random.choice(actions),
            result=np.random.choice(results),
            condition=np.random.choice(conditions),
            trend=np.random.choice(trends),
            sector=np.random.choice(sectors),
            reaction=np.random.choice(reactions),
            issue=np.random.choice(events),
        )

        # Assign labels (0: normal, 1: shock event)
        label = 1 if any(word in text.lower() for word in ["crash", "crisis", "recession"]) else 0

        data.append({"text": text, "label": label})

    return pd.DataFrame(data)
