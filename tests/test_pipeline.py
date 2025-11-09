"""Basic tests for the pipeline."""

import pytest
import numpy as np
from financial_shock_detector.utils import create_sample_financial_data


def test_create_sample_data():
    """Test sample data creation."""
    data = create_sample_financial_data(n_samples=10)

    assert len(data) == 10
    assert "text" in data.columns
    assert "label" in data.columns
    assert data["label"].isin([0, 1]).all()


def test_sample_data_content():
    """Test sample data has proper content."""
    data = create_sample_financial_data(n_samples=5)

    # Check all texts are non-empty strings
    assert all(isinstance(text, str) and len(text) > 0 for text in data["text"])

    # Check labels are binary
    assert set(data["label"].unique()).issubset({0, 1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
