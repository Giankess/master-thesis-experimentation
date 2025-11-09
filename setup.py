"""Setup configuration for financial_shock_detector package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="financial_shock_detector",
    version="0.1.0",
    author="Master Thesis Project",
    description="A package for detecting financial shock events using NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "beautifulsoup4>=4.10.0",
        "requests>=2.27.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "xgboost>=1.5.0",
        "shap>=0.41.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial-shock-detector=financial_shock_detector.cli:main",
        ],
    },
)
