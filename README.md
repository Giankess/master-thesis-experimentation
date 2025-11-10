# Master Thesis — Monetary Policy Text Pipeline

Short overview
- Repository contains notebooks and scripts to ingest central-bank communications (ECB, FED, Bank of England, Bank of Japan), produce financial text embeddings (FinBERT, AS-BERT), reduce dimensionality, cluster, interpret clusters, and train downstream identifiers / clustifiers.

Notebooks
- Data Pipeline.ipynb — main pipeline (ingest, embed, PCA, GMM, DBSCAN, train identifiers).
- playground.ipynb — quick experiments (FinBERT embedding examples).
- (Add new notebooks under /workspaces/master-thesis/notebooks as needed.)

Pipeline steps
1. Getting data
   - Sources: official RSS / APIs and direct scraping (BeautifulSoup/newspaper) for ECB, FED, BOE, BOJ.
2. Tokenization & embeddings
   - Word/token → numeric vectors using transformer models (FIN-BERT, AS-BERT).
3. Dimensionality reduction
   - Apply PCA when many embedding vectors to reduce size.
4. Clustering (unsupervised)
   - GMM (recommended), optional DBSCAN; evaluate via silhouette, AIC/BIC, etc.
5. Cluster interpretation (xAI)
   - Use SHAP/LIME + representative tokens/phrases per cluster.
6. Train identifier
   - Train classifiers on clustered data: Logistic Regression, Random Forest, XGBoost/LightGBM, simple neural net.
7. Clustifier training & validation
   - Train models to predict cluster membership; use cross-validation and hold-out sets.

Quick setup (dev container: Ubuntu 24.04.2 LTS)
1. Create & activate Python env (recommended)
   - python -m venv .venv && source .venv/bin/activate
2. Install dependencies
   - pip install -r requirements.txt
   - If requirements.txt missing, install basics:
     - pip install jupyterlab transformers torch scikit-learn pandas beautifulsoup4 newspaper3k faiss-cpu shap lime xgboost joblib
3. Start Jupyter Lab
   - jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   - In the container, open the token URL on host using:
     - "$BROWSER" <token-url>

GPU note
- If CUDA is available, transformers and torch will use it automatically when device detection is enabled in the notebooks.

Files & artifacts
- Save fitted models/artifacts with joblib or torch.save in /workspaces/master-thesis/artifacts.
- Large embeddings and FAISS index recommended to store under /workspaces/master-thesis/data.

Development tips
- Use the playground notebook for rapid iteration.
- Limit max tokens / chunk long documents before embedding to avoid truncation.
- Tune number of PCA components and GMM clusters; persist PCA & GMM (joblib.dump) for reproducibility.

License & citation
- Add your institution/license details here.

If you want, I can create a new notebook scaffold for the full pipeline next.// filepath: /workspaces/master-thesis/README.md

# Master Thesis — Monetary Policy Text Pipeline

Short overview
- Repository contains notebooks and scripts to ingest central-bank communications (ECB, FED, Bank of England, Bank of Japan), produce financial text embeddings (FinBERT, AS-BERT), reduce dimensionality, cluster, interpret clusters, and train downstream identifiers / clustifiers.

Notebooks
- Data Pipeline.ipynb — main pipeline (ingest, embed, PCA, GMM, DBSCAN, train identifiers).
- playground.ipynb — quick experiments (FinBERT embedding examples).
- (Add new notebooks under /workspaces/master-thesis/notebooks as needed.)

Pipeline steps
1. Getting data
   - Sources: official RSS / APIs and direct scraping (BeautifulSoup/newspaper) for ECB, FED, BOE, BOJ.
2. Tokenization & embeddings
   - Word/token → numeric vectors using transformer models (FIN-BERT, AS-BERT).
3. Dimensionality reduction
   - Apply PCA when many embedding vectors to reduce size.
4. Clustering (unsupervised)
   - GMM (recommended), optional DBSCAN; evaluate via silhouette, AIC/BIC, etc.
5. Cluster interpretation (xAI)
   - Use SHAP/LIME + representative tokens/phrases per cluster.
6. Train identifier
   - Train classifiers on clustered data: Logistic Regression, Random Forest, XGBoost/LightGBM, simple neural net.
7. Clustifier training & validation
   - Train models to predict cluster membership; use cross-validation and hold-out sets.

Quick setup (dev container: Ubuntu 24.04.2 LTS)
1. Create & activate Python env (recommended)
   - python -m venv .venv && source .venv/bin/activate
2. Install dependencies
   - pip install -r requirements.txt
   - If requirements.txt missing, install basics:
     - pip install jupyterlab transformers torch scikit-learn pandas beautifulsoup4 newspaper3k faiss-cpu shap lime xgboost joblib
3. Start Jupyter Lab
   - jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   - In the container, open the token URL on host using:
     - "$BROWSER" <token-url>

GPU note
- If CUDA is available, transformers and torch will use it automatically when device detection is enabled in the notebooks.

Files & artifacts
- Save fitted models/artifacts with joblib or torch.save in /workspaces/master-thesis/artifacts.
- Large embeddings and FAISS index recommended to store under /workspaces/master-thesis/data.

Development tips
- Use the playground notebook for rapid iteration.
- Limit max tokens / chunk long documents before embedding to avoid truncation.
- Tune number of PCA components and GMM clusters; persist PCA & GMM (joblib.dump) for reproducibility.

License & citation
- Add your institution/license details here.

If you want, I can create a new notebook scaffold for the full pipeline next.