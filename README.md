# Kaggle Competition - Data Mining 252
This repository contains the complete implementation and machine learning pipelines designed for **Data Mining 252 Kaggle Competition**. 

The system leverages a multi-modal feature engineering architecture combining structural tabular data extraction, text representations (TF-IDF and Deep Dense Embeddings), graph neural embeddings for author networks, and an optimized ensemble blending layer evaluated across stratified cross-validation splits.

---

## 📌 Project Overview & Structure
The primary goal is to perform predictive analytics on scientific publications by parsing data attributes such as titles, authors, venues, publishing years, and DOIs. The pipeline processes three main data variants:
* `train.csv` — Full training set containing the objective regression/classification `Label`.
* `public_test.csv` — Public evaluation partition used during leadboard benchmarking.
* `private_test.csv` — Private evaluation partition mapping out the final course assessment scores.

### Key Repository Subdirectories
```bash
├── artifacts/             # Cached model embeddings, graph checkpoints, and experimental logs
├── outputs/               # Final submission predictions formatted for Kaggle upload
└── README.md              # Repository documentation and structural guide
```

---

## 🚀 Architectural Pipeline & Methodology

### 1. Robust Data Cleaning & Preprocessing
* **Noise Mitigation (`clean_noise`):** Deduplicates observations by grouping identical lowercased paper titles. Rows with label variance are resolved by computing the rounded aggregate mean of the structural target.
* **Text & Field Standardisation:** Isolates domain semantics by standardizing spaces and casing across titles and venue strings. Missing values are filled with a structured special token (`<missing>`).
* **DOI Structural Component Parsing:** Breaks DOIs or URLs down into core segments including top-level `doi_domain`, `doi_prefix`, string lengths, and flags indicating if the input string is a formatted HTTP web URL.

### 2. High-Dimensional Feature Engineering
* **Graph Topology Node Embeddings:** Constructs localized co-author networks and structural document-author bipartite graphs using `networkx`. Generates dense structural topology representations via `node2vec` walks.
* **Target & Frequency Encodings:** Fits a safe out-of-fold `TargetEncoder` leveraging `StratifiedKFold` to prevent data leakage. Computes structural frequency representations for categorical indicators.
* **Deep Text Space Mining:** Captures rich sentence semantics by extracting high-dimensional embeddings using pre-trained scientific transformers (`allenai/specter2`). Dimensionality is effectively condensed via Principal Component Analysis (PCA).

### 3. Diverse Predictive Modelling Block
The pipeline builds an asynchronous ensemble by capturing interactions from three structurally diverse layers:
1. **Gradient Boosting Decision Trees (GBDT):** Trains sequential leaf-wise tree architectures using `LightGBM` (or `XGBoost`) over structural aggregations and compressed embeddings.
2. **Text N-Gram Ridge Regressor:** Extracts character-level and word-level N-gram sparse matrices using robust TF-IDF Vectorizers mapped into linear ridge penalties.
3. **Deep Language Sequence Classifier:** Finetunes specialized sequence classification transformers (`allenai/scibert_scivocab_uncased`) directly for multi-fold regression targets.

### 4. Cross-Validation & Optimized Blending
* Uses an exact **5-Fold Stratified Cross-Validation Strategy** across all independent models.
* Optimizes out-of-fold root-mean-squared-error (`rmse`) metrics to compute a weighted vector blend (`optimize_blend_weights`).
* Map continuous metrics back to exact categorical classifications using Nelder-Mead threshold boundary optimization routines (`optimize_thresholds`), evaluated by the **Quadratic Weighted Kappa (QWK)** metric.

---

## 🛠️ Operational Setup & Execution

### Prerequisites & Dependencies
Ensure your local execution environment is equipped with a Python 3.12+ interpreter. Install the complete baseline stack via pip:

```bash
pip install lightgbm xgboost node2vec scikit-learn pandas numpy networkx sentence-transformers transformers scipy torch
```

### Pipeline Deployment
To run the end-to-end extraction, model optimization, validation, and prediction generation flow:

1. Place `train.csv`, `public_test.csv`, and `private_test.csv` in the root repository path.
2. Execute the Jupyter notebook pipeline cells sequentially.
3. The training log stream will track fold processing iteratively across GBDT, TF-IDF, and Transformer elements.
4. Retrieve the finalized prediction results from `outputs/submission.csv`.

---

## 📑 Assessment & Milestones
* **Phase 1: Leaderboard Benchmarking (Deadline: 30/05/2026)** — Model deployment, submission validation, and public leaderboard iterations.
* **Phase 2: Private Submission Lock-in (Deadline: 03/06/2026)** — Selection of 3 targeted submission iterations optimized against over-fitting.
* **Phase 3: Final Academic Submission (Deadline: 07/06/2026)** — Final upload to LMS encompassing presentation videos, leaderboard verification screenshots, and a comprehensive project documentation report.
