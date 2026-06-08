# PT1Diagnosis: Graph Neural Network Research for pT1 Colorectal Cancer

## Project Overview

This repository contains the primary experimental framework and research modules developed for my Bachelor's Thesis (TFG). The focus is the application of Graph Neural Networks (GNNs) to model the spatial microenvironment of pT1 Colorectal Cancer for metastasis prediction. The research aims to evaluate the diagnostic potential of graph-based representations of tissue structure.

---

## Technical Architecture

The core research logic is contained within the `src/` directory, organized into modular components for data processing, graph construction, and model training.

### 1. Data Processing and Loading
*   `src/dataset/load_cls.py`: Integrates deep learning embeddings (CLS tokens) with clinical metadata, managing patient-level diagnostic mapping (N0 vs. N1).
*   `src/dataset/graph_loaders.py`: Implements a custom PyTorch Geometric dataset that handles pre-generated graph objects and dynamic feature linking.
*   `src/utils/class_weights.py`: Addresses class imbalance by calculating loss function weights based on patient distribution.

### 2. Graph Construction (`src/graphs/`)
A suite of graph generation modules optimized for histopathological spatial modeling:
*   Connectivity: Support for Euclidean k-NN, Cosine-Similarity k-NN, Radius-based, and Fully Connected graphs.
*   Optimization: GPU-accelerated versions for large-scale datasets and CPU-parallel alternatives for general compatibility.
*   Analysis: Tools in `src/graphs/knn/distance_study/` for analyzing feature distributions to optimize graph hyperparameters.

### 3. Model Architectures (`src/models/`)
*   `models_Graph.py`: Contains various GNN architectures:
    *   GATWeight_batch: Graph Attention Network that utilizes edge similarities as attributes to weight message passing.
    *   GCNWithAgg: Graph Convolutional Network with global mean pooling for patient-level representation.
    *   Hierarchical Pooling: Implementation of TopKPooling layers to capture multi-scale tissue structures.

### 4. Experimental Pipeline
*   `src/main.py`: The central execution script for running multi-experiment GNN benchmarks.
*   `src/training/cross_validation.py`: Implements Stratified 10-Fold Cross-Validation with full MLflow integration.
*   Training Loops: Modular implementations of training and validation logic in `src/training/graph_loops.py` with support for Mixed Precision (AMP).

### 5. Advanced Visualization (`src/visualization/`)
*   `src/visualization/t_SNE.py`: Generates stratified t-SNE projections to visualize feature space clustering across tumor regions and patient categories.

---

## Project Structure

```text
PT1Diagnosis/
├── data/               # Raw and processed datasets (ignored by git)
├── docs/               # Project documentation and research papers
├── results/            # Output plots, metrics, and MLflow runs
├── src/                # Source code
│   ├── analysis/       # Post-hoc analysis scripts
│   ├── dataset/        # Data loading and preprocessing
│   ├── graphs/         # Graph construction pipelines
│   ├── models/         # GNN model definitions
│   ├── training/       # Training loops and cross-validation
│   ├── utils/          # Helper utilities (seeds, weights, etc.)
│   └── visualization/  # Visualization tools (t-SNE)
├── tests/              # Unit tests
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # This file
```

---

## Usage Instructions

### Environment Management
This project uses uv for dependency management and environment isolation.

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Typical Workflow
1.  Graph Generation:
    ```bash
    python src/graphs/graph_creation_pipeline.py
    ```
2.  Benchmarking:
    ```bash
    python src/main.py
    ```
3.  Experimental Analysis:
    ```bash
    mlflow ui --backend-store-uri sqlite:///results/mlruns.db
    ```

---

## Author

Enric Ferrera González
Institution: Universitat Autònoma de Barcelona 
Date: June 2026

---

## License

Copyright (c) 2026 Enric Ferrera González. All rights reserved.

This source code is provided for viewing purposes only as part of a Bachelor's Thesis project. No part of this repository may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the author.
