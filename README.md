# Art Style Classification

This repository provides the implementation of deep learning models for art-style image classification under severe class imbalance, including training, evaluation, and feature analysis.

Detailed methodology, experiments, and results are described in the accompanying technical report `technical_report.pdf`.

## Workflow

- **Data Preparation:** EDA, stratified 70/15/15 split, and offline augmentation with albumentations to address class imbalance.
- **Modeling:** EfficientNet-B0 backbone with two approaches:
  - Direct classification (Cross-Entropy, progressive unfreezing, cosine LR, early stopping)
  - Metric learning (Triplet Margin Loss with hard mining)
- **Hyperparameter Tuning:** Bayesian optimization with Optuna.
- **Feature Analysis:** Embedding extraction, UMAP visualization, K-Means clustering, and Gradient Boosting classification on frozen features.


## Project Structure

### Notebooks

- `0. EDA.ipynb` — class balance, image sizes, quick sanity checks

- `1. Data_Augmentation.ipynb` — preview & export strong aug policies

- `2.1 Hyperparameter_optimization.ipynb` — quick sweeps for LR, bs, aug strength

- `2.2 Model_Training.ipynb` — standard classifier training loop

- `3.1 Metric_Learning.ipynb` — triplet/miner setup & diagnostics

- `3.2 Metric_Learning_Training.ipynb` — metric learning training loop

- `4. Feature_Classification.ipynb` — linear/MLP head on frozen embeddings

### Source

- `data.py` — Albumentations pipelines, torchvision ImageFolder loaders

- `utils.py` — device selection, EfficientNet/backbone helpers, batch preds

- `train.py` — classification train_step/val_step, metric-learning epoch

- `visualization.py` — class distribution, confusion matrix, image size plots


### Data

- `data/images/` — raw images in ImageFolder layout

- `data/augmented_images/` — offline-saved augmented samples

### Outputs

- `outputs/models/` — checkpoints

- `outputs/figs/` — plots

- `outputs/param-optim*/` — sweep logs and summaries
