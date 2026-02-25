# Art Style Classification

## Overview
This repository provides the implementation of deep learning models for art-style image classification under severe class imbalance, including training, evaluation, and feature analysis.

Detailed methodology, experiments, and results are described in the accompanying technical report `art_style_classification.pdf`.

## Project Workflow

### 1. Data Preparation
* **Exploratory Data Analysis (EDA):** Analyzed the dataset to identify key challenges, noting the small sample size and significant class imbalance.
* **Data Split:** The dataset was split into training, validation, and test sets (70/15/15) using stratification to maintain class proportions.
* **Data Augmentation:** Implemented an **offline augmentation** strategy using the `albumentations` library. This was used to up-sample minority classes, creating a more balanced dataset for training.

### 2. Modeling Approaches

Two primary modeling strategies were implemented and compared, both using **EfficientNet-B0** as the backbone architecture.

#### Approach 1: Direct Classification
* **Method:** Fine-tuned the EfficientNet-B0 model for a standard multi-class classification task.
* **Loss:** Standard Cross-Entropy Loss.
* **Techniques:** Employed progressive unfreezing, cosine annealing learning rate, and a custom early stopping mechanism to prevent overfitting.

#### Approach 2: Metric Learning
* **Method:** Trained the model to learn a feature embedding space where similar art styles are clustered together.
* **Loss:** Triplet Margin Loss, combined with a Hard Triplet Miner to find challenging examples.
* **Data Sampling:** Used an `MPerClassSampler` to ensure each batch contained multiple images from several classes, which is necessary for effective triplet mining.

### 3. Hyperparameter Tuning
* **Optuna** was used to perform Bayesian hyperparameter optimization for both modeling approaches, tuning parameters like learning rate, batch size, and weight decay.

### 4. Feature Analysis
* **Feature Extraction:** Embeddings (1280-dimensional) were extracted from the best-performing direct classifier.
* **Visualization:** **UMAP** was used to project the high-dimensional features into 2D, showing clear separation for some classes.
* **Clustering:** **K-Means** clustering was applied to the embeddings.
* **Classification:** A **GradientBoosting Classifier** was trained on the extracted features, achieving results (Macro F1: 0.75) close to the end-to-end model.



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
