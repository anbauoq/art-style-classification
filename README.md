# 🎨 Art Style Classification Project

This project focuses on classifying images into eight distinct art styles: ArtDeco, cartoon, Cubism, Impressionism, Japonism, Naturalism, photo and Rococo.


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


## Results

The **Direct Classification** approach was clearly the most effective and robust for this task. The Gradient Boosting classifier, trained on the features extracted from the direct classifier, performed nearly as well.

| Metric | Direct Classifier | GradientBoosting | Metric Learning |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score** | **0.77** | 0.75 | 0.68 |
| **Accuracy** | **0.81** | 0.79 | 0.71 |
| **Macro Precision** | 0.79 | 0.76 | 0.68 |
| **Macro Recall** | 0.77 | 0.75 | 0.69 |

*Table: Final performance comparison on the test set.*

The metric learning model was found to be highly unstable and overfit almost immediately, likely due to the task's complexity and the limited data. The direct classifier provided the most robust and effective solution.

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

- `outputs/figs/` — plots (loss curves, conf mats, distributions)

- `outputs/param-optim*/` — sweep logs and summaries
