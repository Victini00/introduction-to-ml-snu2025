# Introduction_to_Machine_Learning_SNU2025
#### Assignment Repository of IML; SNUCSE2025 class.

⚠️
Basically, it is based on the source code provided in SNUCSE 4190.428 class, but I made a lot of modifications out of necessity.

This repository contains various machine learning assignments designed to classify handwritten digits from the MNIST dataset (28x28 grayscale images). 

Each assignment applies a different learning strategy to tackle the same classification task, offering insight into the strengths and weaknesses of each approach.

## Overview of Each Assignment

### `pa1_Gaussian_Kernel.py` 
**Kernel-based Learning**

- **Model Type**: Kernel regression-like classification
- **Kernel Function**: Gaussian (RBF)
- **Training**:
  - Each class trains a separate beta vector
  - Uses similarity (kernel) between test image and all training images
- **Pros**:
  - Simple and intuitive
  - Easy to implement
- **Cons**:
  - Computationally slow
  - Requires all training data to be stored

---

### `pa2_Support_Vector_Machine.py`
**Kernel SVM with SMO**

- **Model Type**: Support Vector Machine (SVM)
- **Optimization**: Sequential Minimal Optimization (SMO)
- **Parameters**:
  - `alphas`: Support vector weights
  - `bias`: Class-specific offsets
- **Training**:
  - Checks KKT conditions and updates violating pairs of `alpha`
- **Pros**:
  - Strong generalization
  - Avoids overfitting via margin maximization
- **Cons**:
  - Complex implementation
  - Training time is relatively high

---

### `pa3_Gradient_Descent&Closed-form_OLS.py`
**Logistic Regression & OLS**

#### 1. **Logistic Regression**
- **Model Type**: Binary classifier using logistic loss
- **Optimization**: Gradient Descent
- **Loss Function**: Negative Log-Likelihood (NLL)
- **Pros**:
  - More robust to outliers
  - Tailored for classification problems
- **Cons**:
  - Slower convergence

#### 2. **OLS (Ordinary Least Squares) with Ridge Regularization**
- **Model Type**: Linear regression (adapted for classification)
- **Solution**: Closed-form:
  ```math
  θ = (XᵗX + λI)⁻¹ XᵗY
    (Regularization: Ridge (L2))
- **Pros**:
  - Fast computation
  - Closed-form, no iteration required
- **Cons**:
  - Sensitive to outliers
  - Not originally designed for classification

---

### `pa4_PCA&K-means&Value_Iteration.py`
**PCA + K-means Clustering**

- **Model Type**: Unsupervised Learning
- **Techniques**:
  - PCA (Principal Component Analysis)
  - K-means Clustering
  - Cluster-to-label Matching
- **Training**:
  - Flatten images and reduce dimensions using SVD-based PCA
  - Normalize reduced features to equalize variance
  - Run K-means to group images into 10 clusters
  - Match each cluster to a digit label using greedy matching
- **Pros**:
  - Does not require any labels during training
  - Useful for data compression and exploratory analysis
  - Fast and easy to implement
- **Cons**:
  - Typically lower accuracy than supervised learning
  - Cluster assignments are ambiguous until post-matching
  - Sensitive to initial centroid selection
---


> Note: This file also contains a Value Iteration example for FrozenLake, which is unrelated to MNIST and excluded from this description.