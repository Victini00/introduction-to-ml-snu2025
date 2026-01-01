from pathlib import Path
from typing import Tuple, List
from PIL import Image

import numpy as np
import random 

train_sample_num = 100
eval_sample_num = 10

data_path = "dataset"

learning_rate_default = 0.1
iteration_num_default = 1000
lambda_default = 0.1

'''
We're going to use

1. Logistic Regression
2. OLS (Ordinary Least Square)

In Logistic Regression, it minimizes Loss function NLL.

NLL = mean(y * log(g) + (1 - y) * log(1 - g)) where g = sigmoid(wTx)

In OLS, closed-from solution is determined.

We want to minimize loss function; mean((θ^T * xi + θ0 - yi)^2)

solution is θ = (W^T * W)^(-1) * W^T * Z, where W = X^T, Z = Y^T

#################################################

Logistic Regression)

* It is not so fast..
* but not so sensitive on outliers.
* It is specialized on Classification.

OLS - Linear Regression)

* It is fast - closed-form
* sensitive on outliers..
* It is specialized on Regression; but we can use it on classification problem(this case)

'''

## 1. Prepare the dataset
# this code is same as pa2; function prepare_dataset.

def prepare_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Args:
        path: a path to the dataset.

    Returns:
        tuple of read images and corresponding labels,
            train_images: [np.float32; [B, H, W]]
            train_labels: [np.long; [B]]
            eval_images: [np.float32; [B, H, W]]
            eval_labels: [np.long; [B]]
            
    """
    train_images = []
    train_labels = []

    eval_images = []
    eval_labels = []

    for digit in range(10): # 0 ~ 9

        digit_path = path/str(digit)

        total_files = list(digit_path.glob('*.png'))

        sample_files = random.sample(total_files, train_sample_num + eval_sample_num)

        train_files = sample_files[:train_sample_num]
        eval_files = sample_files[train_sample_num:]

        for file in train_files:
            img = Image.open(file).convert('L') 

            img_array = np.array(img, dtype = np.float32) / 255.0 

            train_images.append(img_array)
            train_labels.append(digit)

        for file in eval_files:
            img = Image.open(file).convert('L')

            img_array = np.array(img, dtype=np.float32) / 255.0

            eval_images.append(img_array)
            eval_labels.append(digit)

    train_data = list(zip(train_images, train_labels))
    random.shuffle(train_data)
    train_images, train_labels = zip(*train_data)
    
    
    eval_data = list(zip(eval_images, eval_labels))
    random.shuffle(eval_data)
    eval_images, eval_labels = zip(*eval_data)

    return (
        np.stack(train_images), 
        np.array(train_labels),
        np.stack(eval_images), 
        np.array(eval_labels)
    )


## sigmoid funtion
# this is activation function.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## 2. Compute the gradient of the loss of the logistic regression.

'''
Compute the the negative log-likelihood and its gradient.

nll = mean(y * log(g) + (1 - y) * log(1 - g)) where g = sigmoid(wTx)

'''
def compute_gradient(
    theta: np.ndarray, X: np.ndarray, Y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Args:
        theta: [np.float32; [H x W]], the given weight vector.

        X: [np.float32; [B, H, W]], the flattened images, in range [0, 1].
        Y: [np.float32; [B]], the corresponding labels, either 0 or 1.

        Directly use function prepare_dataset's output.

    Returns:
        float, the negative log-likelihood.
        [np.float32; [H x W]], the gradient of the negative log-likelihood.
    """
    # Compute the gradient
    # TODO: Compute the gradient of the negative log-likelihood.

    X = X.reshape(X.shape[0], -1)
    B = X.shape[0]

    g = sigmoid(X @ theta)

    # caculate L_ll; Log-likelihood Loss.
    L_ll = np.mean(Y * np.log(g) + (1 - Y) * np.log(1 - g))

    # caculate NLL's gradient.
    gradient = np.zeros_like(theta) 

    for i in range(B):
        gradient += (Y[i] - g[i]) * X[i]
    gradient = gradient / B

    return -L_ll, -gradient
    

## 3. Train a binary classifier using logistic regression formulation and the gradient descent.
def gradient_descent(
    dataset: Tuple[np.ndarray, np.ndarray],
    target_class: int,
    num_iterations: int,
    learning_rate: float,
) -> Tuple[np.ndarray, list[float]]:
    """
    Train a binary logistic regression classifier for one-vs-rest using gradient descent.

    Returns:
        [np.float32; [H x W]], the trained weight vector.
        list[float], the list of the nll values.
    """

    imgs, labels = dataset

    # Reshape images to 2D array [B, H * W]
    B, H, W = imgs.shape
    X = imgs.reshape(B, H * W)

    # Convert labels to binary
    # 1 for target_class, 0 for others
    y = np.where(labels == target_class, 1, 0).astype(np.float32)

    # Initialize the weight vector
    theta = np.zeros(H * W, dtype=np.float32)

    # For string errors
    nlls = []

    # Main training loop - the gradient descent algorithm.
    # TODO: Implement the gradient descent algorithm.

    # 1. Loop through iterations
    for _ in range(num_iterations):

        # 2. Compute the negative log-likelihood and its gradient.
        nll, gradient = compute_gradient(theta, X, y)

        # 3. Collect nlls
        nlls.append(nll)

        # 4. Update weight vector theta with the given step size; learning_rate.
        theta -= (learning_rate * gradient)

    return theta, nlls

## 4. Train a multi-class classifier with multiple binary classifiers.
def train_classifier_with_gradient_descent(
    dataset: Tuple[np.ndarray, np.ndarray],
    num_iterations: int,
    learning_rate: float,
) -> List[np.ndarray]:
    """
    Train a multi-class classifier with gradient descent using one-vs-rest approach.

    Returns:
        list[np.float32; [H x W]], the list of weight vectors trained with `train_binary_classifier`.
    """
    _, labels = dataset

    # Get unique classes
    classes = np.unique(labels)
    num_classes = len(classes)

    # Initialize the list of trained weights.
    weights = []

    # Train multi-label classifier with one-vs-rest approach.
    # 1. Train a binary classifier for each class
    # 2. Return list of models

    for i in range(num_classes):
        weight, _ = gradient_descent(dataset, i, num_iterations, learning_rate)
        weights.append(weight)

    return weights

'''
So far, we implemented..

# 2 -> We computed NLL, gradient.
# 3 -> Using NLL, gradient, train single binary classifier.
# 4 -> Using # 3, train all (in this case, 10) binary classifier.

'''

## 5. Classify the given image
def classify(
    models: List[np.ndarray],
    test_images: np.ndarray,
) -> list[int]:
    
    """
    Classify the given image with the trained logistic regression model.

    Args:
        models: List of Tuples containing trained alphas and bias for each class.
        test_images: [np.float32; [B, H, W]], the target images to classify.

    Returns:
        estimated labels of the images (0-9).
    """
    # Reshape images
    B, H, W = test_images.shape
    X_test = test_images.reshape(B, H * W)

    # TODO: Implement the classification logic
    
    # 1. Compute the probabilities of each labels
    
    '''
    each model's dimension is H * W.

    so, [B, H * W] @ [H * W, num_of_classes] = [B, num_of_classes].

    '''
    values = X_test @ (np.stack(models)).T 

    # 2. Return the class with the highest probability.
    prediction = np.argmax(values, axis = 1)

    prediction_list = prediction.tolist()

    return prediction_list

# from now on, We will compute the closed-form solution...
# of an OLS problem.

## 6. OLS-based binary classifier
def closed_form_solution_ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
   
    # simple!
    # this is implementation of..
    # θ = (W^T * W)^(-1) * W^T * Z, where W = X^T, Z = Y^T

    theta = np.linalg.pinv(X.T @ X) @ (X.T @ Y)

    return theta.astype(np.float32) 

## 6-1. OLS-base binary classifier.. with Ridge Regression.
# in my case, accuracy increases by 0.1!
def closed_form_solution_ridge(X: np.ndarray, Y: np.ndarray, param_lambda: float = lambda_default) -> np.ndarray:

    # simple OLS minimizes ||Xθ - Y||^2 ..
    # but Ridge Regression version minimizes ||Xθ - Y||^2 + λ||θ||^2.

    # so, θ = (W^T * W + λI)^(-1) * W^T * Z

    I = np.eye(X.shape[1], dtype=np.float32)

    theta = np.linalg.pinv(X.T @ X + param_lambda * I) @ (X.T @ Y)

    return theta.astype(np.float32)


## 7. Train the classifier with the closed-form solution of OLS problem.
def train_classifier_with_closed_form_solution(
    dataset: Tuple[np.ndarray, np.ndarray],
) -> List[np.ndarray]:

    images, labels = dataset
    
    B, H, W = images.shape
    images = images.reshape(B, H * W)

    # Get number of classes.
    # MNIST = 10
    classes = np.unique(labels)
    num_classes = len(classes)

    # Initialize the list of trained weights.
    weights = []

    # Train multi-label classifier with closed-form solution of OLS problem.
    for i in range(num_classes):
        
        # 1. Convert labels to binary: 1 for target_class, -1 for others
        Y_convert = np.where(labels == i, 1, -1).astype(np.float32) 

        # 2. Compute the closed-form solution.
        # theta = closed_form_solution_ols(images, Y_convert)
        theta = closed_form_solution_ridge(images, Y_convert, lambda_default)

        # 3. Append the solution(weight matrix) to `weights`
        weights.append(theta)

    return weights
        
############################################################

train_images, train_labels, eval_images, eval_labels = prepare_dataset(Path(data_path))

# first, train logistic regression model.
# you can modify num of iterations and learning rate.

# Also, you can add "L2 Regression(Ridge)".
logistic_models = train_classifier_with_gradient_descent((train_images, train_labels), num_iterations=iteration_num_default, learning_rate=learning_rate_default)

# next, train OLS model.
# I used Ridge Regression version. 
ols_models = train_classifier_with_closed_form_solution((train_images, train_labels))

# Now, predict test datas by both models!
pred_logistic = classify(logistic_models, eval_images)
pred_ols = classify(ols_models, eval_images)

accuracy_logistic = np.mean(np.array(pred_logistic) == eval_labels)
accuracy_ols = np.mean(np.array(pred_ols) == eval_labels)

print(f"Evaluation Accuracy (Logistic Regression): {accuracy_logistic:.2f}")
print(f"Evaluation Accuracy (OLS Closed-form): {accuracy_ols:.2f}")
