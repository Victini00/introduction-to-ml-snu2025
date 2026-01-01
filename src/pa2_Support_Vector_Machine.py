from pathlib import Path
from typing import Tuple, List
from PIL import Image

import numpy as np
import random 

train_sample_num = 100
eval_sample_num = 10

class_num = 10 # 0~9

data_path = "dataset"

sigma_default = 2
C_default = 0.5
iteration_num_default = 1000

## 1. Prepare the dataset
# this code is same as pa1; function prepare_dataset.

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


## 2. Compute the kernel matrix
# this code is also same as pa1; function improved_kernel_function.

def compute_kernel_matrix(a: np.ndarray, b: np.ndarray = None, sigma: float = sigma_default) -> np.ndarray:
    """
    Compute the kernel matrix between X and Y.

    Args:
        X: [np.float32; [N, H, W]], flattened images.
        Y: [np.float32; [M, H, W]], flattened images. Default is None.
        sigma: a bandwidth of the gaussian kernel.
    
    Returns:
        [np.float32; [N, M]], the kernel matrix.
    """

    # Handling Exception
    if b is None:
        b = a

    flat_a = a.reshape((a.shape[0], -1))
    flat_b = b.reshape((b.shape[0], -1))
    
    norm_a = np.sum(flat_a**2, axis=1).reshape(-1, 1)
    norm_b = np.sum(flat_b**2, axis=1).reshape(1, -1)
    
    dist = norm_a + norm_b - 2 * np.dot(flat_a, flat_b.T)

    return np.exp(-dist / (2 * (sigma ** 2)))

## 3. Train a binary SVM classifier, using Sequential Minimal Optimization Algorithm

'''
SVM ~ why we should maximize 'Margins'?

- It prevents overfitting; It is strong to noises.

We can solve this by Lagrange Multipliers; dual problem.

.. and dual problem can be solved by SMO algorithm; updates two alphas simultaneously.

+

this method is smarter than pa1(simple Kernel model)

because this method only selects support vectors, and maximize margin.

'''
def train_svm(
    dataset: Tuple[np.ndarray, np.ndarray],
    target_class: int,
    num_iterations: int,
    C: float = C_default,
    sigma: float = sigma_default,
    tol: float = 1e-4,
    eps: float = 1e-5
) -> Tuple[np.ndarray, float]:
    """
    Train a binary SVM classifier for one-vs-rest classification using SMO-like algorithm.
    
    Args:
        dataset: tuple of read images and corresponding labels,
            [np.float32; [B, H, W]] and [np.long, [B]].
        target_class: the target class for binary classification (one-vs-rest).
        C: regularization parameter.
        sigma: a bandwidth of the gaussian kernel.
        tol: tolerance for KKT conditions.
        eps: numerical stability constant.
    
    Returns:
        Tuple of [np.float32; [B]] and float, the trained alphas and bias.

    """
    imgs, labels = dataset
    
    # Reshape images to 2D array [B, H*W]
    b, h, w = imgs.shape
    X = imgs.reshape(b, h * w)
    
    # Convert labels to binary: 1 for target_class, -1 for others
    y = np.where(labels == target_class, 1, -1).astype(np.float32)
    
    # Compute kernel matrix
    K = compute_kernel_matrix(X, sigma=sigma)
    
    # Initialize alphas and bias
    alphas = np.zeros(b, dtype=np.float32)
    bias = 0.0
    
    # Main training loop - simplified SMO algorithm

    # TODO: Implement the SMO algorithm
    # 1. Loop through iterations
    # 2. Check if example violates KKT conditions
    # 3. Select second alpha
    # 4. Update alphas and bias
    # 5. Update error cache
    
    # Main Idea) 
    # max (Sum(alpha(i)) - 1/2 * (Sum(alpha(i) * alpha(j) * y(i) * y(j) * K(xi, xj)))
    # .. subject to 0 <= alpha(i) <= C, Sum(alpha(i) * y(i)) = 0

    # 1. Loop through iterations
    for _ in range(num_iterations):

        Early_Stop_Checker = True # Early stopping

        for i in range(b):

            # 2. Check if example violates KKT conditions
            Expected_i = np.sum(alphas * y * K[i]) + bias
            Real_i = y[i]

            g_i = Expected_i * Real_i - 1 

            '''
            KKT conditions)

            if alpha(i) == 0: g_i >= 0
            if alpha(i) == C: g_i <= 0
            else ~ g_i == 0

            otherwise... violates KKT!
            so we need to update params.
            '''

            # case 1) gi < 0 and ai < C
            # case 2) gi > 0 and ai > 0
            if (g_i < - tol and alphas[i] < C - eps) or (g_i > tol and alphas[i] > eps):

                # 3. Select second alpha.. random choice.
                # It is SMO algorithm. 
                j_old = i
                j = np.random.choice([k for k in range(b) if k != j_old])

                # calculate range of 'Box'
                if y[i] == y[j]: 
                    Low = max(0, alphas[i] + alphas[j] - C)
                    High = min(C, alphas[i] + alphas[j])

                else: 
                    Low = max(0, alphas[j] - alphas[i])
                    High = min(C, alphas[j] - alphas[i] + C)

                if Low == High: continue

                # 4. Update alphas and bias
                '''
                We want to maximize.. 
                W(alpha) = Sum(alpha(i)) - 1/2 * (Sum(alpha(i) * alpha(j) * y(i) * y(j) * K(xi, xj))

                We selected alpha(i), alpha(j).
                other alphas are fixed.

                dW / d(alpha(j)) = 0.

                We can derive "new alpha(j)"!
                '''

                eta = K[i, i] + K[j, j] - 2 * K[i, j]

                E_i = Expected_i - y[i]
                E_j = np.sum(alphas * y * K[j]) + bias - y[j]

                if eta <= 0: continue # Exception Error

                # 4-1. Update alpha(j), alpha(i)
                new_alpha_j = np.clip(alphas[j] + y[j] * (E_i - E_j) / eta, Low, High)

                # if gap is small, we don't have to update values.
                if abs(new_alpha_j - alphas[j]) < eps: continue

                new_alpha_i = alphas[i] + y[i] * y[j] * (alphas[j] - new_alpha_j)

                # 4-2. Update bias 
                bi = -1 * Expected_i - (y[i] * (new_alpha_i - alphas[i]) * K[i, i]) - (y[j] * (new_alpha_j - alphas[j]) * K[i, j]) + bias

                Expected_j = np.sum(alphas * y * K[j]) + bias - y[j]

                bj = -1 * Expected_j - (y[i] * (new_alpha_i - alphas[i]) * K[i, j]) - (y[j] * (new_alpha_j - alphas[j]) * K[j, j]) + bias

                if (0 < new_alpha_i) and (new_alpha_i < C): bias = bi
                elif (0 < new_alpha_j) and (new_alpha_j < C): bias = bj
                else: bias = (bi + bj) / 2 # if both are not support vector.. we take average of values. 

                alphas[i] = new_alpha_i
                alphas[j] = new_alpha_j

                Early_Stop_Checker = False

        # Early stopping
        if Early_Stop_Checker == True: break

    # Return the model parameters
    return alphas, bias


## 4. Train a multi-class SVM classifier
def train_multi_class_svm(
    dataset: Tuple[np.ndarray, np.ndarray],
    C: float = C_default,
    sigma: float = sigma_default,
) -> List[Tuple[np.ndarray, float]]:
    
    """Train a multi-class SVM classifier using one-vs-rest approach.
    
    Args:
        dataset: tuple of read images and corresponding labels,
            [np.float32; [B, H, W]] and [np.long, [B]].
        C: regularization parameter.
        sigma: a bandwidth of the gaussian kernel.
    
    Returns:
        List of Tuples containing [np.float32; [B]] and float, the trained alphas and bias for each class.
    """
    imgs, labels = dataset
    
    # Get unique classes
    # when using MNIST data, classes = [0, 1, 2, ... 9].
    classes = np.unique(labels)

    class_model = []

    for target_class in classes:
        alphas, bias = train_svm((imgs, labels), target_class = target_class, num_iterations = iteration_num_default, C = C, sigma = sigma) 
        class_model.append((alphas, bias))

    return class_model


## 5. Classify the given image
# Now we know all params!

def classify(
    models: List[Tuple[np.ndarray, float]], 
    train_images: np.ndarray, 
    train_labels: np.ndarray, 
    test_image: np.ndarray,
    sigma: float = sigma_default,
) -> int:
    """Classify the given image with the trained kernel SVM model.
    
    Args:
        models: List of Tuples containing trained alphas and bias for each class.
        train_images: [np.float32; [B, H, W]], the training images.
        train_labels: [np.long; [B]], the training labels.
        test_image: [np.float32; [H, W]], the target image to classify.
        sigma: a bandwidth of the gaussian kernel.
    
    Returns:
        an estimated label of the image (0-9).
    """

    # Reshaping 
    test_image = test_image.reshape(1, *test_image.shape)
    
    # 1. Compute kernel values between test_image and all training images
    K = compute_kernel_matrix(train_images, test_image, sigma = sigma)

    index = 0
    mem = 0

    for i in range(len(models)):

        # 2. Compute decision values for each class
        '''
        decision function fi(x) =

        Sum {j = 1 to N} ( 
            alphas_i(j) * y_i(j) * K(x(j), x) 
        ) + bias_i

        '''
        alphas, bias = models[i]
        y_i = np.array([1 if label == i else -1 for label in train_labels])

        # this is decision function.
        Expected_i = np.sum(alphas * y_i * K.flatten()) + bias

        # 3. Return the class with highest decision value
        if mem <= Expected_i:
            index = i
            mem = Expected_i

    return index


#############################################

'''
C_list = [0.1, 0.5, 1, 2, 5, 10]
sigma_list = [0.1, 0.5, 1, 2, 5]

train_images, train_labels, test_images, test_labels = prepare_dataset(Path(data_path))

# save datas..
best_accuracy = 0.0
best_C = None
best_sigma = None

# test all combinations of C and sigma.
# this code is referenced from chatGPT.

for c_index in range(len(C_list)):
    for s_index in range(len(sigma_list)):

        selected_C = C_list[c_index]
        selected_sigma = sigma_list[s_index]

        print(f"Trying C = {selected_C}, sigma = {selected_sigma}...")

        models = train_multi_class_svm((train_images, train_labels), C = selected_C, sigma = selected_sigma)

        correct = 0
        total = eval_sample_num * class_num

        for number in range(total):
            prediction = classify(models, train_images, train_labels, test_images[number], sigma = selected_sigma)

            if prediction == test_labels[number]:
                correct += 1

        acc = correct / total

        print(f"accuracy = {acc}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_C = selected_C
            best_sigma = selected_sigma

print(f"Best hyperparameters found:")
print(f"C = {best_C}")
print(f"sigma = {best_sigma}")
print(f"accuracy = {best_accuracy}")
'''

# I select C = 0.5, sigma = 2.

train_images, train_labels, test_images, test_labels = prepare_dataset(Path(data_path))

models = train_multi_class_svm((train_images, train_labels), C = C_default, sigma = sigma_default)

ans = 0

for number in range(eval_sample_num * class_num):

    prediction = classify(models, train_images, train_labels, test_images[number], sigma = sigma_default)

    # print(f"Predicted: {prediction}, Actual: {test_labels[number]}") 

    if prediction == test_labels[number]:
        ans += 1

print(f"accuracy : {ans / (eval_sample_num * class_num)}")

