from pathlib import Path
from typing import Tuple
from PIL import Image

import numpy as np
import random

train_sample_num = 100
eval_sample_num = 10

data_path = "dataset"
sigma_default = 3.5

# Kernel-based Learning method.

'''
If a certain image is similar to a new input, we follow the label of that image.

If the new input is similar to xi, then its coressponding beta has a stronger effect

Interesting fact) # of params are determined by "# of datas", not "size of data"!
'''

## 1. Prepare dataset.
# I used original MNIST dataset : https://drive.google.com/file/d/1e2QUgaRp4Dovj-3QKfUsuzmHzxhsb9Bc/view

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
        
        # in data_path, there are 10 folders.
        # their names are "0", "1", ... "9".
        # 28x28 handwritten png images are in each folder.

        total_files = list(digit_path.glob('*.png'))

        sample_files = random.sample(total_files, train_sample_num + eval_sample_num)

        train_files = sample_files[:train_sample_num]
        eval_files = sample_files[train_sample_num:]

        for file in train_files:
            img = Image.open(file).convert('L') # L means monochrome

            img_array = np.array(img, dtype = np.float32) / 255.0 # regularization

            train_images.append(img_array)
            train_labels.append(digit)

        for file in eval_files:
            img = Image.open(file).convert('L')

            img_array = np.array(img, dtype=np.float32) / 255.0

            eval_images.append(img_array)
            eval_labels.append(digit)

    # shuffle
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

# Use this code if needed..
# print(prepare_dataset(Path(data_path)))

## 2. kernel function ~ "Gaussian kernel"
def kernel_function(a: np.ndarray, b: np.ndarray, sigma: float = sigma_default) -> float:

    euclidean_distance = np.sum((a-b)**2)

    kernel_value = np.exp(-euclidean_distance / (2 * (sigma**2)))
    
    return kernel_value

## 2A. kernel function ~ normal kernel function is slow..
def improved_kernel_function(a: np.ndarray, b: np.ndarray, sigma: float = sigma_default) -> np.ndarray:

    flat_a = a.reshape((a.shape[0], -1))

    flat_b = b.reshape((b.shape[0], -1))

    norm_a = np.sum(flat_a**2, axis = 1).reshape(-1, 1)
    norm_b = np.sum(flat_b**2, axis = 1).reshape(1, -1)

    dist = norm_a + norm_b - 2 * np.dot(flat_a, flat_b.T)

    return np.exp(-dist / (2 * sigma ** 2))

## 3. Update the parameters
# We will train 10 binary Classifier.
def train(
    dataset: Tuple[np.ndarray, np.ndarray], # return value of # 1.
    num_training_steps: int, 
    learning_rate: float,
    sigma: float = sigma_default,
) -> np.ndarray:
    
    """
    Returns:
        - beta_matrix: shape [10, N], where N = number of training samples
          ..because there are 10 classes
    """

    images, labels = dataset
    num_of_datas = images.shape[0]
    num_classes = 10

    # set kernel lists
    kernel = improved_kernel_function(images, images, sigma)

    # initialize parameter, beta.
    final_beta = np.zeros((num_classes, num_of_datas), dtype = np.float32)

    # this implements..
    # beta(i) = beta(i) + alpha(y - sigma(beta(j) * Kernel(Xi, Xj)))

    # and also this is Synchronous.
    # pros) Stability
    # cons) Slow..

    # Consider Asynchronous method.. 

    for c in range(num_classes):

        # pre-processing: mapping target class to 1, otherwise -1.
        mapping_label = np.where(labels == c, 1, -1) 

        beta = np.zeros(num_of_datas)

        for _ in range(num_training_steps):

            for i in range(num_of_datas):

                kernel_sum = np.dot(beta, kernel[i])
                beta[i] += learning_rate * (mapping_label[i] - kernel_sum)

        final_beta[c] = beta

    return final_beta
    
## 4. Estimate the labels of the given images
def classify(beta: np.ndarray, 
             test_data: np.ndarray, 
             train_images: np.ndarray, 
             sigma: float = sigma_default) -> int:

    num_classes, num_of_datas = beta.shape

    estimation_score = np.zeros(num_classes)

    # Estimation(j) = sigma(beta(i) * K(Xi, Xj))
    '''
    for c in range(num_classes):
        for i in range(num_of_datas):
            kernel_value = kernel_function(train_images[i], test_data, sigma)
            estimation_score[c] += beta[c, i] * kernel_value
    '''
    kernel_values = improved_kernel_function(train_images, test_data[np.newaxis], sigma)
    kernel_values = kernel_values.flatten()

    for c in range(num_classes):
        estimation_score[c] = np.dot(beta[c], kernel_values)

    return int(np.argmax(estimation_score))

## Overall Test

result_list = []

def test(beta: np.ndarray, 
         train_images: np.ndarray, 
         eval_images: np.ndarray, 
         eval_labels: np.ndarray,
         sigma: float = sigma_default):
    
    correct = 0

    for i in range(len(eval_images)):

        prediction = classify(beta, eval_images[i], train_images, sigma)

        if prediction == eval_labels[i]: 
            correct += 1 # if right!

    accuracy = correct / len(eval_labels)
    result_list.append(f"accuracy: {accuracy}, sigma: {sigma}")

#####################################################################

'''
# select appropriate sigma value.

for i in range(15):
    for j in range(5):
        sigma = 0.5 + i * 0.5
        train_images, train_labels, eval_images, eval_labels = prepare_dataset(Path(data_path))
        params = train((train_images, train_labels), 1000, 0.1, sigma)
        test(params, train_images, eval_images, eval_labels, sigma)

print(result_list)
'''

# I select sigma = 3.5 ! 

train_images, train_labels, eval_images, eval_labels = prepare_dataset(Path(data_path))
params = train((train_images, train_labels), 1000, 0.1, sigma = sigma_default)
test(params, train_images, eval_images, eval_labels, sigma = sigma_default)

print(result_list)