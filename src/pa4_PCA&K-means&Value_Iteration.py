from collections import Counter
from dataclasses import dataclass

from pathlib import Path
from typing import Tuple, List
from PIL import Image

# from sklearn.decomposition import PCA

import numpy as np
import random

train_sample_num = 100

# actually, we don't need eval_sample..
# because k-means algorithm is unsupervised learning!
eval_sample_num = 10

data_path = "dataset"

n_components_default = 15
max_iteration_default = 50

'''
Using PCA, we will reduce dimension of images.

by doing this...

1. Computational efficiency increases.
2. 'Noise' reduces.
3. Overfitting also reduces!

'''

## 1. Prepare the dataset
# this code is same as pa3; function prepare_dataset.

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

## 2. Select principal components

'''
When reducing dimensions, we need to select 'axis' properly.

<Original PCA>
1. Make Covariance matrix from N - dimension data.
2. Find N Eigenvectors, Eigenvalues.
3. Sort Eigenvectors by size of Eigenvalue.
4. Remain N' Eigenvectors, remove other Eigenvectors.
5. Now, remaining N' Eigenvectors are new axis.

but.. we can do PCA more simply.

<SVD based PCA>
1. Mean Centering
2. SVD, Singular Value Decomposition

A = U Σ V^T. 

if A = m * n rectangular matrix,

U = m * m orthogonal matrix, also V^T = n * n orthogonal matrix. 

Σ = m * n diagonal matrix!

3. select N' rows of V^T -> PCA axis!

'''
def pca(data: np.ndarray, n_components: int) -> np.ndarray:

    """
    Returns [np.float32; [N, N' = n_components = reduced_dimension_size]].
    """
    # Implement PCA, by SVD

    U, sigma, V_T = np.linalg.svd(data - np.mean(data, axis = 0), full_matrices=False)

    V_T_C = V_T[:n_components, :] # N' x (original_dimension_size)

    selected_components = np.dot(data - np.mean(data, axis = 0), V_T_C.T) 
    # (N x (original_dimension_size)) x ((original_dimension_size) x N')

    return selected_components.astype(np.float32) # N x N'

## 3. Run the K-means algorithm
def k_means(
    data: np.ndarray, initial_centroid: np.ndarray, max_iteration: int
) -> np.ndarray:
    
    """
    Returns [np.float32; [K, N']], the updated K centroids.
    """
    # Implement K-means algorithm
    
    N = data.shape[0]
    K = initial_centroid.shape[0]

    for _ in range(max_iteration):
        
        # initialize dist matrix.
        dist = np.zeros((N, initial_centroid.shape[0]), dtype = np.float32)

        # caculate distance
        for i in range(N):
            for j in range(K):
                dist[i, j] = np.linalg.norm(data[i] - initial_centroid[j])

        # find nearest point index
        nearest = np.argmin(dist, axis=1) # shape = (N, ).

        new_centroid = np.zeros_like(initial_centroid) # shape = (K, N')

        # update each centroids.
        for i in range(K):
            point = data[nearest == i]

            if len(point) > 0:  
                new_centroid[i] = point.mean(axis=0)
            else:               
                new_centroid[i] = initial_centroid[i]  

        initial_centroid = new_centroid

    return initial_centroid.astype(np.float32) 

## 4. Cluster the given points within the mean vectors.
def cluster(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Cluster the given points to the most likely mode.

    Args:
        data: [np.float32; [N, N']], list of N points.
        mean: [np.float32; [K, N']], the K centroids for each cluster.

    Returns:
        [np.int32; [N]], the indices of the likely cluster.
    """

    # initialize dist matrix.
    dist = np.zeros((data.shape[0], mean.shape[0]), dtype = np.float32) # (N, K)

    # caculate distance
    for i in range(data.shape[0]):
        for j in range(mean.shape[0]): 
            dist[i, j] = np.linalg.norm(data[i] - mean[j])

    # find nearest centroid!
    return np.argmin(dist, axis = 1).astype(np.int32)

####################################################################################

## Match the cluster ID and label.
# this skeleton code originates from SNUCSE 4190.428, Introduction to Machine Learning class.
'''
def match_cluster(
    labels: np.ndarray, assignments: np.ndarray
) -> Tuple[List[int], float]:
    """
    Match the cluster ID and label in greedy manner.
    Args:
        labels: [np.int32; [N]], the actual label of the data, in range[0, K)
        assignments: [np.int32; [N]], the assigned cluster ID of the data, in range[0, K)
    Returns:
        matches: list[int], list of K integers that map label `i` to the cluster ID `matches[i]`.
        accuracy: float, the estimated matching accuracy.
    """
    K = labels.max() + 1
    matches = [None for _ in range(K)]
    # precompute the counters
    counters = [
        Counter(assignments[labels == label]).most_common() for label in range(K)
    ]

    for _ in range(len(matches)):
        try:
            possibles = {
                label: next(
                    (cid, cnt) for cid, cnt in counter if cid not in set(matches)
                )
                for label, (matched, counter) in enumerate(zip(matches, counters))
                if matched is None
            }
        except StopIteration:
            raise ValueError("No more possible matches left.")
        label, (cid, _) = max(possibles.items(), key=lambda x: x[1][1])
        matches[label] = cid.item()

    accuracy = np.mean(
        [matches[label] == cid for label, cid in zip(labels, assignments)]
    )
    return matches, accuracy
'''

# original code has a problem; label-cluster unmapping problem. 
def match_cluster(
    labels: np.ndarray, assignments: np.ndarray
) -> Tuple[List[int], float]:
    """
    Match the cluster ID and label in greedy manner.
    Args:
        labels: [np.int32; [N]], the actual label of the data, in range[0, K)
        assignments: [np.int32; [N]], the assigned cluster ID of the data, in range[0, K)
    Returns:
        matches: list[int], list of K integers that map label `i` to the cluster ID `matches[i]`.
        accuracy: float, the estimated matching accuracy.
    """
    K = labels.max() + 1
    matches = [None for _ in range(K)]
    
    counters = [
        Counter(assignments[labels == label]).most_common() for label in range(K)
    ]

    used_clusters = set()

    for label in range(K):
        for cid, _ in counters[label]:
            if cid not in used_clusters:
                matches[label] = cid
                used_clusters.add(cid)
                break
        else:
            # If there are no matchable cluster, just match non-used cluster
            unused = set(range(K)) - used_clusters
            if unused:
                fallback_cid = unused.pop()
                matches[label] = fallback_cid
                used_clusters.add(fallback_cid)
            else:
                matches[label] = -1

    # remapping expected cluster ID by label, caculate accuracy.
    predicted = np.array([matches[label] for label in labels])
    accuracy = np.mean(predicted == assignments)

    return matches, accuracy

## 5. Value iteration on Frozen-lake environment
class FrozenLake:
    def __init__(
        self,
        desc: list[str] = ["SFFF", "FHFH", "FFFH", "HFFG"],
        actions: List[Tuple[int, int]] = [(1, 0), (0, 1)],
    ):
        """Frozen-lake environment.
        Args:
            desc: a descriptor of the map, e.g. ["SFFF", "FHFH", "FFFH", "HFFG"] represents
                S F F F
                F H F H
                F F F H
                H F F G
                where S is the starting point, H is a hole, F is a frozen field, and G is the goal point.
                The user starts from the starting point and moves towards the goal without falling into the hole.
                The game will end if the user is placed in a hole or goal and gets the reward of 1.0 only when they reach the goal.
            actions: a action space, each represents the position displacement; (dy, dx).
        """
        self.desc = desc # map.
        self.actions = actions # (1,0) means down, (0,1) means right

        self.height = len(desc)
        self.width = len(desc[0])

    @dataclass
    class StepResult:
        next_state: Tuple[int, int] | None = None
        reward: float = 0.0
        done: bool = False

    def is_ended(self, state: Tuple[int, int]) -> bool:
        y, x = state
        if y >= self.height or x >= self.width:
            return True
        return self.desc[y][x] in ("H", "G")

    def step(self, state: Tuple[int, int], action: int) -> StepResult:
        """Step next and return the information.
        Args:
            state: a position of the user; (y, x).
            action: a selected action given by an index of the action space.
        Returns:
            next_state: the next position of the user.
                return None when it is an invalid action (if the user tries to walk out to the map).
            reward: the reward value of the current state-action pair.
            done: whether the game is ended (reached out holes or the goal)
        """

        #####################################

        y, x = state
        dy, dx = self.actions[action]

        # update state
        ny, nx = dy + y, dx + x

        # sanity check
        if ny >= self.height or nx >= self.width:
            return FrozenLake.StepResult(None)
        
        # 1.0 when reach the goal
        reward = float(self.desc[ny][nx] == "G")

        # termination check
        done = self.desc[ny][nx] in ("H", "G")
        return FrozenLake.StepResult((ny, nx), reward, done)


def value_iteration(
    env: FrozenLake,
    num_horizon: int,
) -> np.ndarray:
    # initialize a value table
    # SAVE "VALUE"
    q = np.zeros((env.height, env.width, len(env.actions)))

    # update the value table
    for _ in range(num_horizon):
        q_update = np.zeros_like(q)

        # for each state
        for s in range(env.height * env.width):

            h, w = divmod(s, env.width) # divmod(11, 5) => 2, 1
            if env.is_ended((h, w)): # end condition 
                continue

            # TODO: for each actions, evaluate the reward and update q
            
            ##################################
            # for each box, check all actions
            for index, action in enumerate(env.actions):
                step_result = env.step((h,w), index) # next_state, done, reward available

                if step_result.next_state is None:
                    q_update[h, w, index] = 0
                else:
                    next_h, next_w = step_result.next_state
                    q_update[h, w, index] = step_result.reward + np.max(q[next_h, next_w])

        q = q_update

    return q

def simulate_policy(
    env: FrozenLake,
    value: np.ndarray,
    max_steps: int = 8,
) -> bool:
    state = next(
        (h, w)
        for h in range(env.height)
        for w in range(env.width)
        if env.desc[h][w] == "S"
    )
    for _ in range(max_steps):
        h, w = state
        action = np.argmax(value[h, w])
        result = env.step(state, action.item())
        if result.done:
            return result.reward > 0.0

        if result.next_state is None:
            return False

        state = result.next_state

    return False

########################################################
# hyperparameter test code

'''
train_sample_num_lst = [50, 100, 200]
n_components_default_lst = [5, 10, 15, 20, 25]
max_iteration_default_lst = [50, 100, 200, 300]

for train_sample_num in train_sample_num_lst:
    for n_components_default in n_components_default_lst:
        for max_iteration_default in max_iteration_default_lst:

            avg_acc = 0

            for i in range(5):

                train_images, train_labels, _, _ = prepare_dataset(Path(data_path))

                flat_train = train_images.reshape(len(train_images), -1)  # shape: (N, 28 x 28 = pixels of image)

                reduced_train = pca(flat_train, n_components = n_components_default)  # shape: (N, n_components)
                
                mean = reduced_train.mean(axis=0)
                std = reduced_train.std(axis=0) + 1e-8
                reduced_train = (reduced_train - mean) / std

                _, indices = np.unique(train_labels, return_index=True)
                initial_centroid = reduced_train[np.array(indices)]  # shape: (10 = 0~9, n_components)

                centroids = k_means(reduced_train, initial_centroid, max_iteration=max_iteration_default)

                assignments = cluster(reduced_train, centroids) 

                matches, accuracy = match_cluster(train_labels, assignments)

                avg_acc += accuracy

            print(f"train_sample_num: {train_sample_num}")
            print(f"n_components_default: {n_components_default}")
            print(f"max_iteration_default: {max_iteration_default}")

            print(f"Matching Accuracy: {avg_acc * 100/5:.2f}%, ")


'''
# <PCA & K-means clustering>

# we don't use eval_images, eval_labels.
# "unsupervised learning"

train_images, train_labels, _, _ = prepare_dataset(Path(data_path))

flat_train = train_images.reshape(len(train_images), -1)  # shape: (N, 28 x 28 = pixels of image)

# PCA
reduced_train = pca(flat_train, n_components = n_components_default)  # shape: (N, n_components)

# Normalization
'''
As a result of PCA, Variances depend on axis.

but K-means algorithm is based on "distance".

so, K-means algorithm focuses on axis which has large Variance value.
-> particular axes are more influential.

by normalization, we can solve this problem.
'''
mean = reduced_train.mean(axis=0)
std = reduced_train.std(axis=0) + 1e-8
reduced_train = (reduced_train - mean) / std

# select one data from each class, and set those points -> initial centroids!
_, indices = np.unique(train_labels, return_index=True)
initial_centroid = reduced_train[np.array(indices)]  # shape: (10 = 0~9, n_components)

# k-means clustering.
centroids = k_means(reduced_train, initial_centroid, max_iteration=max_iteration_default)

# allocate each data to centroids.
assignments = cluster(reduced_train, centroids) 

# Matching results!
matches, accuracy = match_cluster(train_labels, assignments)

print(f"train_sample_num: {train_sample_num}")
print(f"n_components_default: {n_components_default}")
print(f"max_iteration_default: {max_iteration_default}")

print(f"Matching Accuracy: {accuracy * 100:.2f}%")

print("###########################################")

# <Value Iteration>

lake = FrozenLake()

value = value_iteration(env=lake, num_horizon=100)

result = simulate_policy(env=lake, value=value)

print(f"Value Iteration result: {result}")