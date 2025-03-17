import numpy as np

from typing import Generator, Tuple, List

DEFAULT_SEED = 42

def splits_generator(dataset_length: int, k: int, seed: int = DEFAULT_SEED) -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Given a dataset length, the number of folds and an optional seed, this function generates k splits of the dataset and return the indices of the training and validation split for each fold.
    
    Args:
        dataset_length (int): The total number of samples in the dataset.
        k (int): The number of folds for cross-validation.
        seed (int, optional): Random seed for reproducibility.

    Yields:
        Tuple[List[int], List[int]]: A tuple containing:
            - training_indices (List[int]): Indices for the training set.
            - testing_indices (List[int]): Indices for the testing set.

    Example:
        >>> for train_idx, test_idx in split_generator(100, 5):
        ...     print(f"Train: {train_idx}, Test: {test_idx}")
    
    Notes:
        - The function randomly shuffles the dataset indices before splitting.
        - Each fold serves as the test set once, while the remaining data forms the training set.
    """
    indices = list(range(dataset_length))

    np.random.seed(seed)
    
    np.random.shuffle(indices)
    
    splits = np.array_split(indices, k)
    
    for i in range(k):
        validation_indices = splits[i]
        training_indices = np.concatenate([splits[j] for j in range(k) if j != i])
        
        yield training_indices, validation_indices