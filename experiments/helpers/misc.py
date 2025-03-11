import os

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from typing import Generator, Tuple, List
from helpers.constants import \
    DATASET_PATH, \
    ANNOTATED_IDS_FILE_NAME, \
    TESTING_PERCENTAGE, \
    FEATURES_EXTRACTORS

from helpers.preparations import video_segments_mapping_generator

def split_generator(k: int, seed: int = 42) -> Generator[Tuple[List[str], List[str]], None, None]:
    annotated_videos_ids = np.loadtxt(os.path.join(DATASET_PATH, ANNOTATED_IDS_FILE_NAME), dtype=str)

    training_videos_ids = np.random.choice(annotated_videos_ids, int(len(annotated_videos_ids) * (1 - TESTING_PERCENTAGE)), replace=False)
    testing_videos_ids = np.setdiff1d(annotated_videos_ids, training_videos_ids)

    np.random.seed(seed)
    
    np.random.shuffle(annotated_videos_ids)
    
    splits = np.array_split(annotated_videos_ids, k)
    
    for i in range(k):
        testing_videos_ids = splits[i]
        training_videos_ids = np.concatenate([splits[j] for j in range(k) if j != i])
        
        yield training_videos_ids, testing_videos_ids
        
def videos_to_indices(videos_ids) -> List[int]:
    video_segments_mapping = video_segments_mapping_generator()
    
    return np.concatenate([video_segments_mapping[video_id][1:] for video_id in videos_ids])

def aggregate_folds_histories(folds_histories):
    best_scores = []

    for index, extractor in enumerate(FEATURES_EXTRACTORS):
        model_training_stats = []
        model_testing_stats = []
        
        for fold_index, fold_histories in enumerate(folds_histories):
            history, best_training_accuracy, best_validation_accuracy, best_epoch = fold_histories[index]
            
            model_testing_stats.append(best_validation_accuracy)
            model_training_stats.append(best_training_accuracy)
            
        average_best_validation_score = np.mean(model_testing_stats)
        average_best_training_score = np.mean(model_training_stats)
        
        variance_best_training_score = np.var(model_training_stats)
        variance_best_validation_score = np.var(model_testing_stats)
        
        best_scores.append((average_best_training_score, average_best_validation_score, variance_best_training_score, variance_best_validation_score))
        
    validation_accuracies = [score[1] * 100 for score in best_scores]
    training_accuracies = [score[0] * 100 for score in best_scores]
    
    histories = folds_histories[-1]
    
    return histories, best_scores, validation_accuracies, training_accuracies

def plot_model_performances(model_names, training_accuracies, validation_accuracies):
    markers = list(Line2D.markers.keys())
    markers = markers[:len(model_names)]

    plt.figure(figsize=(10, 6))

    for i, name, val_acc, train_acc, marker in zip(range(len(model_names)), model_names, validation_accuracies, training_accuracies, markers):
        plt.scatter(val_acc, train_acc, label=name, marker=marker, s=100)
        plt.text(val_acc, train_acc - 2, name, ha='center', fontsize=10, fontweight='bold')

    plt.xlabel("Validation Accuracy (%)")
    plt.ylabel("Training Accuracy (%)")

    plt.title("Comparison of Model Accuracies")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()