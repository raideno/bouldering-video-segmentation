import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from enum import IntFlag, auto

class FilteringMode(IntFlag):
    PERSONLESS = auto()
    NOTHING_CLASS = auto()
    STOPWATCH_CLASS = auto()
    
    ALL = PERSONLESS | NOTHING_CLASS | STOPWATCH_CLASS
    
DEFAULT_FILTERING_MODE = FilteringMode.PERSONLESS | FilteringMode.NOTHING_CLASS

def preliminary(filtering_mode: FilteringMode = DEFAULT_FILTERING_MODE):
    from video_dataset.preprocessor import extract_frames_from_videos

    from helpers.constants import \
        DATASET_PATH, \
        VIDEOS_DIRECTORY_NAME, \
        VIDEOS_FRAMES_DIRECTORY_NAME \
        
    extract_frames_from_videos(
        videos_dir=os.path.join(DATASET_PATH, VIDEOS_DIRECTORY_NAME),
        output_dir=os.path.join(DATASET_PATH, VIDEOS_FRAMES_DIRECTORY_NAME),
        verbose=True
    )
    
    from helpers.constants import \
    FEATURES_EXTRACTORS, \
    DATASET_PATH, \
    VIDEOS_FRAMES_DIRECTORY_NAME, \
    ANNOTATIONS_DIRECTORY_NAME, \
    ANNOTATED_IDS_FILE_NAME
    from utils import LabelEncoderFactory
    from cached_dataset import DiskCachedDataset

    from video_dataset import VideoDataset, VideoShapeComponents
    from video_dataset.video import VideoFromVideoFramesDirectory
    from video_dataset.annotations import AnnotationsFromSegmentLevelCsvFileAnnotations
    
    label_encoder = LabelEncoderFactory.get()
    
    def __aggregate_labels(string_labels):
        """
        Given a list of string labels, returns the most frequent label and the number of unique labels.
        
        NOTE: The number of unique labels might be used later to determine if a video segment contain a transition in action or not.
        """
        labels = label_encoder.transform(string_labels)
        
        unique_elements, counts = np.unique(labels, return_counts=True)

        max_count_index = np.argmax(counts)

        most_frequent_element = unique_elements[max_count_index]
        
        return most_frequent_element, len(unique_elements)
    
    def returns_transform(sample):
        # sample keys: 'frames', 'annotations', 'video_index', 'video_id', 'starting_frame_number_in_video', 'segment_index'
    
        return sample["frames"], sample["annotations"], sample["video_id"], sample["segment_index"]
    
    datasets = []

    for extractor in FEATURES_EXTRACTORS:
        segment_size = 32
        
        step = VideoDataset.compute_step(segment_size, extractor.get_required_number_of_frames())

        dataset = VideoDataset(
            annotations_dir=os.path.join(DATASET_PATH, ANNOTATIONS_DIRECTORY_NAME),
            videos_dir=os.path.join(DATASET_PATH, VIDEOS_FRAMES_DIRECTORY_NAME),
            ids_file=os.path.join(DATASET_PATH, ANNOTATED_IDS_FILE_NAME),
            segment_size=segment_size,
            step=step,
            video_processor=VideoFromVideoFramesDirectory,
            annotations_processor=AnnotationsFromSegmentLevelCsvFileAnnotations,
            annotations_processor_kwargs={"fps": 25, "delimiter": ","},
            video_shape=(VideoShapeComponents.CHANNELS, VideoShapeComponents.TIME, VideoShapeComponents.HEIGHT, VideoShapeComponents.WIDTH),
            frames_transform=extractor.transform_and_extract,
            annotations_transform=__aggregate_labels,
            verbose=False,
            return_transform=returns_transform
        )

        print(f"[extractor-{extractor.get_name()}]:")

        disk_cached_dataset = DiskCachedDataset.load_dataset_or_cache_it(
            dataset=dataset, 
            base_path=os.path.join(DATASET_PATH, "features", extractor.get_name()),
            verbose=True
        )
        
        datasets.append(disk_cached_dataset)
        
        
    yolo_dataset_index = FEATURES_EXTRACTORS.index(next(filter(lambda x: x.get_name() == "yolo", FEATURES_EXTRACTORS)))

    yolo_dataset = datasets[yolo_dataset_index]

    # --- --- ---
    
    # TODO: make the functions to filter return the indices not to keep instead of the indices to keep

    def extract_segments_without_classless_indices(yolo_dataset, classes: list[str]):
        label_encoder = LabelEncoderFactory.get()
        
        return [i for i in range(len(yolo_dataset)) if yolo_dataset[i][1] not in label_encoder.transform(classes)]

    def extract_segments_with_persons_indices(yolo_dataset):
        label_encoder = LabelEncoderFactory.get()
        
        # NOTE: we chose 4 as we take a sample of 32 frames, we then subsample 8 and extract the yolo features from them, thus 4 is the half of 8
        return [i for i in range(len(dataset)) if torch.count_nonzero(torch.sum(yolo_dataset[i][0], dim=1)) >= 4]

    classes_to_filter = [
        "nothing" if FilteringMode.NOTHING_CLASS in filtering_mode else None,
        "chrono" if FilteringMode.STOPWATCH_CLASS in filtering_mode else None
    ]

    # NOTE: this is the indices to keep
    filtered_indices = list(set(extract_segments_without_classless_indices(yolo_dataset, classes_to_filter) + extract_segments_with_persons_indices(yolo_dataset)))

    # --- --- ---

    filtered_datasets = [torch.utils.data.Subset(dataset, filtered_indices) for dataset in datasets]
    
    return datasets, filtered_datasets, FEATURES_EXTRACTORS