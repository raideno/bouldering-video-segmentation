import os
import torch

import numpy as np

from enum import IntFlag, StrEnum, auto

class FilteringMode(IntFlag):
    NO_PERSONLESS = auto()
    NO_NOTHING_CLASS = auto()
    NO_STOPWATCH_CLASS = auto()
    NO_MULTI_CLASS = auto()
    
    ALL = NO_PERSONLESS | NO_NOTHING_CLASS | NO_STOPWATCH_CLASS | NO_MULTI_CLASS
    
    @staticmethod
    def from_string(string: str):
        return FilteringMode.STRING_MAPPINGS[string]
    
    @staticmethod
    def get_mappings():
        return {
            FilteringMode.NO_PERSONLESS: "NO_PERSONLESS",
            FilteringMode.NO_NOTHING_CLASS: "NO_NOTHING_CLASS",
            FilteringMode.NO_STOPWATCH_CLASS: "NO_STOPWATCH_CLASS",
            FilteringMode.NO_MULTI_CLASS: "NO_MULTI_CLASS",
        }
    
    @staticmethod
    def get_str_components(filtering_mode: 'FilteringMode'):
        mappings = FilteringMode.get_mappings()
        
        return [mappings[component] for component in filtering_mode]
    
class FilteringOperator(StrEnum):
    AND = "AND"
    OR = "OR"
        
DEFAULT_FILTERING_MODE = FilteringMode(0)
DEFAULT_FILTERING_OPERATOR = FilteringOperator.OR

def preliminary(filtering_mode: FilteringMode = DEFAULT_FILTERING_MODE, filtering_operator: FilteringOperator = DEFAULT_FILTERING_OPERATOR):
    """
    Prepares and processes datasets by extracting video frames, applying feature extraction, and filtering segments based on specified criteria.

    Parameters
    ----------
    filtering_mode : FilteringMode, optional
        Specifies the filtering criteria for dataset segments. Available options include:
        - FilteringMode.NO_PERSONLESS: Excludes segments with insufficient person detections.
        - FilteringMode.NO_NOTHING_CLASS: Excludes segments labeled as 'nothing'.
        - FilteringMode.NO_STOPWATCH_CLASS: Excludes segments labeled as 'chrono'.
        - FilteringMode.NO_MULTI_CLASS: Excludes segments containing multiple distinct labels.
        - FilteringMode.ALL: Combines all the above filters.
        The default is `DEFAULT_FILTERING_MODE`, which applies no filtering.

    filtering_operator : FilteringOperator, optional
        Determines how multiple filtering criteria are combined. 
        - FilteringOperator.AND: Requires all selected filters to apply for exclusion.
        - FilteringOperator.OR: Requires any one of the selected filters to apply for exclusion.
        The default is `DEFAULT_FILTERING_OPERATOR`, which is `OR`.

    Returns
    -------
    tuple
        A tuple containing:
        - `datasets` (list): The original datasets before filtering.
        - `filtered_datasets` (list): The datasets after applying the specified filtering criteria.
        - `FEATURES_EXTRACTORS` (list): The list of feature extractors used in the dataset processing.

    Notes
    -----
    The function handles:
    - Extracting frames from video files.
    - Aggregating labels for video segments.
    - Caching datasets for improved performance.
    - Filtering segments based on the specified criteria.
    """
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
    
    def get_unwanted_classes_indices(yolo_dataset, unwanted_classes: list[str]):
        """
        Return indices of segments with unwanted classes.
        """
        if not unwanted_classes or all(cls is None for cls in unwanted_classes):
            return []
            
        # NOTE: filter out None values
        unwanted_classes = [cls for cls in unwanted_classes if cls is not None]
        if not unwanted_classes:
            return []
            
        label_encoder = LabelEncoderFactory.get()
        class_indices = label_encoder.transform(unwanted_classes)
        
        # return [i for i in range(len(yolo_dataset)) if yolo_dataset[i][1] in class_indices]
        return [i for i in range(len(yolo_dataset)) if yolo_dataset[i][1][0] in class_indices]

    def get_personless_indices(yolo_dataset):
        """Return indices of segments without sufficient person detections."""
        # NOTE: threshold of 4 as we take a sample of 32 frames, subsample 8 and extract YOLO features
        return [i for i in range(len(yolo_dataset)) if torch.count_nonzero(torch.sum(yolo_dataset[i][0], dim=1)) < 2]
        # return [i for i in range(len(yolo_dataset)) if torch.count_nonzero(torch.sum(yolo_dataset[i][0], dim=1)) < 4]
    
    
    def get_multi_class_indices(yolo_dataset):
        """
        Return indices of segments with multiple classes.
        """
        return [i for i in range(len(yolo_dataset)) if yolo_dataset[i][1][1] > 1]
        
    # --- --- ---
        
    nothing_stopwatch_indices = set()
    personless_indices = set()
    multi_class_indices = set()

    if FilteringMode.NO_NOTHING_CLASS in filtering_mode or FilteringMode.NO_STOPWATCH_CLASS in filtering_mode:
        classes_to_filter = [
            "nothing" if FilteringMode.NO_NOTHING_CLASS in filtering_mode else None,
            "chrono" if FilteringMode.NO_STOPWATCH_CLASS in filtering_mode else None
        ]
        nothing_stopwatch_indices = set(get_unwanted_classes_indices(yolo_dataset, classes_to_filter))

    if FilteringMode.NO_PERSONLESS in filtering_mode:
        personless_indices = set(get_personless_indices(yolo_dataset))

    if FilteringMode.NO_MULTI_CLASS in filtering_mode:
        multi_class_indices = set(get_multi_class_indices(yolo_dataset))

    if filtering_operator == FilteringOperator.OR:
        # NOTE: combine all indices to filter out
        indices_to_filter_out = nothing_stopwatch_indices | personless_indices | multi_class_indices
    elif filtering_operator == FilteringOperator.AND:
        # NOTE: only filter out indices that appear in all active filters
        active_filter_sets = []
        
        if FilteringMode.NO_NOTHING_CLASS in filtering_mode or FilteringMode.NO_STOPWATCH_CLASS in filtering_mode:
            active_filter_sets.append(nothing_stopwatch_indices)
            
        if FilteringMode.NO_PERSONLESS in filtering_mode:
            active_filter_sets.append(personless_indices)
            
        if FilteringMode.NO_MULTI_CLASS in filtering_mode:
            active_filter_sets.append(multi_class_indices)
        
        # NOTE: handle edge case: no active filters
        if not active_filter_sets:
            indices_to_filter_out = set()
        # NOTE: handle edge case: only one active filter
        elif len(active_filter_sets) == 1:
            indices_to_filter_out = active_filter_sets[0]
        # NOTE: normal case: multiple active filters
        else:
            # For AND operation, we need the intersection of all sets
            # (i.e., indices that appear in ALL filter sets)
            indices_to_filter_out = set.intersection(*active_filter_sets)

    indices_to_keep = [i for i in range(len(yolo_dataset)) if i not in indices_to_filter_out]

    # --- --- ---

    filtered_datasets = [torch.utils.data.Subset(dataset, indices_to_keep) for dataset in datasets]
    
    return datasets, filtered_datasets, FEATURES_EXTRACTORS