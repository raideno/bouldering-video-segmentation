import os
import tqdm
import pickle

import numpy as np

from helpers.constants import \
    FeaturesType, \
    DATASET_PATH, \
    VIDEOS_DIRECTORY_NAME, \
    ANNOTATIONS_DIRECTORY_NAME, \
    VIDEOS_FRAMES_DIRECTORY_NAME, \
    ANNOTATED_IDS_FILE_NAME, \
    UNANNOTATED_IDS_FILE_NAME, \
    TESTING_PERCENTAGE, \
    NUMBER_OF_FOLDS, \
    FEATURES_EXTRACTORS

from video_dataset import VideoDataset
from video_dataset.padder import LastValuePadder
from video_dataset.dataset import VideoShapeComponents
from video_dataset.video import VideoFromVideoFramesDirectory
from video_dataset.preprocessor import extract_frames_from_videos
from video_dataset.annotations import AnnotationsFromSegmentLevelCsvFileAnnotations

from enum import IntEnum
from utils import LabelEncoderFactory
from cached_dataset import DiskCachedDataset

label_encoder = LabelEncoderFactory.get()

def __aggregate_labels(str_labels):
    labels = label_encoder.transform(str_labels)
    
    unique_elements, counts = np.unique(labels, return_counts=True)

    max_count_index = np.argmax(counts)

    most_frequent_element = unique_elements[max_count_index]
    
    return most_frequent_element

class DatasetVariants(IntEnum):
    UNANNOTATED = 3
    ANNOTATED = 4

def dataset_with_transform(transform, variant, step):
    if variant == DatasetVariants.UNANNOTATED:
        ids_file = UNANNOTATED_IDS_FILE_NAME
        allow_undefined_annotations = True
    elif variant == DatasetVariants.ANNOTATED:
        ids_file = ANNOTATED_IDS_FILE_NAME
        allow_undefined_annotations = False
    else:
        raise ValueError("Unknown dataset variant")
    
    return VideoDataset(
        annotations_dir=os.path.join(DATASET_PATH, ANNOTATIONS_DIRECTORY_NAME),
        videos_dir=os.path.join(DATASET_PATH, VIDEOS_FRAMES_DIRECTORY_NAME),
        ids_file=os.path.join(DATASET_PATH, ids_file),
        segment_size=32,
        video_processor=VideoFromVideoFramesDirectory,
        annotations_processor=AnnotationsFromSegmentLevelCsvFileAnnotations,
        annotations_processor_kwargs={"fps": 25, "delimiter": ","},
        video_shape=(VideoShapeComponents.CHANNELS, VideoShapeComponents.TIME, VideoShapeComponents.HEIGHT, VideoShapeComponents.WIDTH),
        step=step,
        # padder=LastValuePadder(),
        frames_transform=transform,
        annotations_transform=__aggregate_labels,
        overlap=0,
        allow_undefined_annotations=allow_undefined_annotations
    )
    
def derive_step_from_required_number_of_frames(required_number_of_frames, segment_size):
    return int(np.ceil(segment_size / required_number_of_frames))

def video_segments_mapping_generator():
    CACHE_FILE_PATH = "video_segments_mapping.cache.pkl"

    def load_cache():
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, 'rb') as f:
                return pickle.load(f)
        return None

    def save_cache(mapping):
        with open(CACHE_FILE_PATH, 'wb') as f:
            pickle.dump(mapping, f)

    video_segments_mapping = load_cache()

    if video_segments_mapping is None:
        video_segments_mapping = {}
        dataset = VideoDataset(
            annotations_dir=os.path.join(DATASET_PATH, ANNOTATIONS_DIRECTORY_NAME),
            videos_dir=os.path.join(DATASET_PATH, VIDEOS_FRAMES_DIRECTORY_NAME),
            ids_file=os.path.join(DATASET_PATH, ANNOTATED_IDS_FILE_NAME),
            segment_size=32,
            video_processor=VideoFromVideoFramesDirectory,
            annotations_processor=AnnotationsFromSegmentLevelCsvFileAnnotations,
            annotations_processor_kwargs={"fps": 25, "delimiter": ","},
            load_videos=False,
            load_annotations=False,
            verbose=False
        )

        for i in tqdm.tqdm(iterable=range(len(dataset)), desc="[dataset-mappings-construction]:"):
            frame, annotations, (video_index, video_id, video_starting_frame) = dataset[i]
            
            if video_id not in video_segments_mapping:
                video_segments_mapping[video_id] = [i]
            else:
                video_segments_mapping[video_id].append(i)
            
        save_cache(video_segments_mapping)
        
    return video_segments_mapping

def preparations():
    extract_frames_from_videos(
        videos_dir=os.path.join(DATASET_PATH, VIDEOS_DIRECTORY_NAME),
        output_dir=os.path.join(DATASET_PATH, VIDEOS_FRAMES_DIRECTORY_NAME),
    )
    
    datasets = [
        dataset_with_transform(
            extractor["extractor"].transform_and_extract,
            DatasetVariants.ANNOTATED,
            derive_step_from_required_number_of_frames(extractor["#frames"], 32)
        ) for extractor in FEATURES_EXTRACTORS
    ]
    
    video_segments_mapping = video_segments_mapping_generator()
    
    disk_cached_datasets = []
    
    for dataset, extractor in zip(datasets, FEATURES_EXTRACTORS):
        print(f"[caching-dataset]({extractor['name']}):")
        disk_cached_dataset = DiskCachedDataset.load_dataset_or_cache_it(
            dataset=dataset, 
            base_path=os.path.join(DATASET_PATH, extractor["features-directory-name"]),
            verbose=True
        )
        
        disk_cached_datasets.append(disk_cached_dataset)
    
    return {
        "datasets": datasets,
        "FEATURES_EXTRACTORS": FEATURES_EXTRACTORS,
        "video_segments_mapping": video_segments_mapping,
        "disk_cached_datasets": disk_cached_datasets
    }