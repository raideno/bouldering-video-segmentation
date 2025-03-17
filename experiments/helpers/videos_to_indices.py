import os
import tqdm
import pickle

import numpy as np

from typing import List
from video_dataset import VideoDataset

def video_segments_mapping_generator(video_dataset: VideoDataset):
    """
    The video dataset need to have a special return_transform that gives the video_id. Look into the code for more details.
    """
    CACHE_FILE_PATH = "video_segments_mapping.cache.pkl"

    def load_cache():
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, 'rb') as f:
                return pickle.load(f)
        return None

    def save_cache(mapping):
        with open(CACHE_FILE_PATH, 'wb') as f:
            pickle.dump(mapping, f)

    # video_segments_mapping = load_cache()
    video_segments_mapping = None

    if video_segments_mapping is None:
        video_segments_mapping = {}

        for i in tqdm.tqdm(iterable=range(len(video_dataset)), desc="[dataset-mappings-construction]:"):
            frame, annotations, video_id, segment_index = video_dataset[i]
            
            if video_id not in video_segments_mapping:
                video_segments_mapping[video_id] = [i]
            else:
                video_segments_mapping[video_id].append(i)
            
        save_cache(video_segments_mapping)
        
    return video_segments_mapping

def videos_to_indices(video_dataset: VideoDataset, videos_ids: List[int]) -> List[int]:
    """
    The video dataset need to have a special return_transform that gives the video_id. Look into the code for more details.
    """
    mapping = video_segments_mapping_generator(video_dataset)
    
    return np.concatenate([mapping[list(mapping.keys())[video_id]] for video_id in videos_ids])
    # return np.concatenate([video_segments_mapping[video_id][1:] for video_id in videos_ids])