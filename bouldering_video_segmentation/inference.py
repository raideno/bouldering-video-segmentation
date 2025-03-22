import torch
import tqdm

import numpy as np

from video_dataset.video import read_video
from bouldering_video_segmentation.extractors import FeatureExtractor, X3DFeatureExtractor, X3DModelType
from bouldering_video_segmentation.models import VideoSegmentMlp

NUMBER_OF_CLASSES = 5
DEFAULT_CLASSIFIER = VideoSegmentMlp(output_size=NUMBER_OF_CLASSES)
DEFAULT_FEATURE_EXTRACTOR = X3DFeatureExtractor(model_name=X3DModelType.XS)

SEGMENT_SIZE = 32

def segment_video(
    video_path: str,
    feature_extractor: FeatureExtractor = DEFAULT_FEATURE_EXTRACTOR,
    classifier: torch.nn.Module = DEFAULT_CLASSIFIER,
    verbose: bool=True
):
    video = read_video(video_path)
    
    features = []
    predictions = []

    for i in tqdm.tqdm(iterable=range(0, len(video), SEGMENT_SIZE), desc="[processing-video-segments]:", disable=not verbose):
        segment = video[i:i+SEGMENT_SIZE]
        
        # NOTE: required to be transposed to (Channel, Time, Height, Width)
        segment = segment.transpose(3, 0, 1, 2)
        feature = feature_extractor.transform_and_extract(segment)
        feature = feature.unsqueeze(0)
        prediction = classifier(feature)
        
        features.append(feature)
        predictions.append(prediction)
        
    _, labels = torch.max(torch.stack(predictions), dim=2)
    
    frames_predictions = np.repeat(labels.numpy(), SEGMENT_SIZE)
    
    return frames_predictions