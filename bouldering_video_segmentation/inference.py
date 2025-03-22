import torch
import tqdm

import numpy as np

from enum import StrEnum

from video_dataset.video import read_video

class FeatureExtractorName(StrEnum):
    X3D_XS = "x3d-xs"

class ClassifierType(StrEnum):
    MLP = "mlp"
    LSTM = "lstm"

SEGMENT_SIZE = 32

def segment_video(
    video_path: str,
    feature_extractor_name: FeatureExtractorName = FeatureExtractorName.X3D_XS,
    classifier_type: ClassifierType = ClassifierType.MLP,
    verbose: bool=True
):
    video = read_video(video_path)
    
    extractor,  model = torch.hub.load("raideno/bouldering-video-segmentation", str(classifier_type), backbone_name=str(feature_extractor_name), pretrained=True)
    
    features = []
    predictions = []

    for i in tqdm.tqdm(iterable=range(0, len(video), SEGMENT_SIZE), desc="[processing-video-segments]:", disable=not verbose):
        segment = video[i:i+SEGMENT_SIZE]
        
        # NOTE: required to be transposed to (Channel, Time, Height, Width)
        segment = segment.transpose(3, 0, 1, 2)
        feature = extractor.transform_and_extract(segment)
        feature = feature.unsqueeze(0)
        prediction = model(feature)
        
        features.append(feature)
        predictions.append(prediction)
        
    _, labels = torch.max(torch.stack(predictions), dim=2)
    
    frames_predictions = np.repeat(labels.numpy(), SEGMENT_SIZE)
    
    return frames_predictions