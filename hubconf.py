dependencies = ["torch"]

import os
import torch

from bouldering_video_segmentation.models import VideoSegmentMlp, FullVideoLstm

DEFAULT_NUMBER_OF_CLASSES = 5

def video_segment_mlp(backbone_name:str, number_of_classes:int=DEFAULT_NUMBER_OF_CLASSES, pretrained:bool=False, **kwargs):
    """
    # TODO: complete docstring
    """
    extractor = __get_extractor(backbone_name)

    model = VideoSegmentMlp(output_size=number_of_classes)
    
    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, f"models-weights/mlp.{backbone_name}.pt")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
    
    return model

def video_segment_lstm(backbone_name:str, number_of_classes:int=DEFAULT_NUMBER_OF_CLASSES, pretrained:bool=False, **kwargs):
    """
    # TODO: complete docstring
    """
    extractor = __get_extractor(backbone_name)

    model = FullVideoLstm(
        input_size=extractor.get_features_shape(),
        hidden_size=128,
        # NOTE: the model has been trained on 5 classes, thus the output size is 5 and can't be changed when used with the provided weights
        output_size=number_of_classes
    )

    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, f"models-weights/lstm.{backbone_name}.pt")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
    
    return model

from bouldering_video_segmentation.extractors import \
    FeatureExtractor, \
    ResNet3DFeatureExtractor, \
    DinoFeatureExtractor, \
    I3DFeatureExtractor, \
    ClipFeatureExtractor, \
    YoloFeatureExtractor, \
    IJepaFeatureExtractor, \
    S3DFeatureExtractor, S3DTrainingDataset, \
    X3DFeatureExtractor, X3DModelType, \
    SlowFastFeatureExtractor, \
    ViVitFeatureExtractor

def __get_extractor(backbone_name: str) -> FeatureExtractor:
    FEATURES_EXTRACTORS = {
        "yolo": lambda: YoloFeatureExtractor(average_pool=False, weights_path="extractors-weights/yolo-11n-pose.pt"),
        "dino": lambda: DinoFeatureExtractor(average_pool=False),
        "resnet3d": lambda: ResNet3DFeatureExtractor(),
        "i3d": lambda: I3DFeatureExtractor(weights_path="extractors-weights/i3d.pt"),
        "clip": lambda: ClipFeatureExtractor(average_pool=False),
        "x3d-xs": lambda: X3DFeatureExtractor(X3DModelType.XS),
        "x3d-s": lambda: X3DFeatureExtractor(X3DModelType.S),
        "x3d-m": lambda: X3DFeatureExtractor(X3DModelType.M),
        "x3d-l": lambda: X3DFeatureExtractor(X3DModelType.L),
        "s3d-kinetics": lambda: S3DFeatureExtractor(dataset=S3DTrainingDataset.KINETICS, weights_path="extractors-weights/s3d-kinetics400.pt"),
        "s3d-howto100m": lambda: S3DFeatureExtractor(dataset=S3DTrainingDataset.HOWTO100M, weights_path="extractors-weights/s3d-howto100m.pt"),
        "slowfast": lambda: SlowFastFeatureExtractor(),
        "vivit": lambda: ViVitFeatureExtractor(),
    }
    
    if backbone_name not in FEATURES_EXTRACTORS:
        raise ValueError(f"Backbone must be one of {list(FEATURES_EXTRACTORS.keys())}, but got '{backbone_name}'.")
    
    return FEATURES_EXTRACTORS[backbone_name]()