import torch
import torchvision

from enum import StrEnum

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from bouldering_video_segmentation.utils import UniformTemporalSubsample, ShortSideScale, to_millions
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType, FeatureExtractorNameVersion

class X3DModelType(StrEnum):
    XS = "x3d_xs"
    S = "x3d_s"
    M = "x3d_m"
    L = "x3d_l"

# SOURCE: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
class X3DFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name: X3DModelType):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/pytorchvideo', self.model_name, pretrained=True)
        
        self.__model_transform_params  = {
            X3DModelType.XS: {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            X3DModelType.S: {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            X3DModelType.M: {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            },
            X3DModelType.L: {
                "side_size": 312,
                "crop_size": 312,
                "num_frames": 16,
                "sampling_rate": 5,
            }
        }
        
        self.model.blocks[-1].proj = torch.nn.Identity()
        
        self.model.eval()
        
    def get_name(self, version: FeatureExtractorNameVersion = FeatureExtractorNameVersion.LONG):
        return self.model_name.replace("_", "-")
    
    def get_features_type(self):
        return FeaturesType.TEMPORAL
    
    def get_features_shape(self):
        return (2048)
    
    def get_number_of_params(self):
        return to_millions(sum(parameter.numel() for parameter in self.model.parameters()))
    
    def get_required_number_of_frames(self):
        num_frames = self.__model_transform_params[self.model_name]["num_frames"]
        
        if (num_frames & (num_frames-1) == 0) and num_frames != 0:
            return num_frames
        else:
            power = 1
            while power < num_frames:
                power *= 2
            return power
        
    def transform(self, x):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        transform_params = self.__model_transform_params[self.model_name]

        transform =  torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_samples=transform_params["num_frames"], temporal_dim=1),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(transform_params["crop_size"], transform_params["crop_size"])
            )
        ])
        
        return transform(x)
    
    def extract_features(self, x):
        with torch.no_grad():
            x = x.unsqueeze(0)
            
            return self.model(x).flatten()
        
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))