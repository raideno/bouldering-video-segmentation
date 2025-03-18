import torch
import torchvision

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

from bouldering_video_segmentation.utils import UniformTemporalSubsample
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType
  
# SOURCE: https://github.com/facebookresearch/dinov2
class DinoFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool:bool):
        self.average_pool = average_pool
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        
        self.model.eval()
        
    def get_features_type(self):
        return FeaturesType.FRAME_BY_FRAME
        
    def get_name(self):
        if self.average_pool:
            return "averaged-dino"
        else:
            return "dino"
        
    def get_required_number_of_frames(self):
        return 8
        
    def transform(self, x):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        num_frames = 8
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Resize((252, 252)),
            NormalizeVideo(mean, std),
        ])(x)
        
    def extract_features(self, clip):
        # return self.model.forward_features(clip.permute(1, 0, 2, 3))["x_norm_clstoken"].mean(dim=0).flatten()
            
        # TODO: should probably be moved to the transform in my opinion
        # NOTE: we need it to be [Time, Channel, Height, Width]
        clip = clip.permute(1, 0, 2, 3)    
        
        with torch.no_grad():
            if self.average_pool:
                return self.model.forward_features(clip)["x_norm_clstoken"].mean(dim=0).flatten()
            else:
                return self.model.forward_features(clip)["x_norm_clstoken"]

    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))