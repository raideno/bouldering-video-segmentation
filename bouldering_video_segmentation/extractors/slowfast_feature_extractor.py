import torch
import torchvision

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from bouldering_video_segmentation.utils import UniformTemporalSubsample, ShortSideScale, PackPathway
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType, FeatureExtractorNameVersion


# SOURCE: https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
class SlowFastFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        
        self.model.blocks[-1].proj = torch.nn.Identity()
        
    def transform(self, x):
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        sampling_rate = 2
        frames_per_second = 30
        slowfast_alpha = 4
        num_clips = 10
        num_crops = 3
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway(slowfast_alpha=slowfast_alpha)
        ])(x)
        
    def get_features_shape(self):
        return (2304)
        
    def get_required_number_of_frames(self):
        return 32
        
    def get_name(self, version: FeatureExtractorNameVersion = FeatureExtractorNameVersion.LONG):
        return "slowfast"
    
    def get_features_type(self):
        return FeaturesType.TEMPORAL
        
    def extract_features(self, x):
        with torch.no_grad():
            slow_pathway, fast_pathway = x
            
            slow_pathway = slow_pathway.unsqueeze(0)
            fast_pathway = fast_pathway.unsqueeze(0)
            
            input = [slow_pathway, fast_pathway]
            
            return self.model(input)[0]
        
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))
    
