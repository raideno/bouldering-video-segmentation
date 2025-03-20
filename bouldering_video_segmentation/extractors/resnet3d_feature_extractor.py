import torch
import torchvision

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from bouldering_video_segmentation.utils import UniformTemporalSubsample, ShortSideScale, to_millions
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType, FeatureExtractorNameVersion
    
# SOURCE: https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
class ResNet3DFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        
        self.model.blocks[-1].proj = torch.nn.Identity()
        # del self.model.blocks[-1].output_pool
        
        self.model.eval()
        
    def get_name(self,  version: FeatureExtractorNameVersion = FeatureExtractorNameVersion.LONG):
        return "r3d"
    
    def get_features_type(self):
        return FeaturesType.TEMPORAL
    
    def get_required_number_of_frames(self):
        return 8
    
    def get_number_of_params(self):
        return to_millions(sum(parameter.numel() for parameter in self.model.parameters()))
    
    def get_features_shape(self):
        return (2048)
        
    def transform(self, x):
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ])(x)
        
    def extract_features(self, x):
        with torch.no_grad():
            # NOTE: add batch dimensions
            x = x.unsqueeze(0)

            x = self.model(x)
            
            # NOTE: we remove the added batch dimension
            x = x.flatten()

        return x
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))