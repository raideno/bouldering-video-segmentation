import torch
import open_clip
import torchvision

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

from bouldering_video_segmentation.utils import UniformTemporalSubsample
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType

class ClipFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool:bool):
        self.average_pool = average_pool
        self.model, preprocess_1, preprocess_2 = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        
        self.model.eval()
        
    def get_features_type(self):
        return FeaturesType.FRAME_BY_FRAME
        
    def get_name(self):
        if self.average_pool:
            return "averaged-clip"
        else:
            return "clip"
        
    def get_required_number_of_frames(self):
        return 8
    
    def get_features_shape(self):
        return (self.get_required_number_of_frames(), 512)
        
    def transform(self, x):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        num_frames = 8
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Resize((224, 224)),
            NormalizeVideo(mean, std),
            torchvision.transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
        ])(x)
    
    def extract_features(self, x):
        with torch.no_grad():
            if self.average_pool:
                return self.model.encode_image(x).mean(dim=0).flatten()
            else:
                return self.model.encode_image(x)

    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))