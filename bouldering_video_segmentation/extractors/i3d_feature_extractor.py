import torch
import torchvision

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

from bouldering_video_segmentation.extractors.models.i3d import InceptionI3d
from bouldering_video_segmentation.utils import UniformTemporalSubsample, to_millions
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType, FeatureExtractorNameVersion

DEFAULT_WEIGHTS_PATH = '../extractors-weights/i3d.pt'

# SOURCE: https://github.com/google-deepmind/kinetics-i3d
class I3DFeatureExtractor(FeatureExtractor):
    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH, verbose=True):
        self.verbose = verbose
        self.weights_path = weights_path
        
        self.model = InceptionI3d(
            num_classes=400,
            in_channels=3
        )
        
        missing_keys = self.model.load_state_dict(torch.load(self.weights_path))
        
        if self.verbose:
            print(f"[missing-keys]: {missing_keys}")
        
        self.model.eval()
        
    def get_number_of_params(self):
        return to_millions(sum(parameter.numel() for parameter in self.model.parameters()))
        
    def get_features_shape(self):
        return (1024)
        
    def get_features_type(self):
        return FeaturesType.TEMPORAL
        
    def get_name(self, version: FeatureExtractorNameVersion = FeatureExtractorNameVersion.LONG):
        return "i3d"
    
    def get_required_number_of_frames(self):
        return 16
        
    def transform(self, x):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        num_frames = 16
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Resize((224, 224)),
            NormalizeVideo(mean, std),
        ])(x)
        
    def extract_features(self, x):
        x = x.unsqueeze(0)
        
        return self.model.extract_features(x).flatten()
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))