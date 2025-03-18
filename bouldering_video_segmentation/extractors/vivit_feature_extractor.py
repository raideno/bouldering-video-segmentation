import torch
import torchvision

from transformers import VivitForVideoClassification

from bouldering_video_segmentation.utils import UniformTemporalSubsample
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType

class ViVitFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", torch_dtype=torch.float16)

        self.model.classifier = torch.nn.Identity()

        self.model.eval()
    
    def get_features_type(self) -> FeaturesType:
        return FeaturesType.TEMPORAL
    
    def get_name(self):
        return "vivit"
    
    def get_required_number_of_frames(self):
        return 32
    
    def transform(self, x):
        """
        Expect the clip in the format shape (Channel, Time, Height, Width)
        """
        num_frames = 32
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Lambda(lambda x: x / max(255.0, x.max())),
            # NOTE: the model expects the clip in the format shape (Time, Channel, Height, Width)
            torchvision.transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ])(x)
    
    def extract_features(self, x):
        x = x.unsqueeze(0)
        
        return self.model(x)["logits"][0]
        
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))