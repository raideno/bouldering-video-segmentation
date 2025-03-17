import torch
import torchvision

from transformers import AutoModel, AutoProcessor
from utils import UniformTemporalSubsample

from extractors.feature_extractor import FeatureExtractor, FeaturesType

class IJepaFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool):
        self.average_pool = average_pool
        self.model_id = "facebook/ijepa_vith14_1k"
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)

        self.model.eval()
        
    def get_name(self):
        if self.average_pool:
            return "averaged-ijepa"
        else:
            return "ijepa"
    
    def get_features_type(self):
        return FeaturesType.FRAME_BY_FRAME
    
    def get_required_number_of_frames(self):
        return 8
    
    def transform(self, x):
        num_frames = 8

        return torchvision.transforms.Compose([
            # NOTE: change shape to (Time, Channel, Height, Width)
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
            torchvision.transforms.Lambda(lambda x: x / max(255.0, x.max())),
            torchvision.transforms.Lambda(lambda x: self.processor(x, return_tensors="pt")["pixel_values"])
        ])(x)
    
    @torch.no_grad()
    def extract_features(self, x):
        if self.average_pool:
            return self.model(x).last_hidden_state.mean(dim=1).mean(dim=0).flatten()
        else:
            return self.model(x).last_hidden_state.mean(dim=1)
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))