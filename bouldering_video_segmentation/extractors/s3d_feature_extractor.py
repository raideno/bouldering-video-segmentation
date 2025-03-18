import os
import torch
import torchvision

from enum import StrEnum

from bouldering_video_segmentation.utils import UniformTemporalSubsample
from bouldering_video_segmentation.extractors.feature_extractor import FeatureExtractor, FeaturesType

from bouldering_video_segmentation.extractors.models.s3d import S3D
from bouldering_video_segmentation.extractors.models.s3dg import S3D as S3DG

class S3DTrainingDataset(StrEnum):
    KINETICS = "kinetics"
    HOWTO100M = "howto100m"
    
def load_model(dataset: S3DTrainingDataset, weights_path: str=None, verbose: bool=False):
    
    DEFAULT_WEIGHTS_PATH = '../extractors-weights/s3d-kinetics400.pt' if dataset == S3DTrainingDataset.KINETICS else '../extractors-weights/s3d-howto100m.pt'
    
    weights_path = weights_path if weights_path is not None else DEFAULT_WEIGHTS_PATH
    
    if dataset == S3DTrainingDataset.KINETICS:
        model = S3D(num_class=400)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        if os.path.isfile(weights_path):
            if verbose:
                print('[s3d]: loading weights.')
                
            weight_dict = torch.load(weights_path, map_location=device)
            model_dict = model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        if verbose:
                            print(' size? ' + name, param.size(), model_dict[name].size())
                else:
                    if verbose:
                        print(' name? ' + name)

            if verbose:
                print('[s3d]: loaded weights.')
        else:
            raise ValueError('No weight file.')
        
        model.fc = torch.nn.Identity()
        
        model.eval()
        
        return model
    elif dataset == S3DTrainingDataset.HOWTO100M:
        network = S3DG(num_classes=512)

        missing_keys, unexpected_keys = network.load_state_dict(state_dict=torch.load(weights_path), strict=False)
        
        if verbose:
            print(f'[s3d]: missing keys: {missing_keys}')
            print(f'[s3d]: unexpected keys: {unexpected_keys}')
            
        if len(missing_keys) > 0:
            raise ValueError('Missing keys.')

        network.eval()
        
        return network
    else:
        raise ValueError('Invalid dataset.')

# SOURCE: https://github.com/kylemin/S3D
# SOURCE: https://github.com/antoine77340/S3D_HowTo100M
class S3DFeatureExtractor(FeatureExtractor):
    def __init__(self, dataset:S3DTrainingDataset=S3DTrainingDataset.KINETICS, weights_path:str=None, verbose:bool=False):
        self.dataset = dataset
        self.verbose = verbose
        self.weights_path = weights_path
        
        if self.dataset != S3DTrainingDataset.KINETICS and self.dataset != S3DTrainingDataset.HOWTO100M:
            raise ValueError('Invalid dataset.')
        
        self.model = load_model(self.dataset, self.weights_path, self.verbose)
        
        self.model.eval()
    
    def get_name(self):
        return f"s3d-{self.dataset.value}"
    
    def get_features_type(self):
        return FeaturesType.TEMPORAL
        
    def get_required_number_of_frames(self):
        return 16
    
    def get_features_shape(self):
        return (1024)
        
    def transform(self, x):
        num_frames = 16
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Lambda(lambda x: x / max(255.0, x.max())),
        ])(x)
        
    def extract_features(self, x):
        if self.dataset == S3DTrainingDataset.KINETICS:
            x = x.unsqueeze(0)
            
            return self.model(x)[0]
        elif self.dataset == S3DTrainingDataset.HOWTO100M:
            x = x.unsqueeze(0)
            
            return self.model(x)["mixed_5c"][0]
        else:
            raise ValueError('Invalid dataset.')

    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))