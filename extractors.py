import os
import torch
import open_clip
import torchvision

from ultralytics import YOLO
from models.i3d import InceptionI3d

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from abc import ABC, abstractmethod

from transformers import AutoModel, AutoProcessor
from utils import UniformTemporalSubsample, ShortSideScale

class FeatureExtractor(ABC):
    @abstractmethod
    def transform(self, x):
        """
        Expect the clip in the format shape (Channel, Time, Height, Width)
        """
        pass
    
    @abstractmethod
    def extract_features(self, x):
        """
        Expected x to have passed through a transformation first.
        """
        pass
    
    @abstractmethod
    def transform_and_extract(self, x):
        """
        Expect the clip in the format shape (Channel, Time, Height, Width)
        """
        pass
    
# SOURCE: https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
class ResNet3DFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        
        self.model.blocks[-1].proj = torch.nn.Identity()
        # del self.model.blocks[-1].output_pool
        
        self.model.eval()
        
    def transform(self, x):
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        
        return torchvision.transforms.Compose([
            # UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
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
    
# SOURCE: https://github.com/facebookresearch/dinov2
class DinoFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool:bool):
        self.average_pool = average_pool
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        
        self.model.eval()
        
    def transform(self, x):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        num_frames = 8
        
        return torchvision.transforms.Compose([
            # UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
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

# SOURCE: https://github.com/google-deepmind/kinetics-i3d
class I3DFeatureExtractor(FeatureExtractor):
    def __init__(self, verbose=True):
        self.model = InceptionI3d(
            num_classes=400,
            in_channels=3
        )
        
        missing_keys = self.model.load_state_dict(torch.load('weights/i3d.pt'))
        
        if verbose:
            print(f"[missing-keys]: {missing_keys}")
        
        self.model.eval()
        
    def transform(self, x):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        num_frames = 8
        
        return torchvision.transforms.Compose([
            # UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Resize((224, 224)),
            NormalizeVideo(mean, std),
        ])(x)
        
    def extract_features(self, x):
        x = x.unsqueeze(0)
        
        return self.model.extract_features(x).flatten()
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))    

class ClipFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool:bool):
        self.average_pool = average_pool
        self.model, preprocess_1, preprocess_2 = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        
        # print("[self.model]:")
        # print(self.model)
        # print("[_]:")
        # print(_)
        # print("[self.preprocess]:")
        # print(self.preprocess)
        
        self.model.eval()
        
    def transform(self, x):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        num_frames = 8
        
        return torchvision.transforms.Compose([
            # UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
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

class YoloFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool:bool):
        """
        Initialize YOLO pose estimation model
        :param num_keypoints: Number of keypoints to extract (default is 17 for COCO keypoints)
        """
        self.num_key_points = 17
        self.average_pool = average_pool
        self.model = YOLO('weights/yolov8n-pose.pt', verbose=False)
        
    def transform(self, x):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        num_frames = 8
        
        return torchvision.transforms.Compose([
            # UniformTemporalSubsample(num_frames),
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Resize((640, 640)),
            torchvision.transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
        ])(x)
        
    def extract_features(self, x):
        """
        Extract skeleton keypoints from the input video clip in batch for each frame.
        
        :param x: Transformed video clip tensor (Time, Channel, Height, Width)
        :return: Tensor of keypoint coordinates for each frame, or zeros for frames with no detected person
        """
        results = self.model(x, verbose=False)
        
        # Initialize a list to hold keypoints for each frame
        keypoints_list = []
        
        # Iterate through the results for each frame
        for result in results:
            # Default zero tensor if no person detected
            keypoints = torch.zeros(self.num_key_points * 3)
            
            if len(result.keypoints) > 0:
                # Take the first person's keypoints
                first_person_keypoints = result.keypoints[0].xyn.numpy()
                
                # Flatten keypoints (x, y, confidence for each point)
                keypoints_flat = first_person_keypoints.flatten()
                
                # Ensure we have exactly num_keypoints * 3 values
                if len(keypoints_flat) >= self.num_key_points * 3:
                    keypoints = torch.tensor(keypoints_flat[:self.num_key_points * 3], dtype=torch.float32)
            
            # Append the keypoints (or zero vector) for this frame
            keypoints_list.append(keypoints)
        
        # Convert the list of keypoints to a tensor (Time, num_key_points * 3)
        keypoints_tensor = torch.stack(keypoints_list)
        
        # If average_pool is True, compute the average position of the keypoints across frames
        if self.average_pool:
            # Compute the average of the keypoints across time, ignoring the zero vectors
            non_zero_keypoints = keypoints_tensor[keypoints_tensor.sum(dim=-1) != 0]  # Remove zero vectors
            if len(non_zero_keypoints) > 0:
                keypoints_avg = non_zero_keypoints.mean(dim=0)
            else:
                keypoints_avg = torch.zeros(self.num_key_points * 3)  # If no keypoints detected, return a zero vector
            return keypoints_avg
        
        return keypoints_tensor

    def transform_and_extract(self, x):
        """
        Transform input and extract features in one step
        """
        return self.extract_features(self.transform(x))
    
    
class IJepaFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool):
        self.average_pool = average_pool
        self.model_id = "facebook/ijepa_vith14_1k"
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)

        self.model.eval()
    
    def transform(self, x):
        return torchvision.transforms.Compose([
            # NOTE: change shape to (Time, Channel, Height, Width)
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
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
    

from models.s3d import S3D

# SOURCE: https://github.com/kylemin/S3D?tab=readme-ov-file
class S3DKineticsFeatureExtractor(FeatureExtractor):
    def __init__(self, verbose:bool=False):
        self.verbose = verbose
        self.model = S3D(num_class=400)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        weights_file = 'weights/s3d_kinetics400.pt'
        
        if os.path.isfile(weights_file):
            if self.verbose:
                print('[s3d]: loading weights.')
                
            weight_dict = torch.load(weights_file, map_location=device)
            model_dict = self.model.state_dict()
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

            if self.verbose:
                print('[s3d]: loaded weights.')
        else:
            raise ValueError('No weight file.')
        
        self.model.fc = torch.nn.Identity()
        
        self.model.eval()
        
    def transform(self, x):
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Lambda(lambda x: x / max(255.0, x.max())),
        ])(x)
        
    def extract_features(self, x):
        x = x.unsqueeze(0)
        
        return self.model(x)[0]

    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))
    
from models.s3dg import S3D as S3DG

# SOURCE: https://github.com/antoine77340/S3D_HowTo100M
class S3DHowTo100MFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.network = S3DG(
            dict_path='weights/s3d_dict.npy',
            num_classes=512
        )

        self.network.load_state_dict(torch.load('weights/s3d_howto100m.pth'))

        self.network.eval()
        
    def transform(self, x):
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Lambda(lambda x: x / max(255.0, x.max())),
        ])(x)
        
    def extract_features(self, x):
        x = x.unsqueeze(0)
        
        return self.network(x)["mixed_5c"][0]
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))

# SOURCE: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
class X3DSFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model_name = 'x3d_s'
        self.model = torch.hub.load('facebookresearch/pytorchvideo', self.model_name, pretrained=True)
        
        self.model.blocks[-1].proj = torch.nn.Identity()
        
        
        self.model.eval()
        
    def transform(self, x):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        model_transform_params  = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            }
        }

        transform_params = model_transform_params[self.model_name]

        transform =  torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
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

from utils import PackPathway

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
        
    def extract_features(self, x):
        with torch.no_grad():
            slow_pathway, fast_pathway = x
            
            slow_pathway = slow_pathway.unsqueeze(0)
            fast_pathway = fast_pathway.unsqueeze(0)
            
            input = [slow_pathway, fast_pathway]
            
            return self.model(input)[0]
        
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))