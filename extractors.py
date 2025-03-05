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

            # NOTE: forward pass through all but last block
            for i in range(len(self.model.blocks) - 2):  
                x = self.model.blocks[i](x)

            # NOTE: the second-to-last block contains the global average pooling layer
            x = self.model.blocks[-2](x)

            # NOTE: flatten to (batch_size, 2048) and global average pooling
            x = x.mean(dim=[2, 3, 4])
            
            # NOTE: we remove the added batch dimension
            x = x.flatten()

        return x
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))
    
# SOURCE: https://github.com/facebookresearch/dinov2
class DinoFeatureExtractor(FeatureExtractor):
    def __init__(self):
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
        return self.model.forward_features(clip.permute(1, 0, 2, 3))["x_norm_clstoken"].mean(dim=0).flatten()

    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))

# SOURCE: https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
class SlowFastFeatureExtractor(FeatureExtractor):
    pass

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

# SOURCE: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
class X3DFeatureExtractor(FeatureExtractor):
    pass

class ClipFeatureExtractor(FeatureExtractor):
    def __init__(self):
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
            return self.model.encode_image(x).mean(dim=0).flatten()

    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))

class YoloFeatureExtractor(FeatureExtractor):
    def __init__(self, num_keypoints=17):
        """
        Initialize YOLO pose estimation model
        :param num_keypoints: Number of keypoints to extract (default is 17 for COCO keypoints)
        """
        self.model = YOLO('weights/yolov8n-pose.pt', verbose=False)
        self.num_keypoints = num_keypoints
        
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
        Extract skeleton keypoints from the input video clip in batch
        
        :param x: Transformed video clip tensor (Time, Channel, Height, Width)
        :return: Tensor of keypoint coordinates or zeros
        """
        results = self.model(x)
        
        # NOTE: default zero tensor if no person detected
        keypoints = torch.zeros(self.num_keypoints * 3)
        
        # NOTE: check results for any frame with a detected person
        for result in results:
            if len(result.keypoints) > 0:
                print("found a person")
                # NOTE: take the first person's keypoints
                first_person_keypoints = result.keypoints[0].xyn.numpy()
                
                # NOTE: flatten keypoints (x, y, confidence for each point)
                keypoints_flat = first_person_keypoints.flatten()
                
                # NOTE: ensure we have exactly num_keypoints * 3 values
                if len(keypoints_flat) >= self.num_keypoints * 3:
                    keypoints = torch.tensor(keypoints_flat[:self.num_keypoints * 3], dtype=torch.float32)
                
                # NOTE: stop if a person is found in any frame
                break
        
        return keypoints
    
    def transform_and_extract(self, x):
        """
        Transform input and extract features in one step
        """
        return self.extract_features(self.transform(x))
    
    
class IJepaFeatureExtractor(FeatureExtractor):
    def __init__(self):
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
        return self.model(x).last_hidden_state.mean(dim=1).mean(dim=0)
    
    def transform_and_extract(self, x):
        return self.extract_features(self.transform(x))
    
class S3DFeatureExtractor(FeatureExtractor):
    pass