import torch
import torchvision

from ultralytics import YOLO

from utils import UniformTemporalSubsample
from extractors.feature_extractor import FeatureExtractor, FeaturesType

class YoloFeatureExtractor(FeatureExtractor):
    def __init__(self, average_pool:bool):
        """
        Initialize YOLO pose estimation model
        :param num_keypoints: Number of keypoints to extract (default is 17 for COCO keypoints)
        """
        self.num_key_points = 17
        self.coordinates_dimension = 2
        self.average_pool = average_pool
        self.model = YOLO('../weights/yolo11n-pose.pt', verbose=False)
        
    def get_name(self):
        if self.average_pool:
            return "averaged-yolo"
        else:
            return "yolo"
        
    def get_features_type(self):
        return FeaturesType.FRAME_BY_FRAME
    
    def get_required_number_of_frames(self):
        return 8
        
    def transform(self, x):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        num_frames = 8
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            UniformTemporalSubsample(num_frames),
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
        
        frames_keypoints = []
        
        for result in results:
            if len(result.keypoints) > 0:
                # keypoints = torch.tensor(result.keypoints[0].xyn.flatten())
                keypoints = result.keypoints[0].xyn.flatten()
            else:
                keypoints = torch.zeros(self.num_key_points * self.coordinates_dimension)
                
            if keypoints.shape[0] < self.num_key_points * self.coordinates_dimension:
                keypoints = torch.zeros(self.num_key_points * self.coordinates_dimension)
                
            frames_keypoints.append(keypoints)
        
        # NOTE: (Time, num_key_points * coordinates_dimension)
        frames_keypoints = torch.stack(frames_keypoints)
        
        if self.average_pool:
            # NOTE: take average across time
            frames_keypoints = frames_keypoints.mean(dim=0)
        
        return frames_keypoints

    def count_persons(self, x):
        """
        Count the number of humans detected in the input video clip.
        
        :param x: Transformed video clip tensor (Time, Channel, Height, Width)
        :return: Tensor of the shape (Time, #Persons), where for each frame we get the number of persons
        """
        results = self.model(x, verbose=False)
        
        # Count the number of persons in each frame
        num_persons_per_frame = torch.tensor([len(result.keypoints) for result in results], dtype=torch.int)
        
        return num_persons_per_frame

    def transform_and_extract(self, x):
        """
        Transform input and extract features in one step
        """
        return self.extract_features(self.transform(x))