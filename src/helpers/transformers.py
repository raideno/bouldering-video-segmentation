import math
import torch
import numbers

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, slowfast_alpha):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
    backend: str = "pytorch",
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        An x-like Tensor with scaled spatial dims.
    """  # noqa
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    assert backend in ("pytorch", "opencv")
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    if backend == "pytorch":
        return torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode=interpolation, align_corners=False
        )
    elif backend == "opencv":
        return _interpolate_opencv(x, size=(new_h, new_w), interpolation=interpolation)
    else:
        raise NotImplementedError(f"{backend} backend not supported.")

class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return short_side_scale(
            x, self._size, self._interpolation, self._backend
        )
        

def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)
        
class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return uniform_temporal_subsample(
            x, self._num_samples, self._temporal_dim
        )
        
import math
import torch
import numbers

import numpy as np

from ultralytics import YOLO
from typing import Callable, Protocol

yolo_skeleton_keypoints = {
    "nose": 0,
    "left-eye": 1,
    "right-eye": 2,
    "left-ear": 3,
    "right-ear": 4,
    "left-shoulder": 5,
    "right-shoulder": 6,
    "left-elbow": 7,
    "right-elbow": 8,
    "left-wrist": 9,
    "right-wrist": 10,
    "left-hip": 11,
    "right-hip": 12,
    "left-knee": 13,
    "right-knee": 14,
    "left-ankle": 15,
    "right-ankle": 16
}

class TieBreaker(Protocol):
    def __call__(self, raw_predictions: list) -> int:
        pass

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]

def average_center_crop_video_on_person(clip: torch.types.Tensor, crop_size, yolo: YOLO, tie_breaker:TieBreaker=None, verbose=False):
    """
    This transformer will take in a video, detect the person on each frame of the video using an object recognition model and crop the frame on the center of the person in the video.

    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (Channels, Time, Height, Width)
    Returns:
        torch.tensor: central cropping of video clip. Size is (Channels, Time, crop_size, crop_size)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    
    h, w = clip.size(-2), clip.size(-1)
    
    th, tw = crop_size
    
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")
    
    # NOTE: yolo expected shape (Time, Channel, Height, Width) and we have (Channel, Time, Height, Width)
    raw_predictions = yolo(clip.permute(1, 0, 2, 3), verbose=verbose)
    
    cropping_center_points = []
    
    for index, raw_prediction in enumerate(raw_predictions):
        number_of_persons = len(raw_prediction.keypoints.xy)
        
        person_index = 0
        
        if number_of_persons == 0:
            raise ValueError(f"No person detected in the frame {index + 1} of the clip.")
        
        if number_of_persons > 1:
            if tie_breaker is None:
                raise ValueError(f"More than one person detected in the frame {index + 1} of the clip.")
            else:
                person_index = tie_breaker(raw_prediction)
        
        person_keypoints = raw_prediction.keypoints.xy[person_index]
        
        left_elbow_point = person_keypoints[yolo_skeleton_keypoints["left-ankle"]]
        right_elbow_point = person_keypoints[yolo_skeleton_keypoints["right-ankle"]]
        
        center = (left_elbow_point + right_elbow_point) / 2
        
        i = int(torch.round(center[0] - th / 2.0))
        j = int(torch.round(center[1] - tw / 2.0))
        
        cropping_center_points.append((i, j))

    cropping_center_points = np.array(cropping_center_points)
    
    # TODO: implement a variant that crop each frame individually
    i = int(np.mean(cropping_center_points[:, 0]))
    j = int(np.mean(cropping_center_points[:, 1]))
    
    return crop(clip, i, j, th, tw)
        
class AverageCenterCropVideoOnPerson(torch.nn.Module):
    """
    This transformer will take in a video, detect the person on each frame of the video using an object recognition model and crop the clip on the average center of the person in the video.
    """
    def __init__(self, crop_size, yolo_pose_detection_model_path, tie_breaker:TieBreaker=None, verbose=False):
        super(AverageCenterCropVideoOnPerson, self).__init__()
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        
        self.yolo = YOLO(yolo_pose_detection_model_path)
        self.verbose = verbose
        self.tie_breaker = tie_breaker
            
    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (Channels, Time, Height, Width)
        Returns:
            torch.tensor: central cropping of video clip. Size is (Channels, Time, crop_size, crop_size)
        """
        return average_center_crop_video_on_person(clip, self.crop_size, self.yolo, self.tie_breaker, self.verbose)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"
    
class CenterCropVideoOnPerson(torch.nn.Module):
    """
    This transformer will take in a video, detect the person on each frame of the video using an object recognition model and crop each individual frame of clip on the center of the person in the video.
    """
    def __init__(self, crop_size, yolo_pose_detection_model_path, tie_breaker:TieBreaker=None, verbose=False):
        super(AverageCenterCropVideoOnPerson, self).__init__()
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        
        self.yolo = YOLO(yolo_pose_detection_model_path)
        self.verbose = verbose
        self.tie_breaker = tie_breaker
            
    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (Channels, Time, Height, Width)
        Returns:
            torch.tensor: central cropping of video clip. Size is (Channels, Time, crop_size, crop_size)
        """
        return average_center_crop_video_on_person(clip, self.crop_size, self.yolo, self.tie_breaker, self.verbose)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"