from abc import ABC, abstractmethod

from enum import StrEnum

class FeaturesType(StrEnum):
    TEMPORAL = "TEMPORAL"
    FRAME_BY_FRAME = "FRAME_BY_FRAME"

class FeatureExtractor(ABC):
    @abstractmethod
    def get_features_type(self) -> FeaturesType:
        pass
    
    @abstractmethod
    def get_name(self):
        """
        Return the name of the feature extractor. Might be utilized for saving / loading purposes such as naming a directory, etc.
        """
        pass
    
    @abstractmethod
    def get_required_number_of_frames(self):
        pass
    
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