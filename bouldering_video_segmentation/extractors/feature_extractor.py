from abc import ABC, abstractmethod

from enum import StrEnum

class FeaturesType(StrEnum):
    TEMPORAL = "TEMPORAL"
    FRAME_BY_FRAME = "FRAME_BY_FRAME"

class FeatureExtractorNameVersion(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

class FeatureExtractor(ABC):
    @abstractmethod
    def get_features_type(self) -> FeaturesType:
        """
        Return the type of features extracted. It can either be temporal or frame by frame.
        """
        pass
    
    @abstractmethod
    def get_name(self, version: FeatureExtractorNameVersion = FeatureExtractorNameVersion.LONG) -> str:
        """
        Return the name of the feature extractor. Might be utilized for saving / loading purposes such as naming a directory, etc.
        """
        pass
    
    @abstractmethod
    def get_required_number_of_frames(self):
        """
        Return the required number of frames to pass for frame extraction.
        """
        pass
    
    @abstractmethod
    def get_features_shape(self):
        """
        Return the shape of the extracted features.
        """
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