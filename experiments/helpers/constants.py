from enum import IntEnum

from helpers.extractors import \
    FeatureExtractor, \
    ResNet3DFeatureExtractor, \
    DinoFeatureExtractor, \
    I3DFeatureExtractor, \
    ClipFeatureExtractor, \
    YoloFeatureExtractor, \
    IJepaFeatureExtractor, \
    S3DKineticsFeatureExtractor, \
    S3DHowTo100MFeatureExtractor, \
    X3DSFeatureExtractor, \
    SlowFastFeatureExtractor, \
    X3DModelType

DATASET_PATH = "/Users/nadir/Documents/research-project-dataset"

VIDEOS_DIRECTORY_NAME = "videos"
ANNOTATIONS_DIRECTORY_NAME = "annotations"
VIDEOS_FRAMES_DIRECTORY_NAME = "videos_frames"

ANNOTATED_IDS_FILE_NAME = "annotated_ids.txt"
UNANNOTATED_IDS_FILE_NAME = "unannotated_ids.txt"

TESTING_PERCENTAGE = 0.3

NUMBER_OF_FOLDS = 5

class FeaturesType(IntEnum):
    TEMPORAL = 0
    FRAME_BY_FRAME = 1

FEATURES_EXTRACTORS = [
    {
        "name": "averaged-dino",
        "features-directory-name": "features/averaged-dino-features",
        "extractor": DinoFeatureExtractor(average_pool=True),
        "#frames": 8,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "dino",
        "features-directory-name": "features/dino-features",
        "extractor": DinoFeatureExtractor(average_pool=False),
        "#frames": 8,
        "type": FeaturesType.FRAME_BY_FRAME,
    },
    {
        "name": "resnet-3d",
        "features-directory-name": "features/resnet-3d-features",
        "extractor": ResNet3DFeatureExtractor(),
        "#frames": 8,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "i3d",
        "features-directory-name": "features/i3d-features",
        "extractor": I3DFeatureExtractor(),
        "#frames": 16,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "clip",
        "features-directory-name": "features/clip-features",
        "extractor": ClipFeatureExtractor(average_pool=False),
        "#frames": 8,
        "type": FeaturesType.FRAME_BY_FRAME,
    },
    # {
    #     "name": "averaged-clip",
    #     "features-directory-name": "features/averaged-clip-features",
    #     "extractor": ClipFeatureExtractor(average_pool=True),
    #     "#frames": 8,
    #     "type": FeaturesType.TEMPORAL,
    # },
    # {
    #     "name": "averaged-yolo",
    #     "features-directory-name": "features/averaged-yolo-features",
    #     "extractor": YoloFeatureExtractor(average_pool=True),
    #     "#frames": 8,
    #     type: FeaturesType.TEMPORAL,
    # },
    {
        "name": "x3d-xs",
        "features-directory-name": "features/x3d-xs-features",
        "extractor": X3DSFeatureExtractor(X3DModelType.XS),
        "#frames": 32,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "x3d-s",
        "features-directory-name": "features/x3d-s-features",
        "extractor": X3DSFeatureExtractor(X3DModelType.S),
        "#frames": 32,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "x3d-m",
        "features-directory-name": "features/x3d-m-features",
        "extractor": X3DSFeatureExtractor(X3DModelType.M),
        "#frames": 32,
        "type": FeaturesType.TEMPORAL,
    },
        {
        "name": "x3d-l",
        "features-directory-name": "features/x3d-l-features",
        "extractor": X3DSFeatureExtractor(X3DModelType.L),
        "#frames": 32,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "yolo",
        "features-directory-name": "features/yolo-features",
        "extractor": YoloFeatureExtractor(average_pool=False),
        "#frames": 8,
        "type": FeaturesType.FRAME_BY_FRAME,
    },
    # {
    #     "name": "averaged-i-jepa",
    #     "features-directory-name": "features/averaged-i-jepa-features",
    #     "extractor": IJepaFeatureExtractor(average_pool=True),
    #     "#frames": 8,
    #     "type": FeaturesType.TEMPORAL,
    # },
    # {
    #     "name": "i-jepa",
    #     "features-directory-name": "features/i-jepa-features",
    #     "extractor": IJepaFeatureExtractor(average_pool=False),
    #     "#frames": 8,
    #     "type": FeaturesType.FRAME_BY_FRAME,
    # },
    {
        "name": "s3d-kinetics",
        "features-directory-name": "features/s3d-kinetics-features",
        "extractor": S3DKineticsFeatureExtractor(),
        "#frames": 16,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "s3d-howto100m",
        "features-directory-name": "features/s3d-howto100m-features",
        "extractor": S3DHowTo100MFeatureExtractor(),
        "#frames": 16,
        "type": FeaturesType.TEMPORAL,
    },
    {
        "name": "slowfast",
        "features-directory-name": "features/slowfast-features",
        "extractor": SlowFastFeatureExtractor(),
        "#frames": 32,
        "type": FeaturesType.TEMPORAL,
    },
]