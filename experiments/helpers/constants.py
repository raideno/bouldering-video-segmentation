from extractors import \
    FeatureExtractor, \
    ResNet3DFeatureExtractor, \
    DinoFeatureExtractor, \
    I3DFeatureExtractor, \
    ClipFeatureExtractor, \
    YoloFeatureExtractor, \
    IJepaFeatureExtractor, \
    S3DFeatureExtractor, S3DTrainingDataset, \
    X3DSFeatureExtractor, X3DModelType, \
    SlowFastFeatureExtractor, \
    ViVitFeatureExtractor

DATASET_PATH = "/Users/nadir/Documents/research-project-dataset"

VIDEOS_DIRECTORY_NAME = "videos"
ANNOTATIONS_DIRECTORY_NAME = "annotations"
VIDEOS_FRAMES_DIRECTORY_NAME = "videos_frames"

ANNOTATED_IDS_FILE_NAME = "annotated_ids.txt"
UNANNOTATED_IDS_FILE_NAME = "unannotated_ids.txt"

FEATURES_EXTRACTORS: list[FeatureExtractor] = [
    YoloFeatureExtractor(average_pool=False),
    # DinoFeatureExtractor(average_pool=False),
    ResNet3DFeatureExtractor(),
    I3DFeatureExtractor(),
    ClipFeatureExtractor(average_pool=False),
    X3DSFeatureExtractor(X3DModelType.XS),
    X3DSFeatureExtractor(X3DModelType.S),
    X3DSFeatureExtractor(X3DModelType.M),
    X3DSFeatureExtractor(X3DModelType.L),
    S3DFeatureExtractor(S3DTrainingDataset.KINETICS),
    S3DFeatureExtractor(S3DTrainingDataset.HOWTO100M),
    SlowFastFeatureExtractor(),
    # ViVitFeatureExtractor(),
    # --- --- ---
    # ClipFeatureExtractor(average_pool=True),
    # DinoFeatureExtractor(average_pool=True),
    # YoloFeatureExtractor(average_pool=True),
    # --- --- ---
    # IJepaFeatureExtractor(average_pool=True),
    # IJepaFeatureExtractor(average_pool=False),
]