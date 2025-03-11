import os
import torch

from utils import LabelEncoderFactory
from helpers.constants import DATASET_PATH, FEATURES_EXTRACTORS

from cached_dataset import DiskCachedDataset

label_encoder = LabelEncoderFactory.get()

yolo_dataset_index = FEATURES_EXTRACTORS.index(next(filter(lambda x: x["name"] == "yolo", FEATURES_EXTRACTORS)))

yolo_dataset = DiskCachedDataset(
    base_path=os.path.join(DATASET_PATH, FEATURES_EXTRACTORS[yolo_dataset_index]["features-directory-name"]),
)

def with_ignore_classes(classes: list[str], dataset):
    return torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if dataset[i][1] not in label_encoder.transform(classes)])

def with_ignore_pensionless_segments(dataset):
    return torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if torch.count_nonzero(torch.sum(yolo_dataset[i][0], dim=1)) >= 4])