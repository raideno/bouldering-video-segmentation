# Bouldering Video (Action) Segmentation

This repository contains the code related to my 120 hours research project that consist in Bouldering Video (Action) Segmentation.

Below you'll be able to find instructions related to how to train and do inference using the model / repository's code. For a detailed report on the model and the approach that was used, please refer to the [PDF report](https://google.com) of the research project.

## Pre-Trained Models

A detailed list of pre-trained models and their weights can be found in **[models-weights/README.md](models-weights/README.md)**. A resumed list can also be found just below.

| Backbone Name | Backbone Type    | Model | Accuracy      | Download Link                                                                                                         |
| :------------ | :--------------- | :---- | :------------ | :-------------------------------------------------------------------------------------------------------------------- |
| yolo          | `FRAME_BY_FRAME` | `MLP` | 61.68% ± 1.94 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.yolo.pt)     |
| dino          | `FRAME_BY_FRAME` | `MLP` | 80.00% ± 0.94 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.dino.pt)     |
| r3d           | `TEMPORAL`       | `MLP` | 84.26% ± 2.92 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.r3d.pt)      |
| i3d           | `TEMPORAL`       | `MLP` | 77.47% ± 4.82 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.i3d.pt)      |
| clip          | `FRAME_BY_FRAME` | `MLP` | 74.39% ± 0.37 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.clip.pt)     |
| x3d-xs        | `TEMPORAL`       | `MLP` | 81.65% ± 1.87 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.x3d-xs.pt)   |
| x3d-s         | `TEMPORAL`       | `MLP` | 83.31% ± 2.14 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.x3d-s.pt)    |
| x3d-m         | `TEMPORAL`       | `MLP` | 84.95% ± 3.91 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.x3d-m.pt)    |
| x3d-l         | `TEMPORAL`       | `MLP` | 82.77% ± 2.74 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.x3d-l.pt)    |
| s3d-k         | `TEMPORAL`       | `MLP` | 77.01% ± 2.32 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.s3d-k.pt)    |
| s3d-h         | `TEMPORAL`       | `MLP` | 60.82% ± 0.55 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.s3d-h.pt)    |
| slowfast      | `TEMPORAL`       | `MLP` | 83.98% ± 2.63 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.slowfast.pt) |
| vivit         | `TEMPORAL`       | `MLP` | 77.81% ± 2.16 | [Download Weights](https://github.com/raideno/bouldering-video-segmentation/blob/main/models-weights/mlp.vivit.pt)    |

## Doing Inference

**Note:** Beside the explanation below, a notebook version with a detailed step by step guidecan also be found in the **[`example.inference.ipynb`](example.inference.ipynb)** notebook.

First of all you need to install the package:

```bash
pip install git+https://github.com/raideno/bouldering-video-segmentation.git
```

Now you'll need to download the weights of the model you wish to use from the [Pre-Trained Models](#pretrained-models) section.

Once this is done you can use the code below in order to test the model on a video. An example video can be found here if necessary: [example-bouldering-video.mp4](https://google.com).

```python
import torch

from video_dataset.video import VideoFromVideoFile

from tas_helpers.visualization import SegmentationVisualizer

from bouldering_video_segmentation.models import VideoSegmentMlp
from bouldering_video_segmentation.extractors import ResNet3DFeatureExtractor

# --- --- --- ---

VIDEO_PATH = "./example-video.mp4"
SEGMENT_SIZE = 32
NUMBER_OF_CLASSES = 5
VIDEO_SEGMENT_MLP_MODEL_WEIGHTS_PATH = "./mlp.resnet.weights.pt"

video_dir_path = "/".join(VIDEO_PATH.split("/")[:-1])
video_name, video_extension = VIDEO_PATH.split("/")[-1].split(".")

# --- --- --- ---

extractor = ResNet3DFeatureExtractor()

model = VideoSegmentMlp(
    input_size=extractor.get_features_shape(),
    # NOTE: the model has been trained on 5 classes, thus the output size is 5 and can't be changed when used with the provided weights
    output_size=NUMBER_OF_CLASSES
)

model = model.load_state_dict(torch.load(VIDEO_SEGMENT_MLP_MODEL_WEIGHTS_PATH))

video = VideoFromVideoFile(
    videos_dir_path=video_dir_path,
    id=video_name,
    video_extension=video_extension
)

# --- --- --- ---

predictions = []

for segment in video.get_segments(segment_size=SEGMENT_SIZE):
    features = extractor.transform_and_extract(segment)

    prediction = model(features)

    predictions.append(prediction)

# --- --- --- ---

SegmentationVisualizer(segment, prediction).show()
```

## Training the Model

0. Start by cloning the github repository: `git clone git@github.com:raideno/bouldering-video-segmentation.git`.
1. Download the dataset available at [https://google.com](https://google.com).
2. Follow the instructions in the notebook [`experiments/preliminary.ipynb`](experiments/preliminary.ipynb).
3. Depending on which model you want to train on, follow the instructions in one of the two following notebooks:
   1. Mlp: [`experiments/mlp.experiments.ipynb`](experiments/mlp.experiments.ipynb).
   2. Lstm: [`experiments/lstm.experiments.ipynb`](experiments/lstm.experiments.ipynb).

## Accessing the Dataset

As the dataset contains pictures and videos of climbers who didn't necessarily agree for their videos to be made publicly available, in order to access the dataset please contact me at: [nadirkichou@hotmail.fr](mailto:nadirkichou@hotmail.fr).
