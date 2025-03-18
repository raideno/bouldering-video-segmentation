# Bouldering Video (Action) Segmentation

This repository contains the code related to my 120 hours research project that consist in Bouldering Video (Action) Segmentation.

Below you'll be able to find instructions related to how to train and do inference using the model / repository's code. For a detailed report on the model and the approach that was used, please refer to the [PDF report](https://google.com) of the research project.

## Pre-Trained Models

| Extractor       | Weights                        | Mlp Accuracy | Weights                        | Lstm Accuracy | Weights                        |
| --------------- | ------------------------------ | ------------ | ------------------------------ | ------------- | ------------------------------ |
| Resnet18        | [Download](https://github.com) | 75 ± 1       | [Download](https://github.com) | 75 ± 1        | [Download](https://github.com) |
| Resnet50        | [Download](https://github.com) | 75 ± 1       | [Download](https://github.com) | 75 ± 1        | [Download](https://github.com) |
| Resnet101       | [Download](https://github.com) | 75 ± 1       | [Download](https://github.com) | 75 ± 1        | [Download](https://github.com) |
| EfficientNet-B0 | [Download](https://github.com) | 75 ± 1       | [Download](https://github.com) | 75 ± 1        | [Download](https://github.com) |
| EfficientNet-B1 | [Download](https://github.com) | 75 ± 1       | [Download](https://github.com) | 75 ± 1        | [Download](https://github.com) |

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
