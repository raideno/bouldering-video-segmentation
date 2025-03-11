# Bouldering Video Segmentation

## Context

## Preview of Results

## Pre-Trained Models

Present the pre trained models & their different backbones.

## Usage & Installation

Show how to install and use one of the pre-trained models on some example video.

## Re Produce Training

This section goal is to make it possible for the research team to add new videos and easily re train the model if needed.

---

This repository contains the code for the research project about "Temporal Video Action Segmentation". The idea is to develop an AI model that can take a indoor bouldering video and segment the video based on a predefined set of actions (action-1, action-2, action-3, action-4).

The code is basically a python module that can be used inside of your python project or through the CLI. The module contains commands and functions to do the following things:

- **Raw Annotations Parsing**: Parse the raw annotations received from the sports team. Takes in an excel sheet of annotations of videos and produces a csv file for each video annotation.
- **Integrate the Output of Raw Annotations Parsing into the Dataset**: After parsing the raw annotations you'll get a set of videos and csv files that annotate these videos. You can automatically add them into the right place and right format using the tool.
- **Features Extraction and Generation**: Since you'll most likely not use the raw videos as features for your models, the tool propose you to convert the videos from which we haven't extracted features to extract and automatically add them into the right place of the dataset and also update everything that needs to be (training and testing splits, etc).
- **Integrity Check**: Check whether the raw annotations, partial dataset, or full dataset are in the correct format and ready to be trained on.
- **Training**: Let you train a model on the dataset.
- **Inference**: Let you run a model on a sample.

## Data Structure

### Raw Annotations Structure

This is the structure of the data that must be sent to use by the research team.

The research team must give us a set of `.mov` videos. The video must be named in the following format: `video-climb_{climb-id}-climber_{climber-id}-bloc_{bloc-number}-angle_{face,profile,etc}.mov`. The order of the different chunks (`climb_{climb-id}`, `climber_{climber-id}`, etc) isn't important.

- `climb-id`: An identification string to identify the climb. This can be any alpha numeric character (no spaces or special characters, only alphabet and numbers). It'll be reused in the annotation file to refer to the climb so be careful to reuse the same.
- `climber-id`: An identification string to identify the climber. It can be any alpha numeric character. It'll be reused in the climbers.csv file in case more details want to be added to the climber, ideally it is better to use the climber firstname + lastname. This information is kind of not necessary and redundent but we are leaving it here for the sake of the research teams and in order for them to be easier to find them selves in the annotations especially during the annotation session.
- `bloc-number`: An integer to identify on which bloc the climb has been made. Same as above, the information is kind of redudent since it can be deduced from the climb id but we are going to leave it anyways.
- `angle`: can be a camera id or anything else, in our case we have two pov for the climbs one from the face and another from the profile.

Besides the videos, the annotations must be given in an excel file, where each climb is annotated in a sheet that has as a name the `climb-{climb-od}`.
On top of that and optionally the researchers can provide an excel with a single sheet that describes the climbers using their climber-id. The following columns are considered and any other column is welcome.

## How to use

### First Way

1. Clone the repository into your computer.
2. Install poetry: `pip install poetry`.
3. Navigate into the project's directory.
4. Install the required dependencies `poetry install`.
5. Start using the project: `poetry run manip-chambery --help`

### Second Way

1. Install the project: `pip install manip_chambery`.
2. Start using the project: `manip-chambery --help` or `manip_chambery --help`.
