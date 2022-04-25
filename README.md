# Pedestrian Intent Prediction Using Deep Machine Learning

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Repository for the master thesis project titled "Pedestrian Intent Prediction Using Deep Machine Learning" at the Chalmers University of Technology.

## Abstract
One of the critical requirements for a safe assistive and autonomous driving system is the accurate perception of the ego vehicle's environment. While there have been significant strides in detecting and tracking visible surroundings of the ego vehicle, accurate prediction of vulnerable road users such as pedestrians and cyclists remains a challenge as vulnerable road users can instantly change their direction and speed. Humans make important intuitive decisions based on the interactions in the scene and the sequences of actions to interpret the intent of vulnerable road users. However, the same cannot be assumed for the current assistive and autonomous driving systems, as these intentions are realised through subtle gestures and interactions. Since predicting the future intent of vulnerable road users is essential to warn the driver or automatically perform smoother manoeuvres, our thesis aims to predict pedestrian intent using deep machine learning.

In recent years, the intent prediction problem has been a topic of active research, resulting in several new algorithmic solutions. However, measuring the overall progress towards solving this problem has been difficult. Therefore, this thesis investigates the performance of multiple baseline methods on the joint attention in autonomous driving (JAAD) dataset to tackle this obstacle. Despite achieving state-of-the-art results on curated datasets, most of these methods are developed, disregarding potential deployment in production environments. Our thesis proposes an end-to-end network that attempts to reduce the gap between prototyping and production based on these findings. The proposed end-to-end network predicts the future intent of vulnerable road users up to half a second in the future.

## Datasets
### Joint Attention in Autonomous Driving (JAAD) Dataset [[Link](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/)]:
JAAD is a dataset for studying joint attention in the context of autonomous driving. The focus is on pedestrian and driver behaviors at the point of crossing and factors that influence them. To this end, JAAD dataset provides a richly annotated collection of 346 short video clips (5-10 sec long) extracted from over 240 hours of driving footage. These videos filmed in several locations in North America and Eastern Europe represent scenes typical for everyday urban driving in various weather conditions. Bounding boxes with occlusion tags are provided for all pedestrians making this dataset suitable for pedestrian detection.

Behavior annotations specify behaviors for pedestrians that interact with or require attention of the driver. For each video there are several tags (weather, locations, etc.) and timestamped behavior labels from a fixed list (e.g. stopped, walking, looking, etc.). In addition, a list of demographic attributes is provided for each pedestrian (e.g. age, gender, direction of motion, etc.) as well as a list of visible traffic scene elements (e.g. stop sign, traffic signal, etc.) for each frame.

- **Full Dataset:** http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip
- **Annotations:** https://github.com/ykotseruba/JAAD

### Pedestrian Intent Estimation (PIE) Dataset [[Link](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)]:
PIE is a new dataset for studying pedestrian behavior in traffic. PIE contains over 6 hours of footage recorded in typical traffic scenes with on-board camera. It also provides accurate vehicle information from OBD sensor (vehicle speed, heading direction and GPS coordinates) synchronized with video footage. Rich spatial and behavioral annotations are available for pedestrians and vehicles that potentially interact with the ego-vehicle as well as for the relevant elements of infrastructure (traffic lights, signs and zebra crossings).

There are over 300K labeled video frames with 1842 pedestrian samples making this the largest publicly available dataset for studying pedestrian behavior in traffic.

- **Full Dataset:** http://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/
- **Annotations:** https://github.com/aras62/PIE

## Requirements
The code is based on Python3 (>=3.8) and only supports GPU. There are a few dependencies to run the code. The major libraries are listed as follows:
* Pytorch >= 1.1
* tqdm
* tensorboard >= 2.6

## Installation Guide
To install the anaconda environment, navigate to the repository folder, please update the paths and flags in the script `environment.sh` and run the following command in the terminal:

```
$bash environment.sh
```

## Execution Guide
1. To activate the Conda environment, please run the following command in the terminal:

```
$conda activate $VENV
```

The parameter `$VENV` is the name of the Anaconda environment defined in the file `environment.sh`.

2. Train or Test the network in the Conda environment using the below command in terminal:

```
$python train.py -h     # Command to understand command line arguments
$python test.py -h      # Command to understand command line arguments
```

Presently, there are two different problem formulations for the project: Regression and Classification. Choose the correct file based on the problem formulation you want to try.

3. To deactivate the Conda environment, please run the following command in the terminal:

```
$conda deactivate
```

## Clean-up Guide
To remove the anaconda environment, navigate to the repository folder, and run the following command in the terminal:

```
$conda remove --name $VENV --all
```

The parameter `$VENV` is the name of the Anaconda environment defined in the file `environment.sh`.

## Authors
* Aren Moosakhanian
* Sourab Bapu Sridhar

## Acknowledgements
The setup of this project is based on the [PyTorch project template](https://github.com/victoresque/pytorch-template).

## License
This project is released under the terms of [MIT License](LICENSE).