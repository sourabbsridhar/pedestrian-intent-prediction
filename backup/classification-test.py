# Implementation of Testing
import math

import torch
import sys
import argparse
import numpy as np
from tqdm import tqdm
import model.loss as lossModule
import model.metric as metricModule
from utils.parse_config import ConfigParser
import data_loader.data_loaders as dataModule
import model.social_stgcnn_classification as architectureModule
import data.custom_dataset as customDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, f1_score, accuracy_score, balanced_accuracy_score

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.double)
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
np.random.seed(SEED)

def main(configuration):
    """
    Entry point for testing the experiment.

    Parameters
    ----------
    configuration   : dict
                      User defined configuration for training the experiment

    Returns
    -------
    None
    """
    epoch_range = 1
    savePeriod = 1
    filename = "saved models/Model 1/checkpoint.pth"
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = configuration.initialize_object("model", architectureModule).to(device)
    dataset.to_device(device)

    trainingDataset, validationDataset = dataset.split_dataset(validationSplit=0.2)

    print("Loading Model {}...".format(filename))
    model.load_state_dict(torch.load(filename))

    #Calculate class weights before trainig and setting up loss function
    overallPrediction = list()
    overallGroundTruth = list()
    overallGroundTruthTesting = list()

    for idx_data, (video_name, data) in enumerate(validationDataset.items()):
        for time_frame, frame in enumerate(data):
            pedestrians = frame.classification.count(1)
            y = frame.y.cuda()[[i for i in range(pedestrians)]][:,0].reshape(pedestrians, 1).long()
            overallGroundTruth.append(y.tolist())

    overallGroundTruth = [pedestrianGroundTruth for videoGroundTruth in overallGroundTruth for frameGroundTruth in videoGroundTruth for pedestrianGroundTruth in frameGroundTruth]
    overallGroundTruth = np.array(overallGroundTruth)
    classWeights = compute_class_weight(class_weight="balanced", classes=np.unique(overallGroundTruth), y=overallGroundTruth)
    classWeights = torch.from_numpy(classWeights)
    print("Class Weights: {}".format(classWeights))

    print("Start testing...")
    model.eval()
    correct_each_prediction = [0]
    total_each_prediction = [0]
    correct = 0
    total = 0

    print("Total number of train videos: {}".format(len(trainingDataset)))
    print("Total number of test videos: {}".format(len(validationDataset)))

    print("Calculating final accuracy...")
    with torch.no_grad():
        for idx_video, (_, video) in enumerate(validationDataset.items()):
            sys.stdout.write("\rTesting video {}/{}".format(idx_video+1, len(validationDataset.keys())))
            sys.stdout.flush()

            for time_frame, frame in enumerate(video):

                pedestrians = frame.classification.count(1)

                output = model(frame.cuda(), pedestrians, device)

                y = frame.y.cuda()[[i for i in range(pedestrians)]][:,0].reshape(pedestrians, 1).long()

                prediction = y.detach().clone()

                if not prediction.nelement() == 0:
                    for i in range(output.size()[0]):
                        prediction[i] = torch.argmax(output[i], dim=0)
                    overallGroundTruthTesting.append(y.tolist())
                    overallPrediction.append(prediction.tolist())

    overallGroundTruthTesting = [pedestrianGroundTruth for videoGroundTruth in overallGroundTruthTesting for frameGroundTruth in videoGroundTruth for pedestrianGroundTruth in frameGroundTruth]
    overallPrediction = [pedestrianPrediction for videoPrediction in overallPrediction for framePrediction in videoPrediction for pedestrianPrediction in framePrediction]
    classWeights = classWeights.detach().cpu().numpy().tolist()
    sampleWeights = [classWeights[individualPrediction] for individualPrediction in overallPrediction]

    overallGroundTruthTesting = np.array(overallGroundTruthTesting)
    overallPrediction = np.array(overallPrediction)
    sampleWeights = np.array(sampleWeights)

    accuracy = accuracy_score(overallGroundTruthTesting, overallPrediction, sample_weight=sampleWeights)
    precisionScore = precision_score(overallGroundTruthTesting, overallPrediction, average='weighted', sample_weight=sampleWeights)
    recallScore = recall_score(overallGroundTruthTesting, overallPrediction, average='weighted', sample_weight=sampleWeights)
    f1Score = f1_score(overallGroundTruthTesting, overallPrediction, average='weighted', sample_weight=sampleWeights)

    print("Overall Accuracy: {}".format(accuracy))
    print("Overall Precision Score: {}".format(precisionScore))
    print("Overall Recall Score: {}".format(recallScore))
    print("Overall F1 Score: {}".format(f1Score))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")

    configuration = ConfigParser.from_args(args)
    main(configuration)