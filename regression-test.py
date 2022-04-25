# Implementation of Testing
import torch
import sys
import argparse
import numpy as np
from parse_config import ConfigParser
import model.social_stgcnn_regression as architectureModule
import data.datasets.custom_dataset as customDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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

    filename = "saved models/Model 2/checkpoint.pth"
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = configuration.initialize_object("model", architectureModule).to(device)
    dataset.to_device(device)

    trainingDataset, validationDataset = dataset.split_dataset(validationSplit=0.2)  # Testing with validation dataset

    print("Loading Model {}...".format(filename))
    model.load_state_dict(torch.load(filename))

    # Calculate class weights before training and setting up loss function
    overallPrediction15 = list()
    overallPrediction30 = list()
    overallPrediction45 = list()
    overallGroundTruth = list()
    overallGroundTruthTesting15 = list()
    overallGroundTruthTesting30 = list()
    overallGroundTruthTesting45 = list()

    for idx_data, (video_name, data) in enumerate(validationDataset.items()):
        for time_frame, frame in enumerate(data):
            pedestrians = frame.classification.count(1)
            y = frame.y.cuda()[[i for i in range(pedestrians)]][:, 0].reshape(pedestrians, 1).long()
            overallGroundTruth.append(y.tolist())

    print("Start testing...")
    model.eval()

    print("Total number of train videos: {}".format(len(trainingDataset)))
    print("Total number of test videos: {}".format(len(validationDataset)))

    print("Calculating final accuracy...")
    with torch.no_grad():
        for idx_video, (_, video) in enumerate(validationDataset.items()):
            sys.stdout.write("\rTesting video {}/{}".format(idx_video+1, len(validationDataset.keys())))
            sys.stdout.flush()

            for idx_frame, frame in enumerate(video):
                pedestrians = frame.classification.count(1)
                prediction = torch.round(model(frame.cuda(), device))[[i for i in range(pedestrians)]].long()
                y = torch.cat([frame.y.cuda(),
                               torch.ones(size=[prediction.shape[0]-frame.y.shape[0],
                                                frame.y.shape[1]], device=device)*2], dim=0)[[i for i in range(pedestrians)]].long()

                if not prediction.nelement() == 0:
                    overallGroundTruthTesting15.append(y[:, 0].reshape(-1, 1).tolist())
                    overallPrediction15.append(prediction[:, 0].reshape(-1, 1).tolist())
                    overallGroundTruthTesting30.append(y[:, 1].reshape(-1, 1).tolist())
                    overallPrediction30.append(prediction[:, 1].reshape(-1, 1).tolist())
                    overallGroundTruthTesting45.append(y[:, 2].reshape(-1, 1).tolist())
                    overallPrediction45.append(prediction[:, 2].reshape(-1, 1).tolist())


    overallGroundTruthTesting15 = [pedestrianGroundTruth for videoGroundTruth in overallGroundTruthTesting15 for
                                 frameGroundTruth in videoGroundTruth for pedestrianGroundTruth in frameGroundTruth]
    overallPrediction15 = [pedestrianPrediction for videoPrediction in overallPrediction15 for framePrediction in
                         videoPrediction for pedestrianPrediction in framePrediction]

    overallGroundTruthTesting30 = [pedestrianGroundTruth for videoGroundTruth in overallGroundTruthTesting30 for
                                 frameGroundTruth in videoGroundTruth for pedestrianGroundTruth in frameGroundTruth]
    overallPrediction30 = [pedestrianPrediction for videoPrediction in overallPrediction30 for framePrediction in
                         videoPrediction for pedestrianPrediction in framePrediction]

    overallGroundTruthTesting45 = [pedestrianGroundTruth for videoGroundTruth in overallGroundTruthTesting45 for
                                 frameGroundTruth in videoGroundTruth for pedestrianGroundTruth in frameGroundTruth]
    overallPrediction45 = [pedestrianPrediction for videoPrediction in overallPrediction45 for framePrediction in
                         videoPrediction for pedestrianPrediction in framePrediction]

    overallGroundTruthTesting15 = np.array(overallGroundTruthTesting15)
    overallPrediction15 = np.array(overallPrediction15)

    overallGroundTruthTesting30 = np.array(overallGroundTruthTesting30)
    overallPrediction30 = np.array(overallPrediction30)

    overallGroundTruthTesting45 = np.array(overallGroundTruthTesting45)
    overallPrediction45 = np.array(overallPrediction45)

    print("Overall Ground Truth 15f Shape: {}".format(overallGroundTruthTesting15.shape))
    print("Overall Prediction 15f Shape: {}".format(overallPrediction15.shape))
    print("Overall Ground Truth 30f Shape: {}".format(overallGroundTruthTesting30.shape))
    print("Overall Prediction 30f Shape: {}".format(overallPrediction30.shape))
    print("Overall Ground Truth 45f Shape: {}".format(overallGroundTruthTesting45.shape))
    print("Overall Prediction 45f Shape: {}".format(overallPrediction45.shape))


    ## 15
    accuracy15 = accuracy_score(overallGroundTruthTesting15, overallPrediction15)
    precisionScore15 = precision_score(overallGroundTruthTesting15, overallPrediction15, average=None)
    recallScore15 = recall_score(overallGroundTruthTesting15, overallPrediction15, average=None)
    f1Score15 = f1_score(overallGroundTruthTesting15, overallPrediction15, average=None)

    ## 30
    accuracy30 = accuracy_score(overallGroundTruthTesting30, overallPrediction30)
    precisionScore30 = precision_score(overallGroundTruthTesting30, overallPrediction30, average=None)
    recallScore30 = recall_score(overallGroundTruthTesting30, overallPrediction30, average=None)
    f1Score30 = f1_score(overallGroundTruthTesting30, overallPrediction30, average=None)

    ## 45
    accuracy45 = accuracy_score(overallGroundTruthTesting45, overallPrediction45)
    precisionScore45 = precision_score(overallGroundTruthTesting45, overallPrediction45, average=None)
    recallScore45 = recall_score(overallGroundTruthTesting45, overallPrediction45, average=None)
    f1Score45 = f1_score(overallGroundTruthTesting45, overallPrediction45, average=None)
    # aucScore = auc()

    print("Overall Accuracy 15f: {}".format(accuracy15))
    print("Overall Accuracy 30f: {}".format(accuracy30))
    print("Overall Accuracy 45f: {}".format(accuracy45))
    print("Overall Precision 15f Score: {}".format(precisionScore15))
    print("Overall Precision 30f Score: {}".format(precisionScore30))
    print("Overall Precision 45f Score: {}".format(precisionScore45))
    print("Overall Recall 15f Score: {}".format(recallScore15))
    print("Overall Recall 30f Score: {}".format(recallScore30))
    print("Overall Recall 45f Score: {}".format(recallScore45))
    print("Overall F1 15f Score: {}".format(f1Score15))
    print("Overall F1 30f Score: {}".format(f1Score30))
    print("Overall F1 45f Score: {}".format(f1Score45))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")

    configuration = ConfigParser.from_args(args)
    main(configuration)
