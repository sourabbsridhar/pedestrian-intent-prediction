# Implementation of Training

import torch
import argparse
import numpy as np
import sys
import math
import collections
from trainer import Trainer
import model.loss as lossModule
from utils import prepare_device
import model.metric as metricModule
import torch.nn.functional as F
from utils.parse_config import ConfigParser
import model.social_stgcnn_classification as architectureModule
import data.custom_dataset as customDataset
from sklearn.utils.class_weight import compute_class_weight

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
    Entry point for training the experiment.

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
    filename = "saved models/Model 4/checkpoint.pth"
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset.to_device(device)

    model = configuration.initialize_object("model", architectureModule).to(device)
    print("Build Model Architecture and print to console\n: {}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainingDataset, validationDataset = dataset.split_dataset(validationSplit=0.2)

    #Calculate class weights before trainig and setting up loss function
    overallGroundTruth = list()
    for idx_data, (video_name, data) in enumerate(trainingDataset.items()):
        for time_frame, frame in enumerate(data):
            pedestrians = frame.classification.count(1)
            y = frame.y.cuda()[[i for i in range(pedestrians)]][:,2].reshape(pedestrians, 1).long()
            overallGroundTruth.append(y.tolist())

    overallGroundTruth = [pedestrianGroundTruth for videoGroundTruth in overallGroundTruth for frameGroundTruth in videoGroundTruth for pedestrianGroundTruth in frameGroundTruth]
    overallGroundTruth = np.array(overallGroundTruth)
    print("Overall ground truth : {}".format(overallGroundTruth))
    classWeights = compute_class_weight(class_weight="balanced", classes=np.unique(overallGroundTruth), y=overallGroundTruth)
    classWeights = torch.from_numpy(classWeights)
    classWeights = classWeights.cuda()
    print("Class Weights: {}".format(classWeights))

    # Setup loss function
    lossFunction = torch.nn.NLLLoss()#weight=classWeights)

    print("Start training...")
    model.train()
    for idx_data, (video_name, data) in enumerate(trainingDataset.items()):
        print(dataset.get_video_classification_no(video_name))
        sys.stdout.write("\nTrainging {}, Video: {}/{}, Number of frames:{}".format(video_name, idx_data+1, len(trainingDataset.keys()), len(data)))
        for epoch in range(epoch_range):

            if epoch_range > 1:
                sys.stdout.write("\nEpoch: {}/{}".format(epoch+1, epoch_range))

            total_loss = 0
            correct = 0
            total = 0
            video_pedestrians = 0

            for time_frame, frame in enumerate(data):

                pedestrians = frame.classification.count(1)
                video_pedestrians += pedestrians

                optimizer.zero_grad()
                output = model(frame.cuda(), pedestrians, device)

                y = frame.y.cuda()[[i for i in range(pedestrians)]][:,2].reshape(pedestrians, 1).long()

                loss = lossFunction(output, y)
                prediction = y.detach().clone()

                if not prediction.nelement() == 0:
                    total_loss += loss
                    loss.backward()
                    optimizer.step()
                    for i in range(output.size()[0]):
                        prediction[i] = torch.argmax(output[i], dim=0)

                correct = correct + torch.sub(prediction, y).numel() - torch.count_nonzero(torch.sub(prediction, y))
                total = total + torch.sub(prediction, y).numel()

            accuracy = correct / total
            sys.stdout.write(", Total Loss: {:.4f}, Accuracy: {:.4f}, Pedestrians: {}".format(total_loss, accuracy, video_pedestrians))

    sys.stdout.write("\nSaving Model....")
    torch.save(model.state_dict(), filename)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")


    configuration = ConfigParser.from_args(args)
    main(configuration)
