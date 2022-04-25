# Implementation of Training
import inspect

import torch
import argparse
import numpy as np
import sys
import math
import UpdateConfigFile
from parse_config import ConfigParser
import model.social_stgcnn_regression as architectureModule
import data.datasets.custom_dataset as customDataset

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

    epoch_range = 5
    filename = "saved models/Model 2/checkpoint.pth"
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = configuration.initialize_object("model", architectureModule).to(device)
    dataset.to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)#, weight_decay=5e-5)

    trainingDataset, validationDataset = dataset.split_dataset(validationSplit=0.2)

    print("Start training...")
    model.train()
    for idx_data, (video_name, data) in enumerate(trainingDataset.items()):
        # print(dataset.get_video_classification_no(video_name))
        sys.stdout.write("\nTrainging {}, Video: {}/{}, Number of frames:{}, No of pedestrians: {}, No of vehicles: {}"
                         .format(video_name, idx_data+1, len(trainingDataset.keys()), len(data),
                                 dataset.get_video_classification_no(video_name)[0],
                                 dataset.get_video_classification_no(video_name)[1]))
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
                prediction = model(frame.cuda(), device)[[i for i in range(pedestrians)]]
                y = torch.cat([frame.y.cuda(),
                               torch.ones(size=[prediction.shape[0]-frame.y.shape[0],
                                                frame.y.shape[1]], device=device)*2], dim=0)[[i for i in range(pedestrians)]]

                loss = torch.mean((prediction - y) ** 2)

                if not math.isnan(torch.sum(loss).item()):
                    total_loss += loss
                    loss.backward()
                    optimizer.step()

                prediction = torch.round(prediction)
                correct = correct + torch.sub(prediction, y).numel() - torch.count_nonzero(torch.sub(prediction, y))
                total = total + torch.sub(prediction, y).numel()
            accuracy = correct / total
            sys.stdout.write(", MSE: {:.4f}, Accuracy: {:.4f}, "
                             "Pedestrians: {}".format(total_loss, accuracy, video_pedestrians))
        sys.stdout.write("\n")
    sys.stdout.write("\nSaving Model....")
    torch.save(model.state_dict(), filename)

if __name__ == "__main__":
    UpdateConfigFile.Update(sys.argv[2])
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")

    configuration = ConfigParser.from_args(args)
    main(configuration)
