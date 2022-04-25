# Implementation of Data Loader
import torch

import utils
import pickle
import numpy as np
from base import BaseDataLoader
from torchvision import datasets, transforms
from torch_geometric.data import Data, DataLoader
import data.custom_dataset as customDataset

class MnistDataLoader(BaseDataLoader):
    """
    Class implementation for MnistDataLoader.
    The class is inherited from the class BaseDataLoader.
    """
    def __init__(self, dataDirectory, batchSize, shuffle=True, validationSplit=0.0, numberOfWorkers=1, training=True):
        """
        Method to initialize an object of type MnistDataLoader.

        Parameters
        ----------
        self            : MnistDataLoader
                          Instance of the class
        dataDirectory   : str
                          Directory where the data must be loaded
        batchSize       : int
                          Number of samples per batch to load
        suffle          : bool
                          Set to True to have data resuffled at very epoch
        validationSplit : int/float
                          Number of samples/Percentage of dataset set as validation
        numberOfWorkers : int
                          Number of subprocesses used for data loading
        training        : bool
                          Set to True to have data sampled for training process

        Returns
        -------
        self    : MnistDataLoader
                  Initialized object of class MnistDataLoader
        """
        # Download dataset, perform the required transforms and load the dataset into a dataloader
        requiredTransformations = transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                    ])
        self.dataDirectory = dataDirectory
        self.dataset = datasets.MNIST(self.dataDirectory, train=training, download=True, transform=requiredTransformations)
        super().__init__(self.dataset, batchSize, shuffle, validationSplit, numberOfWorkers)


class JaadDataLoader(BaseDataLoader):
    """
    Class implementation for JaadDataLoader.
    The class is inherited from the class BaseDataLoader
    """
    def __init__(self, annotation_path, root, batchSize, shuffle=True, validationSplit=0.1,  numberOfWorkers=1, training=True):
        """
        Method to initialize an object of type JaadDataLoader

        Parameters
        ----------
        self                    : JaadDataLoader
                                  Instance of the class
        annotation_path             : str
                                  Path to the annotations file
        imageDirectoryFormat    : str
                                  Format of the directory containing images
        batchSize               : int
                                  Number of samples per batch to load
        sequenceLength          : int
                                  Length of the frame sequence for each pedestrian
        predictionLength        : int
                                  Length of the prediction frame sequence for each pedestrian
        suffle                  : bool
                                  Set to True to have data resuffled at very epoch
        validationSplit         : int/float
                                  Number of samples/Percentage of dataset set as validation
        numberOfWorkers         : int
                                  Number of subprocesses used for data loading
        training                : bool
                                  Set to True to have data sampled for training process

        Returns
        -------
        self    : JaadDataLoader
                  Initialized object of class JaadDataLoader
        """

        """
        if modulefocus:

            self.dataset, image_Paths = jaad.JAADDataset(split=trainingSplit, isTrain=isTrain, sequenceLength=bufferFrames, datasetPath=self.datasetDir)
            self.labels = self.dataset.__getitem__(0)
            print(self.dataset.__getitem__(0))
            for i, key in self.dataset.__getitem__().keys():
                self.returnValuelabels.update({i: key})

            self.Image_Paths = self.dataset.pop(self.returnValuelabels.popitem())


        else: self.dataset = dataset.MNIST(self.dataDirectory, train=training, download=True, transforms=requiredTransformations)

        super().__init__(self.dataset, batchSize, shuffle, validationSplit, numberOfWorkers)
        """

        # self.dataset = customDataset.JAAD(annotation_path, root)
        # loader = DataLoader(self.dataset, batch_size=batchSize, shuffle=False)
        # print("Dataset loaded...")
        # for data in loader:
        #     print(data)
        #     break

        self.dataset = customDataset.old_JAAD(annotation_path)
        d = self.dataset.__getitem__()

        utils.save_annotations(utils.jaad_annotation_converter(d), "overall_frame_by_frame_database")
        # d = utils.jaad_annotation_converter(self.dataset.__getitem__())

        '''

        self.graph_dataset = {}

        for video_id, video_value in d.items():
            graph_video = {}
            width = video_value['width']
            height = video_value['height']
            print(video_value.keys())
            for frame_id, frame_value in video_value['frames'].items():
                node_position = np.empty(shape=4)
                node_appearance = np.empty(shape=25)
                node_attributes = np.empty(shape=12)
                node_behavior = np.empty(shape=6)
                node_ground_truth = np.empty(shape=3)
                edge_index = np.empty(shape=[2, 1])
                for object_id, object_value in frame_value.items():

                    node_behavior = np.vstack([node_behavior, np.array(
                        [int(object_behavior_value) for object_behavior_id, object_behavior_value in
                         object_value['behavior'].items()])])
                    node_attributes = np.vstack([node_attributes, np.array(
                        [int(node_attributes_value) for node_attributes_id, node_attributes_value in
                         object_value['attributes'].items() if not node_attributes_id == 'old_id'])])

                    node_appearance = np.vstack([node_appearance, np.array(
                        [int(object_appearance_value) for object_appearance_id, object_appearance_value in
                         object_value['appearance'].items()])])

                    node_position = np.vstack([node_position, object_value['bbox']])

                    node_ground_truth = np.vstack([node_ground_truth, np.array([x if not x is None else 2 for x in object_value['ground_truth']])])

                node_features = np.delete(np.hstack([node_appearance, node_attributes, node_behavior]), 0, 0)
                if node_features.shape[0] > 1:
                    edge_index = np.hstack([edge_index, [[[j, i], [i, j]] for i in range(node_features.shape[0]) for j in range(i+1) if i != j][0]])

                graph_video.update({frame_id: Data(x=torch.as_tensor(node_features),
                            edge_index=torch.as_tensor(np.delete(edge_index, 0, 1), dtype=torch.long),
                            y=torch.as_tensor(np.delete(node_ground_truth, 0, 0)),
                            pos=torch.as_tensor(np.delete(node_position, 0, 0)),
                            width=torch.as_tensor(width),
                            height=torch.as_tensor(height))})
            self.graph_dataset.update({video_id: graph_video})

        print(self.graph_dataset.keys())
        '''
        # super().__init__(self.graph_dataset, batchSize, shuffle, validationSplit, numberOfWorkers)
