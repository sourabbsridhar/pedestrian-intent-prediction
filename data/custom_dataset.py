# Implementation of dataset class for Custom Dataset

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset, Data, download_url, Batch
import os.path as osp



class old_JAAD(Dataset):
    """
    Class implementation for JAAD Dataset. 
    The class is inherited from nn.utils.data.Dataset
    """
    def __init__(self, annotations):
        """
        Method to initialize an object of type JAAD

        Parameters
        ----------
        self                    : JAAD
                                  Instance of the class JAAD
        annotations             : str
                                  Path to the annotations file
        imageDirectoryFormat    : str
                                  Format of the directory containing images
        train                   : bool
                                  Set to True to have data sampled for training process
        sequenceLength          : int
                                  Length of the frame sequence for each pedestrian
        prediction              : bool
                                  Set to True to have prediction made in the future
        predictionLength        : int
                                  Length of the prediction frame sequence for each pedestrian

        Returns
        -------
        self    : JAAD
                  Initialized object of class JAAD
        """
        self.annotations = annotations
        # self.imageDirectoryFormat = imageDirectoryFormat
        # self.training = train
        # self.sequenceLength = sequenceLength
        # self.prediction = prediction
        # self.predictionLength = predictionLength
        with open(self.annotations, "rb") as annotationsFile:
            self.annotations = pickle.load(annotationsFile)


    def __len__(self):
        """
        Method to return lenth attribute of the object.

        Parameters
        ----------
        self        : JAAD
                      Instance of the class JAAD

        Returns
        -------
        length      : int
                      Length attribute of the object
        """
        return len(self.annotations)

    def __getitem__(self):
        """
        Method to allow instances to use the indexer opterators.

        Parameters
        ----------
        self        : JAAD
                      Instance of the class JAAD
        id          : int
                      Index of the instance required

        Returns
        -------
        length      : int
                      Length attribute of the object
        """

        return self.annotations


    """
    Class implementation for JAAD Dataset. 
    The class is inherited from nn.utils.data.Dataset
    def __init__(self, split, isTrain, sequenceLength, datasetPath):
        Split               : float
                                percentage of training/testing data split. The value is the amount of training data
        isTrain             : bool
                                If the dataset should return training data(True) or testing data(False)
        sequenceLength      : float
                                Amount of buffer-frames past pedestrian appearance
        datasetPath         : str
                                percentage of training
        self.split = split
        self.isTrain = isTrain
        self.sequenceLength = sequenceLength
        self.imagePathFormat = datasetPath + r"\{}\{}.jpg"
        print(self.imagePathFormat)

    def __len__(self):
        return len(self.pedstrians)

    def __getitem__(self, idx, frameIdStart = -1):
        pedestrian = self.pedestrian[idx]

        if (frameIdStart != -1):
            frameStart = frameIdStart
        elif (self.isTrain) and (pedestrian['frame_start'] < pedestrian['frame_end'] - self.sequenceLength + 1):
            frameStart = random.randint(pedestrian['frame_start'], pedestrian['frame_end'] - self.sequenceLength + 1)
        else:
            frameStart = self.pedestrian['frame_start']

        if ((self.isTrain) or (frameIdStart != -1)):
            frameIds = [min(pedestrian['frame_end'] - 1, frameStart + i) for i in range(self.sequenceLength)]
        else:
            frameIds = range(pedestrian['frame_start'], pedestrian['frame_end'])

        gtAction = torch.tensor(np.stack([pedestrian['action'][frames] for frames in frameIds]))

        gtBbox = torch.Tensor(np.stack(pedestrian['bbox'][frames].astype(np.float) if len(pedestrian['bbox'][frames]) else np.zeros(4) for frames in frameIds))
        
        objectClass = torch.tensor(np.stack([data['object_class'][frames] for frames in frameIds]))
        
        objectBbox = torch.tensor(np.stack([data['object_bbox'][frames] for frames in frameIds]))

        frames = torch.tensor(np.array(frameIds))

        imagePath = [self.imagePathFormat.format(pedestrian['video'], frames + 1) for frames in frameIds]

        returnValue =   {
                            'GT_Action' : gtAction, 
                            'GT_Bbox' : gtBbox,
                            'Object_Class' : objectClass,
                            'Object_Bbox' : objectBbox,
                            'Frame_Ids' : frames,
                            'Image_Paths' : imagePath,
                        }

        return returnValue
    """



class JAAD(Dataset):
    def __init__(self, original_annotations, root, included_annotations, no_annotations_per_cat, appearence_size,
                 attributes_size, behavior_size, transform=None, pre_transform=None):

        with open(original_annotations, "rb") as annotationsFile:
            self.original_annotations = pickle.load(annotationsFile)
        self.appearence_size = appearence_size
        self.attributes_size = attributes_size
        self.behavior_size = behavior_size
        self.graph_annotations = {}
        self.included_annotations = included_annotations
        self.no_annotations_per_cat = no_annotations_per_cat
        self.dataset_c_nc = {}
        self.dataset_classification_no = {}
        self.total_pedestrians = 0
        super(JAAD, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return list(self.original_annotations.keys())

    @property
    def processed_file_names(self):
        return list(self.original_annotations.keys())

    def process(self):
        max_nodes = 0

        for video_id, video_value in self.original_annotations.items():
            # Video-wise initialization
            graph_video = []
            video_c_nc = [0, 0]  # [Crossing, Not crossing]
            video_classification_no = [0, 0]  # [Pedestrians, vehicles]
            width = video_value['width']
            height = video_value['height']

            for frame_id, frame_value in video_value['frames'].items():
                # Frame-wise initialization
                frame_classification = []  # 1 = pedestrian, 2 = vehicle
                node_bbox = np.empty(shape=[1, 4])  # width = abs(bbox[0] - bbox[2]), height = abs(bbox[1] - bbox[3])
                node_position = np.empty(shape=[1, 2])
                node_appearance = np.empty(shape=[1, self.no_annotations_per_cat[0]])
                node_attributes = np.empty(shape=[1, self.no_annotations_per_cat[2]])
                node_behavior = np.empty(shape=[1, self.no_annotations_per_cat[1]])
                node_ground_truth = np.empty(shape=[1, 3])
                edge_index = np.empty(shape=[2, 1])
                node_vehicle_features = np.empty(shape=[1, 1])

                for object_id, object_value in frame_value.items():

                    if 'behavior' in object_value.keys():
                        # Filters out pedestrians without the behaviour annotation and
                        # therefore only pedestrian with C/NC are left.

                        # Includes behavior annotations if included in regression-config.json
                        node_behavior = np.vstack([node_behavior, np.array(
                            [int(object_behavior_value) for object_behavior_id, object_behavior_value in
                             object_value['behavior'].items() if object_behavior_id in self.included_annotations])])

                        if object_value['behavior']['cross'] == 1:
                            video_c_nc[0] += 1
                        elif object_value['behavior']['cross'] == 0:
                            video_c_nc[1] += 1

                        # Includes attributes annotations if included in regression-config.json
                        node_attributes = np.vstack([node_attributes, np.array(
                            [int(object_attributes_value) for object_attributes_id, object_attributes_value in
                             object_value['attributes'].items() if object_attributes_id in self.included_annotations])])

                        # Includes appearance annotations if included in regression-config.json
                        node_appearance = np.vstack([node_appearance, np.array(
                            [int(object_appearance_value) for object_appearance_id, object_appearance_value in
                             object_value['appearance'].items() if object_appearance_id in self.included_annotations])])

                        frame_classification.append(1)
                        video_classification_no[0] += 1
                        node_ground_truth = np.vstack([node_ground_truth, np.array(
                            [x if not x is None else 2 for x in object_value['ground_truth']])])

                    elif 'vehicle_type' in object_value.keys():
                        node_vehicle_features = np.vstack([node_vehicle_features, np.array(
                            [int(object_value.get('vehicle_type'))])])

                        frame_classification.append(2)
                        video_classification_no[1] += 1

                    node_position = np.vstack([node_position, [object_value['bbox'][2]-object_value['bbox'][0],
                                                               object_value['bbox'][3]-object_value['bbox'][1]]])
                    node_bbox = np.vstack([node_bbox, object_value['bbox']])

                node_features = np.empty(shape=[node_behavior.shape[0], 1])
                if self.no_annotations_per_cat[0] == 0:
                    node_features = np.hstack([node_features, node_appearance])
                if self.no_annotations_per_cat[2] == 0:
                    node_features = np.hstack([node_features, node_attributes])
                if self.no_annotations_per_cat[1] == 0:
                    node_features = np.hstack([node_features, node_behavior])

                node_features = np.delete(node_features, 0, 0)

                if node_features.shape[0] > 1:

                    edge_index = np.hstack([edge_index,
                                            [[[j, i], [i, j]] for i in range(node_features.shape[0]) for j in
                                             range(i + 1) if i != j][0]])

                edge_index = np.delete(edge_index, 0, 1)

                if node_features.shape[0] > 0:
                    edge_index = np.hstack([edge_index,
                                            np.transpose(np.array([[i + node_features.shape[0], j]
                                                                   for i in range(node_vehicle_features.shape[0])
                                                                   for j in range(node_features.shape[0])]))])

                nodes = np.zeros(shape=(1, 12))#node_vehicle_features.shape[1] + node_features.shape[1]))
                if node_features.size != 0 and node_vehicle_features.size != 0:
                    nodes = np.vstack([
                        np.hstack([node_features,
                                   np.zeros(shape=(node_features.shape[0], node_vehicle_features.shape[1]))]),
                        np.hstack([np.zeros(shape=(node_vehicle_features.shape[0], node_features.shape[1])),
                                   node_vehicle_features])
                    ])
                    nodes = np.hstack([nodes, node_bbox])
                elif node_features.size != 0 and node_vehicle_features.size == 0:
                    nodes = node_features
                    nodes = np.hstack([nodes, node_bbox])

                graph_video.append(Data(x=torch.as_tensor(nodes),
                                        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                                        y=torch.as_tensor(np.delete(node_ground_truth, 0, 0)),
                                        pos=torch.as_tensor(np.delete(node_position, 0, 0)),
                                        classification=frame_classification,
                                        width=torch.as_tensor(width),
                                        height=torch.as_tensor(height)))

            if video_classification_no[0] != 0:
                self.graph_annotations.update({video_id: graph_video})
                self.dataset_c_nc.update({video_id: video_c_nc})
                self.dataset_classification_no.update({video_id: video_classification_no})
                self.total_pedestrians += video_classification_no[0]

                if self.pre_filter is not None and not self.pre_filter(graph_video):
                    continue

                if self.pre_transform is not None:
                    graph_video = self.pre_transform(graph_video)

                torch.save(graph_video, osp.join(self.processed_dir, '{}.pt'.format(video_id)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, '{}.pt'.format(list(self.original_annotations.keys())[idx])))
        return data

    def to_device(self, device):
        for video_id, video in self.graph_annotations.items():
            for idx, data in enumerate(video):
                video[idx] = data.to(device)
            self.graph_annotations.update({video_id: video})


    def split_dataset(self, validationSplit=0.2):

        # Check if input is correct
        if validationSplit == 0.0:
            return None, None

        # Create a list of all the videos
        idFull = list(self.original_annotations.keys())

        # Check the value of split and split the videos
        if(isinstance(validationSplit, int)):
            assert validationSplit > 0, "[ERROR] Number of samples is negative!"
            assert validationSplit < len(self.original_annotations), "[ERROR] Number of samples larger than the entire dataset!"
            validationLength = validationSplit
        else:
            validationLength = int(validationSplit * len(self.original_annotations))

        validationKeys = idFull[0:validationLength]
        trainingKeys = list(np.delete(idFull, np.arange(0, validationLength)))

        trainingDataset = {key: value for key, value in self.graph_annotations.items() if key in trainingKeys}
        validationDataset = {key: value for key, value in self.graph_annotations.items() if key in validationKeys}

        return trainingDataset, validationDataset

    def get_video_c_nc(self, video_id):
        if video_id in self.dataset_c_nc.keys():
            return self.dataset_c_nc[video_id]
        else:
            return None

    def get_video_classification_no(self, video_id):
        if video_id in self.dataset_classification_no.keys():
            return self.dataset_classification_no[video_id]
        else:
            return None



