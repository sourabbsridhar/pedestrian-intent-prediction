# Python program to write JSON
# to a file

def Update(config_name):
    import json

    if config_name == "regression-config.json":

        with open(config_name, "r") as jsonFile:
            config = json.load(jsonFile)

        include_annotations_from_appearance = ["pose_front",
                                               "pose_back",
                                               "pose_left",
                                               "pose_right",
                                               "clothes_below_knee",
                                               "clothes_upper_light",
                                               "clothes_upper_dark",
                                               "clothes_lower_light",
                                               "clothes_lower_dark",
                                               "backpack",
                                               "bag_hand",
                                               "bag_elbow",
                                               "bag_shoulder",
                                               "bag_left_side",
                                               "bag_right_side",
                                               "cap",
                                               "hood",
                                               "sunglasses",
                                               "umbrella",
                                               "phone",
                                               "baby",
                                               "object",
                                               "stroller_cart",
                                               "bicycle_motorcycle"
                                               ]

        include_annotations_from_behaviour = ["cross",
                                              "reaction",
                                              "hand_gesture",
                                              "look",
                                              "action",
                                              "nod"
                                              ]

        include_annotations_from_attributes = ["age",
                                               #"old_id",
                                               "num_lanes",
                                               "crossing",
                                               "gender",
                                               "crossing_point",
                                               "decision_point",
                                               "intersection",
                                               "designated",
                                               "signalized",
                                               "traffic_direction",
                                               "group_size",
                                               "motion_direction"
                                               ]
        no_annotations_per_cat = [len(include_annotations_from_appearance), len(include_annotations_from_behaviour), len(include_annotations_from_attributes)]
        include_annotations = include_annotations_from_appearance + include_annotations_from_behaviour + include_annotations_from_attributes

        # Data to be written
        config = {
            "name": "PedestrianIntentPrediction",
            "n_gpu": 1,
            "arch":
                {
                    "type": "social_stgcn",
                    "args": {}
                },
            "dataset":
                {
                    "type": "JAAD",
                    "args":
                        {
                            "original_annotations": "data/datasets/overall_frame_by_frame_database.pkl",
                            "root": "data/datasets/",
                            "included_annotations": include_annotations,
                            "no_annotations_per_cat": no_annotations_per_cat,
                            "appearence_size": 25,
                            "attributes_size": 12,
                            "behavior_size": 6
                        }
                },
            "dataLoader":
                {
                    "type": "JaadDataLoader",
                    "args":
                        {
                            "annotation_path": "data/datasets/overall_database.pkl",
                            "root": "data/datasets/",
                            "batchSize": 1,
                            "shuffle": False,
                            "validationSplit": 0.3,
                            "numberOfWorkers": 1,
                            "training": True
                        }
                },
            "optimizer":
                {
                    "type": "Adam",
                    "args":
                        {
                            "lr": 0.001,
                            "weight_decay": 0,
                            "amsgrad": True
                        }
                },
            "loss": "node_classification_loss",
            "metrics":
                [
                    "accuracy", "top_k_accuracy"
                ],
            "lr_scheduler":
                {
                    "type": "StepLR",
                    "args":
                        {
                            "step_size": 50,
                            "gamma": 0.1
                        }
                },
            "trainer":
                {
                    "epochs": 1,
                    "save_dir": "saved/",
                    "save_period": 1,
                    "verbosity": 2,
                    "monitor": "min val_loss",
                    "early_stop": 15,
                    "tensorboard": True
                },
            "model": {
                "type": "social_stgcn",
                "args": {
                    "input_feat": 17,
                    "Conv_outputs": [45, 40],
                    "LSTM_output": [35, 30, 25],
                    "K": 15,
                    "linear_output": 3
                }
            }
        }


        # Serializing json
        json_config = json.dumps(config, indent=4)

        # Writing to sample.json
        with open(config_name, "w") as outfile:
            outfile.write(json_config)