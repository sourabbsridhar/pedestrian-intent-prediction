from .graph_classification import intent_classifier
from .graph_regression import intent_predictor

def get_model(modelName):
    if modelName == "intent_classifier":
        model = intent_classifier()
    elif modelName == "intent_predictor":
        model = intent_predictor()
    else:
        raise NotImplementedError

    return model