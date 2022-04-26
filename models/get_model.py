from .graph_classification import intent_classifier
from .graph_regression import intent_predictor

def get_model(modelName):
    """
    Function interface to get neural network model.

    Parameters
    ----------
    modelName       : str
                      Name of the neural network model to be imported

    Returns
    -------
    model           : multiple
                      Class defining various neural network models
    """
    if modelName == "intent_classifier":
        model = intent_classifier()
    elif modelName == "intent_predictor":
        model = intent_predictor()
    else:
        raise NotImplementedError

    return model