
def score(data_conf, model_conf, **kwargs):
    """Python score method called by AOA framework batch mode

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """


# Uncomment this code if you want to deploy your model as a Web Service (Real-time / Interactive usage)
# class ModelScorer(object):
#    def __init__(self, config=None):
#        self.model = joblib.load('models/iris_knn.joblib')
#
#    def predict(self, data):
#        return self.model.predict([data])
#